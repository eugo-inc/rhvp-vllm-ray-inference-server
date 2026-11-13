# SPDX-License-Identifier: Apache-2.0
"""The round loop: bring one side up, time its start, load it with GuideLLM, read its GPU power, take it down;
round after round in the plan's order, writing the results file after every round.

The loop knows nothing about Kubernetes or docker: a Backend does the platform work (kube.py, dockerapi.py, or a
fake one in the tests). A round that fails is recorded with its reason and a fixed failure code (ROUND_CODES), and
the run goes on; the comparison then has one pair fewer for the metrics that round would have given.

SIGTERM (a deleted Job, or the Job's time limit; SIGINT too) stops the run without waiting for the round: the round
is cut short and recorded as STOPPED, its side is taken down, the backend cleans up, and the results file is saved as
partial, all within the runner pod's grace period. On OpenShift the pattern's server is asked back first, before the
slow teardown, so a teardown that runs out of time cannot leave it at 0.

One run at a time. Before it touches anything (the results file, the status, the live server), the runner asks the
backend whether another comparison is running, and refuses to start if one is (RUN_ACTIVE): two runs would take the
GPU from each other, and the first to end would bring the live server back under the other.

The live server. The backend may scale the pattern's own server down for the run (kube.py) or stop a container
(dockerapi.py); the runner asks it back however the run ends, and records in run.server_restored whether that
worked. A server that could not be brought back makes the run exit 4, whatever its status, with the command that
brings it back in the log. A server that was running when the run began but had stopped by itself before the backend
came to take it away is left as it is: nothing brings it back, and the run records that in run.server_found_down, a
note, results.md and the log.

Results land in <results>/<run id>/: results.json (the file results.schema.json describes), results.md (the short
table), and rounds/<NN>-<side>-<kind>/ with the server log and the raw GuideLLM report. A run whose results.json
is already complete is not run again (so a Job that Argo CD creates a second time does nothing). See summary.py for
the short views a reader gets without the results volume.
"""

from __future__ import annotations

import json
import os
import signal
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from collections.abc import Callable, Mapping

import httpx

from . import analyze, energy, guidellm, logs, plan as plans, probe, report, summary

HARNESS = {"name": "rhvp_compare", "version": "1"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


#: Fixed failure codes. A round's `error_code`, null when the round is ok:
GPU_NOT_FREE = "GPU_NOT_FREE"          # another pod or program still held the GPU when the round was to start
SIDE_FAILED = "SIDE_FAILED"            # the side failed: bad image, pull failure, no node, or the server exited
SIDE_NEVER_READY = "SIDE_NEVER_READY"  # the side did not list the model, or gave no answer, before its time limit
LOAD_FAILED = "LOAD_FAILED"            # GuideLLM failed or timed out (after its one retry), or its report is unusable
STOPPED = "STOPPED"                    # the runner was told to stop (SIGTERM) during the round
RUNNER_ERROR = "RUNNER_ERROR"          # anything else: the platform's API, or a fault in the runner
ROUND_CODES = (GPU_NOT_FREE, SIDE_FAILED, SIDE_NEVER_READY, LOAD_FAILED, STOPPED, RUNNER_ERROR)
#: How many more times a side is started in one round when its server stopped at vLLM's start-up memory check
#: (logs.memory_profiling_failed); recorded in the round's start_retries. A second such failure is SIDE_FAILED.
START_RETRIES = 1
#: The run's `error_code`, null when every planned round finished:
ROUNDS_FAILED = "ROUNDS_FAILED"  # every round ran and some failed (status partial)
REQUESTS_FAILED = "REQUESTS_FAILED"  # every round ran, but at some load level of some round requests failed (or none
#                                      finished), so that level is left out (status partial)
RUN_TIMEOUT = "RUN_TIMEOUT"      # stopped near the Job's time limit (status partial); STOPPED for any other stop
RUNNER_FAILED = "RUNNER_FAILED"  # the run failed outside a round, for example before the first one (status failed)
PLAN_INVALID = "PLAN_INVALID"    # the config cannot be run; no results file, only the summary in the log and the
#                                  termination message
RUN_ACTIVE = "RUN_ACTIVE"        # another comparison was running, so this one did not start and changed nothing; no
#                                  results file, only the summary in the log and the termination message
RUN_CODES = (ROUNDS_FAILED, REQUESTS_FAILED, STOPPED, RUN_TIMEOUT, RUNNER_FAILED, PLAN_INVALID, RUN_ACTIVE)
#: Exit codes of `python -m rhvp_compare run`.
EXIT_COMPLETE, EXIT_PARTIAL, EXIT_CONFIG, EXIT_FAILED, EXIT_SERVER_NOT_RESTORED, EXIT_RUN_ACTIVE = 0, 1, 2, 3, 4, 5


NOT_RESTORED = "the live server was not brought back"
#: The start of the note a run keeps when its live server had stopped by itself before the backend came to take it
#: away (run.server_found_down).
FOUND_DOWN = "the live server was running when the run began but had stopped by itself"


class RoundError(Exception):
    """A round could not be measured; the message is safe to show and to store, the code is one of ROUND_CODES."""

    def __init__(self, message: str, code: str = RUNNER_ERROR) -> None:
        super().__init__(message)
        self.code = code


class RunActive(Exception):
    """Another comparison is running; the message names it. Raised before anything is changed."""


class Stopped(BaseException):
    """Raised by the SIGTERM handler inside a round, so a stop does not wait for the round to finish. A
    BaseException, so no `except Exception` on the way swallows it."""


class Sampler(Protocol):
    source: str

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def samples(self, start: float, end: float) -> tuple[list[energy.Sample], list[energy.Sample], str | None]:
        """(power samples, energy-counter samples, note) for one window, in seconds since the epoch."""


class Backend(Protocol):
    platform: str

    def prepare(self, plan: Mapping) -> None: ...

    def describe(self) -> dict: ...

    def gpu_free(self, plan: Mapping, side: Mapping) -> None: ...

    def start(self, plan: Mapping, round_: Mapping, side: Mapping, spec: Mapping) -> Any: ...

    def endpoint(self, handle: Any, deadline: float) -> str: ...

    def alive(self, handle: Any) -> str | None: ...

    def run_load(self, plan: Mapping, round_: Mapping, target: str, out_dir: Path) -> Path: ...

    def sampler(self, plan: Mapping, handle: Any) -> Sampler | None: ...

    def logs(self, handle: Any) -> str: ...

    def facts(self, handle: Any) -> dict: ...

    def stop(self, handle: Any) -> None: ...

    def cleanup(self) -> None: ...

    # Optional, each called only when the backend has it:
    # - `check_alone()`: raise RunActive when another comparison is running. Called before anything is changed.
    # - `restore_server() -> bool | None`: bring back the live server the backend took away in prepare(); True once
    #   it is back (and checked), None when there was nothing to bring back; raises when it could not. Called again
    #   until it has worked, so it must do nothing once the server is back.
    # - `restore_before_teardown` (bool): ask the server back as soon as a stop begins, before the round's side is
    #   taken down (Kubernetes: its pod waits for the GPU); otherwise after the teardown (docker: no scheduler).
    # - `on_stop()`: a stop has begun; shorten the waits of what is left (log reads, deletes) to fit the grace period,
    #   and keep prepare() from taking the live server away if it has not yet (dockerapi.py: a stop during the image
    #   pulls leaves the server running). Called from the signal handler, also while prepare() runs, so it only sets
    #   flags. An error it raises is ignored, so it sets the flag prepare() reads before anything that can fail.
    # - `server_found_down` (str or None, read after prepare()): the live server's state when prepare() came to take
    #   it away, when it was running at the start but had stopped by itself by then (docker: "exited", or "gone");
    #   the backend left it as it was. Recorded in run.server_found_down.
    # - `publish(run_id, status_text, result_text_or_None)` keeps the short status where a reader without the
    #   results volume finds it (kube.py: a ConfigMap). Called after every save.


def write_json(path: Path, data: Mapping) -> None:
    """Write atomically: a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def round_dir(round_: Mapping) -> str:
    return f"rounds/{round_['index']:02d}-{round_['side']}-{round_['kind']}"


class Runner:
    def __init__(self, plan: Mapping, backend: Backend, results_root: Path,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep,
                 client: Callable[[str], httpx.Client] = lambda url: httpx.Client(base_url=url),
                 deadline_s: float | None = None) -> None:
        self.plan = plan
        self.client = client
        self.backend = backend
        self.dir = Path(results_root) / plan["run_id"]
        self.path = self.dir / "results.json"
        self.clock, self.wall, self.sleep = clock, wall, sleep
        self.results: dict = {}
        self._stopping = False
        self._stop_begun = False
        self._interruptible = False
        self._publish_failed = False
        self.deadline_s = deadline_s  # the Job's time limit, to tell RUN_TIMEOUT from any other stop
        self._t_start = clock()

    # ---- the results file ----

    def _new_results(self) -> dict:
        return {
            "schema": plans.SCHEMA,
            "run": {"id": self.plan["run_id"], "name": self.plan["run_name"], "status": "running",
                    "error_code": None, "started_at": utc_now(), "finished_at": None,
                    "platform": self.backend.platform, "server_restored": None, "server_found_down": None,
                    "notes": [], "errors": []},
            "plan": self.plan,
            "provenance": {"harness": dict(HARNESS),
                           "load_tool": {"name": "guidellm", "image": self.plan["load"]["image"], "version": None},
                           "seeds": {"load": self.plan["load"]["seed"], "server": self.plan["server"]["seed"]},
                           "sides": {s["key"]: {"image": s["image"], "image_digest": None, "server_version": None}
                                     for s in self.plan["sides"]},
                           "gpu": None, "node": None, "nodes": [], "cluster": None},
            "rounds": [],
            "analysis": None,
        }

    def save(self) -> None:
        self.results["analysis"] = analyze.analyze(self.results)
        write_json(self.path, self.results)
        (self.dir / "results.md").write_text(report.markdown(self.results), encoding="utf-8")
        self._publish()

    def _publish(self) -> None:
        publish = getattr(self.backend, "publish", None)
        if publish is None:
            return
        try:
            publish(self.plan["run_id"], summary.compact(summary.status(self.results, self.wall())),
                    summary.result_text(self.results))
        except Exception as error:  # the results file on the volume is the record; this is a convenience
            if not self._publish_failed:
                self._publish_failed = True
                self.results["run"]["notes"].append(f"the run's status was not published: {type(error).__name__}: "
                                                    f"{error}")

    def already_done(self) -> bool:
        if not self.path.is_file():
            return False
        try:
            old = json.loads(self.path.read_text(encoding="utf-8"))
        except ValueError:
            return False
        return (old.get("run") or {}).get("status") == "complete"

    # ---- the run ----

    def run(self) -> dict:
        if self.already_done():
            self.results = json.loads(self.path.read_text(encoding="utf-8"))
            return self.results
        check_alone = getattr(self.backend, "check_alone", None)
        if check_alone is not None:
            check_alone()  # RunActive: another run is going; nothing here has been touched, and nothing is cleaned up
        if self.path.is_file():  # an earlier attempt that did not finish: keep it beside the new one
            os.replace(self.path, self.dir / f"results.attempt-{int(self.wall())}.json")
        self.results = self._new_results()
        self.save()
        previous: dict = {}
        for sig in (signal.SIGTERM, signal.SIGINT):  # SIGINT: Ctrl-C on scripts/compare-local.sh
            try:
                previous[sig] = signal.signal(sig, self._on_term)
            except ValueError:  # not the main thread (tests)
                break
        try:
            self.backend.prepare(self.plan)
            self._found_down()
            self._describe()
            for round_ in self.plan["rounds"]:
                if self._stopping:
                    self._begin_stop()
                    break
                self.results["rounds"].append(self.measure(round_))
                self.save()
            self._settle()
        except Stopped:  # a stop that landed outside a round's own handling
            self._begin_stop()
            self._settle()
        except Exception as error:  # recorded, then raised for the Job's exit code
            self.results["run"]["status"] = "failed"
            self.results["run"]["error_code"] = RUNNER_FAILED
            self.results["run"]["errors"].append(f"{type(error).__name__}: {error}")
            raise
        finally:
            self._restore()  # before the cleanup's deletes, and before the last save says how it went
            try:
                self.backend.cleanup()
            except Exception as error:  # recorded; the run's own outcome stands
                self.results["run"]["errors"].append(f"cleanup: {type(error).__name__}: {error}")
            self.results["run"]["finished_at"] = utc_now()
            self.save()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        return self.results

    def _restore(self) -> None:
        """Ask the backend to bring the live server back, and record whether it is back (run.server_restored)."""
        restore = getattr(self.backend, "restore_server", None)
        run = self.results["run"]
        if restore is None or run.get("server_restored") is True:
            return
        try:
            back = restore()
        except Exception as error:  # recorded, and the exit code says so (EXIT_SERVER_NOT_RESTORED)
            run["server_restored"] = False
            message = f"{NOT_RESTORED}: {error}"
            if message not in run["errors"]:
                run["errors"].append(message)
            return
        if back and run.get("server_restored") is False:  # a later try went through: keep it as a note
            run["notes"] += [f"{e}; a later try brought it back" for e in run["errors"] if e.startswith(NOT_RESTORED)]
            run["errors"] = [e for e in run["errors"] if not e.startswith(NOT_RESTORED)]
        run["server_restored"] = True if back else None

    def _found_down(self) -> None:
        """Record a live server that had stopped by itself before the backend came to take it away
        (run.server_found_down, and a note). The backend left it as it was, so nothing brings it back after the run
        and run.server_restored stays null."""
        state = getattr(self.backend, "server_found_down", None)
        if not state:
            return
        run = self.results["run"]
        run["server_found_down"] = state
        run["notes"].append(f"{FOUND_DOWN} before the runner came to take it away (its state then: {state}); it was "
                            f"left as it was, and the run does not start it again")

    def _backend_on_stop(self) -> None:
        on_stop = getattr(self.backend, "on_stop", None)
        if on_stop is not None:
            try:
                on_stop()
            except Exception:
                # Ignored, and the stop goes on. A backend sets the flag that keeps prepare() from taking the live
                # server away before anything that can fail, so an error here costs only the shorter waits; one
                # raised before that flag would let prepare() go on to stop the server.
                pass

    def _begin_stop(self) -> None:
        """A stop has begun: bring the live server back now when the backend can do that before the side is taken
        down (on OpenShift its pod waits for the GPU), so a teardown that runs out of time cannot leave it at 0."""
        if self._stop_begun:
            return
        self._stop_begun = True
        self._backend_on_stop()
        if getattr(self.backend, "restore_before_teardown", False):
            self._restore()

    def _settle(self) -> None:
        """The run's status and failure code once the loop has ended."""
        run = self.results["run"]
        failed = [r for r in self.results["rounds"] if r["status"] != "ok"]
        done = len(self.results["rounds"]) == len(self.plan["rounds"])
        if self._stopping:
            late = self.deadline_s is not None and self.clock() - self._t_start >= self.deadline_s - 300
            run["status"], run["error_code"] = "partial", (RUN_TIMEOUT if late else STOPPED)
            run["errors"].append("stopped near the Job's time limit" if late else "stopped by SIGTERM")
        elif failed or not done:
            run["status"], run["error_code"] = "partial", ROUNDS_FAILED
        elif summary.levels_unusable(self.results):
            run["status"], run["error_code"] = "partial", REQUESTS_FAILED
        else:
            run["status"], run["error_code"] = "complete", None

    def _on_term(self, signum, frame) -> None:  # noqa: ARG002 (signal handler signature)
        """Stop: inside a round, cut it short now (Stopped); between rounds, the loop stops before the next one. The
        backend's on_stop is called here at once, not left to _begin_stop: a stop that lands while prepare() runs (the
        image pulls under docker can take minutes) must reach the backend before prepare() goes on to take the live
        server away, and dockerapi.py then leaves the server running. Without this call, a stop during the pulls
        would stop the server and the run's end would start it again. on_stop only sets flags, so it is safe in a
        signal handler."""
        self._stopping = True
        self._backend_on_stop()
        if self._interruptible:
            self._interruptible = False
            raise Stopped()

    def _describe(self) -> None:
        try:
            facts = self.backend.describe() or {}
        except Exception as error:  # provenance is best effort
            self.results["run"]["notes"].append(f"platform facts not read: {type(error).__name__}")
            return
        for key in ("cluster", "node", "gpu"):
            if facts.get(key):
                self.results["provenance"][key] = facts[key]

    # ---- one round ----

    def measure(self, round_: Mapping) -> dict:
        side = next(s for s in self.plan["sides"] if s["key"] == round_["side"])
        spec = plans.side_spec(self.plan, side)
        rel = round_dir(round_)
        out_dir = self.dir / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        record: dict = {"index": round_["index"], "side": round_["side"], "kind": round_["kind"],
                        "pair": round_["pair"], "status": "failed", "error": None, "error_code": None,
                        "started_at": utc_now(), "finished_at": None, "node": None, "image_digest": None,
                        "server_version": None, "startup": None, "levels": [], "metrics": {}, "artifacts": {},
                        "start_retries": 0}
        held: dict = {"handle": None}
        try:
            self._interruptible = True  # SIGTERM from here on raises Stopped
            try:
                if self._stopping:
                    raise Stopped()
                self._round(round_, side, spec, record, out_dir, rel, held)
                record["status"] = "ok"
            finally:
                self._interruptible = False
        except Stopped:
            record["error"], record["error_code"] = "stopped by SIGTERM during the round", STOPPED
            self._begin_stop()
        except RoundError as error:
            record["error"], record["error_code"] = str(error), error.code
        except probe.StartFailed as error:
            record["error"], record["error_code"] = str(error), SIDE_NEVER_READY if error.never_ready else SIDE_FAILED
        except (guidellm.LoadError, guidellm.ReportError) as error:
            record["error"], record["error_code"] = str(error), LOAD_FAILED
        except Exception as error:  # a fault in one round must not lose the rounds before it
            record["error"], record["error_code"] = f"{type(error).__name__}: {error}", RUNNER_ERROR
        finally:
            if self._stopping:  # also a stop that came in after the round's own work: the server back first
                self._begin_stop()
            if held["handle"] is not None:
                self._after(held["handle"], record, out_dir, rel)
            record["finished_at"] = utc_now()
        record.pop("_reported_version", None)
        record.pop("_started_epoch", None)
        return record

    def _round(self, round_: Mapping, side: Mapping, spec: Mapping, record: dict, out_dir: Path, rel: str,
               held: dict) -> None:
        """Wait for the GPU, start the side, time its start, load it; any failure raises. A server that stopped at
        vLLM's start-up memory check is started once more (START_RETRIES), and its start is timed from that start."""
        timeout = self.plan["server"]["warmup_timeout_s" if round_["kind"] == plans.WARMUP else "ready_timeout_s"]
        model = self.plan["model"]["id"]
        while True:
            self.backend.gpu_free(self.plan, side)
            record["_started_epoch"] = self.wall()
            t0 = self.clock()
            handle = held["handle"] = self.backend.start(self.plan, round_, side, spec)
            deadline = t0 + timeout
            target = self.backend.endpoint(handle, deadline)
            alive = lambda: self.backend.alive(handle)  # noqa: E731
            try:
                with self.client(target) as client:
                    ready_s = probe.wait_ready(client, model, t0, deadline, alive)
                    first_s = probe.first_answer(client, model, t0, deadline, alive,
                                                 self.plan["server"]["first_answer_tokens"],
                                                 self.plan["load"]["request_format"])
                    record["_reported_version"] = probe.reported_version(client)
                break
            except probe.StartFailed as error:
                if error.never_ready or record["start_retries"] >= START_RETRIES:
                    raise
                if not self._start_again(handle, record, out_dir, rel, error):
                    raise
                held["handle"] = None  # torn down; the next start sets it
        record["startup"] = {"ready_s": round(ready_s, 3), "first_answer_s": round(first_s, 3),
                             "clock": "runner, from just before the server was asked for", "breakdown": []}
        record["metrics"].update({"startup.ready_s": ready_s, "startup.first_answer_s": first_s})
        if round_["kind"] != plans.WARMUP:
            record["levels"] = self._load(round_, handle, target, out_dir, rel, record)
            record["metrics"].update(guidellm.flat_metrics(record["levels"]))

    def _start_again(self, handle, record: dict, out_dir: Path, rel: str, error: Exception) -> bool:
        """When the server's log shows vLLM's memory-profiling check failing: keep that log as server.attempt-N.log,
        take the side down and return True (the round starts it again). Anything else: False, and the round fails."""
        try:
            text = self.backend.logs(handle)
        except Exception:
            return False
        if not logs.memory_profiling_failed(text):
            return False
        record["start_retries"] += 1
        name = f"server.attempt-{record['start_retries']}.log"
        (out_dir / name).write_text(text, encoding="utf-8")
        record["artifacts"][f"server_log_attempt_{record['start_retries']}"] = f"{rel}/{name}"
        record.setdefault("notes", []).append(
            f"{error} at vLLM's start-up memory check ({logs.MEMORY_PROFILING_ERROR}); the side was started once "
            f"more, and its start is timed from then ({name} is the first try's log)")
        self.backend.stop(handle)
        return True

    def _load(self, round_: Mapping, handle, target: str, out_dir: Path, rel: str, record: dict) -> list[dict]:
        sampler = None
        if self.plan["energy"]["source"] != "none":
            try:
                sampler = self.backend.sampler(self.plan, handle)
                if sampler is not None:
                    sampler.start()
            except Exception as error:
                record.setdefault("notes", []).append(f"power sampling did not start: {type(error).__name__}")
                sampler = None
        try:
            try:
                report_path = self.backend.run_load(self.plan, round_, target, out_dir)
            except guidellm.LoadError as error:
                if not error.timed_out:
                    raise
                # GuideLLM v0.8.1 now and then stops making progress at the end of a level (seen once in about
                # twenty runs against a stand-in server). One more try, against the same server, is recorded on the
                # round.
                record.setdefault("notes", []).append(f"{error}; GuideLLM was run once more against the same server")
                if (out_dir / "guidellm.log").exists():
                    (out_dir / "guidellm.log").replace(out_dir / "guidellm.attempt-1.log")
                report_path = self.backend.run_load(self.plan, round_, target, out_dir)
        finally:
            if sampler is not None:
                try:
                    sampler.stop()
                except Exception:  # the samples taken so far still count
                    pass
        record["artifacts"]["load_report"] = f"{rel}/{report_path.name}"
        raw = json.loads(report_path.read_text(encoding="utf-8"))
        version = guidellm.version(raw)
        if version:
            self.results["provenance"]["load_tool"]["version"] = version
        level_list = guidellm.levels(raw)
        self._energy(level_list, sampler)
        return level_list

    def _energy(self, level_list: list[dict], sampler) -> None:
        source = self.plan["energy"]["source"]
        if source == "none":
            for level in level_list:
                level["energy"] = energy.none_entry("none", "energy is not measured in this run")
            return
        if sampler is None:
            for level in level_list:
                level["energy"] = energy.none_entry(source, "no power source for this platform")
            return
        settle = self.plan["energy"]["settle_s"] if sampler.source == "dcgm" else 0
        last_end = max((lv["end_epoch"] for lv in level_list if lv.get("end_epoch")), default=None)
        if settle and last_end is not None:
            wait = last_end + settle - self.wall()
            if wait > 0:
                self.sleep(wait)
        for level in level_list:
            start, end = level.get("start_epoch"), level.get("end_epoch")
            if start is None or end is None:
                level["energy"] = energy.none_entry(sampler.source, "the load report has no time window")
                continue
            try:
                power, counter, note = sampler.samples(start, end)
            except Exception as error:
                level["energy"] = energy.none_entry(sampler.source, f"power samples not read: {error}")
                continue
            result = energy.window_power(power, start, end, self.plan["energy"]["min_samples"], counter)
            if result.mean_w is None and note:
                result = energy.Power(None, result.samples, None, f"{result.note}; {note}")
            level["energy"] = energy.level_energy(level, result, sampler.source)

    def _after(self, handle, record: dict, out_dir: Path, rel: str) -> None:
        """Logs, facts and teardown, whatever happened in the round."""
        try:
            text = self.backend.logs(handle)
            (out_dir / "server.log").write_text(text, encoding="utf-8")
            record["artifacts"]["server_log"] = f"{rel}/server.log"
            if record.get("startup") is not None:
                record["startup"]["breakdown"] = logs.breakdown(text, record.get("_started_epoch") or 0.0)
            version = record.get("_reported_version")
            source = "endpoint" if version else None
            if not version:
                version = logs.server_version(logs.lines(text))
                source = "log" if version else None
            if version:
                record["server_version"] = {"vllm": version, "source": source}
        except Exception as error:
            record.setdefault("notes", []).append(f"server log not read: {type(error).__name__}")
        try:
            facts = self.backend.facts(handle) or {}
            record["node"] = facts.get("node_name")
            record["image_digest"] = facts.get("image_digest")
            prov = self.results["provenance"]
            side = prov["sides"][record["side"]]
            if record["image_digest"] and not side.get("image_digest"):
                side["image_digest"] = record["image_digest"]
            if record.get("server_version") and not side.get("server_version"):
                side["server_version"] = record["server_version"]
            for key in ("node", "gpu"):
                if facts.get(key) and not prov.get(key):
                    prov[key] = facts[key]
            seen = {"name": facts.get("node_name"), "gpu": (facts.get("gpu") or {}).get("product")}
            if (seen["name"] or seen["gpu"]) and seen not in prov.setdefault("nodes", []):
                prov["nodes"].append(seen)  # every node and GPU the run used; `node` and `gpu` keep the first
        except Exception as error:
            record.setdefault("notes", []).append(f"facts not read: {type(error).__name__}")
        try:
            self.backend.stop(handle)
        except Exception as error:
            record.setdefault("notes", []).append(f"teardown: {type(error).__name__}: {error}")


def _finish(results: Mapping, termination_log: str | None) -> None:
    """The run's last words: the results file as one line between marker lines in the log, and the short summary as
    the container's termination message (when the platform gives a path for it)."""
    for line in summary.log_lines(results):
        print(line, flush=True)
    if termination_log:
        try:
            Path(termination_log).write_text(summary.termination(results), encoding="utf-8")
        except OSError as error:
            print(f"note: the termination message was not written ({type(error).__name__})", flush=True)


def exit_code(results: Mapping) -> int:
    """0 complete, 1 partial, 3 failed; 4 whenever the live server was not brought back."""
    run = results.get("run") or {}
    if run.get("server_restored") is False:
        return EXIT_SERVER_NOT_RESTORED
    return {"complete": EXIT_COMPLETE, "partial": EXIT_PARTIAL}.get(run.get("status"), EXIT_FAILED)


def main_run(config_path: str, results_root: str, backend: Backend, env: Mapping[str, str] | None = None) -> int:
    """Build the plan, run it, print where the results are. The exit code: 0 when every round finished, 1 for a
    partial run (so its Job reads Failed), 3 for a run that failed outside a round, 4 when the live server was not
    brought back (whatever else happened), 5 when another comparison was running (nothing changed). A config error
    raises PlanError (exit 2)."""
    env = os.environ if env is None else env
    termination_log = env.get("RHVP_COMPARE_TERMINATION_LOG") or None
    try:
        the_plan = plans.load(config_path, backend.platform)
    except plans.PlanError as error:
        _finish(summary.not_started(PLAN_INVALID, str(error)), termination_log)
        raise
    floor = the_plan["floor"]
    print(f"run {the_plan['run_id']}: {len(the_plan['rounds'])} rounds, {the_plan['pairs']} scored pairs, "
          f"{floor['family_size']} primary metrics of {len(the_plan['metrics'])}", flush=True)
    if not floor["reachable"]:
        print(f"note: with {the_plan['pairs']} pairs the smallest adjusted p-value is {floor['min_adjusted_p']:.3g}; "
              f"no primary metric can be FASTER or SLOWER below {floor['pairs_needed']} pairs", flush=True)
    deadline = env.get("RHVP_COMPARE_DEADLINE_SECONDS")
    runner = Runner(the_plan, backend, Path(results_root), deadline_s=float(deadline) if deadline else None)
    try:
        runner.run()
    except RunActive as error:
        print(f"not started: {error}", flush=True)
        _finish(summary.not_started(RUN_ACTIVE, str(error), the_plan["run_id"]), termination_log)
        return EXIT_RUN_ACTIVE
    except Exception as error:  # recorded in the results file (status failed, RUNNER_FAILED) by Runner.run
        traceback.print_exc()
        if not runner.results:  # before the results file existed
            _finish(summary.not_started(RUNNER_FAILED, f"{type(error).__name__}: {error}", the_plan["run_id"]),
                    termination_log)
            return EXIT_FAILED
    results = runner.results
    _finish(results, termination_log)
    run = results["run"]
    print(f"results: {runner.path} (status {run['status']}, code {run.get('error_code')})", flush=True)
    for row in (results.get("analysis") or {}).get("metrics") or []:
        print(f"  {row['label']} ({row.get('role', 'primary')}): {row['verdict']} (pairs {row['n_pairs']})",
              flush=True)
    for line in run.get("errors") or []:
        print(f"error: {line}", flush=True)
    for line in run.get("notes") or []:
        if line.startswith(FOUND_DOWN):
            print(f"note: {line}", flush=True)
    return exit_code(results)
