# SPDX-License-Identifier: Apache-2.0
"""The launch loop: bring one side up, time its start, load it with GuideLLM, read its GPU power, take it down;
launch after launch in the plan's order (a round runs every side and its twin once), writing the results file after
every launch.

The loop knows nothing about Kubernetes or docker: a Backend does the platform work (kube.py, dockerapi.py, or a
fake one in the tests). A launch that fails is recorded with its reason and a fixed failure code (LAUNCH_CODES), and
the run goes on; the comparison then has one pair fewer for the cases that launch would have given. A twin is
launched exactly like its side (the same image, arguments and settings); only its name in the lineup differs.

GuideLLM now and then stops making progress at the end of a level (seen once in about twenty runs against a stand-in
server). A launch whose GuideLLM timed out runs once more (LOAD_RETRIES): the side is taken down, started again from
the GPU check and timed from that new start, and GuideLLM runs with the same requests (the same load seed). Sent
again to the same server, those requests could find their prompts in its prefix cache, which no other launch has.
Every figure of the launch then comes from that second try; the first try's files are kept as *.try-1.*, its notes
start with "try 1:", its server records are marked discarded, and nothing it measured stays in the launch's figures,
also when the launch then fails or is stopped (the launch's teardown finishes a discard that a failed stop or a
SIGTERM cut short). The launch's `load_timeouts` counts every time-out and `load_retries` the loads that ran again; a
second time-out is LOAD_FAILED. The analysis counts every time-out against its side, so a server that hangs under
load is never hidden behind a launch that finished on its second try.

The scale round (plan mode scaleout, every scored launch; a warm-up launch is the same in both modes). The side's
first replica starts and is timed as in mode levels; a router (router.py, least-busy) is put in front of it, and
GuideLLM's surge goes through the router. At scale_out.step_after_s after the router saw the surge's first request
(the forced step), the runner starts a second replica of the same side, times it from its start to its first answer,
and adds it to the router. The launch keeps how many requests each replica got (`scale.router`), so the new
replica's share of the surge is on record. A second replica that never answers fails the launch (SIDE_NEVER_READY);
one that answers only after the surge has ended is still timed, with a share of 0. A twin's scale round is its
side's. When GuideLLM times out, the launch runs once more as in mode levels (LOAD_RETRIES): GuideLLM, the router and
the replicas that had started are taken down, and a new first replica is started and timed, then the surge runs with
the same requests and a new second replica.

Every server a launch starts has a server record in the launch's `servers` (servers.py): what the runner ran, and the
environment the server process saw when the backend can read it (`server_facts`).

SIGTERM (a deleted Job, or the Job's time limit; SIGINT too) stops the run without waiting for the launch: the
launch is cut short and recorded as STOPPED, its side is taken down, the backend cleans up, and the results file is
saved as partial, all within the runner pod's grace period. On OpenShift the pattern's server is asked back first,
before the slow teardown, so a teardown that runs out of time cannot leave it at 0.

One run at a time. Before it touches anything (the results file, the status, the live server), the runner asks the
backend whether another comparison is running, and refuses to start if one is (RUN_ACTIVE): two runs would take the
GPU from each other, and the first to end would bring the live server back under the other.

The live server. The backend may scale the pattern's own server down for the run (kube.py) or stop a container
(dockerapi.py); the runner asks it back however the run ends, and records in run.server_restored whether that
worked. A server that could not be brought back makes the run exit 4, whatever its status, with the command that
brings it back in the log. A server that was running when the run began but had stopped by itself during the image
pulls, before the backend would have stopped it (also in a run stopped during the pulls), is left as it is: nothing
brings it back, and the run records that in run.server_found_down, a note, results.md and the log.

Results land in <results>/<run id>/: results.json (the file results.schema.json describes), results.md (the short
table), and launches/<NN>-<side>-<kind>/ with the server log and the raw GuideLLM report. A run whose results.json
is already complete, under this schema, is not run again (so a Job that Argo CD creates a second time does
nothing). See summary.py for the short views a reader gets without the results volume.
"""

from __future__ import annotations

import json
import os
import re
import signal
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from collections.abc import Callable, Iterator, Mapping

import httpx

from . import analyze, energy, guidellm, logs, plan as plans, probe, report, servers, stats, summary

HARNESS = {"name": "rhvp_compare", "version": "1"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


#: Fixed failure codes. A launch's `error_code`, null when the launch is ok:
GPU_NOT_FREE = "GPU_NOT_FREE"          # another pod or program still held the GPU when the launch was to start
SIDE_FAILED = "SIDE_FAILED"            # the side failed: bad image, pull failure, no node, or the server exited
SIDE_NEVER_READY = "SIDE_NEVER_READY"  # the side did not list the model, or gave no answer, before its time limit
LOAD_FAILED = "LOAD_FAILED"            # GuideLLM failed, or timed out in both tries of the launch, or its report is
#                                        unusable
STOPPED = "STOPPED"                    # the runner was told to stop (SIGTERM) during the launch
RUNNER_ERROR = "RUNNER_ERROR"          # anything else: the platform's API, or a fault in the runner
LAUNCH_CODES = (GPU_NOT_FREE, SIDE_FAILED, SIDE_NEVER_READY, LOAD_FAILED, STOPPED, RUNNER_ERROR)
#: How many more times a server is started in one launch when it stopped at vLLM's start-up memory check
#: (logs.memory_profiling_failed): the side, and in mode scaleout also the replica the forced step adds. Recorded in
#: the launch's start_retries and in that server's record. A second such failure of the same start is SIDE_FAILED. A
#: launch whose load runs again (LOAD_RETRIES) starts its servers anew, and each new start gets its own retry.
START_RETRIES = 1
#: How many more times a launch runs after its GuideLLM timed out (LoadError.timed_out), so one stall of the load tool
#: is not a failure of the side. The retry takes the side's servers down and starts the launch again (see the
#: module's docstring); the launch's load_timeouts counts the time-out at once, its load_retries only once the second
#: load has begun. A second time-out is LOAD_FAILED.
LOAD_RETRIES = 1
#: The run's `error_code`, null when every planned launch finished:
LAUNCHES_FAILED = "LAUNCHES_FAILED"  # every launch ran and some failed (status partial)
REQUESTS_FAILED = "REQUESTS_FAILED"  # every launch ran, but at some load level of some launch requests failed (or
#                                      none finished), so that level is left out (status partial)
RUN_TIMEOUT = "RUN_TIMEOUT"      # stopped near the Job's time limit (status partial); STOPPED for any other stop
RUNNER_FAILED = "RUNNER_FAILED"  # the run failed outside a launch, for example before the first one (status failed)
PLAN_INVALID = "PLAN_INVALID"    # the config cannot be run; no results file, only the summary in the log and the
#                                  termination message
RUN_ACTIVE = "RUN_ACTIVE"        # another comparison was running, so this one did not start and changed nothing; no
#                                  results file, only the summary in the log and the termination message
RUN_CODES = (LAUNCHES_FAILED, REQUESTS_FAILED, STOPPED, RUN_TIMEOUT, RUNNER_FAILED, PLAN_INVALID, RUN_ACTIVE)
#: Exit codes of `python -m rhvp_compare run`.
EXIT_COMPLETE, EXIT_PARTIAL, EXIT_CONFIG, EXIT_FAILED, EXIT_SERVER_NOT_RESTORED, EXIT_RUN_ACTIVE = 0, 1, 2, 3, 4, 5


NOT_RESTORED = "the live server was not brought back"
#: The start of the note a run keeps when its live server had stopped by itself before the backend would have stopped
#: it (run.server_found_down).
FOUND_DOWN = "the live server was running when the run began but had stopped by itself"


class RoundError(Exception):
    """A launch could not be measured; the message is safe to show and to store, the code is one of LAUNCH_CODES."""

    def __init__(self, message: str, code: str = RUNNER_ERROR) -> None:
        super().__init__(message)
        self.code = code


class RunActive(Exception):
    """Another comparison is running; the message names it. Raised before anything is changed."""


class Stopped(BaseException):
    """Raised by the SIGTERM handler inside a launch, so a stop does not wait for the launch to finish. A
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

    def start(self, plan: Mapping, launch: Mapping, side: Mapping, spec: Mapping) -> Any: ...

    def endpoint(self, handle: Any, deadline: float) -> str: ...

    def alive(self, handle: Any) -> str | None: ...

    def run_load(self, plan: Mapping, launch: Mapping, target: str, out_dir: Path) -> Path: ...

    # The scale round also needs (plan mode scaleout):
    # - `start(plan, launch, side, spec, replica=2)`: the launch's second replica, next to the first (`replica` is
    #   passed only for 2);
    # - `router(plan, targets, out_dir)`: a router in front of `targets`, with `url`, `add(target) -> dict`,
    #   `stats() -> dict` and `stop()` (router.Process);
    # - `start_load(plan, launch, target, out_dir)`: GuideLLM started in the background (guidellm.Running).

    def sampler(self, plan: Mapping, handle: Any) -> Sampler | None: ...

    def logs(self, handle: Any) -> str: ...

    def facts(self, handle: Any) -> dict: ...

    def stop(self, handle: Any) -> None: ...

    def cleanup(self) -> None: ...

    # Optional, each called only when the backend has it:
    # - `check_alone()`: raise RunActive when another comparison is running. Called before anything is changed.
    # - `check_plan(plan)`: raise PlanError for a plan this backend cannot run as declared (dockerapi.py: a
    #   scaleOut.sharedGpu that does not match where the replicas would run). Called by main_run before anything
    #   is changed, so the run ends PLAN_INVALID.
    # - `restore_server() -> bool | None`: bring back the live server the backend took away in prepare(); True once
    #   it is back (and checked), None when there was nothing to bring back; raises when it could not. Called again
    #   until it has worked, so it must do nothing once the server is back.
    # - `restore_before_teardown` (bool): ask the server back as soon as a stop begins, before the launch's side is
    #   taken down (Kubernetes: its pod waits for the GPU); otherwise after the teardown (docker: no scheduler).
    # - `on_stop()`: a stop has begun; shorten the waits of what is left (log reads, deletes) to fit the grace period,
    #   and keep prepare() from taking the live server away if it has not yet (dockerapi.py: a stop during the image
    #   pulls leaves the server running). Called from the signal handler, also while prepare() runs, so it only sets
    #   flags. An error it raises is ignored, so it sets the flag prepare() reads before anything that can fail.
    # - `server_found_down` (str or None, read after prepare()): the live server's state at prepare()'s last read,
    #   when it was running at the start but had stopped by itself by then (docker: "exited", or "gone"), also in a
    #   run stopped during the pulls; the backend left it as it was. Recorded in run.server_found_down.
    # - `notes` (list of str, read after prepare()): what prepare() could not do and did not need to, such as a state
    #   read that failed during a stop; each becomes a note of the run.
    # - `publish(run_id, status_text, result_text_or_None)` keeps the short status where a reader without the
    #   results volume finds it (kube.py: a ConfigMap). Called after every save.
    # - `server_facts(handle) -> dict`: what a server ran with, read while it runs: `env` (the server process's
    #   environment) and `env_source` (servers.ENV_SOURCES), `image_entrypoint` and `image_cmd`, and the GPU request
    #   the server got (`gpus`) with the UUIDs it reached (`gpu_uuids`).


def write_json(path: Path, data: Mapping) -> None:
    """Write atomically: a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def launch_dir(launch: Mapping) -> str:
    return f"launches/{launch['index']:02d}-{plans.token(launch['side'])}-{launch['kind']}"


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
            "launches": [],
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
        return old.get("schema") == plans.SCHEMA and (old.get("run") or {}).get("status") == "complete"

    # ---- the run ----

    def run(self) -> dict:
        if self.already_done():
            self.results = json.loads(self.path.read_text(encoding="utf-8"))
            return self.results
        check_alone = getattr(self.backend, "check_alone", None)
        if check_alone is not None:
            check_alone()  # RunActive: another run is going; nothing here has been touched, and nothing is cleaned up
        if self.path.is_file():  # an earlier attempt that did not finish, or one of an earlier schema: keep it
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
            for launch in self.plan["launches"]:
                if self._stopping:
                    self._begin_stop()
                    break
                self.results["launches"].append(self.measure(launch))
                self.save()
            self._settle()
        except Stopped:  # a stop that landed outside a launch's own handling
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
        """Record the backend's notes from prepare(), and a live server that had stopped by itself before the
        backend would have stopped it (run.server_found_down, and a note). The backend left it as it was, so nothing
        brings it back after the run and run.server_restored stays null."""
        run = self.results["run"]
        run["notes"] += [str(note) for note in getattr(self.backend, "notes", None) or []]
        state = getattr(self.backend, "server_found_down", None)
        if not state:
            return
        run["server_found_down"] = state
        run["notes"].append(f"{FOUND_DOWN} during the image pulls, before the runner would have stopped it (its state "
                            f"then: {state}); it was left as it was, and the run does not start it again")

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
        failed = [r for r in self.results["launches"] if r["status"] != "ok"]
        done = len(self.results["launches"]) == len(self.plan["launches"])
        if self._stopping:
            late = self.deadline_s is not None and self.clock() - self._t_start >= self.deadline_s - 300
            run["status"], run["error_code"] = "partial", (RUN_TIMEOUT if late else STOPPED)
            run["errors"].append("stopped near the Job's time limit" if late else "stopped by SIGTERM")
        elif failed or not done:
            run["status"], run["error_code"] = "partial", LAUNCHES_FAILED
        elif summary.levels_unusable(self.results):
            run["status"], run["error_code"] = "partial", REQUESTS_FAILED
        else:
            run["status"], run["error_code"] = "complete", None

    def _on_term(self, signum, frame) -> None:  # noqa: ARG002 (signal handler signature)
        """Stop: inside a launch, cut it short now (Stopped); between launches, the loop stops before the next one. The
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

    @contextmanager
    def _held_off(self) -> Iterator[None]:
        """A few quick steps that a stop must not cut half-way (renaming a discarded try's files): a SIGTERM that comes
        in meanwhile is raised as Stopped right after them."""
        was = self._interruptible
        self._interruptible = False
        try:
            yield
        finally:
            self._interruptible = was
        if was and self._stopping:
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

    # ---- one launch ----

    def measure(self, launch: Mapping) -> dict:
        side = next(s for s in self.plan["sides"] if s["key"] == launch["of"])  # a twin launches its side
        spec = plans.side_spec(self.plan, side)
        rel = launch_dir(launch)
        out_dir = self.dir / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        record: dict = {"index": launch["index"], "round": launch["round"], "pos": launch["pos"],
                        "side": launch["side"], "of": launch["of"], "kind": launch["kind"], "status": "failed",
                        "error": None, "error_code": None, "started_at": utc_now(), "finished_at": None,
                        "node": None, "image_digest": None, "server_version": None, "startup": None, "levels": [],
                        "values": {}, "artifacts": {}, "servers": [], "starts": 0, "start_retries": 0,
                        "load_timeouts": 0, "load_retries": 0}
        # Everything the launch has running, so the teardown takes it down whatever happened: the servers in start
        # order (`replicas`, the first replica first), GuideLLM in the background and the router (mode scaleout), and
        # a discard of a try that is not finished yet (`discard`, see _discard_try).
        held: dict = {"replicas": [], "load": None, "router": None, "discard": None}
        try:
            self._interruptible = True  # SIGTERM from here on raises Stopped
            try:
                if self._stopping:
                    raise Stopped()
                self._launch(launch, side, spec, record, out_dir, rel, held)
                record["status"] = "ok"
            finally:
                self._interruptible = False
        except Stopped:
            record["error"], record["error_code"] = "stopped by SIGTERM during the launch", STOPPED
            self._begin_stop()
        except RoundError as error:
            record["error"], record["error_code"] = str(error), error.code
        except probe.StartFailed as error:
            record["error"], record["error_code"] = str(error), SIDE_NEVER_READY if error.never_ready else SIDE_FAILED
        except (guidellm.LoadError, guidellm.ReportError) as error:
            record["error"], record["error_code"] = str(error), LOAD_FAILED
        except Exception as error:  # a fault in one launch must not lose the launches before it
            record["error"], record["error_code"] = f"{type(error).__name__}: {error}", RUNNER_ERROR
        finally:
            if self._stopping:  # also a stop that came in after the launch's own work: the server back first
                self._begin_stop()
            self._stop_scale_helpers(held, record)
            if held["discard"] is not None:  # a discard that failed or was stopped partway: finish it here
                self._finish_discard(spec, record, out_dir, rel, held, quiet=True)
            for entry in reversed(held["replicas"]):  # the added replica first, the first replica last
                if entry["replica"] == 1:
                    self._record_server(entry, spec, record)
                    self._after(entry["handle"], record, out_dir, rel)
                else:
                    self._after_replica(entry, spec, record, out_dir, rel)
            record["servers"].sort(key=lambda item: (not item.get("discarded"), item["replica"]))  # tries in order
            record["finished_at"] = utc_now()
        record.pop("_reported_version", None)
        record.pop("_started_epoch", None)
        return record

    def _launch(self, launch: Mapping, side: Mapping, spec: Mapping, record: dict, out_dir: Path, rel: str,
                held: dict) -> None:
        """Wait for the GPU, start the side, time its start, load it (mode levels) or run the scale round (mode
        scaleout, `_surge`); any failure raises. A load whose GuideLLM timed out runs the launch once more
        (LOAD_RETRIES) from a new start of the side, so the second try meets the same conditions as every other
        launch: a server that has served no load (in mode scaleout, a first replica that has served no surge), and
        the same requests."""
        scale = self.plan.get("mode") == plans.SCALEOUT and launch["kind"] != plans.WARMUP
        tries = 0
        while True:
            first_note = len(record.get("notes") or [])  # the notes from here on belong to this try
            handle, target = self._start_side(launch, side, spec, record, out_dir, rel, held)
            if launch["kind"] == plans.WARMUP:
                return
            try:
                if scale:
                    self._surge(launch, side, spec, record, out_dir, rel, held, target, tries)
                else:
                    record["levels"] = self._load(launch, handle, target, out_dir, rel, record, tries)
            except guidellm.LoadError as error:
                if not error.timed_out:
                    raise
                record["load_timeouts"] += 1  # counted against the side whatever happens next
                if tries >= LOAD_RETRIES:
                    raise
                tries += 1
                self._discard_try(spec, record, out_dir, rel, held, error, tries, first_note)
                continue
            except BaseException as error:  # a stop after a time-out: while GuideLLM or the sampler was stopped
                cause = error.__context__
                if getattr(error, "load_timed_out", False) or (
                        isinstance(cause, guidellm.LoadError) and cause.timed_out):
                    record["load_timeouts"] += 1
                raise
            if not scale:
                record["values"].update(guidellm.flat_values(record["levels"]))
            return

    def _start(self, launch: Mapping, side: Mapping, spec: Mapping, record: dict, held: dict, replica: int,
               retries: int) -> dict:
        """Start one server of the launch (replica 1, the side; replica 2, the one a scale launch adds) and hold it at
        once, so the teardown takes it down whatever happens next. It counts in the launch's `starts` once the
        platform has taken it, and a start again after vLLM's start-up memory check (`retries` 1) counts in
        `start_retries`, with its note, only then too."""
        started_at, epoch = utc_now(), self.wall()
        if replica == 1:
            handle = self.backend.start(self.plan, launch, side, spec)
        else:
            handle = self.backend.start(self.plan, launch, side, spec, replica=replica)
        entry = {"replica": replica, "handle": handle, "started_at": started_at, "epoch": epoch,
                 "start_retries": retries}
        held["replicas"].append(entry)
        record["starts"] += 1
        if retries:  # only now has the server started again
            record["start_retries"] += 1
            record.setdefault("notes", []).append(
                "the side was started once more after vLLM's start-up memory check, and its start is timed from then"
                if replica == 1 else
                f"replica {replica} was started once more after vLLM's start-up memory check; its time to first "
                "answer still counts from when it was first asked for")
        return entry

    def _start_side(self, launch: Mapping, side: Mapping, spec: Mapping, record: dict, out_dir: Path, rel: str,
                    held: dict):
        """Wait for the GPU, start the side (in mode scaleout, its first replica) and time it to its first answer:
        (handle, target). A server that stopped at vLLM's start-up memory check is started once more
        (START_RETRIES), and its start is timed from that start."""
        timeout = self.plan["server"]["warmup_timeout_s" if launch["kind"] == plans.WARMUP else "ready_timeout_s"]
        model = self.plan["model"]["id"]
        retries = 0
        while True:
            self.backend.gpu_free(self.plan, side)
            record["_started_epoch"] = self.wall()
            t0 = self.clock()
            entry = self._start(launch, side, spec, record, held, 1, retries)
            handle = entry["handle"]
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
                if (error.never_ready or retries >= START_RETRIES
                        or not self._start_again(entry, record, out_dir, rel, error, held)):
                    raise
                retries += 1
        record["startup"] = {"ready_s": round(ready_s, 3), "first_answer_s": round(first_s, 3),
                             "clock": "runner, from just before the server was asked for", "breakdown": []}
        record["values"].update({"startup/ready_s": ready_s, "startup/first_answer_s": first_s})
        return handle, target

    def _start_again(self, entry: dict, record: dict, out_dir: Path, rel: str, error: Exception, held: dict) -> bool:
        """When the server's log shows vLLM's memory-profiling check failing: keep that log as
        server.attempt-<n>.log (server-2.attempt-<n>.log for the replica a scale launch adds), take that server down,
        let go of it and return True (the launch starts it again). Anything else: False, and the launch fails."""
        handle, n = entry["handle"], entry["replica"]
        try:
            text = self.backend.logs(handle)
        except Exception:
            return False
        if not logs.memory_profiling_failed(text):
            return False
        attempt = entry["start_retries"] + 1
        stem, key = ("server", "server_log") if n == 1 else (f"server-{n}", f"server_log_{n}")
        name = f"{stem}.attempt-{attempt}.log"
        (out_dir / name).write_text(text, encoding="utf-8")
        record["artifacts"][f"{key}_attempt_{attempt}"] = f"{rel}/{name}"
        who = "the side" if n == 1 else f"replica {n}"
        record.setdefault("notes", []).append(
            f"{error} at vLLM's start-up memory check ({logs.MEMORY_PROFILING_ERROR}); {who} is to be taken down "
            f"and started once more ({name} is the log of the start that stopped)")
        self.backend.stop(handle)
        held["replicas"].remove(entry)  # torn down; the next start holds the new one
        return True

    # ---- the scale round (plan mode scaleout) ----

    def _surge(self, launch: Mapping, side: Mapping, spec: Mapping, record: dict, out_dir: Path, rel: str,
               held: dict, target: str, retry: int = 0) -> None:
        """The surge through the router; the second replica at the forced step; the surge's figures, and how the
        router shared it out. `retry`: the try this is after a GuideLLM time-out (0 for the first)."""
        scale = self.plan["scale_out"]
        for stale in ("guidellm.json", "guidellm.log"):  # from an earlier attempt of the run in a reused folder
            (out_dir / stale).unlink(missing_ok=True)
        router = held["router"] = self.backend.router(self.plan, [target], out_dir)
        if retry:  # only now does the surge run again
            record["load_retries"] = retry
            record.setdefault("notes", []).append(
                f"the launch ran once more from a new first replica, with the same requests (the first try's files "
                f"are the *.try-{retry}.* files)")
        running = held["load"] = self.backend.start_load(self.plan, launch, router.url, out_dir)
        load_deadline = self.clock() + self.plan["load"]["timeout_s"]
        surge_at = None
        while surge_at is None:  # the surge starts when its first request reaches the router
            surge_at = router.stats().get("first_request_at")
            if surge_at is None:
                if running.poll() is not None:
                    running.finish()  # LoadError when it failed
                    raise guidellm.LoadError("GuideLLM ended before it sent a request")
                if self.clock() > load_deadline:  # GuideLLM hung before its first request: a time-out
                    raise guidellm.LoadError("GuideLLM sent no request before its time limit", timed_out=True)
                self.sleep(0.25)
        step_at = surge_at + scale["step_after_s"]
        while (left := step_at - self.wall()) > 0:
            if running.poll() is not None:
                running.finish()
                raise guidellm.LoadError("the surge ended before the forced step")
            self.sleep(min(left, 0.5))
        record["scale"] = {"surge_started_at": _iso(surge_at), "step_after_s": scale["step_after_s"],
                           "step_at_s": None, "new_replica": None, "router": None, "new_replica_share": None,
                           "events": [], "load_triggered_s": None}
        timeout = self.plan["server"]["ready_timeout_s"]
        t0 = self.clock()
        deadline = t0 + timeout
        model = self.plan["model"]["id"]
        retries, asked_epoch = 0, None
        while True:  # a start that stops at vLLM's memory check is tried once more, and its time counts
            entry = self._start(launch, side, spec, record, held, 2, retries)
            if asked_epoch is None:
                asked_epoch = entry["epoch"]
                record["scale"]["step_at_s"] = round(asked_epoch - surge_at, 3)
            entry["epoch"] = asked_epoch  # its log's phases: from the first ask
            handle2 = entry["handle"]
            target2 = self.backend.endpoint(handle2, deadline)
            alive2 = lambda: self.backend.alive(handle2)  # noqa: E731
            try:
                with self.client(target2) as client:
                    ready_s = probe.wait_ready(client, model, t0, deadline, alive2)
                    first_s = probe.first_answer(client, model, t0, deadline, alive2,
                                                 self.plan["server"]["first_answer_tokens"],
                                                 self.plan["load"]["request_format"])
                break
            except probe.StartFailed as error:
                if (error.never_ready or retries >= START_RETRIES
                        or not self._start_again(entry, record, out_dir, rel, error, held)):
                    raise
                retries += 1
        surge_over = running.poll() is not None
        joined = router.add(target2) or {}
        joined_at = joined.get("joined_at") if isinstance(joined.get("joined_at"), (int, float)) else self.wall()
        record["scale"]["new_replica"] = {
            "ready_s": round(ready_s, 3), "first_answer_s": round(first_s, 3),
            "joined_at_s": round(joined_at - surge_at, 3), "joined_during_surge": not surge_over,
            "clock": "runner, from just before the second replica was asked for", "breakdown": []}
        record["values"].update({"scale/ready_s": ready_s, "scale/first_answer_s": first_s})
        report_path = running.finish()  # a time-out: _launch runs the launch once more
        record["artifacts"]["load_report"] = f"{rel}/{report_path.name}"
        raw = json.loads(report_path.read_text(encoding="utf-8"))
        version = guidellm.version(raw)
        if version:
            self.results["provenance"]["load_tool"]["version"] = version
        level = guidellm.surge_level(raw)
        level["energy"] = energy.none_entry("none", "energy is not measured in mode scaleout")
        record["levels"] = [level]
        record["values"].update(guidellm.surge_values(level))
        record["scale"]["router"] = self._router_stats(router.stats(), [target, target2], surge_at)
        sent = [b["sent"] for b in record["scale"]["router"]["backends"]]
        record["scale"]["new_replica_share"] = round(sent[1] / sum(sent), 4) if sum(sent) else None

    def _router_stats(self, counts: Mapping, targets: list[str], surge_at: float) -> dict:
        """The router's counts per replica (replica 1 first), its times in seconds after the surge's start."""
        by_url = {b.get("url"): b for b in counts.get("backends") or []}
        out = []
        for replica, url in enumerate(targets, start=1):
            b = by_url.get(url) or {}

            def since(key: str) -> float | None:
                value = b.get(key)
                return round(value - surge_at, 3) if isinstance(value, (int, float)) else None

            out.append({"replica": replica, "sent": int(b.get("sent") or 0), "finished": int(b.get("finished") or 0),
                        "failed": int(b.get("failed") or 0), "cancelled": int(b.get("cancelled") or 0),
                        "max_in_flight": int(b.get("max_in_flight") or 0), "first_s": since("first_at"),
                        "last_s": since("last_at")})
        return {"policy": counts.get("policy") or "least-busy", "backends": out}

    def _stop_scale_helpers(self, held: dict, record: dict) -> None:
        """Stop the surge and the router, if the launch got that far and they still run. Each is let go of once its
        stop was tried, so a stop that cuts this short leaves the rest to the next call."""
        running, router = held.get("load"), held.get("router")
        if running is not None and running.poll() is None:
            try:
                running.stop()
            except Exception as error:
                record.setdefault("notes", []).append(f"GuideLLM not stopped: {type(error).__name__}")
        held["load"] = None
        if router is not None:
            try:
                router.stop()
            except Exception as error:
                record.setdefault("notes", []).append(f"router not stopped: {type(error).__name__}: {error}")
        held["router"] = None

    def _after_replica(self, entry: dict, spec: Mapping, record: dict, out_dir: Path, rel: str) -> None:
        """The added replica: its server record, its log (server-<n>.log, with its start's breakdown), teardown."""
        handle, n = entry["handle"], entry["replica"]
        self._record_server(entry, spec, record)
        try:
            text = self.backend.logs(handle)
            (out_dir / f"server-{n}.log").write_text(text, encoding="utf-8")
            record["artifacts"][f"server_log_{n}"] = f"{rel}/server-{n}.log"
            new = (record.get("scale") or {}).get("new_replica")
            if new is not None:
                new["breakdown"] = logs.breakdown(text, entry["epoch"])
        except Exception as error:
            record.setdefault("notes", []).append(f"replica {n}'s log not read: {type(error).__name__}")
        try:
            self.backend.stop(handle)
        except Exception as error:
            record.setdefault("notes", []).append(f"teardown of replica {n}: {type(error).__name__}: {error}")

    def _record_server(self, entry: dict, spec: Mapping, record: dict, discarded: bool = False) -> None:
        """The server's record: what the runner ran, and what the backend can read of what the server saw.
        `discarded`: a server of a try that was run again (`_discard_try`). Written once per server: one already
        recorded is skipped."""
        if entry.get("recorded"):
            return
        facts: dict = {}
        read = getattr(self.backend, "server_facts", None)
        if read is not None:
            try:
                facts = read(entry["handle"]) or {}
            except Exception as error:  # the server record says what it could not read
                record.setdefault("notes", []).append(f"server facts of replica {entry['replica']} not read: "
                                                      f"{type(error).__name__}")
        source = facts.get("env_source") if facts.get("env_source") in servers.ENV_SOURCES else "none"
        record["servers"].append(servers.record(
            entry["replica"], spec.get("command"), spec.get("args") or [], facts.get("env"), source,
            started_at=entry["started_at"], image_entrypoint=facts.get("image_entrypoint"),
            image_cmd=facts.get("image_cmd"), start_retries=entry.get("start_retries", 0),
            gpus=facts.get("gpus") if isinstance(facts.get("gpus"), str) else None,
            gpu_uuids=facts.get("gpu_uuids") if isinstance(facts.get("gpu_uuids"), list) else None,
            discarded=discarded))
        entry["recorded"] = True

    # ---- a try discarded after a GuideLLM time-out ----

    def _discard_try(self, spec: Mapping, record: dict, out_dir: Path, rel: str, held: dict, error: Exception,
                     tries: int, first_note: int) -> None:
        """After a load (mode levels) or a surge (mode scaleout) whose GuideLLM timed out: clear what the try
        measured, say why, stop GuideLLM and the router (mode scaleout), then `_finish_discard`: keep every file of
        the try as *.try-N.* (artifact keys *_try_N), mark its notes as the try's, and take its servers down with
        their server records marked discarded. The launch then starts again from the GPU check.

        The figures go first and the note before anything that can fail or be stopped, so a launch cut short here
        has none of the try's figures and says why. While `held["discard"]` is set, every server in `held` belongs to
        the discarded try, and the launch's teardown finishes a discard that a failed stop or a SIGTERM cut short.
        A failed stop here fails the launch, as that server may still hold the GPU and its name."""
        record.pop("_reported_version", None)
        record.pop("scale", None)
        record["startup"], record["levels"], record["values"] = None, [], {}
        notes = record.setdefault("notes", [])
        if self.plan.get("mode") != plans.SCALEOUT:
            what = "the side is to be taken down and the launch run once more from a new start"
        elif len(held["replicas"]) > 1:
            what = "both replicas are to be taken down and the launch run once more from a new first replica"
        else:  # GuideLLM hung before the forced step: only the first replica had started
            what = "the side is to be taken down and the launch run once more from a new first replica"
        notes.append(f"{error}; {what} with the same requests (that try's files are the *.try-{tries}.* files)")
        held["discard"] = {"tries": tries, "first_note": first_note, "note_at": len(notes) - 1, "renamed": None}
        self._stop_scale_helpers(held, record)
        self._finish_discard(spec, record, out_dir, rel, held)

    def _finish_discard(self, spec: Mapping, record: dict, out_dir: Path, rel: str, held: dict,
                        quiet: bool = False) -> None:
        """The rest of a discard (`_discard_try`), each step done once, also when the teardown runs this again after
        a stop or a failed stop: the try's files renamed and its notes marked (in one step a stop cannot cut), then
        its servers taken down. `quiet` (the teardown): a failed stop becomes a note."""
        discard = held["discard"]
        tries = discard["tries"]
        if discard["renamed"] is None:
            with self._held_off():
                renamed: dict[str, str] = {}

                def keep(name: str, key: str) -> None:
                    stem, dot, ext = name.rpartition(".")
                    kept = f"{stem}.try-{tries}{dot}{ext}"
                    if (out_dir / name).exists():
                        (out_dir / name).replace(out_dir / kept)
                        record["artifacts"][f"{key}_try_{tries}"] = f"{rel}/{kept}"
                        renamed[name] = kept

                record["artifacts"].pop("load_report", None)  # kept below under its try's name, if the file is there
                for key, path in list(record["artifacts"].items()):  # such as a memory-check retry's .attempt-1.log
                    if "_try_" not in key:
                        del record["artifacts"][key]
                        keep(path.rpartition("/")[2], key)
                keep("guidellm.log", "guidellm")
                keep("guidellm.json", "load_report")  # a report the stopped GuideLLM left; the load removed any older
                keep("router.log", "router")
                notes = record["notes"]
                for i in range(discard["first_note"], discard["note_at"]):  # such as a sampler that did not start
                    notes[i] = f"try {tries}: {_renamed(notes[i], renamed)}"
                notes[discard["note_at"]] = _renamed(notes[discard["note_at"]], renamed)
                discard["renamed"] = renamed
        for entry in list(reversed(held["replicas"])):  # the added replica first, the first replica last
            self._discard_server(entry, spec, record, out_dir, rel, held, quiet)
        held["discard"] = None

    def _discard_server(self, entry: dict, spec: Mapping, record: dict, out_dir: Path, rel: str, held: dict,
                        quiet: bool) -> None:
        """One server of a discarded try: its server record marked discarded and its log as <stem>.try-N.log, each
        once, then its teardown. `quiet` (the teardown): a failed stop becomes a note."""
        n, tries = entry["replica"], held["discard"]["tries"]
        self._record_server(entry, spec, record, discarded=True)
        if not entry.get("log_kept"):
            stem, key = ("server", "server_log") if n == 1 else (f"server-{n}", f"server_log_{n}")
            name = f"{stem}.try-{tries}.log"
            try:
                (out_dir / name).write_text(self.backend.logs(entry["handle"]), encoding="utf-8")
                record["artifacts"][f"{key}_try_{tries}"] = f"{rel}/{name}"
            except Exception as read_error:
                what = "the server log of that try" if n == 1 else f"the log of replica {n} of that try"
                record.setdefault("notes", []).append(f"{what} not read: {type(read_error).__name__}")
            entry["log_kept"] = True
        try:
            self.backend.stop(entry["handle"])
        except Exception as stop_error:
            if not quiet:
                raise
            record.setdefault("notes", []).append(
                f"teardown of replica {n} of the discarded try: {type(stop_error).__name__}: {stop_error}")
        held["replicas"].remove(entry)

    # ---- the load (mode levels) ----

    def _load(self, launch: Mapping, handle, target: str, out_dir: Path, rel: str, record: dict,
              retry: int = 0) -> list[dict]:
        sampler = None
        if self.plan["energy"]["source"] != "none":
            try:
                sampler = self.backend.sampler(self.plan, handle)
                if sampler is not None:
                    sampler.start()
            except Exception as error:
                record.setdefault("notes", []).append(f"power sampling did not start: {type(error).__name__}")
                sampler = None
        for stale in ("guidellm.json", "guidellm.log"):  # from an earlier attempt of the run in a reused folder
            (out_dir / stale).unlink(missing_ok=True)
        if retry:  # only now does the load run again
            record["load_retries"] = retry
            record.setdefault("notes", []).append(
                f"the launch ran once more from a new start of the side, with the same requests (the first try's "
                f"files are the *.try-{retry}.* files)")
        try:
            report_path = self.backend.run_load(self.plan, launch, target, out_dir)
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
        """Logs, facts and teardown of the side (in mode scaleout, its first replica), whatever happened in the
        launch."""
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
            scale = record.get("scale")
            if scale is not None:  # the scale-ups the server itself logged (the pattern's rhvp-scale lines)
                scale["events"] = logs.scale_events(text)
                surge_at = summary_epoch(scale.get("surge_started_at"))
                up = logs.first_scale_up(scale["events"], surge_at) if surge_at is not None else None
                if up is not None:
                    scale["load_triggered_s"] = round(up, 3)
                    record["values"]["scale/load_triggered_s"] = up
        except Exception as error:
            record.setdefault("notes", []).append(f"server log not read: {type(error).__name__}")
        try:
            facts = self.backend.facts(handle) or {}
            record["node"] = facts.get("node_name")
            record["image_digest"] = facts.get("image_digest")
            prov = self.results["provenance"]
            side = prov["sides"][record["of"]]  # a twin's image is its side's
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


def _renamed(text: str, renamed: Mapping[str, str]) -> str:
    """`text` with each file name in `renamed` (a whole name, not part of a longer one) replaced by its new name."""
    if not renamed:
        return text
    names = "|".join(re.escape(name) for name in sorted(renamed, key=len, reverse=True))
    return re.sub(rf"(?<![\w.-])({names})(?![\w.-])", lambda match: renamed[match.group(1)], text)


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def summary_epoch(text: str | None) -> float | None:
    """Seconds since the epoch of a time the runner wrote (utc_now, _iso)."""
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


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


def plan_lines(the_plan: Mapping) -> list[str]:
    """What a plan allows, before anything runs: the lineup, the launches and the servers they start, and whether a
    verdict is possible. The note names the reason the analysis will give, checked in its order (analyze._decide):
    one round, too few rounds, then a side without a twin."""
    power = the_plan["power"]
    names = ", ".join(e["key"] for e in the_plan["lineup"])
    launches, starts = len(the_plan["launches"]), plans.server_starts(the_plan)
    shown = (f"{launches} server starts" if starts == launches else
             f"{launches} launches, {starts} server starts (two in each scored launch)")
    out = [f"run {the_plan['run_id']}: {the_plan['rounds']} rounds of {names}, {shown}, {power['family_size']} "
           f"primary cases of {len(the_plan['cases'])} ({power['plain']})"]
    no_twin = "no_floor or one_sided_floor"
    if the_plan["rounds"] == 1:
        out.append(f"note: {power['family_size']} primary cases need at least {power['min_rounds']} rounds for any "
                   f"verdict; with 1 round every case gets NO VERDICT (single_measurement)")
    elif the_plan["rounds"] < power["min_rounds"]:
        secondary = stats.min_rounds(1, the_plan["analysis"]["alpha"])
        if the_plan["rounds"] < secondary:
            graded = f"so does every secondary case, which needs {secondary}"
        elif power["both_twins"]:
            graded = "the secondary cases are graded on their own"
        else:
            graded = f"without a twin for both sides so does every secondary case ({no_twin})"
        out.append(f"note: {power['family_size']} primary cases need at least {power['min_rounds']} rounds for any "
                   f"verdict; with {the_plan['rounds']} every primary case gets NO VERDICT (too_few_rounds), and "
                   f"{graded}")
    elif not power["both_twins"]:
        out.append(f"note: without a twin for both sides no case can get a verdict (NO VERDICT: {no_twin})")
    else:
        out.append(f"a verdict on a primary case needs {power['must_agree']} of the {the_plan['rounds']} rounds to "
                   f"lean the same way; at most {min(power['max_primary_cases'], power['family_size'])} primary "
                   f"cases can get one")
    if not the_plan["balance"]["balanced"]:
        out.append(f"note: {the_plan['rounds']} rounds do not balance the run order; multiples of "
                   f"{the_plan['balance']['step']} do")
    return out


def main_run(config_path: str, results_root: str, backend: Backend, env: Mapping[str, str] | None = None) -> int:
    """Build the plan, run it, print where the results are. The exit code: 0 when every launch finished, 1 for a
    partial run (so its Job reads Failed), 3 for a run that failed outside a launch, 4 when the live server was not
    brought back (whatever else happened), 5 when another comparison was running (nothing changed). A config error
    raises PlanError (exit 2)."""
    env = os.environ if env is None else env
    termination_log = env.get("RHVP_COMPARE_TERMINATION_LOG") or None
    try:
        the_plan = plans.load(config_path, backend.platform)
    except plans.PlanError as error:
        _finish(summary.not_started(PLAN_INVALID, str(error)), termination_log)
        raise
    check_plan = getattr(backend, "check_plan", None)
    if check_plan is not None:
        try:
            check_plan(the_plan)
        except plans.PlanError as error:
            _finish(summary.not_started(PLAN_INVALID, str(error), the_plan["run_id"]), termination_log)
            raise
    for line in plan_lines(the_plan):
        print(line, flush=True)
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
    for row in (results.get("analysis") or {}).get("comparisons") or []:
        reason = f", {row['reason']}" if row.get("reason") else ""
        print(f"  {row['case']} ({row['role']}): {row['verdict']}{reason} (rounds {row['n_rounds']})", flush=True)
    for line in run.get("errors") or []:
        print(f"error: {line}", flush=True)
    for line in run.get("notes") or []:
        if line.startswith(FOUND_DOWN):
            print(f"note: {line}", flush=True)
    return exit_code(results)
