# SPDX-License-Identifier: Apache-2.0
"""The round loop: bring one side up, time its start, load it with GuideLLM, read its GPU power, take it down;
round after round in the plan's order, writing the results file after every round.

The loop knows nothing about Kubernetes or docker: a Backend does the platform work (kube.py, dockerapi.py, or a
fake one in the tests). A round that fails is recorded with its reason and the run goes on; the comparison then
has one pair fewer for the metrics that round would have given.

Results land in <results>/<run id>/: results.json (the file results.schema.json describes), results.md (the short
table), and rounds/<NN>-<side>-<kind>/ with the server log and the raw GuideLLM report. A run whose results.json
is already complete is not run again (so a Job that Argo CD creates a second time does nothing).
"""

from __future__ import annotations

import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from collections.abc import Callable, Mapping

import httpx

from . import analyze, energy, guidellm, logs, plan as plans, probe, report

HARNESS = {"name": "rhvp_compare", "version": "1"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class RoundError(Exception):
    """A round could not be measured; the message is safe to show and to store."""


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
                 client: Callable[[str], httpx.Client] = lambda url: httpx.Client(base_url=url)) -> None:
        self.plan = plan
        self.client = client
        self.backend = backend
        self.dir = Path(results_root) / plan["run_id"]
        self.path = self.dir / "results.json"
        self.clock, self.wall, self.sleep = clock, wall, sleep
        self.results: dict = {}
        self._stopping = False

    # ---- the results file ----

    def _new_results(self) -> dict:
        return {
            "schema": plans.SCHEMA,
            "run": {"id": self.plan["run_id"], "name": self.plan["run_name"], "status": "running",
                    "started_at": utc_now(), "finished_at": None, "platform": self.backend.platform,
                    "notes": [], "errors": []},
            "plan": self.plan,
            "provenance": {"harness": dict(HARNESS),
                           "load_tool": {"name": "guidellm", "image": self.plan["load"]["image"], "version": None},
                           "seeds": {"load": self.plan["load"]["seed"], "server": self.plan["server"]["seed"]},
                           "sides": {s["key"]: {"image": s["image"], "image_digest": None, "server_version": None}
                                     for s in self.plan["sides"]},
                           "gpu": None, "node": None, "cluster": None},
            "rounds": [],
            "analysis": None,
        }

    def save(self) -> None:
        self.results["analysis"] = analyze.analyze(self.results)
        write_json(self.path, self.results)
        (self.dir / "results.md").write_text(report.markdown(self.results), encoding="utf-8")

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
            return json.loads(self.path.read_text(encoding="utf-8"))
        if self.path.is_file():  # an earlier attempt that did not finish: keep it beside the new one
            os.replace(self.path, self.dir / f"results.attempt-{int(self.wall())}.json")
        self.results = self._new_results()
        self.save()
        previous = signal.getsignal(signal.SIGTERM) if hasattr(signal, "SIGTERM") else None
        try:
            signal.signal(signal.SIGTERM, self._on_term)
        except ValueError:  # not the main thread (tests)
            previous = None
        try:
            self.backend.prepare(self.plan)
            self._describe()
            for round_ in self.plan["rounds"]:
                if self._stopping:
                    self.results["run"]["errors"].append("stopped by SIGTERM before the last round")
                    break
                self.results["rounds"].append(self.measure(round_))
                self.save()
            failed = [r for r in self.results["rounds"] if r["status"] != "ok"]
            done = len(self.results["rounds"]) == len(self.plan["rounds"])
            self.results["run"]["status"] = "complete" if done and not failed else "partial"
        except Exception as error:  # recorded, then raised for the Job's exit code
            self.results["run"]["status"] = "failed"
            self.results["run"]["errors"].append(f"{type(error).__name__}: {error}")
            raise
        finally:
            try:
                self.backend.cleanup()
            finally:
                self.results["run"]["finished_at"] = utc_now()
                self.save()
                if previous is not None:
                    signal.signal(signal.SIGTERM, previous)
        return self.results

    def _on_term(self, signum, frame) -> None:  # noqa: ARG002 (signal handler signature)
        self._stopping = True

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
                        "pair": round_["pair"], "status": "failed", "error": None, "started_at": utc_now(),
                        "finished_at": None, "node": None, "image_digest": None, "server_version": None,
                        "startup": None, "levels": [], "metrics": {}, "artifacts": {}}
        handle = None
        try:
            self.backend.gpu_free(self.plan, side)
            timeout = self.plan["server"]["warmup_timeout_s" if round_["kind"] == plans.WARMUP else "ready_timeout_s"]
            started_epoch = self.wall()
            record["_started_epoch"] = started_epoch
            t0 = self.clock()
            handle = self.backend.start(self.plan, round_, side, spec)
            deadline = t0 + timeout
            target = self.backend.endpoint(handle, deadline)
            alive = lambda: self.backend.alive(handle)  # noqa: E731
            model = self.plan["model"]["id"]
            with self.client(target) as client:
                ready_s = probe.wait_ready(client, model, t0, deadline, alive)
                first_s = probe.first_answer(client, model, t0, deadline, alive,
                                             self.plan["server"]["first_answer_tokens"],
                                             self.plan["load"]["request_format"])
                record["_reported_version"] = probe.reported_version(client)
            record["startup"] = {"ready_s": round(ready_s, 3), "first_answer_s": round(first_s, 3),
                                 "clock": "runner, from just before the server was asked for", "breakdown": []}
            record["metrics"].update({"startup.ready_s": ready_s, "startup.first_answer_s": first_s})
            if round_["kind"] != plans.WARMUP:
                record["levels"] = self._load(round_, handle, target, out_dir, rel, record)
                record["metrics"].update(guidellm.flat_metrics(record["levels"]))
            record["status"] = "ok"
        except (RoundError, probe.StartFailed, guidellm.LoadError, guidellm.ReportError) as error:
            record["error"] = str(error)
        except Exception as error:  # a fault in one round must not lose the rounds before it
            record["error"] = f"{type(error).__name__}: {error}"
        finally:
            if handle is not None:
                self._after(handle, record, out_dir, rel)
            record["finished_at"] = utc_now()
        record.pop("_reported_version", None)
        record.pop("_started_epoch", None)
        return record

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
                # GuideLLM v0.8.1 now and then stops making progress at the end of a level (seen about once in ten
                # runs against a stand-in server). One more try, against the same server, is recorded on the round.
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
        except Exception as error:
            record.setdefault("notes", []).append(f"facts not read: {type(error).__name__}")
        try:
            self.backend.stop(handle)
        except Exception as error:
            record.setdefault("notes", []).append(f"teardown: {type(error).__name__}: {error}")


def main_run(config_path: str, results_root: str, backend: Backend) -> int:
    """Build the plan, run it, print where the results are. Exit code 0 when every round finished."""
    the_plan = plans.load(config_path, backend.platform)
    floor = the_plan["floor"]
    print(f"run {the_plan['run_id']}: {len(the_plan['rounds'])} rounds, {the_plan['pairs']} scored pairs, "
          f"{floor['family_size']} metrics", flush=True)
    if not floor["reachable"]:
        print(f"note: with {the_plan['pairs']} pairs the smallest adjusted p-value is {floor['min_adjusted_p']:.3g}; "
              f"no metric can be FASTER or SLOWER below {floor['pairs_needed']} pairs", flush=True)
    runner = Runner(the_plan, backend, Path(results_root))
    results = runner.run()
    print(f"results: {runner.path} (status {results['run']['status']})", flush=True)
    for row in (results.get("analysis") or {}).get("metrics") or []:
        print(f"  {row['label']}: {row['verdict']} (pairs {row['n_pairs']})", flush=True)
    return 0 if results["run"]["status"] == "complete" else 1
