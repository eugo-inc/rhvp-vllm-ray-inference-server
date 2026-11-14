# SPDX-License-Identifier: Apache-2.0
"""GuideLLM (https://github.com/vllm-project/guidellm): the command line for one launch, and the launch's figures
out of its JSON report. Written against GuideLLM v0.8.1's `guidellm run` and its report (metadata.version 2).

GuideLLM runs as a child process of the runner: the runner's container is the GuideLLM image, with this package
added (on OpenShift an init container copies it from the notebook image; under docker it is mounted from the repo).
So the load comes from the same place on both platforms, and the report lands straight in the results folder.

One run of GuideLLM per launch measures every concurrency level in turn (the `concurrent` profile with one
`profile.streams` override per level). Each level runs for a fixed time; requests still running when the time is up
are "incomplete", which is expected. A level with any errored request is kept in the results file but marked not
usable, and its figures stay out of the comparison: an average over failed requests would compare failures.

The figures are GuideLLM's own per-level medians. The report keeps a timing record for every request
(benchmarks[].requests.successful, .errored and .incomplete, with time to first token, inter-token latency and token
counts): `--metrics sample_size=0` drops only each request's prompt and output text (GuideLLM v0.8.1 keeps the timing
of every request whatever the sample size), so per-request statistics can be worked out again from the raw report
in each launch's folder.

In mode scaleout one run is one surge instead: GuideLLM's `constant` profile (requests sent at a fixed rate whether
or not earlier ones have finished) for scaleOut.surgeSeconds, with the latency goals passed to `--metrics` as JSON
(`"slo": {"ttft_ms": .., "tpot_ms": ..}`; GuideLLM compares tpot_ms with each request's inter-token latency). GuideLLM
then reports `slo_attainment` (the share of finished and failed requests that met every goal; a failed request
counts as one that missed them, a request still running at the end is left out) and `request_goodput` (the rate of
the requests that met them). The runner starts that run in the background (`start`), because it starts the second
replica while the surge runs. Standard library only.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping


def command(plan: Mapping, target: str, output_path: str) -> list[str]:
    """The arguments after `guidellm` for one launch against the server at `target`: every concurrency level in
    mode levels, the surge (`surge_command`) in mode scaleout."""
    if plan.get("mode") == "scaleout":
        return surge_command(plan, target, output_path)
    load = plan["load"]
    levels = [str(c) for c in load["concurrencies"]]
    args = ["run", "--backend", _backend(plan, target), "--profile", f"kind=concurrent,streams={levels[0]}"]
    if len(levels) > 1:
        args += ["--override", "profile.streams", ",".join(levels)]
    args += ["--constraint", f"kind=max_duration,seconds={load['seconds_per_level']:g}", *_workload(plan),
             "--metrics", "kind=generative,sample_size=0", "--output", f"kind=json,path={output_path}",
             "--disable-console-interactive"]
    return args


def surge_command(plan: Mapping, target: str, output_path: str) -> list[str]:
    """The arguments after `guidellm` for one scale launch's surge: requests at scaleOut.requestRate per second
    for scaleOut.surgeSeconds, scored against the latency goals."""
    scale = plan["scale_out"]
    slo = {key: value for key, value in scale["slo"].items() if value is not None}
    metrics = json.dumps({"kind": "generative", "sample_size": 0, "slo": slo}, separators=(",", ":"))
    return ["run", "--backend", _backend(plan, target), "--profile", f"kind=constant,rate={scale['request_rate']:g}",
            "--constraint", f"kind=max_duration,seconds={scale['surge_s']:g}", *_workload(plan),
            "--metrics", metrics, "--output", f"kind=json,path={output_path}", "--disable-console-interactive"]


def _backend(plan: Mapping, target: str) -> str:
    load, model = plan["load"], plan["model"]
    return f"kind=openai_http,target={target},model={model['id']},request_format={load['request_format']}"


def _workload(plan: Mapping) -> list[str]:
    """The data, tokenizer and seed arguments, the same in both modes."""
    load, model = plan["load"], plan["model"]
    tokenizer = f"kind=huggingface_auto,model={load['tokenizer']}"
    if model.get("revision") and load["tokenizer"] == model["source"]:
        tokenizer += f",load_kwargs.revision={model['revision']}"
    data = f"kind=synthetic_text,prompt_tokens={load['prompt_tokens']},output_tokens={load['output_tokens']}"
    return ["--data", data, "--tokenizer", tokenizer, "--seed", f"kind=static,value={load['seed']}"]


def _iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _stat(metrics: Mapping, name: str, group: str = "successful") -> Mapping:
    value = (metrics.get(name) or {}).get(group)
    return value if isinstance(value, Mapping) else {}


def _summary(stat: Mapping) -> dict:
    percentiles = stat.get("percentiles") or {}
    return {"median": stat.get("median"), "p90": percentiles.get("p90"), "mean": stat.get("mean"),
            "count": stat.get("count")}


class LoadError(Exception):
    """GuideLLM failed or did not finish in time; the message is safe to show."""

    def __init__(self, message: str, timed_out: bool = False) -> None:
        super().__init__(message)
        self.timed_out = timed_out


def run(plan: Mapping, target: str, out_dir: Path, env: Mapping[str, str] | None = None,
        executable: str = "guidellm") -> Path:
    """Run GuideLLM against `target` and return the report's path. Its output goes to guidellm.log next to the
    report. A run past plan.load.timeout_s is stopped (its whole process group) and raises LoadError."""
    return start(plan, target, out_dir, env, executable).finish()


class Running:
    """A GuideLLM run started in the background (`start`). `finish` waits for it; `stop` ends it."""

    def __init__(self, process: subprocess.Popen, log, report: Path, timeout_s: float, started: float) -> None:
        self.process, self._log, self.report = process, log, report
        self.timeout_s, self.started = timeout_s, started

    def poll(self) -> int | None:
        """The exit code once GuideLLM has ended, else None."""
        return self.process.poll()

    def finish(self) -> Path:
        """Wait until GuideLLM ends, at most until plan.load.timeout_s after its start, and return the report's
        path; LoadError when it failed or ran out of time (then it is stopped). A stop of the runner that lands
        while a timed-out GuideLLM is being stopped carries `load_timed_out`, so the runner still counts the
        time-out."""
        remaining = max(0.0, self.timeout_s - (time.monotonic() - self.started))
        try:
            code = self.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                self.stop()
            except BaseException as error:  # the runner was told to stop while GuideLLM was taken down
                error.load_timed_out = True  # the time-out happened all the same: the runner counts it
                raise
            raise LoadError(f"GuideLLM did not finish within {self.timeout_s:g} s and was stopped; see guidellm.log",
                            timed_out=True) from None
        except BaseException:  # the runner was told to stop: take GuideLLM down with it
            self.stop()
            raise
        self._log.close()
        if code != 0 or not self.report.is_file():
            raise LoadError(f"GuideLLM exited with code {code}; see guidellm.log")
        return self.report

    def stop(self) -> None:
        """Stop GuideLLM's whole process group (SIGTERM, then SIGKILL) and close its log. A stop of the runner
        that lands while this waits kills the group at once, then goes on."""
        try:
            _stop(self.process)
        except BaseException:
            _kill(self.process)
            raise
        finally:
            self._log.close()


def start(plan: Mapping, target: str, out_dir: Path, env: Mapping[str, str] | None = None,
          executable: str = "guidellm") -> Running:
    """Start GuideLLM against `target` in its own process group and return at once."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "guidellm.json"
    log = open(out_dir / "guidellm.log", "w", encoding="utf-8")
    try:
        process = subprocess.Popen([executable, *command(plan, target, str(report))], stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   env={**os.environ, **(env or {})}, start_new_session=True)
    except BaseException:
        log.close()
        raise
    return Running(process, log, report, plan["load"]["timeout_s"], time.monotonic())


def _kill(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass


def _stop(process: subprocess.Popen) -> None:
    for sig, wait_s in ((signal.SIGTERM, 15), (signal.SIGKILL, 15)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


class ReportError(ValueError):
    """The GuideLLM report is missing or not in the expected shape."""


def version(report: Mapping) -> str | None:
    return (report.get("metadata") or {}).get("guidellm_version")


def levels(report: Mapping) -> list[dict]:
    """One entry per concurrency level, in run order: its time window (UTC), request counts, TTFT and inter-token
    latency (median, p90, mean, in ms), output tokens and output tokens per second over the window (all requests,
    the incomplete ones at the window's end included, as GuideLLM's own server throughput)."""
    benchmarks = report.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ReportError("the report has no benchmarks")
    out = []
    for bench in benchmarks:
        try:
            strategy = bench["config"]["strategy"]
            metrics = bench["metrics"]
        except (KeyError, TypeError):
            raise ReportError("a benchmark has no config.strategy or metrics") from None
        concurrency = strategy.get("streams") or strategy.get("max_concurrency")
        totals = metrics.get("request_totals") or {}
        errored = int(totals.get("errored") or 0)
        successful = int(totals.get("successful") or 0)
        output_tokens = _stat(metrics, "output_token_count", "total").get("total_sum")
        rate = _stat(metrics, "output_tokens_per_second", "total").get("mean")
        level = {
            "concurrency": int(concurrency) if concurrency is not None else None,
            "started_at": _iso(bench.get("start_time")),
            "finished_at": _iso(bench.get("end_time")),
            "start_epoch": bench.get("start_time"),
            "end_epoch": bench.get("end_time"),
            "duration_s": bench.get("duration"),
            "requests": {"successful": successful, "errored": errored,
                         "incomplete": int(totals.get("incomplete") or 0)},
            "ttft_ms": _summary(_stat(metrics, "time_to_first_token_ms")),
            "itl_ms": _summary(_stat(metrics, "inter_token_latency_ms")),
            "output_tokens": int(output_tokens) if output_tokens is not None else None,
            "output_tokens_per_s": rate,
            "requests_per_s": _stat(metrics, "requests_per_second").get("mean"),
            "usable": errored == 0 and successful > 0,
            "note": None,
        }
        if errored:
            level["note"] = f"{errored} requests failed; this level is left out of the comparison"
        elif not successful:
            level["note"] = "no request finished; this level is left out of the comparison"
        out.append(level)
    return out


def flat_values(level_list: list[Mapping]) -> dict[str, float]:
    """The comparison's per-launch figures from the usable levels (case ids as in plan.case_family). A figure of 0
    is kept (a level whose requests finished without one output token has 0 tokens per second): on side A or B the
    analysis gives that case NO VERDICT (nonpositive_value) instead of leaving the pair out unseen; in a side's
    comparison with its twin, every case whose floor uses that comparison gets it too, with a note."""
    out: dict[str, float] = {}
    for level in level_list:
        c = level.get("concurrency")
        if c is None or not level.get("usable"):
            continue
        pairs = {f"c{c}/ttft_ms_median": (level.get("ttft_ms") or {}).get("median"),
                 f"c{c}/itl_ms_median": (level.get("itl_ms") or {}).get("median"),
                 f"c{c}/output_tokens_per_s": level.get("output_tokens_per_s"),
                 f"c{c}/tokens_per_joule[gpu]": (level.get("energy") or {}).get("tokens_per_joule")}
        out.update({k: float(v) for k, v in pairs.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0})
    return out


def _finite(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def surge_level(report: Mapping) -> dict:
    """The surge of a scale launch, from the report's one benchmark: the fields `levels` gives (with concurrency None
    and the rate the profile sent), plus GuideLLM's SLO figures: `slo_attainment` (0 to 1, None when no request
    could be judged), `slo_determined_requests` (the requests it judged) and `goodput_rps` (requests per second
    that met every goal, the mean of request_goodput over the successful requests)."""
    level = levels(report)[0]
    bench = report["benchmarks"][0]
    metrics, strategy = bench["metrics"], bench["config"]["strategy"]
    determined = metrics.get("slo_determined_requests")
    level.update({
        "concurrency": None,
        "rate": _finite(strategy.get("rate")),
        "slo_attainment": _finite(metrics.get("slo_attainment")),
        "slo_determined_requests": int(determined) if _finite(determined) is not None else None,
        "goodput_rps": _finite(_stat(metrics, "request_goodput").get("mean")),
    })
    errored = level["requests"]["errored"]
    if errored:
        level["note"] = (f"{errored} requests failed; they count as requests that missed the latency goals, and the "
                         "surge's latency medians are left out of the comparison")
    elif not level["requests"]["successful"]:
        level["note"] = "no request finished during the surge; its SLO attainment and goodput count as 0"
    elif level["slo_attainment"] is None:
        level["note"] = ("GuideLLM judged no request against the latency goals (a goal it cannot measure, such as "
                         "ttft_ms without streaming); SLO attainment and goodput are missing")
    return level


def surge_values(level: Mapping) -> dict[str, float]:
    """The scale launch's surge/* figures (case ids as in plan.scale_case_family).

    - surge/slo_attainment and surge/goodput_rps are kept whenever GuideLLM gives them, failed requests included
      (they count as requests that missed the goals). When no request finished or failed in the whole surge,
      GuideLLM gives no attainment; both are then 0, a figure the analysis does not take as a ratio, so the case
      gets NO VERDICT (nonpositive_value) instead of losing the round unseen. A goodput GuideLLM leaves out when the
      attainment is 0 is 0 too.
    - surge/ttft_ms_median, surge/itl_ms_median and surge/output_tokens_per_s follow the rule of a load level: only
      when no request failed and at least one finished."""
    requests = level.get("requests") or {}
    nothing_finished = not requests.get("successful") and not requests.get("errored")
    attainment = level.get("slo_attainment")
    goodput = level.get("goodput_rps")
    if attainment is None and nothing_finished:
        attainment = goodput = 0.0
    elif goodput is None and attainment == 0:
        goodput = 0.0
    found = {"surge/slo_attainment": attainment, "surge/goodput_rps": goodput}
    if level.get("usable"):
        found.update({"surge/ttft_ms_median": (level.get("ttft_ms") or {}).get("median"),
                      "surge/itl_ms_median": (level.get("itl_ms") or {}).get("median"),
                      "surge/output_tokens_per_s": level.get("output_tokens_per_s")})
    return {k: float(v) for k, v in found.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0}
