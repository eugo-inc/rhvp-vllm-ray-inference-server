# SPDX-License-Identifier: Apache-2.0
"""GuideLLM (https://github.com/vllm-project/guidellm): the command line for one round, and the round's figures out
of its JSON report. Written against GuideLLM v0.8.1's `guidellm run` and its report (metadata.version 2).

GuideLLM runs as a child process of the runner: the runner's container is the GuideLLM image, with this package
added (on OpenShift an init container copies it from the notebook image; under docker it is mounted from the repo).
So the load comes from the same place on both platforms, and the report lands straight in the results folder.

One run of GuideLLM per round measures every concurrency level in turn (the `concurrent` profile with one
`profile.streams` override per level). Each level runs for a fixed time; requests still running when the time is up
are "incomplete", which is expected. A level with any errored request is kept in the results file but marked not
usable, and its figures stay out of the comparison: an average over failed requests would compare failures.
Standard library only.
"""

from __future__ import annotations

import os
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping


def command(plan: Mapping, target: str, output_path: str) -> list[str]:
    """The arguments after `guidellm` for one round against the server at `target`."""
    load, model = plan["load"], plan["model"]
    levels = [str(c) for c in load["concurrencies"]]
    tokenizer = f"kind=huggingface_auto,model={load['tokenizer']}"
    if model.get("revision") and load["tokenizer"] == model["source"]:
        tokenizer += f",load_kwargs.revision={model['revision']}"
    args = [
        "run",
        "--backend", f"kind=openai_http,target={target},model={model['id']},request_format={load['request_format']}",
        "--profile", f"kind=concurrent,streams={levels[0]}",
    ]
    if len(levels) > 1:
        args += ["--override", "profile.streams", ",".join(levels)]
    args += [
        "--constraint", f"kind=max_duration,seconds={load['seconds_per_level']:g}",
        "--data", f"kind=synthetic_text,prompt_tokens={load['prompt_tokens']},output_tokens={load['output_tokens']}",
        "--tokenizer", tokenizer,
        "--seed", f"kind=static,value={load['seed']}",
        "--metrics", "kind=generative,sample_size=0",
        "--output", f"kind=json,path={output_path}",
        "--disable-console-interactive",
    ]
    return args


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
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "guidellm.json"
    timeout_s = plan["load"]["timeout_s"]
    with open(out_dir / "guidellm.log", "w", encoding="utf-8") as log:
        process = subprocess.Popen([executable, *command(plan, target, str(report))], stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   env={**os.environ, **(env or {})}, start_new_session=True)
        try:
            code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _stop(process)
            raise LoadError(f"GuideLLM did not finish within {timeout_s:g} s and was stopped; see guidellm.log",
                            timed_out=True) from None
    if code != 0 or not report.is_file():
        raise LoadError(f"GuideLLM exited with code {code}; see guidellm.log")
    return report


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


def flat_metrics(level_list: list[Mapping]) -> dict[str, float]:
    """The comparison's per-round figures from the usable levels (keys as in plan.metric_family)."""
    out: dict[str, float] = {}
    for level in level_list:
        c = level.get("concurrency")
        if c is None or not level.get("usable"):
            continue
        pairs = {f"c{c}.ttft_ms_median": (level.get("ttft_ms") or {}).get("median"),
                 f"c{c}.itl_ms_median": (level.get("itl_ms") or {}).get("median"),
                 f"c{c}.output_tokens_per_s": level.get("output_tokens_per_s"),
                 f"c{c}.tokens_per_joule": (level.get("energy") or {}).get("tokens_per_joule")}
        out.update({k: float(v) for k, v in pairs.items() if isinstance(v, (int, float)) and v > 0})
    return out
