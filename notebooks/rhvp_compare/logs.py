# SPDX-License-Identifier: Apache-2.0
"""What a server's own log says about its start: a breakdown into phases, and the vLLM version it reports.

The log comes with one timestamp per line (Kubernetes `pods/log?timestamps=true` and `docker logs --timestamps` both
put an RFC 3339 time first). Two kinds of entry come out:

- phases the server times itself ("Loading weights took 1.23 seconds"): `seconds` is the server's own figure;
- moments ("Application startup complete."): `at_s` is the line's time since the launch asked for the server, by the
  node's clock.

The patterns follow vLLM's log lines and change between vLLM versions, so a breakdown lists only what it found; a
phase that is missing was not logged in a form this table knows, which says nothing about the server.

`scale_events` reads the rhvp-scale lines the pattern's own server writes when its replica count changes. Standard
library only.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from collections.abc import Iterable

#: Phases a server times itself: (key, label, pattern with the seconds in group "s").
TIMED = (
    ("model_init", "initializing the model", r"Initializing model took (?P<s>[0-9.]+) ?s"),
    ("weights_load", "loading the weights", r"Loading weights took (?P<s>[0-9.]+) ?s"),
    ("model_load", "loading the model", r"Model loading took [0-9.]+ ?G(?:i)?B(?: memory)? and (?P<s>[0-9.]+) ?s"),
    ("dynamo_transform", "torch.compile bytecode transform", r"Dynamo bytecode transform time: (?P<s>[0-9.]+) ?s"),
    ("graph_compile", "compiling the graph", r"Compiling a graph for .*? takes (?P<s>[0-9.]+) ?s"),
    ("torch_compile", "torch.compile in total", r"torch\.compile (?:takes|took) (?P<s>[0-9.]+) ?s in total"),
    ("profile_warmup", "the first profiling and warm-up run", r"Initial profiling/warmup run took (?P<s>[0-9.]+) ?s"),
    ("cuda_graphs", "capturing CUDA graphs", r"Graph capturing finished in (?P<s>[0-9.]+) ?sec"),
    ("engine_init", "engine start (profile, KV cache, warm-up)",
     r"init engine \(profile, create kv cache, warmup model\) took (?P<s>[0-9.]+) ?s"),
    ("ray_app_start", "the Ray Serve LLM app start (this pattern's server)", r"app '[^']+' is RUNNING after "
     r"(?P<s>[0-9.]+) s"),
)

#: Moments: (key, label, pattern). The first line of the log is always the moment "first_log".
MOMENTS = (
    ("api_server_start", "the API server starts", r"vLLM API server version|Starting vLLM API server|"
     r"\[startup\.py:\d+\].*\bversion [0-9]"),
    ("engine_start", "the engine starts", r"Initializing a V\d LLM engine"),
    ("ready_file", "the ready file is written (this pattern's server)", r"ready file written:"),
    ("app_startup_complete", "the HTTP server is up", r"Application startup complete\."),
    ("worker_start", "the HTTP worker starts", r"Starting worker \["),
)

VERSION = (
    re.compile(r"vLLM API server version (?P<v>[0-9][\w.+-]*)"),
    re.compile(r"Initializing a V\d LLM engine \(v(?P<v>[0-9][\w.+-]*)\)"),
    re.compile(r"\bvLLM version (?P<v>[0-9][\w.+-]*)"),
    re.compile(r"\[startup\.py:\d+\].*\bversion (?P<v>[0-9][\w.+-]*)"),
)

#: vLLM's start-up check that the free GPU memory did not grow while it measured (vllm/v1/worker/gpu_worker.py). On a
#: GPU whose memory is the machine's own, another program that frees memory in those seconds fails it.
MEMORY_PROFILING_ERROR = "Error in memory profiling"

_STAMP = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2}))\s?(?P<text>.*)$")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse_time(text: str) -> float:
    """An RFC 3339 time ("Z" or an offset; digits past microseconds are dropped) as seconds since the epoch."""
    head, dot, rest = text.partition(".")
    if dot:
        digits = len(rest) - len(rest.lstrip("0123456789"))
        text = f"{head}.{rest[:min(digits, 6)]}{rest[digits:]}"
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def lines(log: str) -> list[tuple[float | None, str]]:
    """(time since the epoch or None, text) for each line of a timestamped log, colour codes removed."""
    out = []
    for raw in log.splitlines():
        raw = _ANSI.sub("", raw)
        match = _STAMP.match(raw)
        if match:
            try:
                out.append((parse_time(match.group("ts")), match.group("text")))
                continue
            except ValueError:
                pass
        out.append((None, raw))
    return out


def server_version(text_lines: Iterable[tuple[float | None, str]]) -> str | None:
    """The vLLM version the server's log reports, or None."""
    for _, text in text_lines:
        for pattern in VERSION:
            match = pattern.search(text)
            if match:
                return match.group("v")
    return None


def memory_profiling_failed(log: str) -> bool:
    """Whether the server's log shows vLLM's start-up memory-profiling check failing."""
    return MEMORY_PROFILING_ERROR in log


def breakdown(log: str, started_epoch: float) -> list[dict]:
    """The start-up phases found in `log`, in log order. `started_epoch` is when the launch asked for the server."""
    out: list[dict] = []
    seen: set[str] = set()
    timed: dict[str, dict] = {}
    parsed = lines(log)
    first = next((t for t, _ in parsed if t is not None), None)
    if first is not None:
        out.append({"phase": "first_log", "label": "the server writes its first log line", "seconds": None,
                    "at_s": round(first - started_epoch, 3), "source": "log"})
    for at, text in parsed:
        since = round(at - started_epoch, 3) if at is not None else None
        for key, label, pattern in TIMED:
            match = re.search(pattern, text)
            if not match:
                continue
            if key in timed:  # logged more than once (for example one CUDA graph capture per graph): add them up
                entry = timed[key]
                entry["seconds"] = round(entry["seconds"] + float(match.group("s")), 6)
                entry["count"] += 1
                entry["at_s"] = since
                continue
            timed[key] = {"phase": key, "label": label, "seconds": float(match.group("s")), "at_s": since,
                          "source": "log", "count": 1}
            out.append(timed[key])
        for key, label, pattern in MOMENTS:
            if key not in seen and re.search(pattern, text):
                seen.add(key)
                out.append({"phase": key, "label": label, "seconds": None, "at_s": since, "source": "log"})
    return out


# ---- the rhvp-scale line ----

#: The line the pattern's own server writes each time a replica count of its Ray Serve app changes. Its shape is
#: defined once, in inference-server/rhvp_scale.py (another image, so these names are a copy);
#: tests/test_scaleout_contract.py keeps the copy and the parser below in step with it.
SCALE_EVENT = "rhvp-scale"
SCALE_FIELDS = ("event", "ts", "app", "deployment", "from", "to", "reason")
SCALE_REASONS = ("target", "running")
SCALE_MARKER = '{"event":"rhvp-scale"'


def _scale_record(raw: object) -> dict | None:
    """The event in `raw` (a decoded JSON value) if it has the whole shape, else None."""
    if not isinstance(raw, dict) or any(field not in raw for field in SCALE_FIELDS) or raw["event"] != SCALE_EVENT:
        return None
    ts = raw["ts"]
    if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
        return None
    for field in ("app", "deployment"):
        if not isinstance(raw[field], str) or not raw[field]:
            return None
    for field in ("from", "to"):
        if isinstance(raw[field], bool) or not isinstance(raw[field], int) or raw[field] < 0:
            return None
    if raw["reason"] not in SCALE_REASONS:
        return None
    return {field: raw[field] for field in SCALE_FIELDS}


def scale_events(log: str) -> list[dict]:
    """Every rhvp-scale event in a server log, in log order: its fields, with `ts` as a float, plus `log_ts`, the
    log line's own timestamp in seconds since the epoch (None for a line without one). The event may follow a
    prefix on its line. A line with the marker that does not hold a whole, well-typed event is left out."""
    decoder = json.JSONDecoder()
    out = []
    for at, text in lines(log):
        start = text.find(SCALE_MARKER)
        if start < 0:
            continue
        try:
            raw, _ = decoder.raw_decode(text, start)
        except ValueError:
            continue
        record = _scale_record(raw)
        if record is not None:
            record["ts"] = float(record["ts"])
            record["log_ts"] = at
            out.append(record)
    return out


def first_scale_up(events: Iterable[dict], since_epoch: float) -> float | None:
    """Seconds from `since_epoch` to the first decision to add a replica that the server logged at or after it (an
    event with reason target and more replicas after than before), by the event's own `ts`; None when there is
    none."""
    times = [e["ts"] for e in events if e.get("reason") == "target" and e["to"] > e["from"] and e["ts"] >= since_epoch]
    return min(times) - since_epoch if times else None
