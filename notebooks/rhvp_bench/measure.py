# SPDX-License-Identifier: Apache-2.0
"""Timing the server through its OpenAI-compatible API, and reading the server pod's start from Kubernetes.

How a request is timed (sections 2 to 4):

- Each request is a streamed `POST /v1/completions` with `temperature: 0`, `max_tokens: N`, `ignore_eos: true` and
  `stream_options: {include_usage: true}`, on a prompt of its own from prompts.jsonl with a fresh tag in front.
- The clock starts just before the request is sent. Time to first token ends at the first event whose text is not
  empty (a first event with no text, such as a chat stream's role-only event, does not count).
- Tokens are counted from the server's `usage.completion_tokens`, never from the number of events: Ray Serve LLM
  sends tokens in groups, several events in one network write.
- Decode speed is (N - 1) / (time of the last text event - time of the first text event).
- A request whose `completion_tokens` is not exactly N is dropped and counted as dropped; a request that fails (an
  HTTP error, an error event in the stream, a stream that ends before `[DONE]`) is counted as failed.

The startup section reads the server pod through the Kubernetes REST API with the notebook pod's own service account
token (plain httpx, no Kubernetes client package) and takes the time from the pod's PodScheduled condition to its
Ready condition.

Error texts never carry a URL or a host name: the notebook shows them.
"""

import asyncio
import codecs
import json
import math
import os
import secrets
import ssl
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections.abc import Callable, Iterable

import httpx

from . import prompts as prompt_set

MAX_TOKENS = 128
COMPLETIONS = "/v1/completions"
CHAT = "/v1/chat/completions"
MODELS = "/v1/models"
PATHS = {"completions": COMPLETIONS, "chat": CHAT}
READ_TIMEOUT_S = 600.0
CONNECT_TIMEOUT_S = 10.0
#: The seeds that pick each section's prompts from the prompt set; a warm-up uses its section's seed plus WARMUP_SEED.
SEEDS = {"first_token": 2, "decode": 3, "throughput": 4}
WARMUP_SEED = 100
SERVICE_ACCOUNT = Path("/var/run/secrets/kubernetes.io/serviceaccount")

COUNTED, DROPPED, FAILED = "counted", "dropped", "failed"


# ---- reading a stream ----

class EventSplitter:
    """Server-sent events out of network reads: feed the bytes of each read, get back the complete events' data.

    A read can hold several events, or end in the middle of one; comment lines (": ...") are skipped; lines may end
    in CRLF.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""
        self._data: list[str] = []

    def feed(self, chunk: bytes | str) -> list[str]:
        self._buffer += chunk if isinstance(chunk, str) else self._decoder.decode(chunk)
        events = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                if self._data:
                    events.append("\n".join(self._data))
                    self._data = []
                continue
            if line.startswith(":"):
                continue
            name, _, value = line.partition(":")
            if name == "data":
                self._data.append(value[1:] if value.startswith(" ") else value)
        return events

    def close(self) -> list[str]:
        """The last event, when the stream ended without the blank line that closes it."""
        events = self.feed("\n\n") if (self._buffer or self._data) else []
        return events


@dataclass
class StreamResult:
    """One streamed request: when it was sent, when text first and last came back, and the server's token counts."""

    sent_at: float
    finished_at: float = 0.0
    first_token_at: float | None = None
    last_token_at: float | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    text_events: int = 0
    done: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.done

    @property
    def ttft_s(self) -> float | None:
        return None if self.first_token_at is None else self.first_token_at - self.sent_at

    @property
    def decode_tokens_per_s(self) -> float | None:
        """(N - 1) tokens over the time from the first text to the last; None when that time is zero."""
        if self.completion_tokens is None or self.completion_tokens < 2:
            return None
        if self.first_token_at is None or self.last_token_at is None:
            return None
        span = self.last_token_at - self.first_token_at
        return (self.completion_tokens - 1) / span if span > 0 else None


def _choice_text(choice: dict) -> str:
    """The generated text in one choice: `text` (completions) or `delta.content` (chat)."""
    if isinstance(choice.get("text"), str):
        return choice["text"]
    delta = choice.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
        return delta["content"]
    return ""


def _event_error(event: dict) -> str | None:
    """The error an event carries, if it is an error event (vLLM writes {"error": {...}} or {"object": "error"})."""
    error = event.get("error")
    if error is not None:
        if isinstance(error, dict):
            return str(error.get("type") or error.get("code") or "error event")
        return "error event"
    if event.get("object") == "error":
        return str(event.get("type") or event.get("code") or "error event")
    return None


def read_events(timed: Iterable[tuple[float, str]], sent_at: float, finished_at: float | None = None,
                result: StreamResult | None = None) -> StreamResult:
    """Fold a stream's events, each with the time its network read arrived, into a StreamResult."""
    out = result if result is not None else StreamResult(sent_at=sent_at)
    last_time = sent_at
    for at, payload in timed:
        last_time = at
        if out.error is not None:
            continue
        if payload.strip() == "[DONE]":
            out.done = True
            continue
        try:
            event = json.loads(payload)
        except ValueError:
            out.error = "an event that is not JSON"
            continue
        if not isinstance(event, dict):
            out.error = "an event that is not a JSON object"
            continue
        error = _event_error(event)
        if error is not None:
            out.error = f"error in the stream: {error}"
            continue
        for choice in event.get("choices") or []:
            if isinstance(choice, dict) and _choice_text(choice):
                out.text_events += 1
                if out.first_token_at is None:
                    out.first_token_at = at
                out.last_token_at = at
        usage = event.get("usage")
        if isinstance(usage, dict):
            if isinstance(usage.get("completion_tokens"), int):
                out.completion_tokens = usage["completion_tokens"]
            if isinstance(usage.get("prompt_tokens"), int):
                out.prompt_tokens = usage["prompt_tokens"]
    out.finished_at = finished_at if finished_at is not None else last_time
    if out.error is None and not out.done:
        out.error = "the stream ended before [DONE]"
    return out


def outcome(result: StreamResult, max_tokens: int) -> str:
    """COUNTED, DROPPED (did not generate exactly `max_tokens` tokens) or FAILED (an error, or no text at all)."""
    if not result.ok:
        return FAILED
    if result.completion_tokens != max_tokens:
        return DROPPED
    if result.first_token_at is None:
        return FAILED
    return COUNTED


@dataclass
class Tally:
    """A set of requests sorted into counted, dropped and failed."""

    counted: list[StreamResult] = field(default_factory=list)
    dropped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def sent(self) -> int:
        return len(self.counted) + self.dropped + self.failed


def tally(results: Iterable[StreamResult], max_tokens: int) -> Tally:
    out = Tally()
    for result in results:
        kind = outcome(result, max_tokens)
        if kind == COUNTED:
            out.counted.append(result)
        elif kind == DROPPED:
            out.dropped += 1
        else:
            out.failed += 1
            reason = result.error or "no text came back"
            if reason not in out.errors:
                out.errors.append(reason)
    return out


# ---- statistics ----

def percentile(values: Iterable[float], q: float) -> float:
    """The q-th percentile (0-100), interpolating between the two nearest ranks."""
    data = sorted(values)
    if not data:
        raise ValueError("no values")
    if len(data) == 1:
        return data[0]
    rank = (len(data) - 1) * q / 100.0
    low, high = math.floor(rank), math.ceil(rank)
    return data[low] + (data[high] - data[low]) * (rank - low)


def median(values: Iterable[float]) -> float:
    return percentile(values, 50)


def p90(values: Iterable[float]) -> float:
    return percentile(values, 90)


# ---- requests ----

def tagged(prompt: str) -> str:
    """The prompt with a fresh tag in front: no two requests share a prefix for the server's prefix cache."""
    return f"{secrets.token_hex(4)}: {prompt}"


def request_body(model: str, prompt: str, max_tokens: int, api: str = "completions", stream: bool = True) -> dict:
    body: dict = {"model": model, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True, "stream": stream}
    if api == "chat":
        body["messages"] = [{"role": "user", "content": prompt}]
    else:
        body["prompt"] = prompt
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def async_client(endpoint: str, transport: httpx.AsyncBaseTransport | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=endpoint,
        timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
        limits=httpx.Limits(max_connections=128, max_keepalive_connections=128),
        transport=transport,
    )


def describe(error: BaseException) -> str:
    """An exception as the notebook may show it: its kind only (httpx messages carry the URL)."""
    return type(error).__name__


async def stream_one(client: httpx.AsyncClient, model: str, prompt: str, max_tokens: int, api: str = "completions",
                     on_write: Callable[[float, bytes], None] | None = None) -> StreamResult:
    """Send one streamed request and time it. `on_write(seconds since sent, bytes)` sees every network read."""
    body = request_body(model, prompt, max_tokens, api)
    splitter = EventSplitter()
    timed: list[tuple[float, str]] = []
    sent_at = time.perf_counter()
    result = StreamResult(sent_at=sent_at)
    try:
        async with client.stream("POST", PATHS[api], json=body) as response:
            if response.status_code != 200:
                await response.aread()
                result.error = f"HTTP {response.status_code}"
                result.finished_at = time.perf_counter()
                return result
            async for chunk in response.aiter_raw():
                at = time.perf_counter()
                if on_write is not None:
                    on_write(at - sent_at, chunk)
                timed.extend((at, payload) for payload in splitter.feed(chunk))
            at = time.perf_counter()
            timed.extend((at, payload) for payload in splitter.close())
    except httpx.HTTPError as error:
        read_events(timed, sent_at, time.perf_counter(), result)
        result.error = describe(error)
        return result
    return read_events(timed, sent_at, time.perf_counter(), result)


@dataclass
class Progress:
    """How far a measurement has got; written by the measuring thread, read by the notebook's."""

    done: int = 0
    total: int = 0


@dataclass
class Sequential:
    """Requests sent one at a time, after a warm-up."""

    max_tokens: int
    warmup: int
    results: list[StreamResult]

    @property
    def tally(self) -> Tally:
        return tally(self.results, self.max_tokens)

    def ttfts(self) -> list[float]:
        return [r.ttft_s for r in self.tally.counted if r.ttft_s is not None]

    def rates(self) -> list[float]:
        return [r.decode_tokens_per_s for r in self.tally.counted if r.decode_tokens_per_s is not None]

    def prompt_tokens(self) -> int | None:
        counts = [r.prompt_tokens for r in self.tally.counted if r.prompt_tokens is not None]
        return round(median(counts)) if counts else None


@dataclass
class Level:
    """One concurrency level: `streams` requests sent at the same time."""

    streams: int
    max_tokens: int
    results: list[StreamResult]

    @property
    def tally(self) -> Tally:
        return tally(self.results, self.max_tokens)

    @property
    def wall_s(self) -> float:
        """From the first request sent to the last stream finished."""
        return max(r.finished_at for r in self.results) - min(r.sent_at for r in self.results)

    def tokens_per_s(self) -> float | None:
        """All counted requests' tokens over the wall time; None when nothing was counted."""
        counted = self.tally.counted
        if not counted or self.wall_s <= 0:
            return None
        return sum(r.completion_tokens or 0 for r in counted) / self.wall_s

    def prompt_tokens(self) -> int | None:
        counts = [r.prompt_tokens for r in self.tally.counted if r.prompt_tokens is not None]
        return round(median(counts)) if counts else None


def choose(pool: list[str], count: int, warmup: int, seed: int) -> tuple[list[str], list[str]]:
    """A section's prompts and its warm-up prompts: two seeded subsets of the prompt set with nothing in common."""
    measured = prompt_set.pick(pool, count, seed)
    warm = prompt_set.pick(pool, warmup, seed + WARMUP_SEED, avoid=frozenset(measured))
    return [pool[i] for i in measured], [pool[i] for i in warm]


async def sequential(endpoint: str, model: str, texts: list[str], warmup_texts: list[str], max_tokens: int,
                     progress: Progress | None = None, transport: httpx.AsyncBaseTransport | None = None,
                     api: str = "completions") -> Sequential:
    """The warm-up requests (not kept), then the timed requests, each sent once the one before has finished."""
    progress = progress if progress is not None else Progress()
    progress.total = len(warmup_texts) + len(texts)
    async with async_client(endpoint, transport) as client:
        for text in warmup_texts:
            await stream_one(client, model, tagged(text), max_tokens, api)
            progress.done += 1
        results = []
        for text in texts:
            results.append(await stream_one(client, model, tagged(text), max_tokens, api))
            progress.done += 1
    return Sequential(max_tokens=max_tokens, warmup=len(warmup_texts), results=results)


async def levels(endpoint: str, model: str, level_texts: dict[int, list[str]], warmup_texts: list[str],
                 max_tokens: int, progress: Progress | None = None,
                 transport: httpx.AsyncBaseTransport | None = None, api: str = "completions") -> list[Level]:
    """The warm-up requests one at a time (not kept), then each level's requests all at once, level after level."""
    progress = progress if progress is not None else Progress()
    progress.total = len(warmup_texts) + sum(len(t) for t in level_texts.values())
    out = []
    async with async_client(endpoint, transport) as client:
        for text in warmup_texts:
            await stream_one(client, model, tagged(text), max_tokens, api)
            progress.done += 1

        async def one(text: str) -> StreamResult:
            result = await stream_one(client, model, tagged(text), max_tokens, api)
            progress.done += 1
            return result

        for streams, texts in level_texts.items():
            results = await asyncio.gather(*(one(text) for text in texts))
            out.append(Level(streams=streams, max_tokens=max_tokens, results=list(results)))
    return out


def run_async(coroutine, tick: Callable[[], None] | None = None, every_s: float = 0.5):
    """Run `coroutine` to its end on an event loop of its own in a worker thread (the notebook's kernel already runs
    a loop), calling `tick()` in this thread while it runs. Returns the result or raises the error."""
    box: dict = {}

    def work() -> None:
        try:
            box["value"] = asyncio.run(coroutine)
        except BaseException as error:  # handed to the caller below
            box["error"] = error

    worker = threading.Thread(target=work, name="rhvp-measure", daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(every_s)
        if tick is not None and worker.is_alive():
            tick()
    if "error" in box:
        raise box["error"]
    return box["value"]


# ---- readiness ----

@dataclass
class Readiness:
    ready: bool
    step: str
    waited_s: float


def ready_step(client: httpx.Client, model: str) -> str | None:
    """The first check the server does not pass yet ("answers", "listed", "first"), or None when it passes all:
    it answers /v1/models, lists the model, and returns a one-token completion."""
    try:
        response = client.get(MODELS)
    except httpx.HTTPError:
        return "answers"
    if response.status_code != 200:
        return "answers"
    try:
        listed = [m.get("id") for m in response.json().get("data", [])]
    except (ValueError, AttributeError):
        return "answers"
    if model not in listed:
        return "listed"
    try:
        response = client.post(COMPLETIONS, json=request_body(model, "Hello", 1, stream=False))
        choices = response.json().get("choices") if response.status_code == 200 else None
    except (httpx.HTTPError, ValueError, AttributeError):
        return "first"
    return None if choices else "first"


def wait_ready(endpoint: str, model: str, timeout_s: float = 3600.0, poll_s: float = 5.0,
               on_wait: Callable[[str, float], None] | None = None,
               transport: httpx.BaseTransport | None = None) -> Readiness:
    """Check until the server passes every readiness check, or `timeout_s` has gone by."""
    start = time.monotonic()
    with httpx.Client(base_url=endpoint, timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
                      transport=transport) as client:
        while True:
            step = ready_step(client, model)
            waited = time.monotonic() - start
            if step is None:
                return Readiness(ready=True, step="", waited_s=waited)
            if waited >= timeout_s:
                return Readiness(ready=False, step=step, waited_s=waited)
            if on_wait is not None:
                on_wait(step, waited)
            time.sleep(poll_s)


# ---- the server pod's start ----

class PodReadError(Exception):
    """The pods could not be read; the message is safe to show (no URL)."""


def in_cluster() -> bool:
    return bool(os.environ.get("KUBERNETES_SERVICE_HOST")) and (SERVICE_ACCOUNT / "token").is_file()


def read_pods(namespace: str, selector: str, timeout_s: float = 15.0) -> list[dict]:
    """The pods in `namespace` that match the label `selector`, through the in-cluster Kubernetes API."""
    host = os.environ["KUBERNETES_SERVICE_HOST"]
    port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
    if ":" in host:
        host = f"[{host}]"
    token = (SERVICE_ACCOUNT / "token").read_text(encoding="utf-8").strip()
    context = ssl.create_default_context(cafile=str(SERVICE_ACCOUNT / "ca.crt"))
    try:
        with httpx.Client(base_url=f"https://{host}:{port}", verify=context, timeout=timeout_s,
                          headers={"Authorization": f"Bearer {token}"}) as client:
            response = client.get(f"/api/v1/namespaces/{namespace}/pods", params={"labelSelector": selector})
    except httpx.HTTPError as error:
        raise PodReadError(describe(error)) from None
    if response.status_code != 200:
        raise PodReadError(f"HTTP {response.status_code}")
    try:
        return list(response.json().get("items") or [])
    except (ValueError, AttributeError):
        raise PodReadError("an answer that is not a pod list") from None


def parse_time(text: str) -> datetime:
    """A Kubernetes timestamp (RFC 3339, whole seconds, "Z")."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _condition(pod: dict, kind: str) -> dict | None:
    for condition in (pod.get("status") or {}).get("conditions") or []:
        if condition.get("type") == kind:
            return condition
    return None


def _is_ready(pod: dict) -> bool:
    ready = _condition(pod, "Ready")
    return bool(ready) and ready.get("status") == "True"


def choose_pod(items: list[dict]) -> dict | None:
    """The server pod to read: the newest pod that is not being deleted, a Ready one first."""
    live = [pod for pod in items if not (pod.get("metadata") or {}).get("deletionTimestamp")]
    if not live:
        return None
    newest = sorted(live, key=lambda pod: (pod.get("metadata") or {}).get("creationTimestamp") or "", reverse=True)
    for pod in newest:
        if _is_ready(pod):
            return pod
    return newest[0]


def _container(spec_or_status: list[dict], name: str) -> dict | None:
    for item in spec_or_status:
        if item.get("name") == name:
            return item
    return spec_or_status[0] if spec_or_status else None


#: The server container's storage settings that the startup card names, in order.
STORAGE_VARIABLES = ("TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "FLASHINFER_WORKSPACE_BASE", "XDG_CACHE_HOME", "HF_HOME")


def storage(pod: dict, container: str = "server") -> tuple[list[str], list[str]]:
    """The server container's storage settings (STORAGE_VARIABLES) that were empty when the pod started (an emptyDir,
    or no volume at all) and those kept between starts (a persistent volume claim), read from the pod's spec."""
    spec = pod.get("spec") or {}
    box = _container(spec.get("containers") or [], container) or {}
    volumes = {v.get("name"): v for v in spec.get("volumes") or []}
    env = {e.get("name"): e.get("value") for e in box.get("env") or [] if isinstance(e.get("value"), str)}
    mounts = box.get("volumeMounts") or []
    empty, kept = [], []
    for variable in STORAGE_VARIABLES:
        path = env.get(variable)
        if not path:
            continue
        under = [m for m in mounts
                 if path == m.get("mountPath") or path.startswith(str(m.get("mountPath", "")).rstrip("/") + "/")]
        if not under:
            empty.append(variable)
            continue
        mount = max(under, key=lambda m: len(m.get("mountPath", "")))
        volume = volumes.get(mount.get("name")) or {}
        if "emptyDir" in volume:
            empty.append(variable)
        elif "persistentVolumeClaim" in volume:
            kept.append(variable)
    return empty, kept


@dataclass
class Startup:
    """The server pod's latest start. `state` is "ok", "no_pod", "not_ready" or "restarted"."""

    state: str
    to_ready_s: float | None = None
    to_start_s: float | None = None
    start_to_ready_s: float | None = None
    restarts: int = 0
    empty: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)


def startup(items: list[dict], container: str = "server") -> Startup:
    """The newest server pod's time from PodScheduled to Ready, with the container start in between."""
    pod = choose_pod(items)
    if pod is None:
        return Startup(state="no_pod")
    status = pod.get("status") or {}
    box = _container(status.get("containerStatuses") or [], container) or {}
    restarts = int(box.get("restartCount") or 0)
    empty, kept = storage(pod, container)
    scheduled, ready = _condition(pod, "PodScheduled"), _condition(pod, "Ready")
    started = ((box.get("state") or {}).get("running") or {}).get("startedAt")
    if not _is_ready(pod) or not scheduled or not scheduled.get("lastTransitionTime") or not started:
        return Startup(state="not_ready", restarts=restarts, empty=empty, kept=kept)
    if restarts:
        return Startup(state="restarted", restarts=restarts, empty=empty, kept=kept)
    t_scheduled = parse_time(scheduled["lastTransitionTime"])
    t_started = parse_time(started)
    t_ready = parse_time(ready["lastTransitionTime"])
    return Startup(state="ok", to_ready_s=(t_ready - t_scheduled).total_seconds(),
                   to_start_s=(t_started - t_scheduled).total_seconds(),
                   start_to_ready_s=(t_ready - t_started).total_seconds(), restarts=0, empty=empty, kept=kept)
