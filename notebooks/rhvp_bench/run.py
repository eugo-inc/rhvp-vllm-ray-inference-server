# SPDX-License-Identifier: Apache-2.0
"""What the notebook's code cells call: one function per cell.

Settings come from the environment (the notebook chart sets them):

- RHVP_ENDPOINT: the server's base URL (default: the in-cluster Service). Never shown on screen.
- RHVP_MODEL_ID: the model name the server serves; RHVP_MODEL_REVISION, if set, is shown beside it.
- RHVP_SERVER_NAMESPACE and RHVP_SERVER_LABEL_SELECTOR: where the startup section finds the server pod.
- RHVP_INTERNAL=1: an amber INTERNAL strip above every result.
- RHVP_REPLAY=<recording>: REPLAY mode. The stand-in server from replay.py starts in this process, every request
  goes to it, the amber REPLAY strip sits above every result, and the startup section reads the recording's pods.
"""

import os

from . import draw, measure, prompts, replay, words

DEFAULT_ENDPOINT = "http://ray-serve-llm.rhvp-inference.svc:8000"
DEFAULT_MODEL = "qwen2.5-0.5b-instruct"
DEFAULT_NAMESPACE = "rhvp-inference"
DEFAULT_SELECTOR = "app.kubernetes.io/name=ray-serve-llm"
SERVER_CONTAINER = "server"
ON = ("1", "true", "yes", "on")


class Session:
    """This kernel's settings, and the stand-in server when REPLAY is on."""

    def __init__(self) -> None:
        self.internal = os.environ.get("RHVP_INTERNAL", "").strip().lower() in ON
        self.namespace = os.environ.get("RHVP_SERVER_NAMESPACE") or DEFAULT_NAMESPACE
        self.selector = os.environ.get("RHVP_SERVER_LABEL_SELECTOR") or DEFAULT_SELECTOR
        self.recording: dict | None = None
        self.server: replay.FakeServer | None = None
        replay_path = os.environ.get("RHVP_REPLAY", "").strip()
        if replay_path:
            self.recording = replay.load(replay_path)
            meta = self.recording["metadata"]
            self.server = replay.FakeServer(self.recording)
            self.endpoint = self.server.start()
            self.model_id = str(meta.get("model_id") or DEFAULT_MODEL)
            self.revision = str(meta.get("model_revision") or "")
            self.max_tokens = int(meta.get("max_tokens") or measure.MAX_TOKENS)
        else:
            self.endpoint = os.environ.get("RHVP_ENDPOINT") or DEFAULT_ENDPOINT
            self.model_id = os.environ.get("RHVP_MODEL_ID") or DEFAULT_MODEL
            self.revision = os.environ.get("RHVP_MODEL_REVISION", "").strip()
            self.max_tokens = measure.MAX_TOKENS
        self.pool = prompts.load()

    @property
    def replaying(self) -> bool:
        return self.recording is not None

    @property
    def synthetic(self) -> bool:
        return bool(self.recording and self.recording["metadata"].get("synthetic"))

    def strips(self) -> list[str]:
        kinds = (["REPLAY"] if self.replaying else []) + (["INTERNAL"] if self.internal else [])
        return [draw.strip(kind) for kind in kinds]

    def label(self, text: str) -> str:
        return f"{words.IN_THIS_REPLAY if self.replaying else words.IN_THIS_RUN} · {text}"


_SESSION: Session | None = None


def session() -> Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = Session()
    return _SESSION


def _session_or_notice() -> Session | None:
    try:
        return session()
    except replay.RecordingError as error:
        draw.show(draw.notice(words.RECORDING_UNUSABLE.format(reason=str(error))))
        return None


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _whole(seconds: float) -> str:
    return f"{seconds:.0f}"


# ---- the readiness cell ----

def ready(timeout_minutes: float = 60) -> None:
    """Wait until the server answers, lists the model and returns a one-token completion; then show what it runs."""
    s = _session_or_notice()
    if s is None:
        return
    handle = draw.show(*s.strips(), draw.quiet_line(
        words.READY_WAITING.format(step=words.READY_STEP["answers"], seconds=0), ok=False))

    def waiting(step: str, seconds: float) -> None:
        draw.update(handle, *s.strips(), draw.quiet_line(
            words.READY_WAITING.format(step=words.READY_STEP[step], seconds=seconds), ok=False))

    state = measure.wait_ready(s.endpoint, s.model_id, timeout_s=timeout_minutes * 60, on_wait=waiting)
    if not state.ready:
        draw.update(handle, *s.strips(), draw.notice(
            words.NOT_READY.format(minutes=timeout_minutes, step=words.READY_STEP[state.step])))
        return
    lines = [words.READY_ANSWERS, words.READY_LISTED, words.READY_FIRST]
    if s.replaying:
        lines.append(words.REPLAY_READY)
        if s.synthetic:
            lines.append(words.REPLAY_SYNTHETIC)
    lines += [words.READY_SOFTWARE.format(baseline=words.BASELINE), words.READY_MODEL.format(model_id=s.model_id)]
    if s.revision:
        lines.append(words.READY_REVISION.format(revision=s.revision))
    draw.update(handle, *s.strips(), *(draw.quiet_line(line) for line in lines))


# ---- section 1 ----

def startup() -> None:
    """The server pod's latest start, from being placed on a node to Ready, from its Kubernetes conditions."""
    draw.show(draw.side_banner(words.WITHOUT))
    s = _session_or_notice()
    if s is None:
        return
    if s.replaying:
        items = list(s.recording["pods"])
    elif not measure.in_cluster():
        draw.show(draw.notice(words.STARTUP_NOT_IN_CLUSTER))
        return
    else:
        try:
            items = measure.read_pods(s.namespace, s.selector)
        except measure.PodReadError as error:
            draw.show(draw.notice(words.STARTUP_UNREADABLE.format(error=str(error))))
            return
    start = measure.startup(items, SERVER_CONTAINER)
    if start.state == "no_pod":
        draw.show(draw.notice(words.STARTUP_NO_POD))
        return
    if start.state == "not_ready":
        draw.show(draw.notice(words.STARTUP_NOT_READY))
        return
    if start.state == "restarted":
        draw.show(draw.notice(words.STARTUP_RESTARTED.format(restarts=start.restarts)))
        return
    notes = [words.STARTUP_STEPS.format(to_start=_whole(start.to_start_s), to_ready=_whole(start.start_to_ready_s))]
    if start.empty:
        notes.append(words.STARTUP_EMPTY.format(names=_join([words.CACHE_NAMES[v] for v in start.empty])))
    if start.kept:
        notes.append(words.STARTUP_KEPT.format(names=_join([words.CACHE_NAMES[v] for v in start.kept])))
    notes += [words.STARTUP_SECONDS, words.ONE_MEASUREMENT, words.SINGLE_RUN, words.DRAW_MOVES]
    draw.show(*s.strips(), draw.result_card(s.label(words.STARTUP_LABEL), _whole(start.to_ready_s), words.SECONDS,
                                            notes))


# ---- the WITH EUGO side (sections 1 and 2) ----

def with_eugo() -> None:
    """The WITH EUGO side: its banner and the neutral line, nothing else."""
    draw.show(draw.side_banner(words.WITH), draw.notice(words.NO_MEASURED))


# ---- sections 2 to 4 ----

def _measure(s: Session, start, total: int):
    """Run a measurement (`start(progress)` makes its coroutine), showing how far it has got; the result and the
    display handle to put the result in."""
    progress = measure.Progress(total=total)

    def line() -> str:
        return draw.quiet_line(words.MEASURING.format(done=progress.done, total=progress.total), ok=False)

    handle = draw.show(line())
    try:
        result = measure.run_async(start(progress), tick=lambda: draw.update(handle, line()))
    except Exception as error:  # shown without its message: an httpx message carries the URL
        draw.update(handle, draw.notice(words.SERVER_ERROR.format(error=measure.describe(error))))
        return None, handle
    return result, handle


def _requests(n: int) -> str:
    return words.REQUESTS.get(n) or words.REQUESTS_MANY.format(n=n)


def _not_counted(t: measure.Tally, tokens: int) -> list[str]:
    lines = []
    if t.dropped:
        lines.append(words.DROPPED.format(dropped=_requests(t.dropped), tokens=tokens))
    if t.failed:
        lines.append(words.FAILED.format(failed=_requests(t.failed), errors="; ".join(t.errors)))
    return lines


def _shape(prompt_tokens: int | None, tokens: int) -> str:
    if prompt_tokens is None:
        return words.REQUEST_SHAPE_UNCOUNTED.format(tokens=tokens)
    return words.REQUEST_SHAPE.format(prompt_tokens=prompt_tokens, tokens=tokens)


def _nothing_counted(s: Session, handle, t: measure.Tally) -> None:
    if not t.counted and t.failed == t.sent and t.errors:
        draw.update(handle, *s.strips(), draw.notice(words.SERVER_ERROR.format(error=t.errors[0])))
        return
    draw.update(handle, *s.strips(), draw.notice(words.NONE_COUNTED.format(tokens=s.max_tokens),
                                                 _not_counted(t, s.max_tokens)))


def first_token(requests: int = 16, warmup: int = 4) -> None:
    """Time to first token over `requests` requests sent one at a time, after `warmup` requests not counted."""
    draw.show(draw.side_banner(words.WITHOUT))
    s = _session_or_notice()
    if s is None:
        return
    texts, warm = measure.choose(s.pool, requests, warmup, measure.SEEDS["first_token"])
    result, handle = _measure(s, lambda p: measure.sequential(s.endpoint, s.model_id, texts, warm, s.max_tokens, p),
                              len(texts) + len(warm))
    if result is None:
        return
    t = result.tally
    ttfts = result.ttfts()
    if not ttfts:
        _nothing_counted(s, handle, t)
        return
    notes = [words.TTFT_SPREAD.format(p90=measure.p90(ttfts), kept=len(t.counted), sent=t.sent),
             *_not_counted(t, s.max_tokens), words.TTFT_TIMED, _shape(result.prompt_tokens(), s.max_tokens),
             words.ONE_AT_A_TIME, words.WARMUP.format(warmup=result.warmup), words.STREAM_GROUPS_NOTE,
             words.SINGLE_RUN, words.DRAW_MOVES]
    draw.update(handle, *s.strips(), draw.result_card(s.label(words.TTFT_LABEL), f"{measure.median(ttfts):.3f}",
                                                      words.SECONDS, notes))


def decode(streams: int = 4, warmup: int = 2) -> None:
    """Decode speed of one stream at a time, over `streams` requests, after `warmup` requests not counted."""
    draw.show(draw.side_banner(words.WITHOUT))
    s = _session_or_notice()
    if s is None:
        return
    texts, warm = measure.choose(s.pool, streams, warmup, measure.SEEDS["decode"])
    result, handle = _measure(s, lambda p: measure.sequential(s.endpoint, s.model_id, texts, warm, s.max_tokens, p),
                              len(texts) + len(warm))
    if result is None:
        return
    t = result.tally
    if not t.counted:
        _nothing_counted(s, handle, t)
        return
    rates = result.rates()
    if not rates:
        draw.update(handle, *s.strips(), draw.notice(words.NO_SPAN))
        return
    notes = [words.DECODE_SPREAD.format(low=min(rates), high=max(rates), kept=len(t.counted), sent=t.sent),
             *_not_counted(t, s.max_tokens), words.DECODE_TIMED, _shape(result.prompt_tokens(), s.max_tokens),
             words.ONE_AT_A_TIME, words.WARMUP.format(warmup=result.warmup), words.STREAM_GROUPS_NOTE,
             words.SINGLE_RUN, words.DRAW_MOVES]
    draw.update(handle, *s.strips(), draw.result_card(s.label(words.DECODE_LABEL), f"{measure.median(rates):.1f}",
                                                      words.TOKENS_PER_S, notes))


def throughput(levels: tuple[int, ...] = (1, 8, 32), warmup: int = 2) -> None:
    """All generated tokens per second with each number of streams in `levels` running at once."""
    draw.show(draw.side_banner(words.WITHOUT))
    s = _session_or_notice()
    if s is None:
        return
    texts, warm = measure.choose(s.pool, sum(levels), warmup, measure.SEEDS["throughput"])
    level_texts, at = {}, 0
    for streams in levels:
        level_texts[streams] = texts[at:at + streams]
        at += streams
    result, handle = _measure(s, lambda p: measure.levels(s.endpoint, s.model_id, level_texts, warm, s.max_tokens, p),
                              len(texts) + len(warm))
    if result is None:
        return
    if not any(level.tally.counted for level in result):
        _nothing_counted(s, handle, measure.tally([r for level in result for r in level.results], s.max_tokens))
        return
    cards, prompt_tokens = [], None
    for level in result:
        t = level.tally
        rate = level.tokens_per_s()
        prompt_tokens = prompt_tokens or level.prompt_tokens()
        label = words.THROUGHPUT_LABEL.get(level.streams) or words.THROUGHPUT_LABEL_MANY.format(streams=level.streams)
        notes = [words.THROUGHPUT_CARD.format(kept=len(t.counted), sent=t.sent, seconds=level.wall_s),
                 *_not_counted(t, s.max_tokens)]
        cards.append(draw.result_card(s.label(label), f"{rate:.0f}" if rate is not None else words.NO_FIGURE,
                                      words.TOKENS_PER_S, notes))
    shared = [words.THROUGHPUT_TIMED, _shape(prompt_tokens, s.max_tokens), words.WARMUP.format(warmup=len(warm)),
              words.STREAM_GROUPS_NOTE, words.SINGLE_RUN, words.DRAW_MOVES]
    draw.update(handle, *s.strips(), draw.card_row(cards, shared))
