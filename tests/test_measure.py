# SPDX-License-Identifier: Apache-2.0
"""measure.py on recorded streams: how a stream is read, what is counted, dropped or failed, and the startup reading.

Run with: python -m pytest -q tests/test_measure.py
"""

import copy
import json
import re
import sys
from pathlib import Path

import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))

from rhvp_bench import measure, prompts, replay  # noqa: E402

N = 8
TAG = re.compile(r"^[0-9a-f]{8}: ")


def _event(payload: object) -> str:
    return "data: " + (payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)) + "\n\n"


def _text(text: str, finish: str | None = None) -> dict:
    return {"object": "text_completion", "choices": [{"index": 0, "text": text, "finish_reason": finish}]}


def _delta(**delta: str) -> dict:
    return {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


def _usage(completion: int, prompt: int = 300) -> dict:
    return {"object": "text_completion", "choices": [],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}}


def _read(writes: list[tuple[float, str]], sent_at: float = 0.0) -> measure.StreamResult:
    """Feed network writes (time, text) through the event splitter and fold them, as stream_one does."""
    splitter = measure.EventSplitter()
    timed = []
    for at, text in writes:
        timed.extend((at, payload) for payload in splitter.feed(text.encode("utf-8")))
    timed.extend((writes[-1][0], payload) for payload in splitter.close())
    return measure.read_events(timed, sent_at)


# ---- reading one stream ----

def test_role_only_first_chunk_is_not_the_first_token():
    result = _read([
        (0.020, _event(_delta(role="assistant"))),
        (0.050, _event(_delta(content="Hello"))),
        (0.100, _event(_delta(content=" there")) + _event(_usage(2)) + _event("[DONE]")),
    ])
    assert result.ok
    assert result.ttft_s == pytest.approx(0.050)
    assert result.completion_tokens == 2
    assert result.decode_tokens_per_s == pytest.approx(1 / 0.050)


def test_empty_text_event_is_not_the_first_token():
    result = _read([(0.010, _event(_text(""))), (0.030, _event(_text(" a")) + _event(_usage(1)) + _event("[DONE]"))])
    assert result.ttft_s == pytest.approx(0.030)


def test_grouped_events_in_one_write_share_its_time():
    first = "".join(_event(_text(f" w{k}")) for k in range(4))
    second = "".join(_event(_text(f" w{k}")) for k in range(4, 8)) + _event(_usage(N)) + _event("[DONE]")
    result = _read([(0.060, first), (0.110, second)])
    assert result.text_events == 8
    assert result.first_token_at == pytest.approx(0.060)
    assert result.last_token_at == pytest.approx(0.110)
    assert result.decode_tokens_per_s == pytest.approx((N - 1) / 0.050)


def test_an_event_split_across_writes_with_crlf_and_comments():
    whole = _event(_text(" split")).replace("\n", "\r\n")
    cut = len(whole) // 2
    result = _read([
        (0.040, ": keep-alive\r\n\r\n" + whole[:cut]),
        (0.045, whole[cut:]),
        (0.090, _event(_usage(1)) + _event("[DONE]")),
    ])
    assert result.ok
    assert result.text_events == 1
    assert result.ttft_s == pytest.approx(0.045)


def test_a_utf8_character_split_across_network_reads():
    data = _event(_text(" é")).encode("utf-8")
    cut = data.index("é".encode()) + 1
    splitter = measure.EventSplitter()
    events = splitter.feed(data[:cut]) + splitter.feed(data[cut:])
    assert json.loads(events[0])["choices"][0]["text"] == " é"


def test_tokens_come_from_usage_not_from_events():
    result = _read([
        (0.05, _event(_text(" several tokens here"))),
        (0.10, _event(_text(" and more of them"))),
        (0.15, _event(_text(" in each event")) + _event(_usage(12)) + _event("[DONE]")),
    ])
    assert result.text_events == 3
    assert result.completion_tokens == 12
    assert result.prompt_tokens == 300
    assert result.decode_tokens_per_s == pytest.approx(11 / 0.10)


def test_done_marks_the_end_and_its_absence_fails():
    complete = _read([(0.05, _event(_text(" a"))), (0.10, _event(_usage(1)) + _event("[DONE]"))])
    assert complete.done and complete.ok
    unfinished = _read([(0.05, _event(_text(" a"))), (0.10, _event(_usage(1)))])
    assert not unfinished.done
    assert unfinished.error == "the stream ended before [DONE]"
    assert measure.outcome(unfinished, 1) == measure.FAILED


@pytest.mark.parametrize("error", [
    {"error": {"message": "engine died", "type": "InternalServerError", "code": 500}},
    {"object": "error", "message": "engine died", "type": "InternalServerError", "code": 500},
])
def test_an_error_event_mid_stream_fails_the_request(error):
    result = _read([
        (0.05, _event(_text(" a"))),
        (0.10, _event(error) + _event("[DONE]")),
    ])
    assert result.error == "error in the stream: InternalServerError"
    assert measure.outcome(result, 1) == measure.FAILED


def test_an_event_that_is_not_json_fails_the_request():
    result = _read([(0.05, _event("{not json")), (0.10, _event("[DONE]"))])
    assert result.error == "an event that is not JSON"


def test_a_token_count_mismatch_is_dropped_and_counted():
    def stream(tokens: int) -> measure.StreamResult:
        return _read([(0.05, _event(_text(" a"))), (0.10, _event(_text(" b")) + _event(_usage(tokens))
                                                     + _event("[DONE]"))])

    failed = _read([(0.05, _event(_text(" a")))])
    t = measure.tally([stream(N), stream(N - 1), stream(N + 1), failed, stream(N)], N)
    assert len(t.counted) == 2
    assert t.dropped == 2
    assert t.failed == 1
    assert t.sent == 5
    assert t.errors == ["the stream ended before [DONE]"]


def test_a_full_count_with_no_text_is_failed():
    result = _read([(0.05, _event(_usage(N)) + _event("[DONE]"))])
    assert measure.outcome(result, N) == measure.FAILED
    assert measure.tally([result], N).errors == ["no text came back"]


def test_decode_speed_needs_two_tokens_and_a_time_span():
    one_write = _read([(0.05, _event(_text(" a")) + _event(_text(" b")) + _event(_usage(2)) + _event("[DONE]"))])
    assert one_write.decode_tokens_per_s is None
    single = _read([(0.05, _event(_text(" a")) + _event(_usage(1)) + _event("[DONE]"))])
    assert single.decode_tokens_per_s is None


# ---- statistics ----

def test_median_and_p90():
    assert measure.median([3, 1, 2]) == 2
    assert measure.median([1, 2, 3, 4]) == pytest.approx(2.5)
    assert measure.p90(range(1, 11)) == pytest.approx(9.1)
    assert measure.p90([0.5]) == 0.5
    with pytest.raises(ValueError):
        measure.median([])


# ---- requests ----

def test_request_body():
    body = measure.request_body("m", "p", N)
    assert body == {"model": "m", "prompt": "p", "max_tokens": N, "temperature": 0, "ignore_eos": True,
                    "stream": True, "stream_options": {"include_usage": True}}
    chat = measure.request_body("m", "p", N, api="chat")
    assert chat["messages"] == [{"role": "user", "content": "p"}]


def test_errors_are_shown_without_their_message():
    import httpx

    error = httpx.ConnectError("could not reach http://somewhere.example:8000")
    assert measure.describe(error) == "ConnectError"


# ---- over HTTP, against the stand-in server ----

def _recording(*streams: dict) -> dict:
    recording = replay.synthesize(streams=2, max_tokens=N, short=-1)
    if streams:
        recording["streams"] = list(streams)
    return recording


def _grouped_stream(tokens: int = N, first: float = 0.03, every: float = 0.02) -> dict:
    half = tokens // 2
    head = _event(_text("")) + "".join(_event(_text(f" w{k}")) for k in range(half))
    tail = "".join(_event(_text(f" w{k}")) for k in range(half, tokens)) + _event(_usage(tokens)) + _event("[DONE]")
    return {"writes": [[first, head], [first + every, tail]]}


async def _one(url: str, recorder=None) -> measure.StreamResult:
    async with measure.async_client(url) as client:
        return await measure.stream_one(client, "qwen2.5-0.5b-instruct", "p", N, on_write=recorder)


def test_stream_one_over_http_times_the_grouped_writes():
    writes = []
    with replay.FakeServer(_recording(_grouped_stream())) as server:
        result = measure.run_async(_one(server.url, lambda at, data: writes.append(data)))
    assert result.ok
    assert result.completion_tokens == N
    assert result.text_events == N
    assert result.ttft_s >= 0.03
    assert result.last_token_at - result.first_token_at >= 0.015
    assert measure.outcome(result, N) == measure.COUNTED
    assert b"[DONE]" in b"".join(writes)


def test_a_connection_cut_mid_stream_fails_the_request():
    cut = _grouped_stream()
    cut["writes"] = cut["writes"][:1]
    cut["cut"] = True
    with replay.FakeServer(_recording(cut)) as server:
        result = measure.run_async(_one(server.url))
    assert result.error is not None and "://" not in result.error
    assert result.first_token_at is not None
    assert measure.outcome(result, N) == measure.FAILED


def test_an_http_error_fails_the_request():
    with replay.FakeServer(_recording({"status": 503, "writes": []})) as server:
        result = measure.run_async(_one(server.url))
    assert result.error == "HTTP 503"
    assert measure.outcome(result, N) == measure.FAILED


def test_sequential_sends_warmup_then_distinct_tagged_prompts():
    pool = prompts.load()
    texts, warm = measure.choose(pool, 3, 2, seed=measure.SEEDS["first_token"])
    with replay.FakeServer(_recording(_grouped_stream())) as server:
        progress = measure.Progress()
        result = measure.run_async(measure.sequential(server.url, "qwen2.5-0.5b-instruct", texts, warm, N, progress))
        bodies = list(server.requests)
    assert progress.done == progress.total == 5
    assert result.warmup == 2 and len(result.results) == 3
    assert len(result.ttfts()) == 3
    assert result.prompt_tokens() == 300
    assert [b["prompt"][len("01234567: "):] for b in bodies] == warm + texts
    assert all(TAG.match(b["prompt"]) for b in bodies)
    assert len({b["prompt"] for b in bodies}) == 5
    assert all(b["ignore_eos"] is True and b["temperature"] == 0 and b["max_tokens"] == N
               and b["stream_options"] == {"include_usage": True} for b in bodies)


def test_levels_send_each_level_at_once():
    pool = prompts.load()
    texts, warm = measure.choose(pool, 5, 1, seed=measure.SEEDS["throughput"])
    with replay.FakeServer(_recording(_grouped_stream(first=0.1))) as server:
        result = measure.run_async(measure.levels(server.url, "m", {1: texts[:1], 4: texts[1:]}, warm, N))
    assert [level.streams for level in result] == [1, 4]
    four = result[1]
    assert len(four.tally.counted) == 4
    # four streams at once take about as long as one, not four times as long
    assert four.wall_s < 0.1 * 4
    assert four.tokens_per_s() == pytest.approx(4 * N / four.wall_s)


def test_wait_ready():
    with replay.FakeServer(_recording()) as server:
        ready = measure.wait_ready(server.url, "qwen2.5-0.5b-instruct", timeout_s=5, poll_s=0.01)
        missing = measure.wait_ready(server.url, "another-model", timeout_s=0, poll_s=0.01)
    assert ready.ready
    assert not missing.ready and missing.step == "listed"
    nobody = measure.wait_ready("http://127.0.0.1:9", "m", timeout_s=0, poll_s=0.01)
    assert not nobody.ready and nobody.step == "answers"


# ---- the server pod's start ----

def _pod() -> dict:
    return copy.deepcopy(replay.synthesize()["pods"][0])


def test_startup_from_pod_conditions():
    start = measure.startup([_pod()])
    assert start.state == "ok"
    assert start.to_ready_s == 300
    assert start.to_start_s == 60
    assert start.start_to_ready_s == 240
    assert start.empty == ["TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "FLASHINFER_WORKSPACE_BASE", "XDG_CACHE_HOME"]
    assert start.kept == ["HF_HOME"]


def test_startup_after_a_container_restart_shows_no_time():
    pod = _pod()
    pod["status"]["containerStatuses"][0]["restartCount"] = 2
    start = measure.startup([pod])
    assert start.state == "restarted" and start.restarts == 2 and start.to_ready_s is None


def test_startup_of_a_pod_not_ready_yet():
    pod = _pod()
    for condition in pod["status"]["conditions"]:
        if condition["type"] == "Ready":
            condition["status"] = "False"
    assert measure.startup([pod]).state == "not_ready"
    assert measure.startup([]).state == "no_pod"


def test_startup_reads_the_newest_ready_pod_and_skips_deleted_ones():
    old, new, deleting = _pod(), _pod(), _pod()
    new["metadata"]["creationTimestamp"] = "2026-01-02T00:00:00Z"
    new["status"]["conditions"][-1]["lastTransitionTime"] = "2026-01-01T00:10:00Z"
    deleting["metadata"]["creationTimestamp"] = "2026-01-03T00:00:00Z"
    deleting["metadata"]["deletionTimestamp"] = "2026-01-03T00:01:00Z"
    assert measure.startup([old, deleting, new]).to_ready_s == 600


def test_storage_follows_the_longest_mount():
    pod = _pod()
    box = pod["spec"]["containers"][0]
    box["volumeMounts"].append({"name": "weights", "mountPath": "/cache/vllm"})
    pod["spec"]["volumes"].append({"name": "weights", "persistentVolumeClaim": {"claimName": "x"}})
    empty, kept = measure.storage(pod)
    assert "VLLM_CACHE_ROOT" in kept and "VLLM_CACHE_ROOT" not in empty


def test_not_in_cluster_without_a_service_account(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    assert not measure.in_cluster()


# ---- the prompt set and the recordings ----

def test_prompt_set_is_what_the_seed_writes():
    made = prompts.make()
    assert prompts.text(made) == prompts.PATH.read_text(encoding="utf-8")
    assert len(made) == prompts.COUNT == len(set(made))
    assert all(len(p.split()) == prompts.WORDS_PER_PROMPT for p in made)
    assert len({p.split()[0] for p in made}) > 1


def test_warmup_and_measured_prompts_never_overlap():
    pool = prompts.load()
    for seed in measure.SEEDS.values():
        texts, warm = measure.choose(pool, 41, 4, seed)
        assert len(set(texts)) == 41 and not set(texts) & set(warm)
    with pytest.raises(ValueError):
        prompts.pick(pool, len(pool) + 1, seed=1)


def test_every_committed_recording_is_synthetic():
    files = sorted((NOTEBOOKS / "recordings").glob("*.json"))
    assert files
    for path in files:
        recording = replay.load(path)
        assert recording["metadata"]["synthetic"] is True, path.name
        assert recording["metadata"]["note"].startswith("SYNTHETIC"), path.name


def test_the_synthetic_recording_is_what_the_script_writes():
    assert replay.recording_text(replay.synthesize()) == replay.SYNTHETIC.read_text(encoding="utf-8")


def test_a_broken_recording_is_refused(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": "something else"}), encoding="utf-8")
    with pytest.raises(replay.RecordingError):
        replay.load(bad)
