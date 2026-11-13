# SPDX-License-Identifier: Apache-2.0
"""rhvp_compare.router: the least-busy choice (standard library), and the aiohttp server in front of two stand-in
replicas: the counts, a replica added while requests run, streaming, a client that goes away, and the child process
the docker backend starts. The server tests need aiohttp (in the pinned GuideLLM image) and are skipped without it.

Run with: python -m pytest -q tests/test_compare_router.py
"""

import asyncio
import importlib.util
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

from rhvp_compare import router  # noqa: E402


def test_picker_sends_to_the_least_busy_and_ties_take_turns():
    clock = iter(range(100, 200)).__next__
    picker = router.Picker(clock=clock)
    a = picker.add("http://a:8000/")
    assert picker.add("http://a:8000") is a  # one entry per URL
    picker.begin(picker.pick())
    picker.begin(picker.pick())
    assert a.in_flight == 2 and a.sent == 2 and picker.first_request_at == 101
    b = picker.add("http://b:8000")
    assert [picker.pick() is b for _ in range(3)] == [True, True, True]  # b has none in flight
    picker.begin(b)
    picker.begin(b)
    assert {picker.pick().url for _ in range(4)} == {"http://a:8000", "http://b:8000"}  # 2 and 2: turns
    picker.end(a, "finished")
    picker.end(a, "cancelled")
    picker.end(b, "failed")
    stats = picker.stats()
    assert stats["policy"] == "least-busy" and stats["first_request_at"] == 101
    assert [(s["sent"], s["finished"], s["failed"], s["cancelled"], s["in_flight"], s["max_in_flight"])
            for s in stats["backends"]] == [(2, 1, 0, 1, 0, 2), (2, 0, 1, 0, 1, 2)]
    with pytest.raises(LookupError):
        router.Picker().pick()


# ---- the server ----

if importlib.util.find_spec("aiohttp") is not None:
    import aiohttp
    from aiohttp import web
    from aiohttp.test_utils import TestServer

needs_aiohttp = pytest.mark.skipif(importlib.util.find_spec("aiohttp") is None,
                                   reason="the router's server needs aiohttp (in the GuideLLM image)")


def replica(name: str, delay_s: float = 0.05, chunks: int = 3):
    """A stand-in server: /v1/completions streams `chunks` server-sent events, `delay_s` apart."""
    async def completions(request):
        body = await request.json()
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        for i in range(chunks):
            await asyncio.sleep(delay_s)
            await response.write(f'data: {{"name":"{name}","i":{i},"model":"{body["model"]}"}}\n\n'.encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    async def models(request):
        return web.json_response({"data": [{"id": "m"}], "served_by": name})

    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/v1/models", models)
    return app


async def _stream(session, url: str) -> tuple[list[str], float]:
    """The events of one completion request through the router, and when the first one came."""
    t0, first, out = time.monotonic(), None, []
    async with session.post(f"{url}/v1/completions", json={"model": "m", "prompt": "x", "stream": True}) as resp:
        assert resp.status == 200
        async for line in resp.content:
            text = line.decode().strip()
            if text.startswith("data: "):
                first = first if first is not None else time.monotonic() - t0
                out.append(text[6:])
    return out, first


@needs_aiohttp
def test_the_router_counts_streams_and_takes_a_new_replica():
    async def scenario():
        a = TestServer(replica("a", delay_s=0.2))
        b = TestServer(replica("b", delay_s=0.2))
        await a.start_server()
        await b.start_server()
        picker = router.Picker()
        picker.add(str(a.make_url("")).rstrip("/"))
        front = TestServer(router.make_app(picker))
        await front.start_server()
        url = str(front.make_url("")).rstrip("/")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/health") as resp:
                    assert resp.status == 200
                async with session.get(f"{url}/v1/models") as resp:  # passed on, not counted
                    assert (await resp.json())["served_by"] == "a"
                first_wave = [asyncio.create_task(_stream(session, url)) for _ in range(4)]
                await asyncio.sleep(0.1)  # all four in flight at a
                async with session.post(f"{url}/-/router/backends",
                                        json={"url": str(b.make_url("")).rstrip("/")}) as resp:
                    added = await resp.json()
                    assert added["sent"] == 0 and added["joined_at"] > 0
                second_wave = [asyncio.create_task(_stream(session, url)) for _ in range(4)]
                results = await asyncio.gather(*first_wave, *second_wave)
                async with session.get(f"{url}/-/router/stats") as resp:
                    stats = await resp.json()
                async with session.post(f"{url}/-/router/backends", json={}) as resp:
                    assert resp.status == 400
        finally:
            await front.close()
            await a.close()
            await b.close()
        return results, stats

    results, stats = asyncio.run(scenario())
    names = [[e for e in events if e != "[DONE]"] for events, _ in results]
    assert all(len(events) == 3 for events in names)
    served = [events[0].split('"name":"')[1][0] for events in names]
    assert served[:4] == ["a"] * 4 and served[4:] == ["b"] * 4  # the new replica took the next requests
    assert all(first < 0.5 for _, first in results)  # the first event came before the stream ended (0.6 s)
    a_stats, b_stats = stats["backends"]
    assert (a_stats["sent"], a_stats["finished"], a_stats["max_in_flight"], a_stats["in_flight"]) == (4, 4, 4, 0)
    assert (b_stats["sent"], b_stats["finished"], b_stats["in_flight"]) == (4, 4, 0)
    assert stats["first_request_at"] <= a_stats["first_at"] < b_stats["first_at"]


@needs_aiohttp
def test_a_client_that_goes_away_is_cancelled_not_failed():
    async def scenario():
        slow = TestServer(replica("a", delay_s=2.0))
        await slow.start_server()
        picker = router.Picker()
        picker.add(str(slow.make_url("")).rstrip("/"))
        front = TestServer(router.make_app(picker), handler_cancellation=True)
        await front.start_server()
        url = str(front.make_url("")).rstrip("/")
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=0.3)) as session:
                with pytest.raises(asyncio.TimeoutError):
                    await _stream(session, url)
            for _ in range(50):
                if picker.backends[0].in_flight == 0:
                    break
                await asyncio.sleep(0.05)
        finally:
            await front.close()
            await slow.close()
        return picker.stats()

    stats = asyncio.run(scenario())
    only = stats["backends"][0]
    assert (only["sent"], only["cancelled"], only["failed"], only["in_flight"]) == (1, 1, 0, 0)


@needs_aiohttp
def test_a_replica_that_refuses_is_a_failed_request():
    async def scenario():
        picker = router.Picker()
        picker.add("http://127.0.0.1:9")  # nothing listens there
        front = TestServer(router.make_app(picker))
        await front.start_server()
        url = str(front.make_url("")).rstrip("/")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{url}/v1/completions", json={"model": "m"}) as resp:
                    status = resp.status
        finally:
            await front.close()
        return status, picker.stats()

    status, stats = asyncio.run(scenario())
    assert status == 502 and stats["backends"][0]["failed"] == 1 and stats["backends"][0]["in_flight"] == 0


@needs_aiohttp
def test_spawn_starts_the_router_as_a_child_process(tmp_path):
    process = router.spawn(["http://127.0.0.1:9"], tmp_path / "router.log")
    try:
        stats = process.stats()
        assert stats["backends"][0]["url"] == "http://127.0.0.1:9" and stats["first_request_at"] is None
        assert process.add("http://127.0.0.1:10")["url"] == "http://127.0.0.1:10"
        assert len(process.stats()["backends"]) == 2
    finally:
        process.stop()
    assert process.process.returncode == 0
    assert "router listening on 127.0.0.1" in (tmp_path / "router.log").read_text()
