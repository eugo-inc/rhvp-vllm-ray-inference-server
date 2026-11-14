# SPDX-License-Identifier: Apache-2.0
"""The router of the docker scale round: one HTTP endpoint in front of a side's replicas, sending each request to
the replica with the fewest requests in flight (ties take turns).

Why a router: GuideLLM keeps its connections open, so behind a balancer that spreads connections (a Kubernetes
Service) a replica added during the surge would get almost no requests. This router chooses per request, and counts
what each replica got, so a launch can show that the new replica took part of the surge.

The runner starts it as a child process in its own container (`spawn`, used by dockerapi.py), next to GuideLLM:

    python -m rhvp_compare.router --port 8787 --backend http://side-1:8000

- Every path is passed on to the chosen replica, the body streamed back as it comes (server-sent events
  included). Only completion requests (/v1/completions, /v1/chat/completions) are counted.
- GET /health answers 200 as soon as one replica is set (GuideLLM checks it before the surge).
- POST /-/router/backends with {"url": ...} adds a replica; GET /-/router/stats gives the counts: per replica
  `sent`, `finished`, `failed` (an error status or a broken connection), `cancelled` (the client went away, as
  GuideLLM does with the requests still running when the surge ends), `in_flight`, `max_in_flight`, `first_at` and
  `last_at` (seconds since the epoch), and `first_request_at`, when the first completion request came in.
- No timeout on a request and no limit on connections: either would add errors or a queue of the router's own.

The server needs aiohttp, which the pinned GuideLLM image (the runner's image) carries; it is imported only when the
server starts, so the rest of this module is standard library only and the runner can import it anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

POLICY = "least-busy"
#: The requests the router counts.
COUNTED = ("/v1/completions", "/v1/chat/completions")
ADMIN = "/-/router"
#: Headers that belong to one connection, not to the request or response the router passes on.
HOP_BY_HOP = frozenset(("connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
                        "transfer-encoding", "upgrade", "host", "content-length"))


class Backend:
    """One replica and its counts."""

    def __init__(self, url: str, joined_at: float) -> None:
        self.url, self.joined_at = url.rstrip("/"), joined_at
        self.in_flight = self.max_in_flight = 0
        self.sent = self.finished = self.failed = self.cancelled = 0
        self.first_at: float | None = None
        self.last_at: float | None = None

    def stats(self) -> dict:
        return {"url": self.url, "joined_at": self.joined_at, "sent": self.sent, "finished": self.finished,
                "failed": self.failed, "cancelled": self.cancelled, "in_flight": self.in_flight,
                "max_in_flight": self.max_in_flight, "first_at": self.first_at, "last_at": self.last_at}


class Picker:
    """Least busy first: the replica with the fewest counted requests in flight; among equals, the one after the
    last pick, so ties take turns. Not thread-safe: the server calls it from one event loop."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.backends: list[Backend] = []
        self.clock = clock
        self.first_request_at: float | None = None
        self._turn = 0

    def add(self, url: str) -> Backend:
        for backend in self.backends:
            if backend.url == url.rstrip("/"):
                return backend
        backend = Backend(url, self.clock())
        self.backends.append(backend)
        return backend

    def pick(self) -> Backend:
        if not self.backends:
            raise LookupError("no replica to send the request to")
        least = min(b.in_flight for b in self.backends)
        n = len(self.backends)
        for step in range(n):
            backend = self.backends[(self._turn + step) % n]
            if backend.in_flight == least:
                self._turn = (self.backends.index(backend) + 1) % n
                return backend
        raise AssertionError("unreachable")

    def begin(self, backend: Backend) -> None:
        """A counted request goes to `backend`."""
        now = self.clock()
        if self.first_request_at is None:
            self.first_request_at = now
        backend.sent += 1
        backend.in_flight += 1
        backend.max_in_flight = max(backend.max_in_flight, backend.in_flight)
        backend.first_at = now if backend.first_at is None else backend.first_at
        backend.last_at = now

    def end(self, backend: Backend, outcome: str) -> None:
        """A counted request ended: finished, failed or cancelled."""
        backend.in_flight -= 1
        if outcome == "finished":
            backend.finished += 1
        elif outcome == "cancelled":
            backend.cancelled += 1
        else:
            backend.failed += 1

    def stats(self) -> dict:
        return {"policy": POLICY, "first_request_at": self.first_request_at,
                "backends": [b.stats() for b in self.backends]}


# ---- the server (aiohttp) ----

def make_app(picker: Picker):
    """The aiohttp application around `picker`."""
    import asyncio

    from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

    session_key = web.AppKey("session", ClientSession)

    async def on_startup(app) -> None:
        app[session_key] = ClientSession(timeout=ClientTimeout(total=None, sock_connect=30),
                                         connector=TCPConnector(limit=0), auto_decompress=False)

    async def on_cleanup(app) -> None:
        await app[session_key].close()

    async def stats(request):
        return web.json_response(picker.stats())

    async def add(request):
        try:
            url = (await request.json())["url"]
        except (ValueError, KeyError, TypeError):
            return web.json_response({"error": 'send {"url": "http://host:port"}'}, status=400)
        return web.json_response(picker.add(str(url)).stats())

    async def health(request):
        return web.Response(status=200 if picker.backends else 503, text="ok" if picker.backends else "no replica")

    async def proxy(request):
        try:
            backend = picker.pick()
        except LookupError as error:
            return web.json_response({"error": str(error)}, status=503)
        counted = request.method == "POST" and request.path in COUNTED
        body = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        if counted:
            picker.begin(backend)
        outcome, prepared = "failed", False
        try:
            async with request.app[session_key].request(request.method, backend.url + request.path_qs,
                                                      headers=headers, data=body or None) as upstream:
                response = web.StreamResponse(status=upstream.status, reason=upstream.reason, headers={
                    k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP})
                await response.prepare(request)
                prepared = True
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                outcome = "finished" if upstream.status < 400 else "failed"
                return response
        except asyncio.CancelledError:  # the client went away (handler_cancellation): close the upstream request
            outcome = "cancelled"
            raise
        except ConnectionResetError:  # the client went away while the answer streamed
            outcome = "cancelled"
            raise
        except Exception as error:  # the replica refused or broke the connection
            if prepared:  # too late for an error status: the client sees the stream break
                raise
            return web.json_response({"error": f"replica {backend.url}: {type(error).__name__}"}, status=502)
        finally:
            if counted:
                picker.end(backend, outcome)

    app = web.Application(client_max_size=64 * 2**20)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get(f"{ADMIN}/stats", stats)
    app.router.add_post(f"{ADMIN}/backends", add)
    app.router.add_get("/health", health)
    app.router.add_route("*", "/{tail:.*}", proxy)
    return app


def serve(host: str, port: int, backends: list[str]) -> None:
    """Run the router until SIGTERM or SIGINT."""
    import asyncio

    from aiohttp import web

    picker = Picker()
    for url in backends:
        picker.add(url)

    async def main() -> None:
        runner = web.AppRunner(make_app(picker), handler_cancellation=True, shutdown_timeout=5)
        await runner.setup()
        await web.TCPSite(runner, host, port).start()
        print(f"router listening on {host}:{port}, policy {POLICY}, replicas {backends}", flush=True)
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        await stop.wait()
        print(f"router stopping: {json.dumps(picker.stats())}", flush=True)
        await runner.cleanup()

    asyncio.run(main())


# ---- the runner's side: start the router as a child process and talk to it ----

class RouterError(RuntimeError):
    """The router did not start or did not answer; the message is safe to show."""


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Process:
    """A router started by `spawn`: its `url` (for GuideLLM), `add`, `stats` and `stop`."""

    def __init__(self, process: subprocess.Popen, port: int, log) -> None:
        import httpx

        self.process, self.port, self._log = process, port, log
        self.url = f"http://127.0.0.1:{port}"
        self.http = httpx.Client(base_url=self.url, timeout=httpx.Timeout(10.0))

    def add(self, target: str) -> dict:
        return self._json(self.http.post(f"{ADMIN}/backends", json={"url": target}), "add a replica")

    def stats(self) -> dict:
        return self._json(self.http.get(f"{ADMIN}/stats"), "give its counts")

    def _json(self, response, what: str) -> dict:
        if response.status_code != 200:
            raise RouterError(f"the router could not {what}: HTTP {response.status_code}")
        return response.json()

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.http.close()
        self._log.close()


def spawn(targets: list[str], log_path: Path, port: int | None = None, start_timeout_s: float = 30.0,
          python: str = sys.executable) -> Process:
    """Start the router in front of `targets` on 127.0.0.1 and wait until it answers."""
    import httpx

    port = port or free_port()
    log = open(log_path, "w", encoding="utf-8")
    args = [python, "-m", "rhvp_compare.router", "--host", "127.0.0.1", "--port", str(port)]
    for target in targets:
        args += ["--backend", target]
    package_root = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (package_root, os.environ.get("PYTHONPATH"))
                                                       if p)}
    process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
    router = Process(process, port, log)
    deadline = time.monotonic() + start_timeout_s
    while True:
        if process.poll() is not None:
            router.stop()
            raise RouterError(f"the router exited with code {process.returncode} at its start; see {log_path.name}")
        try:
            router.stats()
            return router
        except (httpx.HTTPError, RouterError):
            if time.monotonic() > deadline:
                router.stop()
                raise RouterError(f"the router did not answer within {start_timeout_s:g} s; see {log_path.name}")
            time.sleep(0.2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m rhvp_compare.router", description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--backend", action="append", default=[], help="a replica's base URL; repeat for more")
    args = parser.parse_args(argv)
    serve(args.host, args.port, args.backend)


if __name__ == "__main__":
    main()
