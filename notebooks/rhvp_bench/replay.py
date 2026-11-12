# SPDX-License-Identifier: Apache-2.0
"""REPLAY mode: record a server's streamed answers, and serve them again from a small stand-in server.

With RHVP_REPLAY set to a recording, the notebook starts `FakeServer` in its own process and sends every request
there: an OpenAI-compatible server on 127.0.0.1 (stdlib http.server, one thread per request) that answers
`GET /v1/models`, a non-streamed completion, and streamed completions. Each streamed request gets the next recorded
stream in turn, written back in the same network writes, at the same offsets from the request as when it was
recorded. Nothing reaches a GPU, so CI can run the whole notebook.

A recording is one JSON file:

    {"format": "rhvp-replay/1",
     "metadata": {"synthetic": true or false, "note": ..., "model_id": ..., "model_revision": ..., "max_tokens": N},
     "models": <the /v1/models answer>, "completion": <a non-streamed one-token completion>,
     "pods": [<server pods as the Kubernetes API lists them>],
     "streams": [{"writes": [[seconds after the request, "text written"], ...]}, ...]}

A stream may also carry "cut": true (the connection closes before the stream's end) or "status": <HTTP status>
(the request is refused with that status).

The repository commits only synthetic recordings (`synthesize`, written by make_notebook.py, every figure in it made
up); a test checks that every committed recording says so. `record` captures a live server's streams for local use;
a live recording stays out of the repository.

    python -m rhvp_bench.replay synthesize --out recordings/synthetic.json
    python -m rhvp_bench.replay serve --recording recordings/synthetic.json --port 8000
    python -m rhvp_bench.replay record --out /tmp/run.json --requests 8
"""

import argparse
import codecs
import itertools
import json
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FORMAT = "rhvp-replay/1"
HERE = Path(__file__).resolve().parent
SYNTHETIC = HERE.parent / "recordings" / "synthetic.json"
SYNTHETIC_NOTE = ("SYNTHETIC: written by rhvp_bench.replay synthesize to check the notebook's layout. Every time "
                  "and count in it is made up; it is not a measurement.")
LIVE_NOTE = "Recorded from a live server for local use. Not a measurement to publish; keep it out of the repository."
MODEL_ID = "qwen2.5-0.5b-instruct"
MODELS_PATH = "/v1/models"
STREAM_PATHS = ("/v1/completions", "/v1/chat/completions")
#: The synthetic streams' shape: when the first text comes back, how often a group of tokens is written (Ray Serve
#: LLM's default interval), and one made-up token time. Round numbers, all of them invented.
SYNTHETIC_FIRST_S = 0.05
SYNTHETIC_GROUP_S = 0.05
SYNTHETIC_TOKEN_S = 0.004
SYNTHETIC_PROMPT_TOKENS = 300


class RecordingError(ValueError):
    """A recording file the notebook cannot replay."""


def load(path: str | os.PathLike) -> dict:
    """A recording, checked for the parts the stand-in server and the notebook read."""
    try:
        recording = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RecordingError(f"the recording cannot be read: {type(error).__name__}") from None
    if not isinstance(recording, dict) or recording.get("format") != FORMAT:
        raise RecordingError(f"the recording is not in the {FORMAT} format")
    for key, kind in (("metadata", dict), ("models", dict), ("completion", dict), ("pods", list), ("streams", list)):
        if not isinstance(recording.get(key), kind):
            raise RecordingError(f"the recording has no {key}")
    if not recording["streams"]:
        raise RecordingError("the recording has no streams")
    return recording


class FakeServer:
    """The stand-in OpenAI-compatible server. Use as a context manager, or start() and stop()."""

    def __init__(self, recording: dict, host: str = "127.0.0.1", port: int = 0) -> None:
        self.recording = recording
        self.requests: list[dict] = []
        self._turn = itertools.count()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.httpd = ThreadingHTTPServer((host, port), _handler(self))
        self.httpd.daemon_threads = True

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def next_stream(self) -> dict:
        streams = self.recording["streams"]
        with self._lock:
            return streams[next(self._turn) % len(streams)]

    def start(self) -> str:
        if self._thread is None:
            self._thread = threading.Thread(target=self.httpd.serve_forever, name="rhvp-replay", daemon=True)
            self._thread.start()
        return self.url

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self._thread = None

    def __enter__(self) -> "FakeServer":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()


def _handler(server: FakeServer) -> type:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args) -> None:  # quiet: the notebook shows no server log
            pass

        def _json(self, status: int, payload: object) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _not_found(self) -> None:
            self._json(404, {"error": {"message": "not found", "type": "NotFoundError", "code": 404}})

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == MODELS_PATH:
                self._json(200, server.recording["models"])
            elif path == "/-/healthz":
                self._json(200, "success")
            else:
                self._not_found()

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                body = None
            if path not in STREAM_PATHS or not isinstance(body, dict):
                self._not_found()
                return
            with server._lock:
                server.requests.append(body)
            if not body.get("stream"):
                self._json(200, server.recording["completion"])
                return
            self._stream(server.next_stream())

        def _stream(self, stream: dict) -> None:
            if stream.get("status"):
                self._json(int(stream["status"]), {"error": {"message": "refused", "type": "ReplayError",
                                                             "code": int(stream["status"])}})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            start = time.perf_counter()
            try:
                for offset, text in stream["writes"]:
                    wait = float(offset) - (time.perf_counter() - start)
                    if wait > 0:
                        time.sleep(wait)
                    data = text.encode("utf-8")
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                    self.wfile.flush()
                if stream.get("cut"):
                    self.close_connection = True
                    return
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    return Handler


# ---- the synthetic recording ----

def _event(payload: object) -> str:
    return "data: " + json.dumps(payload, separators=(",", ":")) + "\n\n"


def _chunk(index: int, text: str, finish: str | None = None) -> dict:
    return {"id": f"cmpl-synthetic-{index}", "object": "text_completion", "created": 0, "model": MODEL_ID,
            "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}]}


def _synthetic_stream(index: int, tokens: int, rng: random.Random) -> dict:
    from .prompts import VOCABULARY

    words = [" " + rng.choice(VOCABULARY) for _ in range(tokens)]
    first = SYNTHETIC_FIRST_S + 0.01 * (index % 3)
    times = [first + k * SYNTHETIC_TOKEN_S for k in range(tokens)]
    writes: list[list] = []
    flush_at = first
    pending = _event(_chunk(index, ""))  # an opening event with no text
    k = 0
    while k < tokens:
        while k < tokens and times[k] <= flush_at + 1e-9:
            pending += _event(_chunk(index, words[k], "length" if k == tokens - 1 else None))
            k += 1
        if k == tokens:
            usage = {"prompt_tokens": SYNTHETIC_PROMPT_TOKENS, "completion_tokens": tokens,
                     "total_tokens": SYNTHETIC_PROMPT_TOKENS + tokens}
            pending += _event({"id": f"cmpl-synthetic-{index}", "object": "text_completion", "created": 0,
                               "model": MODEL_ID, "choices": [], "usage": usage})
            pending += "data: [DONE]\n\n"
        writes.append([round(flush_at, 6), pending])
        pending = ""
        flush_at += SYNTHETIC_GROUP_S
    return {"writes": writes}


def _synthetic_pod() -> dict:
    env = [{"name": "RHVP_CONFIG", "value": "/config/llm.yaml"}, {"name": "HF_HOME", "value": "/models/hf"},
           {"name": "TRITON_CACHE_DIR", "value": "/cache/triton"}, {"name": "VLLM_CACHE_ROOT", "value": "/cache/vllm"},
           {"name": "FLASHINFER_WORKSPACE_BASE", "value": "/cache/flashinfer"},
           {"name": "XDG_CACHE_HOME", "value": "/cache/xdg"}, {"name": "RAY_TMPDIR", "value": "/tmp/r"}]
    mounts = [{"name": "models", "mountPath": "/models"}, {"name": "cache", "mountPath": "/cache"},
              {"name": "ray-tmp", "mountPath": "/tmp/r"}, {"name": "dshm", "mountPath": "/dev/shm"},
              {"name": "config", "mountPath": "/config"}]
    volumes = [{"name": "models", "persistentVolumeClaim": {"claimName": "ray-serve-llm-models"}},
               {"name": "cache", "emptyDir": {}}, {"name": "ray-tmp", "emptyDir": {}},
               {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "4Gi"}},
               {"name": "config", "configMap": {"name": "ray-serve-llm-config"}}]
    day = "2026-01-01T"
    return {
        "metadata": {"name": "ray-serve-llm-synthetic", "namespace": "vllm-inference",
                     "creationTimestamp": day + "00:00:00Z", "labels": {"app.kubernetes.io/name": "ray-serve-llm"}},
        "spec": {"containers": [{"name": "server", "env": env, "volumeMounts": mounts}], "volumes": volumes},
        "status": {
            "phase": "Running",
            "conditions": [
                {"type": "PodScheduled", "status": "True", "lastTransitionTime": day + "00:00:00Z"},
                {"type": "Initialized", "status": "True", "lastTransitionTime": day + "00:00:00Z"},
                {"type": "ContainersReady", "status": "True", "lastTransitionTime": day + "00:05:00Z"},
                {"type": "Ready", "status": "True", "lastTransitionTime": day + "00:05:00Z"},
            ],
            "containerStatuses": [{"name": "server", "ready": True, "restartCount": 0,
                                   "state": {"running": {"startedAt": day + "00:01:00Z"}}}],
        },
    }


def synthesize(seed: int = 11, streams: int = 6, max_tokens: int = 32, short: int = 5) -> dict:
    """The synthetic recording: `streams` made-up streams of `max_tokens` tokens each, written in groups, except
    stream `short`, which stops one token early (the notebook drops it); one made-up server pod."""
    rng = random.Random(seed)
    return {
        "format": FORMAT,
        "metadata": {"synthetic": True, "note": SYNTHETIC_NOTE, "seed": seed, "model_id": MODEL_ID,
                     "model_revision": "", "max_tokens": max_tokens, "api": "completions"},
        "models": {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": 0,
                                               "owned_by": "synthetic"}]},
        "completion": {"id": "cmpl-synthetic", "object": "text_completion", "created": 0, "model": MODEL_ID,
                       "choices": [{"index": 0, "text": " Hello", "logprobs": None, "finish_reason": "length"}],
                       "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        "pods": [_synthetic_pod()],
        "streams": [_synthetic_stream(i, max_tokens - 1 if i == short else max_tokens, rng) for i in range(streams)],
    }


def recording_text(recording: dict) -> str:
    return json.dumps(recording, indent=1, ensure_ascii=False) + "\n"


# ---- recording a live server ----

def record(endpoint: str, model: str, count: int = 8, max_tokens: int | None = None) -> dict:
    """A recording of `count` streamed requests to a live server (plus its model list, a one-token completion, and
    the server pods when run inside the cluster)."""
    import httpx

    from . import measure, prompts

    max_tokens = max_tokens or measure.MAX_TOKENS
    pool = prompts.load()
    texts = [pool[i] for i in prompts.pick(pool, count, seed=measure.SEEDS["first_token"])]

    async def capture() -> list[dict]:
        streams = []
        async with measure.async_client(endpoint) as client:
            for text in texts:
                writes: list[list] = []
                decoder = codecs.getincrementaldecoder("utf-8")()

                def keep(at: float, data: bytes, writes=writes, decoder=decoder) -> None:
                    writes.append([round(at, 6), decoder.decode(data)])

                await measure.stream_one(client, model, measure.tagged(text), max_tokens, on_write=keep)
                streams.append({"writes": writes})
        return streams

    streams = measure.run_async(capture())
    with httpx.Client(base_url=endpoint, timeout=60) as client:
        models = client.get(measure.MODELS).json()
        completion = client.post(measure.COMPLETIONS, json=measure.request_body(model, "Hello", 1, stream=False)).json()
    pods = []
    if measure.in_cluster():
        pods = measure.read_pods(os.environ.get("RHVP_SERVER_NAMESPACE", "vllm-inference"),
                                 os.environ.get("RHVP_SERVER_LABEL_SELECTOR", "app.kubernetes.io/name=ray-serve-llm"))
    return {"format": FORMAT,
            "metadata": {"synthetic": False, "note": LIVE_NOTE, "model_id": model,
                         "model_revision": os.environ.get("RHVP_MODEL_REVISION", ""), "max_tokens": max_tokens,
                         "api": "completions"},
            "models": models, "completion": completion, "pods": pods, "streams": streams}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record, replay or synthesize streamed server answers.")
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("synthesize", help="write the synthetic recording")
    make.add_argument("--out", type=Path, default=SYNTHETIC)
    serve = sub.add_parser("serve", help="serve a recording on 127.0.0.1 until interrupted")
    serve.add_argument("--recording", type=Path, default=SYNTHETIC)
    serve.add_argument("--port", type=int, default=8000)
    rec = sub.add_parser("record", help="record a live server's streams")
    rec.add_argument("--out", type=Path, required=True)
    rec.add_argument("--endpoint", default=os.environ.get("RHVP_ENDPOINT", "http://ray-serve-llm.vllm-inference.svc:8000"))
    rec.add_argument("--model", default=os.environ.get("RHVP_MODEL_ID", MODEL_ID))
    rec.add_argument("--requests", type=int, default=8)
    rec.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args(argv)

    if args.command == "synthesize":
        args.out.write_text(recording_text(synthesize()), encoding="utf-8")
        print(f"wrote {args.out.name}")
    elif args.command == "serve":
        server = FakeServer(load(args.recording), port=args.port)
        print(f"serving {args.recording.name} on port {server.httpd.server_address[1]}; Ctrl-C stops it", flush=True)
        try:
            server.httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.httpd.server_close()
    else:
        args.out.write_text(recording_text(record(args.endpoint, args.model, args.requests, args.max_tokens)),
                            encoding="utf-8")
        print(f"wrote {args.out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
