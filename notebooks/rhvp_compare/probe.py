# SPDX-License-Identifier: Apache-2.0
"""Start-up timing of one side, on the runner's own clock, the same way for every image:

- time to ready: from just before the runner asks for the server until `GET /v1/models` answers 200 and lists the
  model;
- time to first answer: from the same moment until the first generated text of a streamed completion comes back.
  Requests start once the server is ready and are repeated every half second until one answers.

Some servers list the model before it is loaded (this pattern's Ray Serve LLM server does), so time to first answer
is the figure that compares any two images fairly; time to ready is what the server says about itself.
Error texts never carry a URL.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import httpx

POLL_S = 0.5


class StartFailed(Exception):
    """The server died, never got a node, or did not answer in time. The message is safe to show."""


def _check(alive: Callable[[], str | None], deadline: float, step: str) -> None:
    problem = alive()
    if problem:
        raise StartFailed(problem)
    if time.monotonic() > deadline:
        raise StartFailed(f"no {step} before the time limit")


def wait_ready(client: httpx.Client, model: str, t0: float, deadline: float,
               alive: Callable[[], str | None]) -> float:
    """Seconds from `t0` (time.monotonic) until /v1/models lists `model`."""
    while True:
        try:
            response = client.get("/v1/models", timeout=5.0)
            if response.status_code == 200:
                ids = [m.get("id") for m in (response.json().get("data") or []) if isinstance(m, dict)]
                if model in ids:
                    return time.monotonic() - t0
        except (httpx.HTTPError, ValueError, AttributeError):
            pass
        _check(alive, deadline, "answer from /v1/models")
        time.sleep(POLL_S)


def _first_text(client: httpx.Client, model: str, tokens: int, request_format: str) -> bool:
    """Send one streamed request; True once text comes back (the stream is then closed)."""
    body: dict = {"model": model, "max_tokens": tokens, "temperature": 0, "stream": True}
    if request_format == "/v1/chat/completions":
        body["messages"] = [{"role": "user", "content": "Say hello."}]
    else:
        body["prompt"] = "Hello, my name is"
    with client.stream("POST", request_format, json=body, timeout=httpx.Timeout(600.0, connect=5.0)) as response:
        if response.status_code != 200:
            return False
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return False
            try:
                event = json.loads(payload)
            except ValueError:
                continue
            for choice in event.get("choices") or []:
                text = choice.get("text")
                if text is None:
                    text = (choice.get("delta") or {}).get("content")
                if text:
                    return True
    return False


def first_answer(client: httpx.Client, model: str, t0: float, deadline: float, alive: Callable[[], str | None],
                 tokens: int = 16, request_format: str = "/v1/completions") -> float:
    """Seconds from `t0` until the first generated text of a streamed completion."""
    while True:
        try:
            if _first_text(client, model, tokens, request_format):
                return time.monotonic() - t0
        except httpx.HTTPError:
            pass
        _check(alive, deadline, "generated text")
        time.sleep(POLL_S)


def reported_version(client: httpx.Client) -> str | None:
    """The version from `GET /version` (vLLM's OpenAI server answers {"version": ...}), or None."""
    try:
        response = client.get("/version", timeout=5.0)
        if response.status_code == 200:
            value = response.json().get("version")
            return str(value) if value else None
    except (httpx.HTTPError, ValueError, AttributeError):
        pass
    return None
