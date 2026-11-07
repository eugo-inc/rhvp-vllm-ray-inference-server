# SPDX-License-Identifier: Apache-2.0
"""Every sentence and label the notebook shows, as constants.

Nothing on screen is built from a host name, a URL, an image name or a file path: every label is written here by
hand, and run figures are filled into these templates. Results are given in seconds, tokens per second and percent;
never as a ratio. No sentence here says why one side differs from the other.
"""

WITHOUT = "WITHOUT EUGO"
WITH = "WITH EUGO"

#: The software the WITHOUT EUGO side runs: the server image built from inference-server/ (a test checks it against
#: the pins in inference-server/requirements.in, the two lock files and the Containerfile).
BASELINE = "Stock vLLM 0.29.0 · Ray 2.60.0 · PyTorch 2.13.0 from PyPI · Python 3.12 · NVIDIA CUDA 13.0.3 UBI 9 image"

#: Shown on the WITH EUGO side, and nowhere else, whatever the circumstances: it says nothing about why and promises
#: nothing.
NO_MEASURED = "No measured result is shown for this comparison."

#: Every live result is labelled with the run it comes from, and followed by the draw line.
IN_THIS_RUN = "In this run"
IN_THIS_REPLAY = "In this replay"
SINGLE_RUN = "A single run can come out either way."
DRAW_MOVES = "These figures move from run to run."
ONE_MEASUREMENT = "One measurement, no p-value."
STREAM_GROUPS_NOTE = "Ray Serve LLM sends streamed tokens in groups, every 50 ms by default."

#: The amber strips above a result.
INTERNAL_STRIP = ("INTERNAL: run on a shared machine, for checking only. Not a result to show outside Eugo.")
REPLAY_STRIP = ("REPLAY: no request reaches a real server. The results below replay a recorded run, only to check "
                "the layout; they are not measurements.")
STRIP = {"INTERNAL": INTERNAL_STRIP, "REPLAY": REPLAY_STRIP}

# ---- the notebook's markdown cells (showing.py) ----

TITLE = "Stock vLLM inference server: WITHOUT EUGO and WITH EUGO"
#: A markdown-level banner line, so the two sides show even where styles are stripped (GitHub's notebook view).
TITLE_BANNER = "# 🟥 WITHOUT EUGO · 🟪 WITH EUGO"
INTRO = ("This notebook times the inference server this pattern installs: stock vLLM served by Ray Serve LLM, built "
         "only from stock packages. It talks to the server inside the cluster through its OpenAI-compatible API. "
         "Each section measures the WITHOUT EUGO side live. Sections 1 and 2 also have a WITH EUGO side. A WITH EUGO "
         "result appears only when it was measured in the same run as WITHOUT EUGO, on the same hardware, with the "
         "same requests and server settings.")
INTRO_HOW = ("Run the cells from the top. The first cell waits until the server answers a one-token completion. To "
             "check the layout without a GPU, start JupyterLab with RHVP_REPLAY set to a recording from the "
             "recordings folder.")
SECTION_HEADING = "## {number} · {title}"
SECTION_TITLE = {
    1: "Server start to ready",
    2: "Time to first token",
    3: "Decode speed (one stream)",
    4: "Throughput at 1, 8 and 32 concurrent streams",
}
SECTION_TEXT = {
    1: ("How long the server pod takes from being placed on a node to Ready, read from the pod's own Kubernetes "
        "conditions. The pod turns Ready once the server has loaded its model and is serving. One start of the pod "
        "is one measurement."),
    2: ("The time from just before a request is sent to the first generated text coming back, for requests sent one "
        "at a time. Every prompt differs from its first word on, and vLLM's prefix cache (on by default) has nothing "
        "to reuse."),
    3: ("How fast one stream generates text once it has started: the tokens after the first, divided by the time "
        "from the first generated text to the last. Requests are sent one at a time. This section has no WITH EUGO "
        "side."),
    4: ("All generated tokens per second, with 1, 8 and 32 streams running at the same time, every stream generating "
        "the same number of tokens. This section has no WITH EUGO side."),
}
SIDE_HEADING = {WITHOUT: "### 🟥 WITHOUT EUGO", WITH: "### 🟪 WITH EUGO"}

# ---- the readiness cell ----

READY_ANSWERS = "The server answers."
READY_LISTED = "The model is listed."
READY_FIRST = "A one-token completion came back."
READY_SOFTWARE = "Server software: {baseline}"
READY_MODEL = "Model: {model_id}"
READY_REVISION = "Hugging Face revision: {revision}"
READY_WAITING = "Waiting for the server: {step}. {seconds:.0f} s so far."
READY_STEP = {
    "answers": "waiting for it to answer",
    "listed": "waiting for the model to be listed",
    "first": "waiting for a one-token completion",
}
NOT_READY = "The server is not ready after {minutes:.0f} minutes: still {step}."
REPLAY_READY = "The stand-in server is up. No request reaches a real server."
REPLAY_SYNTHETIC = "The recording is synthetic: written by a script, never measured."

# ---- section 1: server start to ready ----

STARTUP_LABEL = "the server pod's latest start, from being placed on a node to Ready"
STARTUP_STEPS = ("Placed on a node to container start: {to_start} s, including any image pull. Container start to "
                 "Ready: {to_ready} s.")
STARTUP_EMPTY = "Empty when the pod started: {names}."
STARTUP_KEPT = "Kept between starts: {names}."
STARTUP_SECONDS = "Kubernetes records these times in whole seconds."
#: What each of the server container's storage settings holds, by environment variable.
CACHE_NAMES = {
    "TRITON_CACHE_DIR": "the Triton kernel cache",
    "VLLM_CACHE_ROOT": "the vLLM cache",
    "FLASHINFER_WORKSPACE_BASE": "the FlashInfer cache",
    "XDG_CACHE_HOME": "the other caches",
    "HF_HOME": "the downloaded model files",
}
STARTUP_NOT_IN_CLUSTER = ("Not measured here: the server pod's start is read from the Kubernetes API, which this "
                          "notebook reaches only inside the cluster.")
STARTUP_NO_POD = "No server pod was found."
STARTUP_NOT_READY = "The server pod is not Ready yet. Run this cell again once the first cell has passed."
STARTUP_RESTARTED = ("No start time is shown: the server container has restarted inside its pod ({restarts} times), "
                     "and a restart keeps the pod's caches.")
STARTUP_UNREADABLE = "The server pod's conditions could not be read ({error})."

# ---- sections 2 to 4 ----

TTFT_LABEL = "median time to first token"
TTFT_SPREAD = "p90 {p90:.3f} s · {kept} of {sent} requests counted."
TTFT_TIMED = "Timed from just before each request is sent to the first generated text that comes back."
DECODE_LABEL = "median decode speed, one stream at a time"
DECODE_SPREAD = "Slowest {low:.1f} · fastest {high:.1f} tokens/s · {kept} of {sent} requests counted."
DECODE_TIMED = "The tokens after the first, divided by the time from the first generated text to the last."
THROUGHPUT_LABEL = {1: "1 stream"}
THROUGHPUT_LABEL_MANY = "{streams} concurrent streams"
THROUGHPUT_CARD = ("{kept} of {sent} requests counted · {seconds:.2f} s from the first request sent to the last stream "
                   "finished.")
THROUGHPUT_TIMED = ("All generated tokens of the counted requests, divided by the time from the first request sent to "
                    "the last stream finished.")
TOKENS_PER_S = "tokens/s"
SECONDS = "s"
REQUEST_SHAPE = ("Each request: its own prompt of about {prompt_tokens} tokens (the server's own count), {tokens} "
                 "tokens generated, temperature 0.")
REQUEST_SHAPE_UNCOUNTED = "Each request: its own prompt, {tokens} tokens generated, temperature 0."
ONE_AT_A_TIME = "Requests sent one at a time."
WARMUP = "Before timing: {warmup} warm-up requests with other prompts, not counted."
#: A number of requests, in words ("1 request", "3 requests").
REQUESTS = {1: "1 request"}
REQUESTS_MANY = "{n} requests"
DROPPED = "Not counted: {dropped} that did not generate exactly {tokens} tokens."
FAILED = "Not counted: {failed} that failed ({errors})."
NONE_COUNTED = "No result: no request generated exactly {tokens} tokens."
NO_SPAN = "No result: every counted request's text came back in a single network write."
NO_FIGURE = "none"
SERVER_ERROR = "The server did not answer ({error}). Run the first cell, then this one again."
MEASURING = "Measuring: {done} of {total} requests done."
RECORDING_UNUSABLE = "REPLAY is on, but {reason}."
