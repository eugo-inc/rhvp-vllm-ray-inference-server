#!/usr/bin/env bash
# Build the inference server image and serve the model with docker, the way
# OpenShift runs it: an arbitrary UID in group 0, a 4 GiB /dev/shm, fresh
# compile caches, the model cache on a volume, and the config file mounted at
# /config/llm.yaml. Needs docker, curl and one idle NVIDIA GPU with the NVIDIA
# Container Toolkit; nothing else on the host.
#
# What it checks, in order:
#   1. The GPU is idle: it refuses to start when nvidia-smi shows more than 10%
#      utilization or any compute process (FORCE=1 skips this check).
#   2. The image builds (SKIP_BUILD=1 reuses an existing $IMAGE).
#   3. The server starts. It logs when GET /v1/models first answers and when
#      the ready file /tmp/r/rhvp-ready appears; /v1/models is expected to
#      answer first, which is why the probes use the ready file.
#   4. One /v1/chat/completions request (max_tokens 16) answers; the reply is
#      printed.
#   5. rhvp_status.py inside the container prints RUNNING.
#   6. docker stop (SIGTERM) shuts the server down cleanly (exit code 0).
#
# Settings (environment variables):
#   IMAGE                         image tag (default rhvp-server:dev)
#   SKIP_BUILD=1                  do not build; use $IMAGE as it is
#   FORCE=1                       start even if the GPU looks busy
#   SMOKE_PORT                    host port for the server (default 8000)
#   SMOKE_GPU_MEMORY_UTILIZATION  vLLM gpu_memory_utilization (default 0.20,
#                                 lower than the server's 0.85 so the test fits
#                                 next to other work on the same GPU)
#   SMOKE_MAX_MODEL_LEN           vLLM max_model_len (default 8192)
#   SMOKE_MODEL_ID / SMOKE_MODEL_SOURCE / SMOKE_MODEL_REVISION
#                                 default: global.model.* from values-global.yaml
#   SMOKE_MODELS_VOLUME           docker volume for /models (default
#                                 rhvp-smoke-models; kept between runs so the
#                                 weights download once; remove it with
#                                 docker volume rm rhvp-smoke-models)
#   SMOKE_TIMEOUT                 seconds to wait for the ready file (default 3600)
#   SMOKE_LOG_DIR                 where logs go (default ${TMPDIR:-/tmp}/rhvp-smoke)
#   KEEP=1                        leave the container running at the end
#   HF_TOKEN                      passed to the container when set (gated models)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-rhvp-server:dev}"
PORT="${SMOKE_PORT:-8000}"
GMU="${SMOKE_GPU_MEMORY_UTILIZATION:-0.20}"
MAX_LEN="${SMOKE_MAX_MODEL_LEN:-8192}"
MODELS_VOLUME="${SMOKE_MODELS_VOLUME:-rhvp-smoke-models}"
TIMEOUT="${SMOKE_TIMEOUT:-3600}"
LOG_DIR="${SMOKE_LOG_DIR:-${TMPDIR:-/tmp}/rhvp-smoke}"
NAME="rhvp-smoke-$(date +%Y%m%d-%H%M%S)"
READY_FILE=/tmp/r/rhvp-ready

mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
TIMELINE="$LOG_DIR/timeline.log"
: > "$TIMELINE"

say() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$TIMELINE"; }
die() { say "FAIL: $*"; exit 1; }

gpu_is_idle() {
    local util apps line
    command -v nvidia-smi > /dev/null || { say "nvidia-smi not found"; return 1; }
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)" \
        || { say "nvidia-smi failed"; return 1; }
    apps="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)" \
        || { say "nvidia-smi failed"; return 1; }
    while IFS= read -r line; do
        line="${line//[[:space:]]/}"
        if ! [[ "$line" =~ ^[0-9]+$ ]]; then
            say "GPU utilization is not a number ('$line')"
            return 1
        fi
        if (( line > 10 )); then
            say "GPU utilization is ${line}% (limit 10%)"
            return 1
        fi
    done <<< "$util"
    if [[ -n "${apps//[[:space:]]/}" ]]; then
        say "GPU compute processes are running: $(echo "$apps" | tr '\n' ';')"
        return 1
    fi
    return 0
}

require_idle_gpu() {
    if [[ "${FORCE:-0}" == 1 ]]; then
        say "FORCE=1: skipping the GPU idle check"
        return 0
    fi
    gpu_is_idle || die "the GPU is busy; try later, or set FORCE=1 if you are sure"
    say "GPU is idle"
}

port_in_use() { (exec 3<> "/dev/tcp/127.0.0.1/$1") 2> /dev/null; }

http_code() {
    curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$@" 2> /dev/null || true
}

cleanup() {
    local rc=$?
    if docker container inspect "$NAME" > /dev/null 2>&1; then
        docker logs "$NAME" > "$LOG_DIR/server.log" 2>&1 || true
        if [[ "${KEEP:-0}" == 1 && $rc -eq 0 ]]; then
            say "KEEP=1: container $NAME left running on port $PORT"
        else
            docker stop -t 90 "$NAME" > /dev/null 2>&1 || true
            docker logs "$NAME" > "$LOG_DIR/server.log" 2>&1 || true
            docker rm "$NAME" > /dev/null 2>&1 || true
        fi
    fi
    say "logs: $LOG_DIR"
    exit "$rc"
}

# ---------------------------------------------------------------- 1. GPU idle
require_idle_gpu
port_in_use "$PORT" && die "port $PORT is already in use; set SMOKE_PORT"

# ---------------------------------------------------------------- 2. build
if [[ "${SKIP_BUILD:-0}" == 1 ]]; then
    docker image inspect "$IMAGE" > /dev/null 2>&1 || die "SKIP_BUILD=1 but $IMAGE does not exist"
    say "using existing image $IMAGE"
else
    say "building $IMAGE (log: $LOG_DIR/build.log)"
    started=$SECONDS
    docker build -t "$IMAGE" -f "$REPO/inference-server/Containerfile" "$REPO/inference-server" \
        > "$LOG_DIR/build.log" 2>&1 \
        || die "docker build failed; see $LOG_DIR/build.log"
    say "built $IMAGE in $(( SECONDS - started )) s"
fi

# ---------------------------------------------------------------- config file
# Model defaults come from values-global.yaml (global.model.*), read with the
# image's own Python so the host needs no YAML parser.
read_model_value() {
    docker run --rm -i --entrypoint python "$IMAGE" -c '
import sys, yaml
data = yaml.safe_load(sys.stdin) or {}
model = (data.get("global") or {}).get("model") or {}
value = model.get(sys.argv[1])
print("" if value is None else value)
' "$1" < "$REPO/values-global.yaml"
}
MODEL_ID="${SMOKE_MODEL_ID:-$(read_model_value id)}"
MODEL_SOURCE="${SMOKE_MODEL_SOURCE:-$(read_model_value source)}"
MODEL_REVISION="${SMOKE_MODEL_REVISION-$(read_model_value revision)}"
MODEL_ID="${MODEL_ID:-qwen2.5-0.5b-instruct}"
MODEL_SOURCE="${MODEL_SOURCE:-Qwen/Qwen2.5-0.5B-Instruct}"

cat > "$LOG_DIR/llm.yaml" <<EOF
app_name: llm
model_id: "$MODEL_ID"
model_source: "$MODEL_SOURCE"
model_revision: "$MODEL_REVISION"
engine_kwargs: {max_model_len: $MAX_LEN, gpu_memory_utilization: $GMU, tensor_parallel_size: 1}
accelerator_type: null
min_replicas: 1
max_replicas: 1
object_store_memory_gb: 2
http_host: 0.0.0.0
http_port: 8000
EOF
chmod 0644 "$LOG_DIR/llm.yaml"
say "config: model_id=$MODEL_ID model_source=$MODEL_SOURCE revision=${MODEL_REVISION:-latest} gpu_memory_utilization=$GMU"
docker run --rm -v "$LOG_DIR/llm.yaml:/config/llm.yaml:ro" --entrypoint python "$IMAGE" \
    /opt/app-root/src/serve_app.py --print-config > "$LOG_DIR/resolved-config.json" \
    || die "serve_app.py rejected the config; see $LOG_DIR/resolved-config.json"

# ---------------------------------------------------------------- 3. start
require_idle_gpu   # again: the build can take a long time
trap cleanup EXIT

run_args=(
    -d --name "$NAME"
    --gpus all
    --user 123456:0          # an arbitrary UID in group 0, like OpenShift
    --shm-size 4g            # the chart's /dev/shm emptyDir is 4Gi
    --init                   # reaps child processes, as the pod's shared PID namespace does
    --tmpfs "/cache:rw,exec,mode=1777,size=16g"   # compile caches start empty, like the emptyDir
    --tmpfs "/tmp/r:rw,exec,mode=1777,size=8g"    # RAY_TMPDIR and the ready file
    -v "$MODELS_VOLUME:/models"
    -v "$LOG_DIR/llm.yaml:/config/llm.yaml:ro"
    -p "127.0.0.1:$PORT:8000"
)
if [[ -n "${HF_TOKEN:-}" ]]; then
    run_args+=(-e HF_TOKEN)
fi

say "starting container $NAME"
start=$SECONDS
docker run "${run_args[@]}" "$IMAGE" > /dev/null

models_at=""
ready_at=""
while (( SECONDS - start < TIMEOUT )); do
    if [[ "$(docker container inspect -f '{{.State.Running}}' "$NAME" 2> /dev/null)" != true ]]; then
        docker logs "$NAME" > "$LOG_DIR/server.log" 2>&1 || true
        die "the container stopped before it was ready; see $LOG_DIR/server.log"
    fi
    if [[ -z "$models_at" && "$(http_code "http://127.0.0.1:$PORT/v1/models")" == 200 ]]; then
        models_at=$(( SECONDS - start ))
        say "GET /v1/models first answered 200 at +${models_at} s"
    fi
    if docker exec "$NAME" test -f "$READY_FILE"; then
        ready_at=$(( SECONDS - start ))
        say "ready file $READY_FILE appeared at +${ready_at} s"
        break
    fi
    sleep 2
done
[[ -n "$ready_at" ]] || die "no ready file after ${TIMEOUT} s"
if [[ -z "$models_at" && "$(http_code "http://127.0.0.1:$PORT/v1/models")" == 200 ]]; then
    models_at=$ready_at
fi
if [[ -n "$models_at" && "$models_at" -lt "$ready_at" ]]; then
    say "as expected, /v1/models answered before the model was loaded (+${models_at} s vs +${ready_at} s)"
else
    say "note: /v1/models did not answer before the ready file (/v1/models at +${models_at:-never} s)"
fi

# ---------------------------------------------------------------- 4. chat
payload="$(printf '{"model": "%s", "messages": [{"role": "user", "content": "Say hello in five words."}], "max_tokens": 16, "temperature": 0}' "$MODEL_ID")"
code="$(curl -s -o "$LOG_DIR/chat.json" -w '%{http_code}' --max-time 120 \
    -H 'Content-Type: application/json' -d "$payload" \
    "http://127.0.0.1:$PORT/v1/chat/completions" || true)"
[[ "$code" == 200 ]] || die "chat completion returned HTTP $code; see $LOG_DIR/chat.json"
reply="$(docker exec -i "$NAME" python -c '
import json, sys
body = json.load(sys.stdin)
print(body["choices"][0]["message"]["content"].strip().replace("\n", " "))
print("completion_tokens=%s" % body.get("usage", {}).get("completion_tokens"))
' < "$LOG_DIR/chat.json")" || die "could not read the chat reply; see $LOG_DIR/chat.json"
say "chat reply: $reply"

# ---------------------------------------------------------------- 5. status
status_rc=0
docker exec "$NAME" python rhvp_status.py > "$LOG_DIR/status.txt" 2>> "$LOG_DIR/status.txt" || status_rc=$?
say "rhvp_status.py: $(head -n 1 "$LOG_DIR/status.txt") (exit $status_rc)"
[[ "$status_rc" -eq 0 ]] || die "rhvp_status.py did not report RUNNING"

docker exec "$NAME" sh -c 'echo "uid=$(id -u) gid=$(id -g)"; env | grep -E "^(RAY_USAGE_STATS_ENABLED|VLLM_NO_USAGE_STATS|DO_NOT_TRACK|RAYLLM_ENABLE_REQUEST_PROMPT_LOGS|RAY_TMPDIR|HF_HOME)=" | sort' \
    > "$LOG_DIR/env.txt"
say "container identity and settings: $(tr '\n' ' ' < "$LOG_DIR/env.txt")"

# ---------------------------------------------------------------- 6. stop
if [[ "${KEEP:-0}" == 1 ]]; then
    say "PASS (container kept running)"
    exit 0
fi
say "stopping with SIGTERM"
stop_start=$SECONDS
docker stop -t 90 "$NAME" > /dev/null
exit_code="$(docker container inspect -f '{{.State.ExitCode}}' "$NAME")"
say "container exited with code $exit_code after $(( SECONDS - stop_start )) s"
[[ "$exit_code" == 0 ]] || die "the server did not shut down cleanly"
say "PASS"
