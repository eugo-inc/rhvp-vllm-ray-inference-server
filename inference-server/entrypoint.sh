#!/usr/bin/env bash
# Entrypoint of the inference server image. It replaces the base image's
# /opt/nvidia/nvidia_entrypoint.sh, which prints NVIDIA's banner and license
# notice, warns when no GPU driver is visible, and then runs its arguments.
# The NVIDIA license notice stays in the image at /NGC-DL-CONTAINER-LICENSE.
#
# 1. Creates the writable folders the server uses, when no volume is mounted
#    there (OpenShift mounts emptyDirs at /cache and /tmp/r and a PVC at /models).
# 2. Removes a ready file left by an earlier container in the same pod: the
#    /tmp/r emptyDir survives a container restart, and a stale file would make
#    the startup and readiness probes pass before the model is loaded.
# 3. Keeps usage telemetry and request-prompt logging off unless the operator
#    set these variables explicitly.
# 4. Unsets HF_TOKEN unless it looks like a Hugging Face token (starts with
#    "hf_"); the pattern's placeholder secret is "none".
# 5. Runs serve_app.py as this process (exec), so it receives SIGTERM directly.
set -euo pipefail

: "${RAY_TMPDIR:=/tmp/r}"
: "${HOME:=/opt/app-root/home}"
: "${HF_HOME:=/models/hf}"
: "${TRITON_CACHE_DIR:=/cache/triton}"
: "${VLLM_CACHE_ROOT:=/cache/vllm}"
: "${FLASHINFER_WORKSPACE_BASE:=/cache/flashinfer}"
: "${XDG_CACHE_HOME:=/cache/xdg}"
export RAY_TMPDIR HOME HF_HOME TRITON_CACHE_DIR VLLM_CACHE_ROOT FLASHINFER_WORKSPACE_BASE XDG_CACHE_HOME

for dir in "$RAY_TMPDIR" "$HOME" "$HF_HOME" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" \
           "$FLASHINFER_WORKSPACE_BASE" "$XDG_CACHE_HOME"; do
    if ! mkdir -p "$dir" 2>/dev/null || [ ! -w "$dir" ]; then
        echo "entrypoint: $dir is not writable by uid $(id -u) gid $(id -g);" \
             "mount a volume there that group 0 can write" >&2
        exit 1
    fi
done

rm -f "$RAY_TMPDIR/rhvp-ready"

: "${RAY_USAGE_STATS_ENABLED:=0}"
: "${VLLM_NO_USAGE_STATS:=1}"
: "${DO_NOT_TRACK:=1}"
: "${RAYLLM_ENABLE_REQUEST_PROMPT_LOGS:=0}"
export RAY_USAGE_STATS_ENABLED VLLM_NO_USAGE_STATS DO_NOT_TRACK RAYLLM_ENABLE_REQUEST_PROMPT_LOGS

case "${HF_TOKEN:-}" in
    hf_*) ;;
    *) unset HF_TOKEN ;;
esac

exec python /opt/app-root/src/serve_app.py "$@"
