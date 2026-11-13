#!/usr/bin/env bash
# Compare two inference-server images on one machine with docker, no cluster: the same runner, rounds, GuideLLM
# load and analysis as the OpenShift Job (see README, "Compare two images"), with GPU power from nvidia-smi instead
# of DCGM. Needs docker and, for GPU sides, one NVIDIA GPU with the NVIDIA Container Toolkit; nothing else on the
# host.
#
#   scripts/compare-local.sh <config.json> <results folder>
#
# The config file is JSON with the keys of global.compare in values-global.yaml, plus "model" (the keys of
# global.model). tests/fixtures/compare-local-mock.json is a small one that needs no GPU: two GuideLLM mock
# servers with different speeds. tests/fixtures/compare-scaleout-mock.json is the same in mode scaleout: mock
# servers that take a while to start, one surge per round, and a second replica at the forced step.
#
# What it does:
#   1. Creates the docker network $NETWORK if it is missing.
#   2. Runs the runner (notebooks/rhvp_compare, mounted read-only) in a container of the GuideLLM image on that
#      network, with the docker socket mounted. The runner starts each side and the nvidia-smi sampler as sibling
#      containers on the same network, one round at a time, and runs GuideLLM as its own child process, as the
#      OpenShift Job does.
#   3. Prints where results.json and results.md are. The exit code is the runner's: 0 when every round finished,
#      1 for a partial run, 2 for a config that cannot run, 3 for a run that failed outside a round, 4 when the
#      server container (SERVER_CONTAINER) could not be started again, 5 when another comparison was running.
#
# One comparison at a time: the runner container carries the label rhvp-compare.runner, and a run refuses to start
# while an older one is running. Stopping the runner (docker stop, or Ctrl-C on this script) cuts the current round
# short, takes its side down and saves the results as partial, within STOP_TIMEOUT.
#
# Each side runs as an arbitrary UID in group 0 (as on OpenShift), with its caches on a fresh tmpfs and the model
# weights on the docker volume $MODELS_VOLUME, kept between runs (remove it with docker volume rm). A side with
# compileCache: warm gets its own cache volume instead (named after $CACHE_VOLUME and the run), filled by its warm-up
# round and removed when the run ends.
#
# Mode scaleout (mode: scaleout in the config): each round starts a second container of the side, named like the
# first plus -2, at the forced step; the runner puts a small router in front of the two (router.py, a child process
# of the runner, like GuideLLM). Both replicas get GPUS unless REPLICA_GPUS gives each its own; the run is refused
# when that does not match scaleOut.sharedGpu (true: one GPU for both, false: a GPU of its own for each).
#
# Settings (environment variables):
#   LOAD_IMAGE      the GuideLLM image the runner runs in (default: the pinned release); set it to the same
#                   image when the config sets load.image
#   NETWORK         docker network (default rhvp-compare)
#   MODELS_VOLUME   docker volume for /models (default rhvp-compare-models)
#   GPUS            "all" (default) or a device id such as 0; give one id on a machine with several GPUs
#   REPLICA_GPUS    mode scaleout: one device id per replica, comma-separated (for example 0,1, or two MIG
#                   devices such as 0:0,0:1), for scaleOut.sharedGpu: false; default empty, every replica gets
#                   GPUS (one GPU shared, as scaleOut.sharedGpu: true expects)
#   CPUSET          CPUs the runner and every container it starts may use, as docker writes them (for example
#                   0-3; default all)
#   CACHE_VOLUME    name prefix of the warm compile-cache volumes (default rhvp-compare-cache)
#   HF_TOKEN        passed on when it starts with hf_ (gated models)
#   RUNNER_MEMORY   memory limit of the runner container, GuideLLM included (default 8g)
#   SIDE_MEMORY     memory limit of each side's container, as docker writes it (default none, for example 32g)
#   SERVER_CONTAINER  a container that normally holds the GPU: stopped for the run, started again after it (one
#                   that stopped by itself before the run took it is left as it is; results.md says so)
#   STOP_TIMEOUT    seconds the runner has to stop cleanly after docker stop (default 120)
set -euo pipefail

if [ "$#" -ne 2 ]; then
  sed -n '2,13p' "$0" >&2
  exit 2
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
mkdir -p "$2"
RESULTS="$(cd "$2" && pwd)"
NETWORK="${NETWORK:-rhvp-compare}"
MODELS_VOLUME="${MODELS_VOLUME:-rhvp-compare-models}"
GPUS="${GPUS:-all}"
REPLICA_GPUS="${REPLICA_GPUS:-}"
CPUSET="${CPUSET:-}"
CACHE_VOLUME="${CACHE_VOLUME:-rhvp-compare-cache}"
RUNNER_MEMORY="${RUNNER_MEMORY:-8g}"
SIDE_MEMORY="${SIDE_MEMORY:-}"
SERVER_CONTAINER="${SERVER_CONTAINER:-}"
STOP_TIMEOUT="${STOP_TIMEOUT:-120}"
RUNNER_NAME="rhvp-compare-runner-$$"
SOCKET=/var/run/docker.sock

[ -f "$CONFIG" ] || { echo "no config file at $CONFIG" >&2; exit 2; }
[ -S "$SOCKET" ] || { echo "no docker socket at $SOCKET" >&2; exit 2; }

# The pinned GuideLLM release (GUIDELLM_IMAGE in notebooks/rhvp_compare/plan.py; a test keeps the two equal).
LOAD_IMAGE="${LOAD_IMAGE:-ghcr.io/vllm-project/guidellm:v0.8.1@sha256:015b22441020e6014b4c06d89882921eb29e9b9a6f2e274b253afef4029aa081}"
docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK" >/dev/null

token_env=()
case "${HF_TOKEN:-}" in hf_*) token_env=(-e HF_TOKEN) ;; esac
cpuset_flag=()
[ -z "$CPUSET" ] || cpuset_flag=(--cpuset-cpus "$CPUSET")

docker run --rm --name "$RUNNER_NAME" --network "$NETWORK" \
  --label rhvp-compare.runner=1 --memory "$RUNNER_MEMORY" --stop-timeout "$STOP_TIMEOUT" --init \
  "${cpuset_flag[@]}" \
  --user "$(id -u):$(id -g)" --group-add "$(stat -c %g "$SOCKET")" \
  -v "$SOCKET:/var/run/docker.sock" \
  -v "$RESULTS:/results" \
  -v "$CONFIG:/config/compare.json:ro" \
  -v "$REPO/notebooks/rhvp_compare:/runner/rhvp_compare:ro" \
  -e PYTHONPATH=/runner \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e HOME=/tmp/home \
  -e HF_HOME=/tmp/hf \
  -e RHVP_COMPARE_NETWORK="$NETWORK" \
  -e RHVP_COMPARE_HOST_RESULTS="$RESULTS" \
  -e RHVP_COMPARE_MODELS_VOLUME="$MODELS_VOLUME" \
  -e RHVP_COMPARE_GPUS="$GPUS" \
  -e RHVP_COMPARE_REPLICA_GPUS="$REPLICA_GPUS" \
  -e RHVP_COMPARE_CPUSET="$CPUSET" \
  -e RHVP_COMPARE_CACHE_VOLUME="$CACHE_VOLUME" \
  -e RHVP_COMPARE_MEMORY="$SIDE_MEMORY" \
  -e RHVP_COMPARE_SERVER_CONTAINER="$SERVER_CONTAINER" \
  -e RHVP_COMPARE_RUNNER_NAME="$RUNNER_NAME" \
  "${token_env[@]}" \
  -w /runner --entrypoint python "$LOAD_IMAGE" \
  -m rhvp_compare run --backend docker --config /config/compare.json --results /results
