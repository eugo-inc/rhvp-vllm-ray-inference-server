#!/usr/bin/env bash
# Lint and render every chart of this pattern the way the cluster renders it.
#
# helm runs inside the Validated Patterns utility container, so the host needs only
# docker or podman, plus network access to charts.validatedpatterns.io for the
# clustergroup chart. Every helm call gets the values the Validated Patterns operator
# injects at deploy time (repo URL, branch, platform AWS, OpenShift 4.20, cluster
# domains), after values-global.yaml, variants/prod/values-prod.yaml and the override
# files the operator would include through clusterGroup.sharedValueFiles.
#
# Output, in ${RENDER_DIR:-.render} (a relative path is relative to the repo root):
#   charts/<chart>.yaml          each charts/all/<chart>, as Argo CD renders it
#   charts/<chart>-variant.yaml  ray-serve-llm and notebook with image.override set,
#                                plus route.enabled=true / notebook.internal=true
#   charts/notebook-compare*.yaml the notebook chart with global.compare.enabled=true
#                                (also with energy.source=none and image.override, with
#                                bigger side pods: a new Job with the same config, and with
#                                mode scaleout and a nested block, passed through as given)
#   charts/nvidia-gpu-config-timeslicing.yaml  with the opt-in GPU time-slicing on
#   charts/ray-serve-llm-scaling.yaml  in-pod autoscaling from 1 to 2 replicas sharing the GPU
#   charts/ray-serve-llm-cache-<mode>.yaml  compileCache.mode node and shared
#   charts/<chart>-keda.yaml     every chart with global.scaleOut.mode=keda and that case's
#                                values files, and notebook-compare-keda.yaml with the
#                                comparison on too
#   refused/<case>.txt           helm's error for each setting the charts must refuse (the
#                                script fails if one renders, or fails without its message)
#   clustergroup/<case>.yaml     the clustergroup chart (the app of apps) for:
#                                  default    AWS, global.gpuMachineSet from values-global.yaml
#                                  none       AWS, global.gpuMachineSet=none
#                                  gcp        GCP, global.gpuMachineSet from values-global.yaml
#                                  keda       AWS, global.scaleOut.mode=keda
#   clustergroup/<case>.valuefiles  the values files used for that case
#
# Any helm error (lint, template, or the clustergroup values schema) fails the script.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The utility container's v1.0.5 release, pinned by digest. Pin a release tag's digest, not latest's: quay
# deletes a latest digest once latest is rebuilt (the earlier pin vanished that way in November 2026).
UTILITY_IMAGE="${UTILITY_IMAGE:-quay.io/validatedpatterns/utility-container@sha256:556083f92b08642868ca64ade6cf9553295bd394f7a54d5e10ca12d143907214}"
CLUSTERGROUP_CHART_VERSION="${CLUSTERGROUP_CHART_VERSION:-0.9.58}"
RENDER_DIR="${RENDER_DIR:-.render}"
case "${RENDER_DIR}" in
  /*) ;;
  *) RENDER_DIR="${REPO_ROOT}/${RENDER_DIR}" ;;
esac

if [[ "${RHVP_RENDER_INSIDE:-}" != "1" ]]; then
  if command -v podman >/dev/null 2>&1; then
    ENGINE=(podman run --rm --userns=keep-id --security-opt label=disable)
  elif command -v docker >/dev/null 2>&1; then
    ENGINE=(docker run --rm --user "$(id -u):$(id -g)")
  else
    echo "render-charts.sh: needs podman or docker" >&2
    exit 2
  fi
  mkdir -p "${RENDER_DIR}"
  exec "${ENGINE[@]}" \
    -e HOME=/tmp \
    -e RHVP_RENDER_INSIDE=1 \
    -e RENDER_DIR="${RENDER_DIR}" \
    -e CLUSTERGROUP_CHART_VERSION="${CLUSTERGROUP_CHART_VERSION}" \
    -v "${REPO_ROOT}:${REPO_ROOT}" \
    -v "${RENDER_DIR}:${RENDER_DIR}" \
    -w "${REPO_ROOT}" \
    --entrypoint bash \
    "${UTILITY_IMAGE}" "${REPO_ROOT}/scripts/render-charts.sh"
fi

# ---- inside the utility container ------------------------------------------------
cd "${REPO_ROOT}"
export HELM_CACHE_HOME=/tmp/helm/cache HELM_CONFIG_HOME=/tmp/helm/config HELM_DATA_HOME=/tmp/helm/data

REPO_URL="https://github.com/eugo-inc/vllm-ray-inference-server"
VARIANT="$(yq -r '.main.variant' values-global.yaml)"
PATTERN="$(yq -r '.global.pattern' values-global.yaml)"
VARIANT_FILE="variants/${VARIANT}/values-${VARIANT}.yaml"
mkdir -p "${RENDER_DIR}/charts" "${RENDER_DIR}/clustergroup" "${RENDER_DIR}/refused"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# Values the operator passes to the clustergroup chart and, through it, to every app
# (patterns-operator newApplicationParameters; clustergroup _helpers.tpl). Sets the
# array SETS for one platform.
operator_sets() {
  local platform="$1"
  SETS=(
    --set "global.repoURL=${REPO_URL}"
    --set "global.originURL=${REPO_URL}"
    --set "global.targetRevision=main"
    --set "global.pattern=${PATTERN}"
    --set-string "global.clusterVersion=4.20"
    --set "global.clusterPlatform=${platform}"
    --set "global.clusterDomain=example.com"
    --set "global.localClusterDomain=apps.example.com"
    --set "global.hubClusterDomain=apps.example.com"
    --set "global.localClusterName=local-cluster"
    --set "global.vpArgoNamespace=vp-gitops"
    --set-string "global.vpNewFolderDir=true"
    --set-string "global.multiSourceSupport=true"
    --set "global.multiSourceRepoUrl=https://charts.validatedpatterns.io"
    --set "global.multiSourceTargetRevision=${CLUSTERGROUP_CHART_VERSION}"
    --set-string "global.privateRepo=false"
  )
}

# The values files the operator and the clustergroup chart pass for one case: global,
# the variant, then each clusterGroup.sharedValueFiles entry, templated with helm's own
# tpl and kept only if the file exists (Argo CD skips missing value files). Sets the
# arrays FILES (paths) and FARGS (-f arguments). Extra arguments go to helm.
value_files() {
  local platform="$1"; shift
  local chart="${WORK}/svf" f
  mkdir -p "${chart}/templates"
  printf 'apiVersion: v2\nname: svf\nversion: 0.0.1\n' > "${chart}/Chart.yaml"
  cat > "${chart}/templates/files.yaml" <<'TPL'
files:
{{- range .Values.clusterGroup.sharedValueFiles }}
  - {{ tpl . $ | quote }}
{{- end }}
TPL
  operator_sets "${platform}"
  helm template svf "${chart}" -f values-global.yaml -f "${VARIANT_FILE}" "${SETS[@]}" "$@" \
    --show-only templates/files.yaml > "${WORK}/svf.yaml"
  FILES=(values-global.yaml "${VARIANT_FILE}")
  while IFS= read -r f; do
    f="${f#/}"
    if [[ -n "${f}" && -f "${f}" ]]; then FILES+=("${f}"); fi
  done < <(yq -r '.files[]' "${WORK}/svf.yaml")
  FARGS=()
  for f in "${FILES[@]}"; do FARGS+=(-f "${f}"); done
}

# ---- local charts ------------------------------------------------------------------
value_files AWS
operator_sets AWS
echo "values files (AWS): ${FILES[*]}"
# The namespace of a chart's app: in the variant file, or in an override file for an app that
# only an opt-in adds (keda-config, overrides/values-scaleout-keda.yaml).
app_namespace() {
  local chart="$1" f ns
  for f in "${VARIANT_FILE}" overrides/*.yaml; do
    ns="$(yq -r ".clusterGroup.applications // {} | .[] | select(.path == \"charts/all/${chart}\") | .namespace" "${f}")"
    if [[ -n "${ns}" && "${ns}" != "null" ]]; then
      echo "${ns}"
      return 0
    fi
  done
  return 1
}

CHARTS=()
for dir in charts/all/*/; do
  chart="$(basename "${dir}")"
  if ! ns="$(app_namespace "${chart}")"; then
    echo "render-charts.sh: charts/all/${chart} has no application in ${VARIANT_FILE} or overrides/" >&2
    exit 1
  fi
  CHARTS+=("${chart}:${ns}")
  helm lint --strict "${dir}" --namespace "${ns}" "${FARGS[@]}" "${SETS[@]}"
  helm template "${chart}" "${dir}" --namespace "${ns}" "${FARGS[@]}" "${SETS[@]}" \
    > "${RENDER_DIR}/charts/${chart}.yaml"
  echo "rendered charts/${chart}.yaml (namespace ${ns})"
done

# Variants: prebuilt image (no BuildConfig, ImageStream or trigger) and the opt-ins.
helm template ray-serve-llm charts/all/ray-serve-llm --namespace vllm-inference "${FARGS[@]}" "${SETS[@]}" \
  --set image.override=quay.io/example/ray-serve-llm:1.0 --set route.enabled=true \
  > "${RENDER_DIR}/charts/ray-serve-llm-variant.yaml"
helm template notebook charts/all/notebook --namespace vllm-notebook "${FARGS[@]}" "${SETS[@]}" \
  --set image.override=quay.io/example/notebook:1.0 --set notebook.internal=true \
  --set notebook.replay=/opt/app-root/notebooks/replay/example \
  > "${RENDER_DIR}/charts/notebook-variant.yaml"
# The image comparison switched on (global.compare.enabled), with and without energy.
helm template notebook charts/all/notebook --namespace vllm-notebook "${FARGS[@]}" "${SETS[@]}" \
  --set global.compare.enabled=true > "${RENDER_DIR}/charts/notebook-compare.yaml"
helm template notebook charts/all/notebook --namespace vllm-notebook "${FARGS[@]}" "${SETS[@]}" \
  --set global.compare.enabled=true --set global.compare.energy.source=none \
  --set image.override=quay.io/example/notebook:1.0 \
  > "${RENDER_DIR}/charts/notebook-compare-noenergy.yaml"
helm template notebook charts/all/notebook --namespace vllm-notebook "${FARGS[@]}" "${SETS[@]}" \
  --set global.compare.enabled=true --set compare.side.resources.limits.memory=32Gi \
  > "${RENDER_DIR}/charts/notebook-compare-sides.yaml"
helm template notebook charts/all/notebook --namespace vllm-notebook "${FARGS[@]}" "${SETS[@]}" \
  --set global.compare.enabled=true --set image.pullPolicy=Always \
  > "${RENDER_DIR}/charts/notebook-compare-pull.yaml"
helm template notebook charts/all/notebook --namespace vllm-notebook "${FARGS[@]}" "${SETS[@]}" \
  --set global.compare.enabled=true --set global.compare.mode=scaleout \
  --set-json 'global.compare.passThrough={"nested":{"rate":2.5,"names":["a","b"]}}' \
  > "${RENDER_DIR}/charts/notebook-compare-scaleout.yaml"
helm template nvidia-gpu-config charts/all/nvidia-gpu-config --namespace nvidia-gpu-operator "${FARGS[@]}" "${SETS[@]}" \
  --set timeSlicing.enabled=true > "${RENDER_DIR}/charts/nvidia-gpu-config-timeslicing.yaml"
echo "rendered charts/*-variant.yaml, charts/notebook-compare*.yaml and charts/nvidia-gpu-config-timeslicing.yaml"

# Scale-out settings (charts/all/ray-serve-llm/templates/_settings.tpl). --set reads 0.5 as a
# string, so the GPU share goes in as JSON. Two replicas on half a GPU each must each claim at
# most half of its memory, so the memory share is lowered too.
helm template ray-serve-llm charts/all/ray-serve-llm --namespace vllm-inference "${FARGS[@]}" "${SETS[@]}" \
  --set server.autoscaling.maxReplicas=2 --set server.autoscaling.targetOngoingRequests=4 \
  --set-json server.gpuPerReplica=0.5 --set-json server.engineKwargs.gpu_memory_utilization=0.4 \
  > "${RENDER_DIR}/charts/ray-serve-llm-scaling.yaml"
for cache in node shared; do
  helm template ray-serve-llm charts/all/ray-serve-llm --namespace vllm-inference "${FARGS[@]}" "${SETS[@]}" \
    --set compileCache.mode="${cache}" > "${RENDER_DIR}/charts/ray-serve-llm-cache-${cache}.yaml"
done

# Settings the charts must refuse: the render must fail, and say why.
refuse() {
  local name="$1" chart="$2" ns="$3" want="$4"
  shift 4
  if helm template "${chart}" "charts/all/${chart}" --namespace "${ns}" "${FARGS[@]}" "${SETS[@]}" "$@" \
      > /dev/null 2> "${RENDER_DIR}/refused/${name}.txt"; then
    echo "render-charts.sh: ${name}: the ${chart} chart rendered, but must refuse $*" >&2
    exit 1
  fi
  if ! grep -qF -- "${want}" "${RENDER_DIR}/refused/${name}.txt"; then
    echo "render-charts.sh: ${name}: the ${chart} chart failed without saying '${want}':" >&2
    cat "${RENDER_DIR}/refused/${name}.txt" >&2
    exit 1
  fi
}
refuse scaleout-kuberay ray-serve-llm vllm-inference "kuberay mode is designed but not built yet" \
  --set global.scaleOut.mode=kuberay
refuse scaleout-unknown ray-serve-llm vllm-inference "global.scaleOut.mode must be none or keda" \
  --set global.scaleOut.mode=hpa
refuse autoscaling-no-target ray-serve-llm vllm-inference "targetOngoingRequests must be set when maxReplicas" \
  --set server.autoscaling.maxReplicas=2
refuse autoscaling-over-gpus ray-serve-llm vllm-inference "is more than server.gpuCount" \
  --set server.autoscaling.maxReplicas=2 --set server.autoscaling.targetOngoingRequests=4
refuse autoscaling-shared-gpu-memory ray-serve-llm vllm-inference \
  "server.engineKwargs.gpu_memory_utilization (0.85) must be at most 0.95 times server.gpuPerReplica (0.5)" \
  --set server.autoscaling.maxReplicas=2 --set server.autoscaling.targetOngoingRequests=4 \
  --set-json server.gpuPerReplica=0.5
refuse autoscaling-shared-gpu-full-share ray-serve-llm vllm-inference \
  "server.engineKwargs.gpu_memory_utilization (0.5) must be at most 0.95 times server.gpuPerReplica (0.5)" \
  --set server.autoscaling.maxReplicas=2 --set server.autoscaling.targetOngoingRequests=4 \
  --set-json server.gpuPerReplica=0.5 --set-json server.engineKwargs.gpu_memory_utilization=0.5
refuse gpu-per-replica-one-worker ray-serve-llm vllm-inference \
  "server.gpuPerReplica (2) must be at most 1 when a replica has one worker" \
  --set server.gpuPerReplica=2 --set server.gpuCount=2
refuse gpu-per-replica-workers ray-serve-llm vllm-inference \
  "server.gpuPerReplica (0.5) must be 1 or equal engineKwargs.tensor_parallel_size times pipeline_parallel_size (2)" \
  --set-json server.gpuPerReplica=0.5 --set server.engineKwargs.tensor_parallel_size=2 --set server.gpuCount=2
refuse replicas-moved ray-serve-llm vllm-inference "moved to server.autoscaling" --set server.maxReplicas=2
refuse compare-mode-unknown notebook vllm-notebook "global.compare.mode must be pairs or scaleout" \
  --set global.compare.enabled=true --set global.compare.mode=surge
refuse keda-threshold ray-serve-llm vllm-inference "scaleOut.waitingRequestsPerPod must be a number above 0" \
  --set global.scaleOut.mode=keda --set scaleOut.waitingRequestsPerPod=0
refuse keda-thanos-url ray-serve-llm vllm-inference "scaleOut.thanosUrl must be an http:// or https:// address" \
  --set global.scaleOut.mode=keda --set scaleOut.thanosUrl=thanos-querier:9092
refuse keda-cooldown ray-serve-llm vllm-inference "scaleOut.cooldownSeconds must be a whole number from 0 to 3600" \
  --set global.scaleOut.mode=keda --set scaleOut.cooldownSeconds=7200
refuse timeslicing-replicas nvidia-gpu-config nvidia-gpu-operator "timeSlicing.replicas must be a whole number of at least 2" \
  --set timeSlicing.enabled=true --set timeSlicing.replicas=1
echo "rendered charts/ray-serve-llm-scaling.yaml, charts/ray-serve-llm-cache-*.yaml; refused/*.txt as expected"

# global.scaleOut.mode=keda, with the values files the operator includes in that case.
value_files AWS --set global.scaleOut.mode=keda
operator_sets AWS
for entry in "${CHARTS[@]}"; do
  chart="${entry%%:*}" ns="${entry#*:}"
  helm lint --strict "charts/all/${chart}" --namespace "${ns}" "${FARGS[@]}" "${SETS[@]}" \
    --set global.scaleOut.mode=keda
  helm template "${chart}" "charts/all/${chart}" --namespace "${ns}" "${FARGS[@]}" "${SETS[@]}" \
    --set global.scaleOut.mode=keda > "${RENDER_DIR}/charts/${chart}-keda.yaml"
done
helm template notebook charts/all/notebook --namespace vllm-notebook "${FARGS[@]}" "${SETS[@]}" \
  --set global.scaleOut.mode=keda --set global.compare.enabled=true > "${RENDER_DIR}/charts/notebook-compare-keda.yaml"
echo "rendered charts/*-keda.yaml with: ${FILES[*]}"

# ---- clustergroup chart (app of apps) ---------------------------------------------
render_clustergroup() {
  local name="$1" platform="$2"; shift 2
  value_files "${platform}" "$@"
  operator_sets "${platform}"
  printf '%s\n' "${FILES[@]}" > "${RENDER_DIR}/clustergroup/${name}.valuefiles"
  helm template "${PATTERN}-${VARIANT}" clustergroup \
    --repo https://charts.validatedpatterns.io --version "${CLUSTERGROUP_CHART_VERSION}" \
    "${FARGS[@]}" "${SETS[@]}" "$@" \
    > "${RENDER_DIR}/clustergroup/${name}.yaml"
  echo "rendered clustergroup/${name}.yaml with: ${FILES[*]}"
}
render_clustergroup default AWS
render_clustergroup none AWS --set global.gpuMachineSet=none
render_clustergroup gcp GCP
render_clustergroup keda AWS --set global.scaleOut.mode=keda

echo "render-charts.sh: OK, output in ${RENDER_DIR}"
