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
#   clustergroup/<case>.yaml     the clustergroup chart (the app of apps) for:
#                                  default    AWS, global.gpuMachineSet from values-global.yaml
#                                  none       AWS, global.gpuMachineSet=none
#                                  gcp        GCP, global.gpuMachineSet from values-global.yaml
#   clustergroup/<case>.valuefiles  the values files used for that case
#
# Any helm error (lint, template, or the clustergroup values schema) fails the script.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UTILITY_IMAGE="${UTILITY_IMAGE:-quay.io/validatedpatterns/utility-container@sha256:cdac62c3746f65d69f05b71fd99a347f7dbdf345776c56d9a6893adb1e9aa040}"
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
mkdir -p "${RENDER_DIR}/charts" "${RENDER_DIR}/clustergroup"
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
for dir in charts/all/*/; do
  chart="$(basename "${dir}")"
  ns="$(yq -r ".clusterGroup.applications[] | select(.path == \"charts/all/${chart}\") | .namespace" "${VARIANT_FILE}")"
  if [[ -z "${ns}" || "${ns}" == "null" ]]; then
    echo "render-charts.sh: charts/all/${chart} has no application in ${VARIANT_FILE}" >&2
    exit 1
  fi
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
echo "rendered charts/*-variant.yaml"

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

echo "render-charts.sh: OK, output in ${RENDER_DIR}"
