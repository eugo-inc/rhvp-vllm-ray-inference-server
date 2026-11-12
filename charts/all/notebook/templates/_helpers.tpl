{{/* Common labels. */}}
{{- define "notebook.labels" -}}
app.kubernetes.io/name: notebook
app.kubernetes.io/part-of: vllm-ray-inference-server
{{- end }}

{{/* True when the image is built in the cluster (no image.override). */}}
{{- define "notebook.inClusterBuild" -}}
{{- if not .Values.image.override }}true{{ end }}
{{- end }}

{{/* Image the notebook container runs. */}}
{{- define "notebook.image" -}}
{{- if .Values.image.override -}}
{{ .Values.image.override }}
{{- else -}}
image-registry.openshift-image-registry.svc:5000/{{ .Release.Namespace }}/notebook:latest
{{- end -}}
{{- end }}

{{/* True while the image comparison is on (global.compare.enabled). */}}
{{- define "notebook.compareEnabled" -}}
{{- if (.Values.global.compare | default dict).enabled }}true{{ end }}
{{- end }}

{{/* The comparison's config file: global.compare without "enabled", plus global.model. */}}
{{- define "notebook.compareConfig" -}}
{{- $cfg := omit (.Values.global.compare | default dict) "enabled" -}}
{{- $_ := set $cfg "model" (pick .Values.global.model "id" "source" "revision") -}}
{{- toPrettyJson $cfg -}}
{{- end }}

{{/* A short hash of that config: a changed config is a new Job. */}}
{{- define "notebook.compareHash" -}}
{{- include "notebook.compareConfig" . | sha256sum | trunc 8 -}}
{{- end }}

{{/* The GuideLLM image the runner runs in: global.compare.load.image, or the pinned
     release (GUIDELLM_IMAGE in notebooks/rhvp_compare/plan.py; a test keeps them equal). */}}
{{- define "notebook.compareLoadImage" -}}
{{- ((.Values.global.compare | default dict).load | default dict).image | default "ghcr.io/vllm-project/guidellm:v0.8.1@sha256:015b22441020e6014b4c06d89882921eb29e9b9a6f2e274b253afef4029aa081" -}}
{{- end }}

{{/* The energy source the config asks for (auto when not set). */}}
{{- define "notebook.compareEnergy" -}}
{{- ((.Values.global.compare | default dict).energy | default dict).source | default "auto" -}}
{{- end }}
