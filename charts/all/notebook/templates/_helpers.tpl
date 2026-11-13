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

{{/* The comparison's config file: global.compare without "enabled", plus global.model.
     "mode" is left out when it is pairs, the default, so a pairs run keeps the config,
     and so the run id, it had before the setting existed. */}}
{{- define "notebook.compareConfig" -}}
{{- $cfg := omit (.Values.global.compare | default dict) "enabled" -}}
{{- if hasKey $cfg "mode" -}}
{{- if not (and (kindIs "string" $cfg.mode) (has $cfg.mode (list "pairs" "scaleout"))) -}}
{{- fail (printf "global.compare.mode must be pairs or scaleout; got %v" $cfg.mode) -}}
{{- end -}}
{{- if eq $cfg.mode "pairs" }}{{ $cfg = omit $cfg "mode" }}{{ end -}}
{{- end -}}
{{- $_ := set $cfg "model" (pick .Values.global.model "id" "source" "revision") -}}
{{- toPrettyJson $cfg -}}
{{- end }}

{{/* A short hash of that config: a changed config is a new ConfigMap. */}}
{{- define "notebook.compareHash" -}}
{{- include "notebook.compareConfig" . | sha256sum | trunc 8 -}}
{{- end }}

{{/* A short hash of the Job's whole rendered pod template (compare-job.yaml), which
     takes in the config's hash too. A Job's pod template cannot be changed, so any
     change to it is a new Job. A new Job with an unchanged config finds its run
     complete and exits at once, or, while the first still runs, refuses to start. */}}
{{- define "notebook.compareJobHash" -}}
{{- include "notebook.comparePodTemplate" . | sha256sum | trunc 8 -}}
{{- end }}

{{/* The GuideLLM image the runner runs in: global.compare.load.image, or the pinned
     release (GUIDELLM_IMAGE in notebooks/rhvp_compare/plan.py; a test keeps them equal). */}}
{{- define "notebook.compareLoadImage" -}}
{{- ((.Values.global.compare | default dict).load | default dict).image | default "ghcr.io/vllm-project/guidellm:v0.8.1@sha256:015b22441020e6014b4c06d89882921eb29e9b9a6f2e274b253afef4029aa081" -}}
{{- end }}

{{/* The server's ScaledObject that the runner pauses for a run: compare.serverScaledObject while
     global.scaleOut.mode is keda and the runner manages the server (compare.serverDeployment); empty
     otherwise. */}}
{{- define "notebook.compareScaledObject" -}}
{{- if and (eq (toString (.Values.global.scaleOut | default dict).mode) "keda") .Values.compare.serverDeployment -}}
{{- .Values.compare.serverScaledObject -}}
{{- end -}}
{{- end }}

{{/* The energy source the config asks for (auto when not set). */}}
{{- define "notebook.compareEnergy" -}}
{{- ((.Values.global.compare | default dict).energy | default dict).source | default "auto" -}}
{{- end }}
