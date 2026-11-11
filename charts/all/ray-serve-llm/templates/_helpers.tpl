{{/* Common labels. */}}
{{- define "ray-serve-llm.labels" -}}
app.kubernetes.io/name: ray-serve-llm
app.kubernetes.io/part-of: vllm-ray-inference-server
{{- end }}

{{/* True when the image is built in the cluster (no image.override). */}}
{{- define "ray-serve-llm.inClusterBuild" -}}
{{- if not .Values.image.override }}true{{ end }}
{{- end }}

{{/* Image the server container runs. */}}
{{- define "ray-serve-llm.image" -}}
{{- if .Values.image.override -}}
{{ .Values.image.override }}
{{- else -}}
image-registry.openshift-image-registry.svc:5000/{{ .Release.Namespace }}/ray-serve-llm:latest
{{- end -}}
{{- end }}
