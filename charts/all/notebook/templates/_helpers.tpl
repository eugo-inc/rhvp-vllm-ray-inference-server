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
