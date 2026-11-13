{{/* Checks of the scale-out settings. Each rule fails the render with a message that
     names the key; the check itself renders nothing (templates/settings-check.yaml).
     scripts/pattern_settings.py holds the same rules for scripts/validate-values.sh;
     keep the two in step. */}}

{{/* True when the value is a whole number (YAML gives float64, --set gives int64). */}}
{{- define "ray-serve-llm.isWhole" -}}
{{- if and (or (kindIs "float64" .) (kindIs "int64" .) (kindIs "int" .)) (eq (float64 .) (floor (float64 .))) }}true{{ end }}
{{- end }}

{{/* True when the value is a number. */}}
{{- define "ray-serve-llm.isNumber" -}}
{{- if or (kindIs "float64" .) (kindIs "int64" .) (kindIs "int" .) }}true{{ end }}
{{- end }}

{{/* True when the compile caches live on a volume that outlives the pod (compileCache.mode node or
     shared): deployment.yaml then mounts PVC ray-serve-llm-compile-cache (compile-cache-pvc.yaml) at
     /cache and sets the CUDA driver's cache variables. */}}
{{- define "ray-serve-llm.compileCacheKept" -}}
{{- if has (toString (.Values.compileCache | default dict).mode) (list "node" "shared") }}true{{ end }}
{{- end }}

{{- define "ray-serve-llm.checkSettings" -}}
{{- $scale := .Values.global.scaleOut | default dict -}}
{{- $mode := "none" -}}
{{- if hasKey $scale "mode" }}{{ $mode = $scale.mode }}{{ end -}}
{{- if eq (toString $mode) "kuberay" -}}
{{- fail "global.scaleOut.mode: kuberay mode is designed but not built yet; use none or keda" -}}
{{- else if not (and (kindIs "string" $mode) (has $mode (list "none" "keda"))) -}}
{{- fail (printf "global.scaleOut.mode must be none or keda (kuberay is designed but not built yet); got %v" $mode) -}}
{{- end -}}
{{- $srv := .Values.server -}}
{{- if or (hasKey $srv "minReplicas") (hasKey $srv "maxReplicas") -}}
{{- fail "server.minReplicas and server.maxReplicas moved to server.autoscaling.minReplicas and server.autoscaling.maxReplicas" -}}
{{- end -}}
{{- $as := $srv.autoscaling | default dict -}}
{{- $min := $as.minReplicas -}}
{{- $max := $as.maxReplicas -}}
{{- if not (and (include "ray-serve-llm.isWhole" $min) (ge (float64 $min) 1.0)) -}}
{{- fail (printf "server.autoscaling.minReplicas must be a whole number of at least 1; got %v (%s)" $min (kindOf $min)) -}}
{{- end -}}
{{- if not (and (include "ray-serve-llm.isWhole" $max) (ge (float64 $max) (float64 $min))) -}}
{{- fail (printf "server.autoscaling.maxReplicas must be a whole number of at least minReplicas (%v); got %v (%s)" $min $max (kindOf $max)) -}}
{{- end -}}
{{- $target := $as.targetOngoingRequests -}}
{{- if and (not (kindIs "invalid" $target)) (not (and (include "ray-serve-llm.isNumber" $target) (gt (float64 $target) 0.0))) -}}
{{- fail (printf "server.autoscaling.targetOngoingRequests must be a number above 0, or null; got %v (%s)" $target (kindOf $target)) -}}
{{- end -}}
{{- if and (gt (float64 $max) 1.0) (kindIs "invalid" $target) -}}
{{- fail "server.autoscaling.targetOngoingRequests must be set when maxReplicas is above 1: Ray Serve LLM's default target is 10^9 ongoing requests per replica, so the replica count would never grow" -}}
{{- end -}}
{{- $gpr := $srv.gpuPerReplica -}}
{{- if not (and (include "ray-serve-llm.isNumber" $gpr) (gt (float64 $gpr) 0.0)) -}}
{{- fail (printf "server.gpuPerReplica must be a number above 0; got %v (%s)" $gpr (kindOf $gpr)) -}}
{{- end -}}
{{- /* The server's rules for gpu_per_replica against a replica's workers; gpuPerReplica 1 is not
       written to llm.yaml (configmap.yaml), so the server then uses the worker count. */ -}}
{{- $ek := $srv.engineKwargs | default dict -}}
{{- $tp := $ek.tensor_parallel_size -}}
{{- if not (include "ray-serve-llm.isNumber" $tp) -}}{{- $tp = 1 -}}{{- end -}}
{{- $pp := $ek.pipeline_parallel_size -}}
{{- if not (include "ray-serve-llm.isNumber" $pp) -}}{{- $pp = 1 -}}{{- end -}}
{{- $workers := mulf $tp $pp -}}
{{- if and (ne (float64 $gpr) 1.0) (gt $workers 1.0) (ne (float64 $gpr) $workers) -}}
{{- fail (printf "server.gpuPerReplica (%v) must be 1 or equal engineKwargs.tensor_parallel_size times pipeline_parallel_size (%v): each of a replica's workers needs a whole GPU" $gpr $workers) -}}
{{- end -}}
{{- if and (le $workers 1.0) (gt (float64 $gpr) 1.0) -}}
{{- fail (printf "server.gpuPerReplica (%v) must be at most 1 when a replica has one worker (engineKwargs.tensor_parallel_size and pipeline_parallel_size 1)" $gpr) -}}
{{- end -}}
{{- if gt (mulf $max $gpr) (addf $srv.gpuCount 0.000001) -}}
{{- fail (printf "server.autoscaling.maxReplicas (%v) times server.gpuPerReplica (%v) is more than server.gpuCount (%v)" $max $gpr $srv.gpuCount) -}}
{{- end -}}
{{- /* Replicas that share a GPU: vLLM refuses to start when the GPU's free memory is below
       gpu_memory_utilization times its total memory, and a starting replica needs a little
       memory of its own first, so each must claim at most 0.95 of its share (serve_app.py
       refuses the same at start; the comparison's plan.py SHARED_GPU_LIMIT is the same 0.95). */ -}}
{{- $gmu := $ek.gpu_memory_utilization -}}
{{- if and (lt (float64 $gpr) 1.0) (gt (float64 $max) 1.0) (include "ray-serve-llm.isNumber" $gmu) -}}
{{- if gt (float64 $gmu) (addf (mulf $gpr 0.95) 0.000000001) -}}
{{- fail (printf "server.engineKwargs.gpu_memory_utilization (%v) must be at most 0.95 times server.gpuPerReplica (%v) when up to %v replicas share a GPU: each vLLM replica claims that fraction of the whole GPU's memory when it starts, and needs a little more before it can, so at the full share a later replica would not fit" $gmu $gpr $max) -}}
{{- end -}}
{{- end -}}
{{- if not (and (include "ray-serve-llm.isWhole" $srv.metricsPort) (ge (float64 $srv.metricsPort) 1.0) (le (float64 $srv.metricsPort) 65535.0) (ne (float64 $srv.metricsPort) (float64 $srv.httpPort))) -}}
{{- fail (printf "server.metricsPort must be a port number (1 to 65535) other than server.httpPort; got %v (%s)" $srv.metricsPort (kindOf $srv.metricsPort)) -}}
{{- end -}}
{{- $cache := (.Values.compileCache | default dict).mode -}}
{{- if not (and (kindIs "string" $cache) (has $cache (list "none" "node" "shared"))) -}}
{{- fail (printf "compileCache.mode must be none, node or shared; got %v" $cache) -}}
{{- end -}}
{{- $access := .Values.modelStorage.accessMode -}}
{{- if not (and (kindIs "string" $access) (has $access (list "ReadWriteOnce" "ReadWriteMany"))) -}}
{{- fail (printf "modelStorage.accessMode must be ReadWriteOnce or ReadWriteMany; got %v" $access) -}}
{{- end -}}
{{- if eq $mode "keda" -}}
{{- $pods := .Values.scaleOut | default dict -}}
{{- if not (and (include "ray-serve-llm.isWhole" $pods.minPods) (ge (float64 $pods.minPods) 1.0)) -}}
{{- fail (printf "scaleOut.minPods must be a whole number of at least 1; got %v" $pods.minPods) -}}
{{- end -}}
{{- if not (and (include "ray-serve-llm.isWhole" $pods.maxPods) (ge (float64 $pods.maxPods) (float64 $pods.minPods))) -}}
{{- fail (printf "scaleOut.maxPods must be a whole number of at least scaleOut.minPods (%v); got %v" $pods.minPods $pods.maxPods) -}}
{{- end -}}
{{- end -}}
{{- end }}
