# AGENTS.md

Instructions for coding agents (and people) who change or deploy this repository. `CLAUDE.md` imports this file;
keep this file the only copy of these instructions.

## Read this first: the default install costs money

- On AWS, the default install creates **one billable GPU instance: `g6.2xlarge` with one NVIDIA L4**. An
  imperative job applies a MachineSet for it, and AWS bills the instance while it runs.
- To stop that, set `global.gpuMachineSet: none` in `values-global.yaml` **in your fork**, commit and push it,
  **before** you install. Install from a fork so this value can be changed first.
- Off AWS, nothing is created; bring your own GPU node.
- Uninstalling the pattern does **not** delete that MachineSet. Argo CD did not create it, so Argo CD does not
  remove it. See [Uninstall](#uninstall).

## What this pattern is

`vllm-ray-inference-server` is a community sandbox Validated Pattern by Eugo. It is not produced, supported or
validated by Red Hat. Red Hat OpenShift GitOps (Argo CD), driven by the Validated Patterns Operator, deploys it onto
one OpenShift cluster:

- **The inference server.** One pod runs a Ray head, Ray Serve and the vLLM engine through Ray Serve LLM
  (`ray.serve.llm.build_openai_app`). It serves an OpenAI-compatible API on port 8000. The cluster builds the image
  itself from `inference-server/`, using stock packages only: PyPI wheels on NVIDIA's CUDA UBI 9 image. There is no
  KubeRay and no OpenShift AI.
- **GPU enablement.** Node Feature Discovery and the NVIDIA GPU Operator. On AWS, an imperative job also creates
  one GPU node (see above).
- **Secrets.** HashiCorp Vault plus the External Secrets Operator carry an optional Hugging Face token.
- **A notebook.** A small JupyterLab behind OpenShift login. Its notebook times the server.
- **An image comparison**, off by default (`global.compare`). A Job compares two inference-server images on the
  server's GPU and takes the server down while it runs (README, "Compare two images"; here, "How the image
  comparison runs").
- **Scale out**, off by default: more vLLM replicas inside the server pod (`server.autoscaling`, Ray Serve's own
  autoscaler), more server pods (`global.scaleOut.mode: keda`, Red Hat's Custom Metrics Autoscaler), and the
  comparison's mode `scaleout` (README, "Scale out"; here, "How scale out works").

### Names that must stay in sync

The charts, `inference-server/serve_app.py`, the notebook and these docs all rely on these names. If you change
one, change it everywhere and run the tests.

| Thing | Name |
|---|---|
| Pattern (`global.pattern`) | `vllm-ray-inference-server`; also `name` in `pattern-metadata.yaml`, the `app.kubernetes.io/part-of` label in both charts, both Containerfiles' labels, and the secrets file name |
| GitHub repository | `https://github.com/eugo-inc/vllm-ray-inference-server`: `pattern-metadata.yaml`, the `global.repoURL` stubs in both charts' `values.yaml`, `scripts/render-charts.sh` and both Containerfiles' labels |
| Cluster group / variant | `prod` (`variants/prod/values-prod.yaml`) |
| Server namespace | `vllm-inference` |
| Notebook namespace | `vllm-notebook` |
| Server Deployment, Service, ImageStream, BuildConfig | `ray-serve-llm` (label `app.kubernetes.io/name=ray-serve-llm`, container `server`) |
| Server Service port | `8000`, port name `http` |
| Server config | ConfigMap `ray-serve-llm-config`, key `llm.yaml`, mounted at `/config/llm.yaml` |
| Model cache | PVC `ray-serve-llm-models`, mounted at `/models` |
| Hugging Face token | Vault `secret/data/hub/hfmodel`, property `hftoken` → ExternalSecret and Secret `huggingface-secret`, key `hftoken` |
| Server build | BuildConfig `contextDir: inference-server`, `dockerfilePath: Containerfile` |
| Notebook Deployment, Service, Route | `notebook` (label `app.kubernetes.io/name=notebook`, containers `notebook` and `oauth-proxy`); Secrets `notebook-tls` (serving certificate) and `notebook-proxy` (cookie secret) |
| Notebook build | BuildConfig and ImageStream `notebook`, `contextDir: notebooks`, `dockerfilePath: Containerfile` |
| Notebook ServiceAccount | `notebook`, with a Role and RoleBinding `notebook-read-pods` in `vllm-inference` that allow get/list/watch on pods |
| Rebuild permission | Role and RoleBinding `rebuild-on-commit` in both namespaces, for ServiceAccount `imperative/imperative-sa`; the namespaces and BuildConfig names are also listed in `ansible/rebuild-on-commit.yaml` |
| Image comparison runner | Job `compare-<runName>-<hash of the pod template>` in `vllm-notebook` (pod label `app.kubernetes.io/name=image-compare`), ServiceAccount `image-compare`, config ConfigMap `image-compare-config-<hash of the config>` (key `compare.json`) |
| Image comparison permissions | Role and RoleBinding `image-compare-sides` in `vllm-inference` (pods, pods/log, Secret `huggingface-secret`, `deployments/scale` of `ray-serve-llm`); Role and RoleBinding `image-compare-status` in `vllm-notebook` (ConfigMaps; list pods, for one run at a time); ClusterRole and ClusterRoleBinding `vllm-notebook-image-compare` (nodes, clusterversions); ClusterRoleBinding `vllm-notebook-image-compare-monitoring` (`cluster-monitoring-view`, unless energy is `none`); in keda mode also get and patch on ScaledObject `ray-serve-llm` (`scaledobjects.keda.sh`) in `image-compare-sides` |
| Image comparison results | PVC `image-compare-results` in `vllm-notebook`, mounted at `/results` in the runner and read-only in the notebook (`RHVP_COMPARE_RESULTS`); status ConfigMap `compare-<run id>` (labels `app.kubernetes.io/component=compare-run`, `compare-run-id=<run id>`) |
| Image comparison sides | Pods `cmp-<last 8 characters of the run id>-r<NN>-<a or b>` in `vllm-inference`, labels `app.kubernetes.io/name=image-compare-side` and `rhvp-compare/run=<run id>` |
| DCGM metrics | namespace `nvidia-gpu-operator` labelled `openshift.io/cluster-monitoring: "true"` (`variants/prod/values-prod.yaml`) |
| Server metrics | Ray's Prometheus metrics on port `8080` (`server.metricsPort`, `metrics_port` in `llm.yaml`, `ray.init(_metrics_export_port=...)`) in every install; container port and Service port named `metrics` only in keda mode; vLLM's waiting-request gauge is `ray_vllm_num_requests_waiting` |
| Replica count log line | one JSON line `{"event":"rhvp-scale",...}` per change (`inference-server/rhvp_scale.py` defines it; `notebooks/rhvp_compare/logs.py` keeps its own copy of the names); deployments as Ray Serve LLM names them, `LLMServer:<model id, "/" as "--" and "." as "_">` and `OpenAiIngress` |
| Compile cache volume | PVC `ray-serve-llm-compile-cache` at `/cache` while `compileCache.mode` is `node` (ReadWriteOnce) or `shared` (ReadWriteMany); `none` keeps the emptyDir |
| Autoscaler install (keda mode) | namespace `openshift-keda` (OperatorGroup for all namespaces, labelled `openshift.io/cluster-monitoring: "true"`), Subscription key `keda` for package `openshift-custom-metrics-autoscaler-operator` (channel `stable`), app `keda-config` (`charts/all/keda-config`) with KedaController `keda`; all in `overrides/values-scaleout-keda.yaml`, which `sharedValueFiles` includes as `/overrides/values-scaleout-<global.scaleOut.mode>.yaml` |
| Autoscaler objects (keda mode) | in `vllm-inference`: ScaledObject `ray-serve-llm` (on Deployment `ray-serve-llm`), TriggerAuthentication `keda-trigger-auth-prometheus`, ServiceAccount `thanos`, Roles and RoleBindings `thanos-metrics-reader` and `keda-operator-token-creator` (for ServiceAccount `openshift-keda/keda-operator`), ServiceMonitor `ray-serve-llm` (port `metrics`) |
| Autoscaler pause | annotation `autoscaling.keda.sh/paused-replicas: "0"` on ScaledObject `ray-serve-llm`, set by the comparison's runner (env `RHVP_COMPARE_SERVER_SCALED_OBJECT`, from `compare.serverScaledObject` in the notebook chart) and ignored by Argo CD (`overrides/values-scaleout-keda.yaml`) |
| GPU time-slicing | ConfigMap `time-slicing-config` (key `any`) in `nvidia-gpu-operator`, named by the ClusterPolicy's `devicePlugin.config` only while `timeSlicing.enabled` is true |
| Comparison with docker, scale-out mode | second replica container `<first replica's name>-2`; warm compile-cache volumes `<CACHE_VOLUME>-<last 8 characters of the run id>-<a or b>`; per round `server-2.log` (artifact `server_log_2`) and `router.log`; env `RHVP_COMPARE_REPLICA_GPUS`, `RHVP_COMPARE_CPUSET`, `RHVP_COMPARE_CACHE_VOLUME` (from `REPLICA_GPUS`, `CPUSET`, `CACHE_VOLUME` in `scripts/compare-local.sh`); helper containers `rhvp-cmp-<last 8 characters of the run id>-gpu-probe-<1 or 2>` (nvidia-smi, before the first round); launch-record fields `gpus` and `gpu_uuids` |

## Repo map

```
AGENTS.md  CLAUDE.md (@AGENTS.md)  README.md  LICENSE (Apache-2.0)  LICENSES/BSD-3-Clause.txt
pattern-metadata.yaml  .gitignore
values-global.yaml            pattern name, global.gpuMachineSet, global.model.*, main.variant
values-secret.yaml.template   secrets file format: hfmodel / hftoken, value "none"
variants/prod/values-prod.yaml   the cluster group: namespaces, operator subscriptions, Argo CD apps, imperative jobs
overrides/values-gpu-machineset-create-AWS.yaml   create-gpu-machineset imperative job and its RBAC
overrides/values-scaleout-keda.yaml   keda mode: the Custom Metrics Autoscaler's namespace, Subscription and app,
                                      and the ray-serve-llm app's ignore rules restated with the pause annotation
ansible/rebuild-on-commit.yaml   imperative job: start a build when that build's source folder changed
ansible/rebuild_on_commit.py     the rebuild decision (which BuildConfigs to start), tested in tests/test_charts.py
charts/all/nfd-config/         NodeFeatureDiscovery instance
charts/all/nvidia-gpu-config/  NVIDIA ClusterPolicy (tolerates the nvidia.com/gpu taint); time-slicing-config.yaml
                               (opt-in GPU time-slicing)
charts/all/ray-serve-llm/      BuildConfig, ImageStream, Deployment, Service, Route (off by default), PVC,
                               ConfigMap, ExternalSecret; compile-cache-pvc.yaml (compileCache node or shared);
                               scaledobject.yaml, triggerauthentication.yaml, servicemonitor.yaml (keda mode
                               only); _settings.tpl checks the scale-out settings (settings-check.yaml renders
                               nothing, it only fails a bad render)
charts/all/keda-config/        KedaController keda (an app only in keda mode)
charts/all/notebook/           BuildConfig, ImageStream, Deployment (JupyterLab + oauth-proxy), Service, Route,
                               ServiceAccount and RBAC; compare-job.yaml, compare-config.yaml (ConfigMap, PVC)
                               and compare-rbac.yaml, rendered only while global.compare.enabled is true
inference-server/              Containerfile, requirements.in, requirements-pip.lock (pip itself),
                               requirements-x86_64.lock, requirements-aarch64.lock, serve_app.py, entrypoint.sh,
                               rhvp_status.py, rhvp_scale.py (the rhvp-scale log line's shape)
notebooks/                     Containerfile, entrypoint.sh, .dockerignore, requirements.in, requirements.lock,
                               make_notebook.py, rhvp_bench/ (words, draw, showing, measure, replay, prompts,
                               run), prompts.jsonl, recordings/synthetic.json,
                               inference_timings.ipynb (generated)
notebooks/rhvp_compare/        the image comparison: plan, runner, probe, guidellm, logs, energy, stats, analyze,
                               report, summary, kube (OpenShift backend), dockerapi (docker backend), example,
                               words, show, cells, launch (what each server start ran with), router (the
                               scale-out mode's request router, docker only); results.schema.json.
                               compare_images.ipynb and
                               compare-recordings/synthetic.json are generated by make_notebook.py
scripts/                       render-charts.sh, validate-values.sh (with pattern_settings.py, the scale-out
                               rules), local-smoke.sh, compare-local.sh
tests/                         test_serve_config.py, test_secrets_template.py, test_charts.py,
                               test_server_chart.py, test_scaleout_contract.py (conftest.py and
                               chart_helpers.py share one chart render),
                               test_notebook_rules.py, test_measure.py, test_compare_stats.py,
                               test_compare_parsers.py, test_compare_runner.py, test_compare_notebook.py,
                               test_compare_router.py, fixtures/ (trimmed GuideLLM reports, one with latency
                               goals; the mock configs for compare-local.sh, pairs and scaleout)
.github/workflows/             lint.yml, jsonschema.yaml, notebook-check.yml
pattern.sh  Makefile  Makefile-common  ansible.cfg  .claude/skills/pattern-author/  .cursor/skills/pattern-author/
```

Patternizer manages `pattern.sh`, `Makefile-common`, `ansible.cfg` and both `pattern-author` skill folders. Do not
edit them by hand; `patternizer upgrade` rewrites them. You may add targets to `Makefile`.

### How the server starts

1. `entrypoint.sh` replaces the base image's NVIDIA entrypoint. It unsets `HF_TOKEN` unless the value starts
   with `hf_`, then starts `serve_app.py`.
2. `serve_app.py` reads `/config/llm.yaml` (the path in `RHVP_CONFIG`) and then the environment overrides
   `RHVP_MODEL_ID`, `RHVP_MODEL_SOURCE`, `RHVP_MODEL_REVISION`, `RHVP_GPU_MEMORY_UTILIZATION` and
   `RHVP_MAX_MODEL_LEN`.
3. It starts Ray with a 2 GiB object store and its Prometheus metrics on port 8080 (`metrics_port`), checks that
   Ray sees enough GPUs for `max_replicas` replicas (exit 2 when it does not and `max_replicas` is above 1), starts
   Serve on `0.0.0.0:8000`, and runs the Ray Serve LLM app `llm`. `serve.run` returns once the app is RUNNING,
   which means the model is loaded.
4. It then writes the ready file `/tmp/r/rhvp-ready` and waits for SIGTERM. Until then it checks the app every 10
   seconds (every second while `max_replicas` is above `min_replicas`): it removes the ready file when the app stops
   being RUNNING, and logs one `rhvp-scale` line for each replica count that changed. Adding or removing a replica
   leaves the app RUNNING, so the ready file stays.
5. The startup and readiness probes check that ready file every 2 seconds. The kubelet runs the readiness probe
   as soon as the startup probe passes, so the pod turns Ready up to about 2 seconds after the file appears. That
   is the resolution of the notebook's start-to-Ready figure, which Kubernetes also records in whole seconds. The
   liveness probe calls `/-/healthz` every 30 seconds.

`llm.yaml` (the chart renders it from `values-global.yaml`):

```yaml
app_name: llm
model_id: qwen2.5-0.5b-instruct          # global.model.id; the name clients send as "model"
model_source: Qwen/Qwen2.5-0.5B-Instruct # global.model.source
model_revision: <global.model.revision, or empty>
engine_kwargs: {max_model_len: 8192, gpu_memory_utilization: 0.85, tensor_parallel_size: 1}
accelerator_type: null
min_replicas: 1
max_replicas: 1
object_store_memory_gb: 2
http_host: 0.0.0.0
http_port: 8000
# Written only when they differ from these defaults (so a default install renders the llm.yaml above):
# target_ongoing_requests: null  # server.autoscaling.targetOngoingRequests; required when max_replicas > 1
# gpu_per_replica: null          # server.gpuPerReplica; null: one GPU per tensor- and pipeline-parallel worker
# metrics_port: 8080             # server.metricsPort
```

`serve_app.py` refuses a config it cannot run, with a message that names the key: among others a `max_replicas`
above 1 without `target_ongoing_requests` (Ray Serve LLM's own target is 10^9), a `gpu_per_replica` that is not the
worker count of a tensor-parallel replica (or above 1 for a replica with one worker), and, for replicas that share a
GPU, a `gpu_memory_utilization` above 0.95 times `gpu_per_replica` (`SHARED_GPU_MARGIN`, the same 0.95 in the chart,
`scripts/pattern_settings.py` and the comparison's `plan.SHARED_GPU_LIMIT`). The chart and
`scripts/pattern_settings.py` refuse the same settings before a sync.

Server container environment:

- `RHVP_CONFIG=/config/llm.yaml`
- `HF_HOME=/models/hf`: model weights on the PVC, kept across restarts.
- Compile caches under `/cache`: `TRITON_CACHE_DIR=/cache/triton`, `VLLM_CACHE_ROOT=/cache/vllm`,
  `FLASHINFER_WORKSPACE_BASE=/cache/flashinfer`, `XDG_CACHE_HOME=/cache/xdg`. With `compileCache.mode: none` (the
  default) `/cache` is an emptyDir, emptied on every new pod. With `node` or `shared` it is PVC
  `ray-serve-llm-compile-cache`, kept for later pods; the pod then also gets `CUDA_CACHE_PATH=/cache/cuda`,
  `CUDA_CACHE_MAXSIZE=4294967296` (4 GiB) and `fsGroupChangePolicy: OnRootMismatch`. There is no hostPath option:
  the restricted-v2 SCC the server runs under does not allow hostPath volumes.
- `RAY_TMPDIR=/tmp/r` on its own emptyDir. Keep it short: Ray puts Unix sockets under it, and socket paths are
  limited to 108 bytes.
- `/dev/shm`: an emptyDir with `medium: Memory`, 4Gi.
- `HF_TOKEN`: from `secretKeyRef` `huggingface-secret` / `hftoken`, `optional: true`.

Notebook environment (set by the chart, read by `rhvp_bench`):

- `RHVP_ENDPOINT=http://ray-serve-llm.vllm-inference.svc:8000`
- `RHVP_MODEL_ID=<global.model.id>`
- `RHVP_MODEL_REVISION=<global.model.revision>`, set only when the revision is not empty.
- `RHVP_SERVER_NAMESPACE=vllm-inference`
- `RHVP_SERVER_LABEL_SELECTOR=app.kubernetes.io/name=ray-serve-llm`
- Optional `RHVP_INTERNAL=1` adds an amber INTERNAL strip. It is off by default.
- Optional `RHVP_REPLAY=<path to a recorded run>` turns on REPLAY mode.

The notebook image bakes in the `notebooks/` folder at `/opt/app-root/notebooks`. Nothing is cloned at runtime.
When the pod starts, it copies that folder into a writable emptyDir at `/work`. It then runs JupyterLab on
`127.0.0.1:8888` with `root_dir` `/work`, behind an OpenShift oauth-proxy sidecar. The sidecar listens on port 8443
and uses a serving certificate; the Route uses `reencrypt` termination.

### How the image comparison runs

- Off by default. `global.compare` in `values-global.yaml` is the config; the notebook chart renders it, plus
  `global.model`, into ConfigMap `image-compare-config-<hash>`, and renders the Job and its RBAC.
- The Job's pod runs on the pinned GuideLLM image (`ghcr.io/vllm-project/guidellm:v0.8.1@sha256:015b...`); an init
  container copies `notebooks/rhvp_compare` into it from the notebook image. It runs on the notebook pod's node,
  because the results PVC is ReadWriteOnce and the notebook mounts it too.
- The runner (`python -m rhvp_compare run --backend kube`) first lists the runner pods in `vllm-notebook`
  (`app.kubernetes.io/name=image-compare`) and refuses to start (exit 5, `RUN_ACTIVE`, nothing changed) while an
  older one is Pending or Running. It then scales Deployment `ray-serve-llm` to 0, and for each round creates a side
  pod in `vllm-inference` (every one after the first pinned to the first one's node), times its start, runs GuideLLM
  as its own child process, reads GPU power from the Thanos Querier, deletes the pod and saves `results.json`. A
  side whose server exits at vLLM's start-up memory check (`Error in memory profiling` in its log) is started once
  more in the same round (`runner.START_RETRIES`, the round's `start_retries`, and the launch record's own
  `start_retries`); in mode `scaleout` the second replica gets the same one retry, timed from its first ask. When
  the run ends it scales the server back to `compare.serverReplicas` (1), tries again for about a minute, reads the
  scale back, and records `run.server_restored`; a server that did not come back is exit 4.
- The ray-serve-llm app ignores the Deployment's `/spec/replicas` on every install, also with the comparison off: a
  sync never changes the server's replica count, so a server at 0 stays at 0 until someone scales it.
- SIGTERM (a deleted Job or `activeDeadlineSeconds`) cuts the current round short and asks the server back at once,
  before the side is taken down (its pod waits for the GPU); every API call then gets a 10 s timeout. The pod's
  `terminationGracePeriodSeconds` (300) is above the worst case of that path (`kube.stop_budget_seconds()`, checked
  by `tests/test_charts.py`).
- A Job's pod template cannot change, so the Job's name carries a hash of the whole rendered pod template
  (`notebook.comparePodTemplate`, hashed by `notebook.compareJobHash`). The run id carries a hash of the config
  only, so a new Job with the same config finds its run complete and exits at once, or refuses while the first
  still runs.
- The verdict rule (`stats.verdict`, then `analyze.py`): TIE needs the median ratio and its whole interval inside
  the tie band; FASTER or SLOWER needs the corrected p-value at most alpha and the test leaning the same way as the
  median; a metric that lost a pair to one side failing on its own account (a value of 0, which gives no ratio,
  counts as such a loss), and every metric of a run whose rounds used more than one node, is INCONCLUSIVE.
- `notebooks/rhvp_compare/analyze.py` is the boundary to the statistics: it reads `plan` and `rounds` of the results
  file and writes `analysis`. Another tool that writes the same `analysis` shape can replace `analyze.py` and
  `stats.py`.
- Every sentence the comparison notebook shows is in `rhvp_compare/words.py`.

Checked on a plain Kubernetes cluster (kind) with GuideLLM mock servers: the Job and its RBAC, the server's scale
to 0 and back, the status ConfigMap, the termination message, and the stop path (Job deleted mid-round). That run
came before the refusal of a second run, the scale read-back and the early restore on a stop, which on Kubernetes
are covered by unit tests only (under docker, the refusal and the stop path have run against the mock server).
Checked with real servers, under docker on one GPU (`scripts/compare-local.sh`): two 2-pair smoke runs of the
example's two vLLM images with smaller settings. The second finished all 6 rounds, measured by GuideLLM, with tokens
per joule from nvidia-smi, one of them after a second start past vLLM's start-up memory check (README, "On one
machine"); the first, from before that second start, lost a round to that check.
Not checked yet: the Kubernetes backend with a real server; tokens per joule from DCGM (its reader is tested only
against hand-written answers, and the DCGM label names it matches on have not been checked against a real DCGM
exporter); a run on OpenShift, with or without a GPU (restricted SCC, internal registry, Argo CD health of the
Job); the example's own settings; and the length of a run (an estimate of 2 to 3 hours with the defaults).

### How scale out works

Three parts, each off by default. With every default, the local charts render exactly as before they existed
(`scripts/render-charts.sh`; the tests compare the renders).

- **Inside the pod** (`server.autoscaling`, `server.gpuPerReplica`): Ray Serve's own autoscaler, in the locked Ray
  release, adds and removes vLLM replicas in the one server pod. `serve_app.py` passes `target_ongoing_requests`
  into `autoscaling_config` and, when replicas share a GPU, a placement bundle of that share per worker
  (`placement_group_config`) and the same share as `VLLM_RAY_PER_WORKER_GPUS` in `LLMConfig.runtime_env`. vLLM
  0.29.0 reads that variable in its engine process, a child of the Serve replica, to size each worker's GPU request
  (1.0 when unset); Ray Serve LLM 2.60.0 sets it only for the workers, and LLMConfig.runtime_env is what reaches the
  replica (`LLMServer.get_deployment_options`). It does not depend on `global.scaleOut.mode`.
- **Across pods** (`global.scaleOut.mode: keda`): `variants/prod/values-prod.yaml` adds
  `/overrides/values-scaleout-<mode>.yaml` to `sharedValueFiles`; only the keda file exists. It installs Red Hat's
  Custom Metrics Autoscaler (operator, KedaController through the `keda-config` app), and the server chart renders
  the ScaledObject, TriggerAuthentication with its ServiceAccount and Roles, the ServiceMonitor and the `metrics`
  ports (see the names table). The trigger sums `ray_vllm_num_requests_waiting` over the server pods in
  `vllm-inference`, read from the Thanos Querier on port 9092 (namespace-scoped), and the HPA that KEDA creates
  divides it by the pod count. Between `minPods` and `maxPods` the HPA reads the metric on its own 15-second sync
  period: `pollingIntervalSeconds` only sets how often KEDA refreshes the ScaledObject's status, so a pod is added
  at the earliest about 15 seconds after requests start to wait. `cooldownSeconds` is the HPA's scale-down
  stabilization window. Prerequisites: monitoring for user-defined projects enabled on the cluster, and a second
  GPU (another GPU node, or `timeSlicing.enabled` in `charts/all/nvidia-gpu-config/values.yaml`, with
  `gpu_memory_utilization` lowered to leave room for the other pod; nothing checks that). Server pods on more than
  one node need `modelStorage.accessMode: ReadWriteMany` and `compileCache.mode: shared`; a PVC's access mode
  cannot change after it is created.
- **The comparison's mode `scaleout`** (`global.compare.mode`): docker only for now (`kube.py` refuses it with
  `PLAN_INVALID`). `scaleOut.sharedGpu` only sets the memory default and the 0.475 refusal; the docker backend places
  the replicas from `RHVP_COMPARE_REPLICA_GPUS` (else `RHVP_COMPARE_GPUS`) and refuses, as `PLAN_INVALID` before
  anything changes, a plan whose `sharedGpu` does not match (`DockerBackend.check_plan`); `prepare()` reads each
  replica's GPUs with nvidia-smi into the launch records (`gpus`, `gpu_uuids`). `scaleOut.requestRate`'s default (4)
  is a placeholder: the README's "Picking the surge rate" derives a rate from a pairs run of the same sides, and
  `summary.surge_rate_fit` makes `results.md` and the notebook say when every scored surge of both sides met the goals
  or failed requests. In keda mode, the OpenShift runner pauses the ScaledObject before it scales the server to 0
  (annotation `autoscaling.keda.sh/paused-replicas: "0"`; its Role allows get and patch on that one ScaledObject),
  and removes the pause when it brings the server back; the by-hand restore command it prints includes the
  `oc annotate` step. `kube.stop_budget_seconds()` is 290 s, under the runner pod's 300 s grace period.
- **`kuberay`** is a third value of `global.scaleOut.mode`, designed but not built. `scripts/pattern_settings.py`
  (through `scripts/validate-values.sh`) and `charts/all/ray-serve-llm/templates/_settings.tpl` refuse it, and
  any unknown mode, with the same message; keep the two in step (`tests/test_scaleout_contract.py`).

Switching keda mode off removes nothing: the apps sync without pruning, so the ScaledObject keeps scaling the
server until someone deletes it (README, "Scale out", has the commands), and the operator, `openshift-keda` and the
`keda-config` application stay.

The Subscription follows `stable` with automatic updates. Upstream KEDA 2.21 refuses a bound ServiceAccount token,
the kind `keda-trigger-auth-prometheus` uses, unless the operator maps that ServiceAccount to a token audience
(KEDA's 2.21 migration notes); the ScaledObject then never turns Ready and the server stops scaling while the
operator looks healthy. Red Hat's current operator is on the 2.19 line, which needs no mapping. If a Red Hat
release adds the rule, map the audience in the KedaController or move the TriggerAuthentication to a long-lived
token Secret.

Checked: unit tests and chart renders for all three parts. On CPU, in the server image, a stand-in Ray Serve app
with replicas that take 20 seconds to start scaled from 1 to 2 under `serve_app.serve_and_watch`: the `rhvp-scale`
lines showed the target and then the running count, the app stayed RUNNING, the ready file stayed, no request
failed, and `rhvp_status.py` printed the counts. On kind with upstream KEDA 2.19 (upstream 2.21 refused the bound
token, as above), the chart's own keda objects with stand-ins for the Thanos Querier and the server: 1 to 2 pods on
a waiting count above the threshold, held at 0 by the pause for a minute, back to 1 pod when the pause was removed;
every query carried the bearer token, the namespace and the rendered query. Through a Service, a new pod got none
of the requests of a client that kept 8 connections open; through `rhvp_compare/router.py`, it got a share. The
comparison's mode ran end to end with docker against GuideLLM mock servers
(`tests/fixtures/compare-scaleout-mock.json`): schema-valid results, the second replica took part of every surge,
and the notebook rendered.

Not checked yet:

- No GPU scale event: no real vLLM replica or server pod has been added by either autoscaler. The GPU check at
  start and the deployment names of a real Ray Serve LLM app have not run.
- No replica on a share of a GPU (`gpu_per_replica` below 1) has started on a real GPU. The one try (one GPU,
  `gpu_per_replica: 0.5`) failed at the first replica's engine start with `Cannot schedule RayWorkerProc with the
  placement group ... {'CPU': 0, 'GPU': 1.0} cannot fit into ... [{'CPU': 1.0, 'GPU': 0.5}]`: the engine process did
  not see `VLLM_RAY_PER_WORKER_GPUS`. The fix (the variable in `LLMConfig.runtime_env`) was checked on CPU only: Ray
  2.60.0's Ray Serve LLM builder with a stand-in server class and a stand-in for vLLM's import surface, on a Ray node
  with one logical GPU; the replica's spawned child asked for one worker in the replica's placement group as
  `ray_executor_v2` does, and got the same error without the variable and a placed worker with it.
- Nothing of keda mode on OpenShift: the operator install (package and channel are from Red Hat's docs and catalog,
  not from a cluster), the KedaController that the operator and Argo CD both manage, the operator's trust in the
  Thanos Querier's certificate, the ServiceMonitor (kind has no CRD for it), the labels on the real metric, the
  runner's own Role (the kind check paused as cluster admin), the Argo CD rule that keeps the pause annotation,
  Red Hat's operator build, and GPU time-slicing.
- No server pod has been seen exporting `ray_vllm_num_requests_waiting` under that name; KEDA reads an empty answer
  as 0, so a wrong name means the server never scales, without an error.
- `compileCache.mode` `node` and `shared`, and `fsGroupChangePolicy: OnRootMismatch`, on OpenShift.
- A surge in which some requests miss the latency goals, end to end (unit tests and a trimmed report only); the
  OpenShift scale round of the comparison, which is not built; the launch records of OpenShift sides carry no
  server environment (`env_source: "none"`).
- An upscale replica that keeps failing: whether Ray then marks the app UNHEALTHY, which would remove the ready file
  and take the one serving replica out of the Service.

## How to deploy

The pattern deploys through Red Hat OpenShift GitOps and the Validated Patterns Operator. Argo CD deploys what is
in **your fork on GitHub**, not the files on your disk: commit and push every change before you install or expect
it to sync.

### Prerequisites

- OpenShift 4.18 or newer, and cluster-admin access.
- A cluster that does not already run Node Feature Discovery or the NVIDIA GPU Operator. The pattern installs both
  itself, with its own OperatorGroups and settings, and can clash with existing installs of them. It has not been
  tried on a cluster that runs OpenShift AI.
- A StorageClass that provisions volumes dynamically, for the model PVC.
- The internal image registry set to `Managed`, and the `Build` and `ImageRegistry` cluster capabilities enabled.
  The cluster builds both images itself. To check:
  ```sh
  oc get configs.imageregistry.operator.openshift.io cluster -o jsonpath='{.spec.managementState}'
  oc get clusterversion version -o jsonpath='{.status.capabilities.enabledCapabilities}'
  ```
  On bare metal and vSphere the registry starts as `Removed`; see [Troubleshooting](#troubleshooting).
- A GPU. On AWS with `global.gpuMachineSet: create`, the pattern creates the node. Anywhere else, add a node with
  an NVIDIA GPU that the NVIDIA GPU Operator supports. One AWS exception: on a cluster installed into existing
  subnets (`platform.aws.subnets` or `vpc.subnets` in its install-config), the worker MachineSets name their subnet
  by ID, and the `create_machineset` job's template reads subnet filters, so it fails on every run. Set
  `gpuMachineSet: none` there and add the GPU MachineSet yourself. The server requests `nvidia.com/gpu: 1` and tolerates the
  `nvidia.com/gpu` taint.
- The CPU type of that GPU node. Both Deployments and both BuildConfigs select nodes with
  `kubernetes.io/arch: amd64`, from the `arch` value in `charts/all/ray-serve-llm/values.yaml` and
  `charts/all/notebook/values.yaml`. If your GPU node is arm64, set `arch: arm64` in
  `charts/all/ray-serve-llm/values.yaml` in your fork before you install; otherwise the server pod and its build
  never get a node. Leave the notebook chart on the CPU type of your ordinary workers: it needs no GPU and does not
  tolerate the GPU taint. The server's build also needs a node of that CPU type without the `nvidia.com/gpu`
  taint: a BuildConfig cannot carry tolerations (only a cluster-wide build override can add them), so a lone,
  tainted arm64 GPU node cannot run it.
- On your workstation:
  - podman 4.3 or newer. `pattern.sh` runs everything in a podman container; docker alone is not enough.
  - git.
  - `oc`, logged in to the cluster.
  - If `KUBECONFIG` is set, it must point inside `$HOME`, because only `$HOME` is mounted into the container.
- A fork of this repository, cloned locally, with your branch pushed and an upstream set. The installer reads the
  repo URL from the branch's upstream remote, and fails if the branch has no upstream.

### Path 1: command line (recommended)

1. Fork `https://github.com/eugo-inc/vllm-ray-inference-server` on GitHub, then clone your fork:
   ```sh
   git clone https://github.com/<you>/vllm-ray-inference-server.git
   cd vllm-ray-inference-server
   ```
2. Edit `values-global.yaml`:
   - `global.gpuMachineSet`: `create` (the default: one AWS g6.2xlarge with one L4) or `none`.
   - `global.model.id`: the name clients send as `model`.
   - `global.model.source`: the Hugging Face repo id.
   - `global.model.revision`: the Hugging Face commit to pin.

   Then commit and push:
   ```sh
   git commit -am "Configure my install"
   git push            # for a new branch: git push -u origin <branch>
   ```
3. Secrets. These are optional: the default model is not gated, and the installer falls back to the repo's
   template (token `none`). For a gated model, or to avoid picking up a leftover `~/values-secret.yaml` from
   another pattern, create the pattern's own secrets file:
   ```sh
   mkdir -p ~/.config/validated-patterns
   cp values-secret.yaml.template \
      ~/.config/validated-patterns/values-secret-vllm-ray-inference-server.yaml
   ```
   Put your `hf_...` token in the `hftoken` field's `value`. Never commit a filled-in secrets file;
   `.gitignore` ignores `values-secret*` except the template.
4. Log in and install:
   ```sh
   oc login --token=<token> --server=<api-url>
   ./pattern.sh make install
   ```
5. What `make install` does, step by step:
   1. `pattern.sh` starts `quay.io/validatedpatterns/utility-container` with podman, mounts the repo and
      `$HOME`, and runs `make` inside it.
   2. `make install` runs the Ansible playbook `rhvp.cluster_utils.install`. In order, it loads bootstrap
      secrets, deploys the operator, loads secrets, and then waits for every Argo CD application to be Synced and
      Healthy. The wait is at most 60 tries, 60 seconds apart.
   3. The operator step checks prerequisites first: cluster access, a StorageClass, the OpenShift version, and the
      name-length limit. It then renders the `pattern-install` Helm chart
      (`oci://quay.io/validatedpatterns/pattern-install`) with `values-global.yaml` plus your branch's upstream
      URL and branch name, and applies the result.
   4. That chart subscribes the cluster to the **Validated Patterns Operator** (`patterns-operator`, from
      `community-operators`). It also creates the **Pattern** custom resource: variant `prod`, Git target = your
      fork and branch.
   5. The operator installs **Red Hat OpenShift GitOps** and creates the Argo CD instance **`vp-gitops`** (in
      namespace `vp-gitops`). It then creates the app-of-apps: the **clustergroup** Helm chart (version `0.9.*`
      from charts.validatedpatterns.io), with `values-global.yaml` and `variants/prod/values-prod.yaml` from your
      fork.
   6. The clustergroup chart creates:
      - the namespaces and OperatorGroups;
      - the operator Subscriptions: External Secrets, NFD and the NVIDIA GPU Operator;
      - one Argo CD Application per app: `vault`, `openshift-external-secrets`, `nfd-config`,
        `nvidia-gpu-config`, `ray-serve-llm` and `notebook`;
      - the imperative CronJob, which runs two jobs: `create-gpu-machineset` (the playbook
        `rhvp.cluster_utils.create_machineset`) on AWS when `gpuMachineSet: create`, and `rebuild-on-commit`.
   7. The apps then come up. The BuildConfigs build the server and notebook images inside the cluster. The server
      pod waits for a GPU node, downloads the model to its PVC, and turns Ready once the model is loaded.
6. `./pattern.sh make load-secrets` loads the secrets file again. Run it after you edit that file;
   `make install` already loaded it once. The secret travels from Vault (`secret/data/hub/hfmodel`, property
   `hftoken`) through ExternalSecret `huggingface-secret` to the Secret `huggingface-secret`. The server reads
   `HF_TOKEN` only when its container starts, so delete the pod to pick up a new token:
   `oc delete pod -l app.kubernetes.io/name=ray-serve-llm -n vllm-inference`.
7. `./pattern.sh make argo-healthcheck` checks again that every Argo CD application is Synced and Healthy. Use
   it after an install that timed out.

Secrets file search order. The first file found wins:

1. `$VALUES_SECRET`
2. `~/.config/hybrid-cloud-patterns/values-secret-vllm-ray-inference-server.yaml`
3. `~/.config/validated-patterns/values-secret-vllm-ray-inference-server.yaml`
4. `~/values-secret-vllm-ray-inference-server.yaml`
5. `~/values-secret.yaml`
6. the repo's `values-secret.yaml.template`

The template's `hftoken` value must stay a non-empty string (`"none"`). With `null`, the secrets loader aborts.

### Path 2: OpenShift web console

1. Go to **OperatorHub**, search for "Validated Patterns" and install the **Validated Patterns Operator**. It is a
   community operator, so accept the community-operator notice.
2. Go to **Installed Operators → Validated Patterns Operator → Pattern → Create instance**.
3. Fill in the form:
   - Name: `vllm-ray-inference-server`
   - **Cluster Group Name**: `prod`
   - Git Config → **Target Repo**: your fork's URL
   - Target Revision: your branch
4. Click **Create**.
5. Load secrets afterwards. The console path loads none. Do this even for the default, ungated model: until
   `secret/data/hub/hfmodel` exists, ExternalSecret `huggingface-secret` is not Ready and Argo CD shows the
   `ray-serve-llm` app as Degraded. The server pod still starts, because the token is optional. Two ways:
   - From a clone of your fork, run `./pattern.sh make load-secrets`.
   - In the Vault UI, in the KV engine `secret`, create path `hub/hfmodel` with key `hftoken` (value `none`, or
     your token). The Vault root token is in Secret `vaultkeys` in namespace `imperative`.

### Path 3: private forks

Argo CD needs credentials to read a private fork:

1. Create an Argo CD repository Secret. It carries the label `argocd.argoproj.io/secret-type: repository` and
   your SSH key or token. You can create it through `bootstrap_secrets` in your secrets file.
2. Install with that Secret:
   ```sh
   ./pattern.sh make TOKEN_SECRET=<secret-name> TOKEN_NAMESPACE=<secret-namespace> install
   ```
3. See https://validatedpatterns.io/learn/private-repos/ for the Secret format.

That token covers Argo CD only. The in-cluster BuildConfigs (`ray-serve-llm`, `notebook`) also clone the
repo from `global.repoURL`, and they do not use that Secret. For a private fork, give each build its own git
credentials:

1. Create a source secret in each build's namespace, for example with a GitHub token:
   ```sh
   oc create secret generic pattern-git -n vllm-inference --type=kubernetes.io/basic-auth \
     --from-literal=username=<you> --from-literal=password=<token>
   oc create secret generic pattern-git -n vllm-notebook --type=kubernetes.io/basic-auth \
     --from-literal=username=<you> --from-literal=password=<token>
   ```
2. Set `build.sourceSecret: pattern-git` in `charts/all/ray-serve-llm/values.yaml` and
   `charts/all/notebook/values.yaml`, then push.

Or keep the fork public, or set `image.override` to a prebuilt image. On a running install, a new `image.override`
needs extra steps; see Troubleshooting.

### Updating a running install

- **Config and charts.** Push to your branch. Argo CD syncs automatically (automated sync, with retries).
- **Model or engine settings.** Changing `global.model.*` changes ConfigMap `ray-serve-llm-config`. Its checksum
  annotation then restarts the server pod, which downloads the new model. The Deployment uses `Recreate`, so the
  server is down until the new pod is Ready.
- **Images.** A BuildConfig's ConfigChange trigger fires only when the BuildConfig is created, so a push alone
  does not rebuild an image. There are two ways to rebuild:
  - **Automatic.** The `rebuild-on-commit` imperative job (`ansible/rebuild-on-commit.yaml`) runs every 10
    minutes, for both `ray-serve-llm` and `notebook`. It starts a build only when that build's folder
    (`inference-server/` or `notebooks/`) changed between the last successful build's commit and the head of the
    target branch. A failed build is not retried automatically. The full rule list is at the top of
    `ansible/rebuild_on_commit.py`. An error on one build (for example a Forbidden `start-build` when
    `rebuildOnCommit.enabled` is false in that chart) is printed as an `ERROR` line in the job's log, and the job
    still checks the other build and exits 0. The imperative CronJob runs its jobs one after another and stops at
    the first that fails, so `rebuild-on-commit` runs first and exits 0 on any `oc` or `git` error. Only its
    300-second job timeout can still fail the pod, and that also skips the GPU MachineSet job for that run.
  - **By hand:**
    ```sh
    oc start-build ray-serve-llm -n vllm-inference --follow
    oc start-build notebook -n vllm-notebook --follow
    ```

  A finished build updates the ImageStream. The image trigger annotation then rolls the Deployment.
- **Installs made under the old name.** Until November 2026 this pattern was named
  `rhvp-vllm-ray-inference-server`, and its namespaces were `rhvp-inference` and `rhvp-notebook`. An install made
  under the old name keeps working, and its Pattern resource and clustergroup app keep that name.
  - Merging the rename into your install's branch moves both apps to the new namespaces, `vllm-inference` and
    `vllm-notebook`, and nothing deletes the old ones. The `ray-serve-llm` and `notebook` apps sync without pruning
    (`automated: {}` in `variants/prod/values-prod.yaml`), and the Validated Patterns Operator syncs the
    clustergroup app without pruning too. So Argo CD creates a second server and a second notebook in the new
    namespaces, builds both images there (the server image takes about 18 minutes), and creates a new model PVC, so
    the new server downloads the model again. The old server keeps running in `rhvp-inference` and keeps its GPU.
    With one GPU, the new server pod stays Pending until you delete the old namespaces (see "Old namespaces" below).
    This is what the charts and the operator's code imply; it has not been tried on a cluster.
  - The installer takes the pattern name from `PATTERN_NAME` when it is set, and from `global.pattern` otherwise,
    and finds the Pattern resource and the secrets file by that name. So once your fork has the new
    `global.pattern`, set `PATTERN_NAME=rhvp-vllm-ray-inference-server` on every `./pattern.sh make` target you run
    on the old install, until you move it. Without it, nothing warns you:
    - `load-secrets` no longer finds a secrets file named for the old name. It loads `~/values-secret.yaml` if
      that exists, and the repo's template otherwise, and replaces a real token in Vault with that file's value
      (`none`, from the template).
    - `install` creates a second Pattern resource, `vllm-ray-inference-server`, next to the old one.
  - `display-secrets-info` is the exception: it always takes the name from `global.pattern` and ignores
    `PATTERN_NAME`. For it, set `VALUES_SECRET` to your old secrets file's path instead.

  To move the install to the new name, uninstall it first and merge the rename after:
  1. Download any notebook work you want to keep.
  2. Uninstall it under the old name: `PATTERN_NAME=rhvp-vllm-ray-inference-server ./pattern.sh make uninstall`
     (see [Uninstall](#uninstall)).
  3. Once the uninstall has finished, merge the rename into your branch, if you have not already, and push it. The
     uninstall has finished when the Pattern resource is gone and the operator has removed the apps:
     `oc get patterns -A` no longer lists `rhvp-vllm-ray-inference-server`, and `oc get applications.argoproj.io -A`
     lists none of its apps. Step 6 takes the pattern name from `global.pattern` in your clone, and Argo CD deploys
     from your fork, so the rename must be in both. Merging only after the uninstall keeps the old install from
     syncing the rename and starting a second server in the new namespaces. This order follows from the charts and
     the installer; it has not been tried on a cluster.
  4. Check for the old namespaces and delete them (see "Old namespaces" below).
  5. Rename your secrets file, if you have one:
     ```sh
     mv ~/.config/validated-patterns/values-secret-rhvp-vllm-ray-inference-server.yaml \
        ~/.config/validated-patterns/values-secret-vllm-ray-inference-server.yaml
     ```
  6. Install again: `./pattern.sh make install`. The new install creates `vllm-inference` and `vllm-notebook`,
     builds both images, and downloads the model again into a new PVC, `vllm-inference/ray-serve-llm-models`.

  Old namespaces. Neither path is sure to delete `rhvp-inference` and `rhvp-notebook`. A merge never does. The
  uninstall is experimental and has not been checked here; the framework's announcement says its last phase removes
  the app of apps, and with it the subscriptions and namespaces. Delete them only after the merge has synced or the uninstall has finished; before that, Argo CD can
  create them again. To check:
  ```sh
  oc get namespace rhvp-inference rhvp-notebook
  oc get pvc ray-serve-llm-models -n rhvp-inference
  ```
  To delete them with everything in them, including the old server, the old notebook and the downloaded model:
  ```sh
  oc delete namespace rhvp-inference rhvp-notebook
  ```
  Then check `oc get pv` for a volume whose claim was `rhvp-inference/ray-serve-llm-models`. A volume with the
  reclaim policy `Retain` stays after its claim is deleted; delete it with `oc delete pv <name>`. That deletes only
  the Kubernetes object: the backing disk (an EBS volume on AWS) stays until you delete it in your storage backend.

## Before changing structure, study how existing patterns do it

This pattern copies its structure from these sources. Read the matching one before you add an operator, an app, a
GPU knob, a build or a new serving mode:

- **The template.** https://github.com/validatedpatterns/multicloud-gitops
  - `values-global.yaml`
  - `variants/`: `variants/standalone/values-standalone.yaml` is the single-cluster shape.
  - `Makefile-common`
- **GPU enablement.** https://github.com/validatedpatterns-sandbox/ai-quickstart-rag
  - `overrides/values-gpu.yaml`: the NFD and GPU Operator namespaces, subscriptions and apps.
  - `overrides/values-gpu-AWS.yaml`: the `rhvp.cluster_utils.create_machineset` job and its `clusterRoleYaml`.
  - `charts/nvidia-config`: a ClusterPolicy that tolerates `nvidia.com/gpu`.
- **vLLM and the Hugging Face token.** https://github.com/validatedpatterns/rag-llm-gitops
  - `charts/all/vllm-inference-service`: vLLM on KServe, and the `hf_` token check.
  - `values-secret.yaml.template`: the `hfmodel` secret.
- **In-cluster builds under Argo CD.** https://github.com/validatedpatterns-sandbox/qna-chat-amd
  - `charts/all/llm/templates/llm-buildconfig.yaml`: a BuildConfig with `dockerfilePath`.
  - `values-hub.yaml`: `ignoreDifferences` for image-triggered Deployments and BuildConfig `.spec.paused`.
- **A future RayService variant through the KubeRay operator.**
  https://github.com/kuberay-project/kuberay/blob/master/ray-operator/config/samples/ray-service.llm-serve.yaml
  This is `ray.serve.llm:build_openai_app` as a RayService. The repo moved there from `ray-project/kuberay`.
- **The framework itself.** The pattern-author skill in `.claude/skills/pattern-author/` (`SKILL.md`,
  `reference.md`) covers the clustergroup values schema, the values-file order, sharedValueFiles and imperative
  jobs. `.cursor/skills/pattern-author/` holds the same skill.

Framework facts that catch people out:

- Namespaces are maps, and maps merge across values files. Lists replace each other: a values file that sets
  `clusterGroup.imperative.jobs` replaces the whole job list from earlier files.
- A per-app `syncPolicy` replaces the chart's default (`automated: {}` plus `retry`) completely, so restate
  both.
- Missing `sharedValueFiles` are skipped. That is how `gpuMachineSet: none`, or any platform other than AWS,
  turns the MachineSet job off, and how `global.scaleOut.mode: none` adds no autoscaler: each app's valueFiles
  then lists `/overrides/values-scaleout-none.yaml`, which does not exist. A value named in a `sharedValueFiles`
  entry must be set, or the clustergroup chart's render fails; `scripts/validate-values.sh` checks it.
- Keep app keys equal to chart folder names.

## Local development loop

Install the Python tools:

```sh
python3.12 -m pip install pytest pyyaml jsonschema nbformat nbclient nbconvert ipykernel httpx aiohttp==3.14.3
```

Then run these, all from the repo root:

| Command | What it does | Where it runs |
|---|---|---|
| `scripts/render-charts.sh` | renders every local chart and the clustergroup chart with the values the operator injects | needs docker or podman; uses the utility container |
| `scripts/validate-values.sh` | schema-checks the merged values (global, then `prod`, then each override) and the secrets template, and checks the scale-out settings | anywhere |
| `python3.12 -m pytest -q tests/` | runs the unit tests | anywhere; the chart tests (those that use the render in `tests/conftest.py`) are skipped without docker or podman and access to charts.validatedpatterns.io |
| `python3.12 notebooks/make_notebook.py --check` | regenerates both notebooks and their synthetic files and fails if one differs from the committed file | anywhere |
| `scripts/local-smoke.sh` | builds the server image and serves the model with docker | a machine with an NVIDIA GPU only |
| `scripts/compare-local.sh tests/fixtures/compare-local-mock.json <folder>` | the image comparison with docker, against two GuideLLM mock servers | needs docker; GPU sides need an NVIDIA GPU |
| `scripts/compare-local.sh tests/fixtures/compare-scaleout-mock.json <folder>` | the comparison's scale-out mode with docker, against GuideLLM mock servers that take a while to start | needs docker; no GPU |
| `python3.12 -m rhvp_compare validate --config <file>` (from `notebooks/`) | checks a comparison config and prints its rounds and whether the pairs can reach alpha | anywhere |

## Bumping Ray and vLLM

Ray decides the vLLM version. The `llm` extra in Ray's `python/setup.py` pins exactly one vLLM: Ray 2.60.0 pins
`vllm[audio]==0.29.0`, which pins `torch==2.13.0`. Never bump vLLM alone.

1. Read `python/setup.py` at the new Ray tag (`https://github.com/ray-project/ray/blob/ray-<version>/python/setup.py`)
   for the vLLM pin. Then read that vLLM's requirements for its torch pin, and that torch's `cuda-toolkit` pin.
2. Edit `inference-server/requirements.in`: set `ray[llm]==<version>`. Set `cuda-toolkit[nvcc,cccl]` to torch's
   CUDA toolkit version.
3. If the CUDA version changed, move the `FROM` in `inference-server/Containerfile` to the matching
   `nvcr.io/nvidia/cuda:<cuda>-devel-ubi9` image, pinned by digest.
4. Regenerate both locks from `inference-server/` (the command is also in each lock file's header):
   ```sh
   uv pip compile requirements.in --python-version 3.12 --python-platform x86_64-manylinux_2_34 \
     --only-binary :all: --generate-hashes -o requirements-x86_64.lock
   uv pip compile requirements.in --python-version 3.12 --python-platform aarch64-manylinux_2_34 \
     --only-binary :all: --generate-hashes -o requirements-aarch64.lock
   ```
5. If the install step fails in hash mode, check `inference-server/requirements-pip.lock`: the image installs that
   pip first, and an older pip can reject a dependency that is asked for with an extra.
6. Update the `BASELINE` label in `notebooks/rhvp_bench/words.py`, then regenerate the notebook, and update the
   `version`, `description` and `io.k8s.description` image labels in `inference-server/Containerfile`. A test checks
   their Ray, vLLM, PyTorch and CUDA versions against the lock files. Update the versions in `README.md`, and the vLLM tag in the GPU support link below.
7. Run the whole local loop, then `scripts/local-smoke.sh` on a GPU machine.

## Rules

1. **Stock packages only.** Allowed sources:
   - PyPI wheels;
   - public NVIDIA and Red Hat UBI base images;
   - Red Hat and certified operators;
   - Validated Patterns charts.

   Never add Eugo packages, Eugo images or Eugo registries, and never add a private package index.

   The image comparison also runs GuideLLM's public release image, pinned by digest, and the two side images the
   user puts in `global.compare.sides`. Examples, defaults and tests use only public upstream images (two vLLM
   releases) or made-up `registry.example/...` names.
2. **Notebooks.** `notebooks/inference_timings.ipynb` and `notebooks/compare_images.ipynb` are generated by
   `notebooks/make_notebook.py` and are committed without outputs. Never hand-edit an `.ipynb`: change
   `make_notebook.py`, `rhvp_bench/` or `rhvp_compare/`, regenerate, and run `--check`.
   - `inference_timings.ipynb` times the stock server only. Its results are given in seconds and tokens per
     second, never as a ratio; `tests/test_notebook_rules.py` checks this.
   - `compare_images.ipynb` shows B/A ratios and the verdicts of the comparison's pre-declared rule, and nothing
     else that ranks the two images; `tests/test_compare_notebook.py` checks it.
   - In both, no sentence gives a cause for a result.
3. **Telemetry stays off.** Keep these settings in the server image: `RAY_USAGE_STATS_ENABLED=0`,
   `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`, `RAYLLM_ENABLE_REQUEST_PROMPT_LOGS=0`.
4. **Installer name limit.** `len(global.pattern) + 2 * len(variant)` must be under 47. Today it is
   25 + 2 * 4 = 33, so a variant name of up to 10 characters, such as `standalone` (25 + 2 * 10 = 45), still fits.
5. **Secrets.** Never commit real secret values.
6. **Keep the probes.** Readiness and startup check the ready file. Do not replace them with an HTTP check (see
   Troubleshooting).
7. **Keep `strategy: Recreate`** on the server Deployment.
8. **Imperative jobs must be idempotent.** They run every 10 minutes.
9. **License headers.** Server files that carry the Eugo BSD-3-Clause header keep it. Do not name private
   repositories in any file.

## Troubleshooting

- **Server crashes with `PermissionError` while writing a file, or the entrypoint exits with "... is not
  writable by uid ...".**
  - Why: OpenShift runs the container as an arbitrary UID in group 0. The image's virtualenv is read-only. Only
    `/opt/app-root/home`, `/cache`, `/models` and `/tmp/r` are writable by group 0. The chart mounts emptyDirs at
    `/cache` (or PVC `ray-serve-llm-compile-cache`, see `compileCache.mode`) and `/tmp/r`, and the PVC at `/models`.
  - Fix: point any new tool's cache at `/cache/<tool>` through its environment variable. For any new image folder
    that must be writable, run `chgrp 0` and `chmod g=u` in the same `RUN` that creates it.
- **Ray warns that it is using `/tmp` instead of `/dev/shm`, or the pod is OOMKilled.**
  - Why: Ray's object store lives in `/dev/shm`. The chart mounts a 4Gi memory-backed emptyDir there, and
    `object_store_memory_gb: 2` keeps the store inside it. A memory-backed emptyDir counts against the pod's
    memory limit.
  - Fix: if you raise either size, raise the `/dev/shm` `sizeLimit` and the memory limit with it.
- **Connections to the Service are refused while Ray reports the app as RUNNING.**
  - Why: Ray Serve's HTTP proxy listens on localhost by default. `serve_app.py` calls
    `serve.start(http_options={"host": "0.0.0.0", "port": 8000})` with `http_host` and `http_port` from
    `llm.yaml`.
  - Fix: keep `http_host: 0.0.0.0`.
- **The pod turned Ready before the model was loaded, and requests hang.**
  - Why: `/v1/models` and `/-/healthz` both answer before the model is loaded.
  - Fix: the readiness and startup probes must check the ready file `/tmp/r/rhvp-ready`. `serve_app.py` writes
    it only after `serve.run` returns with the app RUNNING. Use `/-/healthz` for liveness only.
- **FlashInfer errors about `nvcc`, or every pod start takes a long time.**
  - Why: FlashInfer compiles GPU kernels just in time with `nvcc`. The base is NVIDIA's CUDA *devel* UBI 9 image,
    which ships `nvcc`; `CUDA_HOME=/usr/local/cuda`; and the locks pin `cuda-toolkit[nvcc,cccl]` to the same CUDA
    version. With `compileCache.mode: none` (the default) the compile caches live in the `/cache` emptyDir, so
    every new pod compiles the kernels again.
  - Fix: do not switch to a `-runtime` or `-base` image. To keep the compiled kernels for later pods, set
    `compileCache.mode` to `node` (one GPU node) or `shared` (a ReadWriteMany StorageClass).
- **A request through a Route fails with 504 or times out.**
  - Why: the OpenShift router times out after 30 seconds by default. The optional server Route sets
    `haproxy.router.openshift.io/timeout: 10m`.
  - Note: the notebook talks to the Service directly, not through a Route.
- **After an update, the new server pod stays Pending, or the Deployment reports `ProgressDeadlineExceeded`.**
  - Why: a rolling update would need a second GPU and the same single-node (ReadWriteOnce) PVC. With one GPU it
    deadlocks.
  - Fix: keep `strategy: Recreate`, and expect the server to be down during each rollout.
- **The build fails to push, or no build ever runs.**
  - Why: the internal image registry must be `Managed`. On bare metal and vSphere it starts as `Removed`, because
    there is no shared object storage. The `Build` capability must also be enabled.
  - Fix: give the registry storage and set it to `Managed`. Or set `image.override` in both charts to prebuilt
    images that you host; the charts then render no BuildConfig and no image trigger (on a running install, see the
    `image.override` entry below for the extra steps). Without the registry, also
    set `oauthProxy.image` in `charts/all/notebook/values.yaml` to a Red Hat image such as
    `registry.redhat.io/openshift4/ose-oauth-proxy-rhel9:v4.19`, because the default proxy image is pulled
    through the internal registry.
- **You set or changed `image.override` on a running install, and the pod still runs the old image.**
  - Why: the Argo CD apps ignore the image of the image-triggered container (`server`, `notebook`), and with
    `RespectIgnoreDifferences=true` an ignored field keeps its live value even when Git changes it. The rule names
    only that container, so an `oauthProxy.image` change does sync; an `image.override` change does not.
  - Fix, in this order, when the install was building the image in the cluster (`image.override` was empty). To
    change one prebuilt image for another, only step 2 is needed: there is no trigger, and nothing to remove.
    1. Wait until Argo CD has applied the commit: this prints nothing once the image-trigger annotation is gone
       (before that, the trigger would put the old image back):
       `oc get deployment/ray-serve-llm -n vllm-inference -o jsonpath='{.metadata.annotations.image\.openshift\.io/triggers}'`
       (for the notebook, `deployment/notebook -n vllm-notebook`). The app still shows OutOfSync at this
       point, because of the objects step 3 removes.
    2. Set the image on the Deployment yourself. Argo CD ignores that one field, so the value stays:
       `oc set image deployment/ray-serve-llm server=<your image> -n vllm-inference`
       (for the notebook: `oc set image deployment/notebook notebook=<your image> -n vllm-notebook`).
       Wait until the new pod runs that image.
    3. Remove what the chart no longer renders. The apps sync automatically without pruning, so Argo CD leaves the
       BuildConfig, ImageStream and the rebuild Role and RoleBinding in place; the app stays OutOfSync and
       `rebuild-on-commit` keeps starting builds of that BuildConfig. Either sync once with Prune checked in the
       Argo CD UI, or delete them:
       `oc delete buildconfig/ray-serve-llm imagestream/ray-serve-llm role/rebuild-on-commit rolebinding/rebuild-on-commit -n vllm-inference`
       (for the notebook, `buildconfig/notebook imagestream/notebook` and the same Role and RoleBinding,
       in `vllm-notebook`). Do this only after step 2: the old pod's image is pulled through that ImageStream.
  - Do not delete the Deployment instead: these apps sync automatically without self-heal, so Argo CD recreates a
    deleted Deployment only on the next commit or a manual sync, and the server stays down until then.
  - A fresh install with `image.override` set from the start needs none of this: no BuildConfig, ImageStream,
    trigger or rebuild Role is ever created.
- **The image did not change after a push, or the first build failed.**
  - Why: ConfigChange triggers fire only when the BuildConfig is created. `rebuild-on-commit` starts a new build
    only after a push changes the build's folder, and it never retries a failed build on its own.
  - Fix: read the build log with `oc logs -f bc/ray-serve-llm -n vllm-inference`, fix the cause, then start the
    build by hand: `oc start-build ray-serve-llm -n vllm-inference --follow` (for the notebook:
    `oc start-build notebook -n vllm-notebook --follow`).
- **The `ray-serve-llm` Argo CD app shows OutOfSync.**
  - Why: the image trigger rewrites the Deployment's container image to a digest. The app entry in
    `variants/prod/values-prod.yaml` handles this with `ignoreDifferences` on the triggered container's image
    (`.spec.template.spec.containers[] | select(.name == "server") | .image`, and `"notebook"` for the notebook app)
    and `.spec.paused` (BuildConfig), plus `RespectIgnoreDifferences=true`.
  - Fix: if you add another image-triggered resource, add the same rules.
- **`make install` fails after about an hour, with apps not Healthy.**
  - Why: the installer waits at most 60 minutes, and a first install can take longer. Everything before the wait
    has already run: the pattern is installed and the secrets are loaded.
  - Fix: watch progress with `oc get builds -n vllm-inference` and `oc get pods -n vllm-inference`, then run
    `./pattern.sh make argo-healthcheck`. To skip the wait on a later install, run
    `VP_SKIP_ARGO_HEALTHCHECK=true ./pattern.sh make install`, then run `make argo-healthcheck` once things
    settle.
- **The server pod stays Pending with "Insufficient nvidia.com/gpu".**
  - Why: no node advertises `nvidia.com/gpu` yet. The chain is: the MachineSet job (AWS) → the node joins → NFD
    labels it → the GPU Operator installs the driver and device plugin. The ClusterPolicy must tolerate the
    `nvidia.com/gpu` taint. Argo CD sync waves do not order apps in this framework; apps retry instead.
  - Check:
    ```sh
    oc get nodes -l nvidia.com/gpu.present=true
    oc get pods -n nvidia-gpu-operator
    oc describe node <gpu-node> | grep nvidia.com/gpu
    ```
- **Right after the install, ClusterSecretStore `vault-backend` reports "Vault is sealed" (code 503) or a 403,
  and ExternalSecret `huggingface-secret` reports `ClusterSecretStore "vault-backend" is not ready`.**
  - Why: the External Secrets Operator checks the store as soon as it exists, and Vault starts sealed. The
    framework's own `unsealvault-cronjob` (namespace `imperative`, every 5 minutes; separate from the pattern's
    imperative CronJob, which runs every 10) unseals Vault and sets up the Kubernetes login the store uses;
    `make install` waits for that, then loads the secrets. Until then the `openshift-external-secrets` and
    `ray-serve-llm` apps can show Degraded.
  - Fix: on the command-line install, none; it clears once the unseal job has run and `make install` has loaded the
    secrets. On the two live AWS installs the store turned Valid within 5 minutes of its first error, and the
    Secret was created within 7 minutes after that. If the store is still not Valid after 10 minutes, or the Secret
    is still missing after 15, check `oc get clustersecretstore vault-backend` and `oc get jobs -n imperative`. On
    the web-console install the store turns Valid by itself, but ExternalSecret `huggingface-secret` stays not Ready
    until you load the secrets ([Path 2](#path-2-openshift-web-console), step 5).
- **A gated model fails to download with 401 or 403.**
  - Why: the token is missing or not loaded. The entrypoint drops `HF_TOKEN` unless it starts with `hf_`.
  - Fix: put your `hf_` token in the secrets file, run `./pattern.sh make load-secrets`, wait for the Secret to
    refresh, then delete the server pod.

## GPU support

The server image is built for the GPU architectures in vLLM's release build list
(https://github.com/vllm-project/vllm/blob/v0.29.0/.buildkite/release-pipeline.yaml). On OpenShift we have tested only
the NVIDIA L4 on the default AWS node (README, "What this pattern was verified against").

## Uninstall

- `./pattern.sh make uninstall` is marked experimental by the framework
  (https://validatedpatterns.io/blog/2026-02-16-pattern-uninstall/). It deletes the Pattern resource, and the
  operator then removes the apps. Find the Pattern with `oc get patterns -A`.
- The Validated Patterns Operator, OpenShift GitOps and operator CRDs stay installed.
- Delete the GPU MachineSet yourself, after the pattern is gone or after `gpuMachineSet: none` has synced;
  otherwise the job may create it again:
  ```sh
  oc get machinesets -n openshift-machine-api \
    -o custom-columns=NAME:.metadata.name,TYPE:.spec.template.spec.providerSpec.value.instanceType
  oc delete machineset <name-of-the-g6.2xlarge-machineset> -n openshift-machine-api
  ```
- Check `oc get pv` for a leftover volume from claim `vllm-inference/ray-serve-llm-models`.
