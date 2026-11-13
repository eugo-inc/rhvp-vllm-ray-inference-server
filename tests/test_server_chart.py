"""The rendered server chart (charts/all/ray-serve-llm): Deployment, probes, GPU scheduling, ConfigMap, Service,
PVC, ExternalSecret, build and Route. The render comes from tests/conftest.py and is skipped the same way."""

from __future__ import annotations

import json
import re
import sys

import pytest
import yaml

from chart_helpers import GLOBAL, READY_FILE, REPO, container, docs, env_map, kinds, one, quantity

sys.path.insert(0, str(REPO / "inference-server"))

import serve_app  # noqa: E402

COMPILE_CACHE_PVC = "ray-serve-llm-compile-cache"


@pytest.fixture(scope="module")
def server_variant(render) -> list[dict]:
    return docs(render / "charts/ray-serve-llm-variant.yaml")


def render_of(render, name: str) -> list[dict]:
    return docs(render / f"charts/ray-serve-llm-{name}.yaml")


def llm_yaml(items: list[dict]) -> str:
    return one(items, "ConfigMap", "ray-serve-llm-config")["data"]["llm.yaml"]


def test_server_deployment_rollout(server):
    deploy = one(server, "Deployment", "ray-serve-llm")
    assert deploy["spec"]["strategy"] == {"type": "Recreate"}
    assert deploy["spec"]["progressDeadlineSeconds"] == 7200
    assert deploy["spec"]["replicas"] == 1
    pod = deploy["spec"]["template"]
    assert re.fullmatch(r"[0-9a-f]{64}", pod["metadata"]["annotations"]["checksum/config"])
    assert pod["metadata"]["labels"]["app.kubernetes.io/name"] == "ray-serve-llm"
    assert pod["spec"]["shareProcessNamespace"] is True
    assert pod["spec"]["terminationGracePeriodSeconds"] == 120


def test_server_image_and_trigger(server):
    deploy = one(server, "Deployment", "ray-serve-llm")
    c = container(deploy, "server")
    assert c["image"] == "image-registry.openshift-image-registry.svc:5000/vllm-inference/ray-serve-llm:latest"
    # The pod names the entrypoint itself, so an image CMD mangled by the builder is never passed to the server.
    containerfile = (REPO / "inference-server/Containerfile").read_text()
    assert f'ENTRYPOINT {json.dumps(c["command"])}' in containerfile
    assert c["command"] == ["/opt/app-root/src/entrypoint.sh"] and "args" not in c
    [trigger] = json.loads(deploy["metadata"]["annotations"]["image.openshift.io/triggers"])
    assert trigger["from"] == {"kind": "ImageStreamTag", "name": "ray-serve-llm:latest"}
    assert trigger["fieldPath"] == 'spec.template.spec.containers[?(@.name=="server")].image'


def test_server_probes(server):
    c = container(one(server, "Deployment", "ray-serve-llm"), "server")
    startup, ready, live = c["startupProbe"], c["readinessProbe"], c["livenessProbe"]
    assert startup["exec"]["command"] == READY_FILE
    assert ready["exec"]["command"] == READY_FILE
    budget = startup["periodSeconds"] * startup["failureThreshold"]
    assert 3500 <= budget <= 3700, budget
    # A short period keeps the notebook's start-to-Ready figure close to the moment the server is serving.
    assert startup["periodSeconds"] == 2 and ready["periodSeconds"] == 2
    assert ready["periodSeconds"] * ready["failureThreshold"] == 30
    # The notebook's words state the period; keep them in step with the chart.
    words = (REPO / "notebooks/rhvp_bench/words.py").read_text()
    assert f"every {ready['periodSeconds']} seconds" in words
    assert live["httpGet"] == {"path": "/-/healthz", "port": "http"}
    assert live["initialDelaySeconds"] >= 60
    assert c["ports"] == [{"name": "http", "containerPort": 8000, "protocol": "TCP"}]


def test_server_gpu_scheduling(server):
    pod = one(server, "Deployment", "ray-serve-llm")["spec"]["template"]["spec"]
    res = container({"spec": {"template": {"spec": pod}}}, "server")["resources"]
    assert res["requests"]["nvidia.com/gpu"] == 1
    assert res["limits"]["nvidia.com/gpu"] == 1
    assert {"key": "nvidia.com/gpu", "operator": "Exists"} in pod["tolerations"]
    [term] = pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
    assert {"key": "nvidia.com/gpu.present", "operator": "In", "values": ["true"]} in term["matchExpressions"]
    assert pod["nodeSelector"] == {"kubernetes.io/arch": "amd64"}


def test_server_memory_covers_shm(server):
    deploy = one(server, "Deployment", "ray-serve-llm")
    vols = {v["name"]: v for v in deploy["spec"]["template"]["spec"]["volumes"]}
    shm = vols["dshm"]["emptyDir"]
    assert shm["medium"] == "Memory" and shm["sizeLimit"] == "4Gi"
    res = container(deploy, "server")["resources"]
    limit = quantity(res["limits"]["memory"])
    assert limit >= quantity(res["requests"]["memory"])
    assert limit >= quantity(shm["sizeLimit"]) + 8 * 2**30


def test_server_security_context(server):
    pod = one(server, "Deployment", "ray-serve-llm")["spec"]["template"]["spec"]
    assert pod["securityContext"] == {"runAsNonRoot": True, "seccompProfile": {"type": "RuntimeDefault"}}
    sc = pod["containers"][0]["securityContext"]
    assert sc["allowPrivilegeEscalation"] is False
    assert sc["capabilities"] == {"drop": ["ALL"]}
    # restricted-v2 assigns the UID and fsGroup; a fixed one would be refused.
    assert "runAsUser" not in sc and "runAsUser" not in pod["securityContext"]
    assert "privileged" not in sc


def test_server_env_and_volumes(server):
    deploy = one(server, "Deployment", "ray-serve-llm")
    c = container(deploy, "server")
    env = env_map(c)
    assert env["RHVP_CONFIG"] == "/config/llm.yaml"
    assert env["HF_HOME"] == "/models/hf"
    assert env["TRITON_CACHE_DIR"] == "/cache/triton"
    assert env["VLLM_CACHE_ROOT"] == "/cache/vllm"
    assert env["FLASHINFER_WORKSPACE_BASE"] == "/cache/flashinfer"
    assert env["XDG_CACHE_HOME"] == "/cache/xdg"
    assert env["RAY_TMPDIR"] == "/tmp/r"
    assert env["HF_TOKEN"] == {"secretKeyRef": {"name": "huggingface-secret", "key": "hftoken", "optional": True}}
    mounts = {m["mountPath"]: m["name"] for m in c["volumeMounts"]}
    vols = {v["name"]: v for v in deploy["spec"]["template"]["spec"]["volumes"]}
    assert vols[mounts["/config"]]["configMap"] == {
        "name": "ray-serve-llm-config", "items": [{"key": "llm.yaml", "path": "llm.yaml"}]}
    assert vols[mounts["/models"]]["persistentVolumeClaim"] == {"claimName": "ray-serve-llm-models"}
    assert "emptyDir" in vols[mounts["/cache"]]
    assert "emptyDir" in vols[mounts["/tmp/r"]]
    assert vols[mounts["/dev/shm"]]["emptyDir"]["medium"] == "Memory"


def test_server_config_map(server):
    cm = one(server, "ConfigMap", "ray-serve-llm-config")
    assert yaml.safe_load(cm["data"]["llm.yaml"]) == {
        "app_name": "llm",
        "model_id": GLOBAL["global"]["model"]["id"],
        "model_source": GLOBAL["global"]["model"]["source"],
        "model_revision": GLOBAL["global"]["model"]["revision"],
        "engine_kwargs": {"max_model_len": 8192, "gpu_memory_utilization": 0.85, "tensor_parallel_size": 1},
        "accelerator_type": None,
        "min_replicas": 1,
        "max_replicas": 1,
        "object_store_memory_gb": 2,
        "http_host": "0.0.0.0",
        "http_port": 8000,
    }


def test_server_service_pvc_secret(server):
    svc = one(server, "Service", "ray-serve-llm")
    assert svc["spec"]["selector"] == {"app.kubernetes.io/name": "ray-serve-llm"}
    assert svc["spec"]["ports"] == [{"name": "http", "port": 8000, "targetPort": "http", "protocol": "TCP"}]
    pvc = one(server, "PersistentVolumeClaim", "ray-serve-llm-models")
    assert pvc["spec"]["accessModes"] == ["ReadWriteOnce"]
    assert pvc["spec"]["resources"]["requests"]["storage"] == "50Gi"
    es = one(server, "ExternalSecret", "huggingface-secret")
    assert es["apiVersion"] == "external-secrets.io/v1"
    assert es["metadata"]["annotations"]["argocd.argoproj.io/sync-options"] == "SkipDryRunOnMissingResource=true"
    assert es["spec"]["secretStoreRef"] == {"name": "vault-backend", "kind": "ClusterSecretStore"}
    assert es["spec"]["target"]["name"] == "huggingface-secret"
    assert es["spec"]["data"] == [
        {"secretKey": "hftoken", "remoteRef": {"key": "secret/data/hub/hfmodel", "property": "hftoken"}}]


def test_server_no_route_by_default(server):
    assert "Route" not in kinds(server)


def test_server_build(server):
    bc = one(server, "BuildConfig", "ray-serve-llm")["spec"]
    assert bc["strategy"] == {"type": "Docker", "dockerStrategy": {"dockerfilePath": "Containerfile"}}
    assert bc["source"]["contextDir"] == "inference-server"
    assert bc["source"]["git"] == {"uri": "https://github.com/eugo-inc/vllm-ray-inference-server", "ref": "main"}
    assert bc["triggers"] == [{"type": "ConfigChange"}]
    assert bc["runPolicy"] == "SerialLatestOnly"
    assert bc["resources"]["requests"]["ephemeral-storage"] == "50Gi"
    assert bc["completionDeadlineSeconds"] == 10800
    assert bc["nodeSelector"] == {"kubernetes.io/arch": "amd64"}
    assert bc["output"]["to"] == {"kind": "ImageStreamTag", "name": "ray-serve-llm:latest"}
    assert one(server, "ImageStream", "ray-serve-llm")["spec"]["lookupPolicy"] == {"local": True}


def test_server_image_override_hides_build(server_variant):
    assert not kinds(server_variant) & {"BuildConfig", "ImageStream"}
    deploy = one(server_variant, "Deployment", "ray-serve-llm")
    assert "image.openshift.io/triggers" not in (deploy["metadata"].get("annotations") or {})
    assert container(deploy, "server")["image"] == "quay.io/example/ray-serve-llm:1.0"
    assert not [d for d in server_variant if d["kind"] == "Role" and d["metadata"]["name"] == "rebuild-on-commit"]


def test_server_route_when_enabled(server_variant):
    route = one(server_variant, "Route", "ray-serve-llm")
    assert route["metadata"]["annotations"]["haproxy.router.openshift.io/timeout"] == "10m"
    assert route["spec"]["tls"]["termination"] == "edge"
    assert route["spec"]["to"]["name"] == "ray-serve-llm"
    assert route["spec"]["port"] == {"targetPort": "http"}


# ---- scale-out settings: each renders only away from its default ------------------------------------------


def test_default_render_has_no_scale_out_parts(server):
    deploy = one(server, "Deployment", "ray-serve-llm")
    c = container(deploy, "server")
    assert [p["name"] for p in c["ports"]] == ["http"]
    assert [p["name"] for p in one(server, "Service", "ray-serve-llm")["spec"]["ports"]] == ["http"]
    env = env_map(c)
    assert "CUDA_CACHE_PATH" not in env and "CUDA_CACHE_MAXSIZE" not in env
    vols = {v["name"]: v for v in deploy["spec"]["template"]["spec"]["volumes"]}
    assert vols["cache"] == {"name": "cache", "emptyDir": {"sizeLimit": "20Gi"}}
    assert not [d for d in server if d["kind"] == "PersistentVolumeClaim" and d["metadata"]["name"] == COMPILE_CACHE_PVC]
    llm = yaml.safe_load(llm_yaml(server))
    assert not {"target_ongoing_requests", "gpu_per_replica", "metrics_port"} & set(llm)


def test_scaling_render_reaches_ray_serve(server, render):
    scaling = render_of(render, "scaling")
    llm = yaml.safe_load(llm_yaml(scaling))
    assert (llm["min_replicas"], llm["max_replicas"], llm["target_ongoing_requests"], llm["gpu_per_replica"]) == (
        1, 2, 4, 0.5)
    checksum = "checksum/config"
    pod = one(scaling, "Deployment", "ray-serve-llm")["spec"]["template"]
    default_pod = one(server, "Deployment", "ray-serve-llm")["spec"]["template"]
    assert pod["metadata"]["annotations"][checksum] != default_pod["metadata"]["annotations"][checksum]
    # The pod still asks for one GPU; the two replicas share it through Ray.
    assert container({"spec": {"template": pod}}, "server")["resources"]["limits"]["nvidia.com/gpu"] == 1


def test_rendered_llm_yaml_loads_in_the_server(server, server_variant, render, tmp_path):
    for name, items in [("default", server), ("variant", server_variant), ("keda", render_of(render, "keda")),
                        ("cache-node", render_of(render, "cache-node")),
                        ("cache-shared", render_of(render, "cache-shared"))]:
        path = tmp_path / f"{name}.yaml"
        path.write_text(llm_yaml(items))
        cfg = serve_app.load_config(str(path), {})
        assert cfg["metrics_port"] == 8080 and cfg["gpu_per_replica"] == 1, name
    # Two replicas on half a GPU each must each claim at most half of its memory (serve_app, the chart and the
    # values check refuse more), so the scaling render sets a memory share that fits.
    path = tmp_path / "scaling.yaml"
    path.write_text(llm_yaml(render_of(render, "scaling")))
    cfg = serve_app.load_config(str(path), {})
    assert cfg["engine_kwargs"]["gpu_memory_utilization"] == 0.4
    kwargs = serve_app.build_llm_config_kwargs(cfg)
    assert kwargs["deployment_config"]["autoscaling_config"] == {
        "min_replicas": 1, "max_replicas": 2, "target_ongoing_requests": 4}
    assert kwargs["placement_group_config"] == {"bundle_per_worker": {"GPU": 0.5}}
    assert serve_app.poll_interval_s(cfg) == serve_app.SCALE_POLL_INTERVAL_S


@pytest.mark.parametrize("mode, access", [("node", "ReadWriteOnce"), ("shared", "ReadWriteMany")])
def test_compile_cache_modes_keep_the_caches(server, render, mode, access):
    items = render_of(render, f"cache-{mode}")
    pvc = one(items, "PersistentVolumeClaim", COMPILE_CACHE_PVC)
    assert pvc["spec"]["accessModes"] == [access]
    assert pvc["spec"]["resources"]["requests"]["storage"] == "20Gi"
    assert "storageClassName" not in pvc["spec"]
    deploy = one(items, "Deployment", "ray-serve-llm")
    c = container(deploy, "server")
    env = env_map(c)
    assert env["CUDA_CACHE_PATH"] == "/cache/cuda"
    assert env["CUDA_CACHE_MAXSIZE"] == "4294967296"  # a string: env values are strings; 4 GiB, the driver's most
    # Every compile cache is under /cache, which is now the kept volume.
    for name in ("TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "FLASHINFER_WORKSPACE_BASE", "XDG_CACHE_HOME",
                 "CUDA_CACHE_PATH"):
        assert env[name].startswith("/cache/"), name
    mounts = {m["mountPath"]: m["name"] for m in c["volumeMounts"]}
    vols = {v["name"]: v for v in deploy["spec"]["template"]["spec"]["volumes"]}
    assert vols[mounts["/cache"]] == {"name": "cache", "persistentVolumeClaim": {"claimName": COMPILE_CACHE_PVC}}
    pod = deploy["spec"]["template"]["spec"]
    assert pod["securityContext"]["fsGroupChangePolicy"] == "OnRootMismatch"
    # Restricted-v2 has no hostPath; nothing else in the pod changes.
    assert not [v for v in pod["volumes"] if "hostPath" in v]
    default_deploy = one(server, "Deployment", "ray-serve-llm")
    assert env_map(container(default_deploy, "server")).items() <= env.items()
    assert llm_yaml(items) == llm_yaml(server)


def test_keda_mode_opens_the_metrics_port(server, render):
    items = render_of(render, "keda")
    c = container(one(items, "Deployment", "ray-serve-llm"), "server")
    assert c["ports"] == [{"name": "http", "containerPort": 8000, "protocol": "TCP"},
                          {"name": "metrics", "containerPort": 8080, "protocol": "TCP"}]
    svc = one(items, "Service", "ray-serve-llm")
    assert svc["spec"]["ports"][1] == {"name": "metrics", "port": 8080, "targetPort": "metrics", "protocol": "TCP"}
    # The server exports on that port in every mode (serve_app.ray_init_kwargs); keda only opens it.
    assert llm_yaml(items) == llm_yaml(server)
    assert serve_app.DEFAULTS["metrics_port"] == 8080


def test_model_storage_access_mode_is_the_setting(server):
    values = yaml.safe_load((REPO / "charts/all/ray-serve-llm/values.yaml").read_text())
    assert one(server, "PersistentVolumeClaim", "ray-serve-llm-models")["spec"]["accessModes"] == [
        values["modelStorage"]["accessMode"]]
    template = (REPO / "charts/all/ray-serve-llm/templates/pvc.yaml").read_text()
    assert "- {{ .Values.modelStorage.accessMode }}" in template
