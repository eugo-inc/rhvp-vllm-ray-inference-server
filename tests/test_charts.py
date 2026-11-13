"""Rendered charts, the clustergroup app of apps, and the rebuild-on-commit job.

The chart tests run scripts/render-charts.sh once (helm inside the Validated Patterns
utility container) and check the rendered YAML. They are skipped when neither docker nor
podman is available, or when charts.validatedpatterns.io cannot be reached.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
# Same pin as scripts/render-charts.sh: the v1.0.5 release digest (a latest digest is deleted on rebuild).
UTILITY_IMAGE = (
    "quay.io/validatedpatterns/utility-container@sha256:"
    "556083f92b08642868ca64ade6cf9553295bd394f7a54d5e10ca12d143907214"
)
GLOBAL = yaml.safe_load((REPO / "values-global.yaml").read_text())
READY_FILE = ["test", "-f", "/tmp/r/rhvp-ready"]


def container_engine() -> str | None:
    for engine in ("podman", "docker"):
        if shutil.which(engine):
            probe = subprocess.run([engine, "info"], capture_output=True, timeout=60)
            if probe.returncode == 0:
                return engine
    return None


@pytest.fixture(scope="module")
def render(tmp_path_factory) -> Path:
    if container_engine() is None:
        pytest.skip("needs docker or podman")
    try:
        urllib.request.urlopen("https://charts.validatedpatterns.io/index.yaml", timeout=30).close()
    except (urllib.error.URLError, TimeoutError) as err:
        pytest.skip(f"charts.validatedpatterns.io unreachable: {err}")
    out = tmp_path_factory.mktemp("render")
    proc = subprocess.run(
        [str(REPO / "scripts/render-charts.sh")],
        env={**os.environ, "RENDER_DIR": str(out)},
        capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return out


def docs(path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all(path.read_text()) if d]


def one(items: list[dict], kind: str, name: str | None = None) -> dict:
    found = [d for d in items if d["kind"] == kind and (name is None or d["metadata"]["name"] == name)]
    assert len(found) == 1, f"expected one {kind} {name or ''}, found {len(found)}"
    return found[0]


def kinds(items: list[dict]) -> set[str]:
    return {d["kind"] for d in items}


def container(deploy: dict, name: str) -> dict:
    [c] = [c for c in deploy["spec"]["template"]["spec"]["containers"] if c["name"] == name]
    return c


def env_map(c: dict) -> dict:
    return {e["name"]: e.get("value", e.get("valueFrom")) for e in c.get("env", [])}


def quantity(value: str) -> float:
    units = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "k": 1e3, "M": 1e6, "G": 1e9}
    match = re.fullmatch(r"([0-9.]+)([A-Za-z]*)", str(value))
    assert match, value
    return float(match.group(1)) * units.get(match.group(2), 1)


# ---- ray-serve-llm ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def server(render) -> list[dict]:
    return docs(render / "charts/ray-serve-llm.yaml")


@pytest.fixture(scope="module")
def server_variant(render) -> list[dict]:
    return docs(render / "charts/ray-serve-llm-variant.yaml")


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


def test_rebuild_rbac(server, notebook):
    for items, ns in ((server, "vllm-inference"), (notebook, "vllm-notebook")):
        role = one(items, "Role", "rebuild-on-commit")
        assert "namespace" not in role["metadata"]  # lands in the app's namespace
        rules = {(tuple(r["resources"]), tuple(r["verbs"])) for r in role["rules"]}
        assert (("buildconfigs/instantiate", "builds/docker"), ("create",)) in rules
        binding = one(items, "RoleBinding", "rebuild-on-commit")
        assert binding["subjects"] == [{"kind": "ServiceAccount", "name": "imperative-sa", "namespace": "imperative"}]


# ---- notebook ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def notebook(render) -> list[dict]:
    return docs(render / "charts/notebook.yaml")


@pytest.fixture(scope="module")
def notebook_variant(render) -> list[dict]:
    return docs(render / "charts/notebook-variant.yaml")


def test_notebook_jupyter_on_loopback(notebook):
    deploy = one(notebook, "Deployment", "notebook")
    assert deploy["spec"]["strategy"] == {"type": "Recreate"}
    assert deploy["spec"]["progressDeadlineSeconds"] >= 1800
    c = container(deploy, "notebook")
    # The chart runs the image's own entrypoint, with no arguments, so the base
    # image's CMD never reaches it; that script is what binds Jupyter to loopback.
    assert c["command"] == ["/opt/app-root/notebooks/entrypoint.sh"]
    assert "args" not in c
    script = (REPO / "notebooks/entrypoint.sh").read_text()
    assert "--ServerApp.ip=127.0.0.1" in script
    assert "--ServerApp.port=8888" in script
    assert 'work="${RHVP_WORK:-/work}"' in script
    assert '--ServerApp.root_dir="$work"' in script
    assert 'src="${RHVP_NOTEBOOKS:-/opt/app-root/notebooks}"' in script
    assert "git clone" not in script
    containerfile = (REPO / "notebooks/Containerfile").read_text()
    assert "COPY . /opt/app-root/notebooks" in containerfile
    assert not re.search(r"^CMD\b", containerfile, re.M)  # OpenShift's builder mangles `CMD []`
    assert "ports" not in c  # Jupyter is reachable only through the proxy
    mounts = {m["mountPath"] for m in c["volumeMounts"]}
    assert "/work" in mounts
    assert c["image"] == "image-registry.openshift-image-registry.svc:5000/vllm-notebook/notebook:latest"
    [trigger] = json.loads(deploy["metadata"]["annotations"]["image.openshift.io/triggers"])
    assert trigger["from"] == {"kind": "ImageStreamTag", "name": "notebook:latest"}
    assert trigger["fieldPath"] == 'spec.template.spec.containers[?(@.name=="notebook")].image'


def test_notebook_oauth_proxy(notebook):
    deploy = one(notebook, "Deployment", "notebook")
    pod = deploy["spec"]["template"]["spec"]
    assert pod["serviceAccountName"] == "notebook"
    proxy = container(deploy, "oauth-proxy")
    args = proxy["args"]
    assert "--provider=openshift" in args
    assert "--openshift-service-account=notebook" in args
    assert "--upstream=http://127.0.0.1:8888" in args
    assert "--https-address=:8443" in args
    [sar] = [a for a in args if a.startswith("--openshift-sar=")]
    assert json.loads(sar.split("=", 1)[1])["namespace"] == "vllm-notebook"
    assert proxy["ports"] == [{"name": "https", "containerPort": 8443, "protocol": "TCP"}]
    vols = {v["name"]: v for v in pod["volumes"]}
    tls = {m["mountPath"]: m["name"] for m in proxy["volumeMounts"]}["/etc/tls/private"]
    assert vols[tls]["secret"]["secretName"] == "notebook-tls"
    assert one(notebook, "Secret", "notebook-proxy")["data"]["session_secret"]
    svc = one(notebook, "Service", "notebook")
    assert svc["metadata"]["annotations"]["service.beta.openshift.io/serving-cert-secret-name"] == "notebook-tls"
    assert svc["spec"]["ports"][0]["port"] == 8443
    route = one(notebook, "Route", "notebook")
    assert route["spec"]["tls"]["termination"] == "reencrypt"
    assert route["spec"]["to"]["name"] == "notebook"
    sa = one(notebook, "ServiceAccount", "notebook")
    ref = json.loads(sa["metadata"]["annotations"]["serviceaccounts.openshift.io/oauth-redirectreference.primary"])
    assert ref["reference"] == {"kind": "Route", "name": "notebook"}


def test_notebook_reads_server_pods(notebook):
    role = one(notebook, "Role", "notebook-read-pods")
    assert role["metadata"]["namespace"] == "vllm-inference"
    assert role["rules"] == [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch"]}]
    binding = one(notebook, "RoleBinding", "notebook-read-pods")
    assert binding["metadata"]["namespace"] == "vllm-inference"
    assert binding["subjects"] == [{"kind": "ServiceAccount", "name": "notebook", "namespace": "vllm-notebook"}]


def test_notebook_env(notebook, notebook_variant):
    env = env_map(container(one(notebook, "Deployment", "notebook"), "notebook"))
    assert env["RHVP_ENDPOINT"] == "http://ray-serve-llm.vllm-inference.svc:8000"
    assert env["RHVP_MODEL_ID"] == GLOBAL["global"]["model"]["id"]
    assert env["RHVP_MODEL_REVISION"] == GLOBAL["global"]["model"]["revision"]
    assert env["RHVP_SERVER_NAMESPACE"] == "vllm-inference"
    assert env["RHVP_SERVER_LABEL_SELECTOR"] == "app.kubernetes.io/name=ray-serve-llm"
    assert "RHVP_INTERNAL" not in env and "RHVP_REPLAY" not in env
    venv = env_map(container(one(notebook_variant, "Deployment", "notebook"), "notebook"))
    assert venv["RHVP_INTERNAL"] == "1"
    assert venv["RHVP_REPLAY"] == "/opt/app-root/notebooks/replay/example"


def test_notebook_build(notebook, notebook_variant):
    bc = one(notebook, "BuildConfig", "notebook")["spec"]
    assert bc["strategy"]["dockerStrategy"] == {"dockerfilePath": "Containerfile"}
    assert bc["source"]["contextDir"] == "notebooks"
    assert bc["triggers"] == [{"type": "ConfigChange"}]
    assert bc["output"]["to"] == {"kind": "ImageStreamTag", "name": "notebook:latest"}
    assert one(notebook, "ImageStream", "notebook")["spec"]["lookupPolicy"] == {"local": True}
    assert not kinds(notebook_variant) & {"BuildConfig", "ImageStream"}
    deploy = one(notebook_variant, "Deployment", "notebook")
    assert "image.openshift.io/triggers" not in (deploy["metadata"].get("annotations") or {})
    assert container(deploy, "notebook")["image"] == "quay.io/example/notebook:1.0"


# ---- the image comparison (global.compare) --------------------------------------------------


@pytest.fixture(scope="module")
def compare(render) -> dict[str, list[dict]]:
    return {name: docs(render / f"charts/{name}.yaml")
            for name in ("notebook-compare", "notebook-compare-noenergy", "notebook-compare-sides",
                         "notebook-compare-pull")}


COMPARE_KINDS = {"Job", "PersistentVolumeClaim", "ServiceAccount", "ClusterRole", "ClusterRoleBinding"}


def test_compare_is_off_by_default(server, notebook):
    assert GLOBAL["global"]["compare"]["enabled"] is False
    assert one(server, "Deployment", "ray-serve-llm")["spec"]["replicas"] == 1
    assert not [d for d in notebook if "compare" in d["metadata"]["name"]]
    assert not kinds(notebook) & {"Job", "PersistentVolumeClaim", "ClusterRole"}
    deploy = one(notebook, "Deployment", "notebook")
    assert "RHVP_COMPARE_RESULTS" not in env_map(container(deploy, "notebook"))
    assert "affinity" not in deploy["spec"]["template"]["spec"]


def test_compare_leaves_the_server_replicas_to_the_runner(server, cg):
    # The runner scales the server to 0 for a run and back when it ends; Git keeps 1 and Argo CD ignores replicas.
    assert one(server, "Deployment", "ray-serve-llm")["spec"]["replicas"] == 1
    apps = {d["metadata"]["name"]: d for d in cg["default"] if d["kind"] == "Application"}
    replicas = [i for i in apps["ray-serve-llm"]["spec"]["ignoreDifferences"] if i.get("jsonPointers")]
    assert replicas == [{"group": "apps", "kind": "Deployment", "name": "ray-serve-llm",
                         "jsonPointers": ["/spec/replicas"]}]


def test_compare_job(compare):
    items = compare["notebook-compare"]
    [job] = [d for d in items if d["kind"] == "Job"]
    config = one(items, "ConfigMap")
    digest = config["metadata"]["name"].rsplit("-", 1)[1]
    assert job["metadata"]["labels"]["rhvp-compare/config"] == digest
    assert re.fullmatch(rf"compare-{GLOBAL['global']['compare']['runName']}-[0-9a-f]{{8}}", job["metadata"]["name"])
    assert len(job["metadata"]["name"]) <= 63
    assert json.loads(job["metadata"]["annotations"]["rhvp-compare/config"]) == json.loads(
        config["data"]["compare.json"])
    spec = job["spec"]
    assert spec["backoffLimit"] == 0 and "ttlSecondsAfterFinished" not in spec
    pod = spec["template"]["spec"]
    assert pod["serviceAccountName"] == "image-compare" and pod["restartPolicy"] == "Never"
    assert pod["terminationGracePeriodSeconds"] == 300
    sys.path.insert(0, str(REPO / "notebooks"))
    from rhvp_compare import kube

    # a stop inside a round (GuideLLM stopped, the server asked back, the side taken down, the last saves) fits
    assert kube.stop_budget_seconds() < pod["terminationGracePeriodSeconds"]
    assert pod["securityContext"] == {"runAsNonRoot": True, "seccompProfile": {"type": "RuntimeDefault"}}
    affinity = pod["affinity"]["podAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][0]
    assert affinity == {"labelSelector": {"matchLabels": {"app.kubernetes.io/name": "notebook"}},
                        "topologyKey": "kubernetes.io/hostname"}
    [copy] = pod["initContainers"]
    assert copy["image"] == "image-registry.openshift-image-registry.svc:5000/vllm-notebook/notebook:latest"
    assert copy["command"] == ["cp", "-R", "/opt/app-root/notebooks/rhvp_compare", "/runner/"]
    [runner] = pod["containers"]
    assert runner["image"].startswith("ghcr.io/vllm-project/guidellm:v0.8.1@sha256:")
    assert runner["command"] == ["python", "-m", "rhvp_compare", "run", "--backend", "kube", "--config",
                                 "/config/compare.json", "--results", "/results"]
    env = env_map(runner)
    assert env["RHVP_COMPARE_NAMESPACE"] == "vllm-inference" and env["PYTHONPATH"] == "/runner"
    assert env["RHVP_COMPARE_MODELS_PVC"] == "ray-serve-llm-models"
    assert json.loads(env["RHVP_COMPARE_SIDE_RESOURCES"])["limits"]["memory"] == "24Gi"
    assert env["RHVP_COMPARE_SERVER_DEPLOYMENT"] == "ray-serve-llm" and env["RHVP_COMPARE_SERVER_REPLICAS"] == "1"
    assert env["RHVP_COMPARE_DEADLINE_SECONDS"] == str(spec["activeDeadlineSeconds"])
    assert env["RHVP_COMPARE_TERMINATION_LOG"] == "/dev/termination-log"
    assert runner["terminationMessagePolicy"] == "FallbackToLogsOnError"
    status_ns = next(e for e in runner["env"] if e["name"] == "RHVP_COMPARE_STATUS_NAMESPACE")
    assert status_ns["valueFrom"] == {"fieldRef": {"fieldPath": "metadata.namespace"}}
    pod_name = next(e for e in runner["env"] if e["name"] == "RHVP_COMPARE_POD_NAME")  # one run at a time
    assert pod_name["valueFrom"] == {"fieldRef": {"fieldPath": "metadata.name"}}
    assert spec["template"]["metadata"]["labels"]["app.kubernetes.io/name"] == "image-compare"
    for c in pod["initContainers"] + pod["containers"]:
        assert c["securityContext"] == {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}}
    assert "runAsUser" not in json.dumps(job)
    volumes = {v["name"]: v for v in pod["volumes"]}
    assert volumes["config"]["configMap"]["name"] == config["metadata"]["name"]
    assert volumes["results"]["persistentVolumeClaim"]["claimName"] == "image-compare-results"
    pvc = one(items, "PersistentVolumeClaim", "image-compare-results")
    assert pvc["spec"]["accessModes"] == ["ReadWriteOnce"]


def test_compare_job_name_follows_the_pod_template(compare):
    # A Job's pod template cannot change: bigger side pods, or another pull policy for the init container, must be a
    # new Job, with the same config (and run id). The name hashes the whole rendered pod template.
    [job] = [d for d in compare["notebook-compare"] if d["kind"] == "Job"]
    for variant in ("notebook-compare-sides", "notebook-compare-pull"):
        [other] = [d for d in compare[variant] if d["kind"] == "Job"]
        assert job["metadata"]["name"] != other["metadata"]["name"]
        assert one(compare["notebook-compare"], "ConfigMap")["metadata"]["name"] == one(
            compare[variant], "ConfigMap")["metadata"]["name"]
    [bigger] = [d for d in compare["notebook-compare-sides"] if d["kind"] == "Job"]
    env = env_map(bigger["spec"]["template"]["spec"]["containers"][0])
    assert json.loads(env["RHVP_COMPARE_SIDE_RESOURCES"])["limits"]["memory"] == "32Gi"
    [pulled] = [d for d in compare["notebook-compare-pull"] if d["kind"] == "Job"]
    assert pulled["spec"]["template"]["spec"]["initContainers"][0]["imagePullPolicy"] == "Always"


def test_compare_config_is_global_compare_plus_the_model(compare):
    config = json.loads(one(compare["notebook-compare"], "ConfigMap")["data"]["compare.json"])
    want = {k: v for k, v in GLOBAL["global"]["compare"].items() if k != "enabled"}
    assert {k: v for k, v in config.items() if k != "model"} == json.loads(json.dumps(want))
    assert config["model"] == {k: GLOBAL["global"]["model"][k] for k in ("id", "source", "revision")}
    sys.path.insert(0, str(REPO / "notebooks"))
    from rhvp_compare import plan as plans

    plan = plans.build(config, "openshift")
    assert plan["floor"]["reachable"] and plan["energy"]["source"] == "dcgm"
    for side in plan["sides"]:
        assert side["image"].startswith("docker.io/vllm/vllm-openai:v") and "@sha256:" in side["image"]


def test_compare_rbac(compare):
    items = compare["notebook-compare"]
    role = one(items, "Role", "image-compare-sides")
    assert role["metadata"]["namespace"] == "vllm-inference"
    assert role["rules"] == [
        {"apiGroups": [""], "resources": ["pods"], "verbs": ["create", "delete", "get", "list", "watch"]},
        {"apiGroups": [""], "resources": ["pods/log"], "verbs": ["get"]},
        {"apiGroups": [""], "resources": ["secrets"], "resourceNames": ["huggingface-secret"], "verbs": ["get"]},
        {"apiGroups": ["apps"], "resources": ["deployments/scale"], "resourceNames": ["ray-serve-llm"],
         "verbs": ["get", "patch"]}]
    status = one(items, "Role", "image-compare-status")
    assert "namespace" not in status["metadata"]  # the release namespace, vllm-notebook
    assert status["rules"] == [{"apiGroups": [""], "resources": ["configmaps"], "verbs": ["create", "get", "patch"]},
                               {"apiGroups": [""], "resources": ["pods"], "verbs": ["list"]}]
    assert one(items, "RoleBinding", "image-compare-status")["subjects"] == [
        {"kind": "ServiceAccount", "name": "image-compare", "namespace": "vllm-notebook"}]
    binding = one(items, "RoleBinding", "image-compare-sides")
    assert binding["subjects"] == [{"kind": "ServiceAccount", "name": "image-compare", "namespace": "vllm-notebook"}]
    cluster_role = one(items, "ClusterRole", "vllm-notebook-image-compare")
    assert cluster_role["rules"] == [
        {"apiGroups": [""], "resources": ["nodes"], "verbs": ["get"]},
        {"apiGroups": ["config.openshift.io"], "resources": ["clusterversions"], "verbs": ["get"]}]
    monitoring = one(items, "ClusterRoleBinding", "vllm-notebook-image-compare-monitoring")
    assert monitoring["roleRef"]["name"] == "cluster-monitoring-view"
    names = {d["metadata"]["name"] for d in compare["notebook-compare-noenergy"] if d["kind"] == "ClusterRoleBinding"}
    assert names == {"vllm-notebook-image-compare"}


def test_compare_notebook_mounts_the_results_read_only(compare):
    deploy = one(compare["notebook-compare"], "Deployment", "notebook")
    c = container(deploy, "notebook")
    assert env_map(c)["RHVP_COMPARE_RESULTS"] == "/results"
    assert {"name": "compare-results", "mountPath": "/results", "readOnly": True} in c["volumeMounts"]
    volume = next(v for v in deploy["spec"]["template"]["spec"]["volumes"] if v["name"] == "compare-results")
    assert volume["persistentVolumeClaim"] == {"claimName": "image-compare-results", "readOnly": True}
    assert deploy["spec"]["strategy"] == {"type": "Recreate"}
    # a notebook pod that restarts during a run prefers the runner's node (the results volume is ReadWriteOnce)
    [term] = deploy["spec"]["template"]["spec"]["affinity"]["podAffinity"][
        "preferredDuringSchedulingIgnoredDuringExecution"]
    assert term["podAffinityTerm"] == {"labelSelector": {"matchLabels": {"app.kubernetes.io/name": "image-compare"}},
                                       "topologyKey": "kubernetes.io/hostname"}
    variant = compare["notebook-compare-noenergy"]
    [job] = [d for d in variant if d["kind"] == "Job"]
    assert job["spec"]["template"]["spec"]["initContainers"][0]["image"] == "quay.io/example/notebook:1.0"


def test_gpu_operator_namespace_is_scraped_by_cluster_monitoring(cg):
    [ns] = [d for d in cg["default"] if d["kind"] == "Namespace" and d["metadata"]["name"] == "nvidia-gpu-operator"]
    assert ns["metadata"]["labels"]["openshift.io/cluster-monitoring"] == "true"


# ---- GPU enablement ------------------------------------------------------------------------


def test_gpu_operator_config(render):
    policy = one(docs(render / "charts/nvidia-gpu-config.yaml"), "ClusterPolicy")
    assert {"effect": "NoSchedule", "key": "nvidia.com/gpu", "value": "true"} in policy["spec"]["daemonsets"]["tolerations"]
    assert policy["metadata"]["annotations"]["argocd.argoproj.io/sync-options"] == "SkipDryRunOnMissingResource=true"
    nfd = one(docs(render / "charts/nfd-config.yaml"), "NodeFeatureDiscovery", "nfd-instance")
    assert nfd["spec"]["operand"]["image"] == "registry.redhat.io/openshift4/ose-node-feature-discovery-rhel9:v4.20"
    assert nfd["metadata"]["annotations"]["argocd.argoproj.io/sync-options"] == "SkipDryRunOnMissingResource=true"


# ---- clustergroup (app of apps) ----------------------------------------------------------


@pytest.fixture(scope="module")
def cg(render) -> dict[str, list[dict]]:
    return {case: docs(render / f"clustergroup/{case}.yaml") for case in ("default", "none", "gcp")}


def imperative_jobs(items: list[dict]) -> list[str]:
    cron = one(items, "CronJob", "imperative-cronjob")
    init = cron["spec"]["jobTemplate"]["spec"]["template"]["spec"]["initContainers"]
    return [c["name"] for c in init if c["name"] not in ("fetch-ca", "git-init")]


def test_applications(cg):
    apps = {d["metadata"]["name"]: d for d in cg["default"] if d["kind"] == "Application"}
    assert set(apps) == {"vault", "openshift-external-secrets", "nfd-config", "nvidia-gpu-config",
                         "ray-serve-llm", "notebook"}
    want = {"nfd-config": "openshift-nfd", "nvidia-gpu-config": "nvidia-gpu-operator",
            "ray-serve-llm": "vllm-inference", "notebook": "vllm-notebook"}
    for name, ns in want.items():
        assert apps[name]["spec"]["source"]["path"] == f"charts/all/{name}"
        assert apps[name]["spec"]["destination"]["namespace"] == ns
        files = apps[name]["spec"]["source"]["helm"]["valueFiles"]
        assert files[:2] == ["/values-global.yaml", "/variants/prod/values-prod.yaml"]
        assert "/overrides/values-gpu-machineset-create-AWS.yaml" in files


def test_image_triggered_apps_ignore_image_drift(cg):
    apps = {d["metadata"]["name"]: d for d in cg["default"] if d["kind"] == "Application"}
    triggered = {"ray-serve-llm": ("ray-serve-llm", "server"), "notebook": ("notebook", "notebook")}
    for name, (deployment, container) in triggered.items():
        spec = apps[name]["spec"]
        ignore = {(i.get("group", ""), i["kind"]): i for i in spec["ignoreDifferences"] if not i.get("jsonPointers")
                  or i["kind"] != "Deployment"}
        # Only the image-triggered container: the oauth-proxy sidecar's image must stay compared.
        assert ignore[("apps", "Deployment")]["name"] == deployment
        assert ignore[("apps", "Deployment")]["jqPathExpressions"] == [
            f'.spec.template.spec.containers[] | select(.name == "{container}") | .image']
        assert ignore[("build.openshift.io", "BuildConfig")]["jqPathExpressions"] == [".spec.paused"]
        assert spec["syncPolicy"] == {"automated": {}, "retry": {"limit": 20},
                                      "syncOptions": ["RespectIgnoreDifferences=true"]}
    notebook_ignore = {i["kind"]: i for i in apps["notebook"]["spec"]["ignoreDifferences"]}
    assert notebook_ignore["Secret"]["jsonPointers"] == ["/data/session_secret"]


def test_shared_value_files_switch(cg, render):
    assert (render / "clustergroup/default.valuefiles").read_text().split() == [
        "values-global.yaml", "variants/prod/values-prod.yaml", "overrides/values-gpu-machineset-create-AWS.yaml"]
    for case in ("none", "gcp"):
        assert "overrides/" not in (render / f"clustergroup/{case}.valuefiles").read_text()
    prod = yaml.safe_load((REPO / "variants/prod/values-prod.yaml").read_text())
    assert prod["clusterGroup"]["sharedValueFiles"] == [
        "/overrides/values-gpu-machineset-{{ $.Values.global.gpuMachineSet }}-{{ $.Values.global.clusterPlatform }}.yaml"]
    assert GLOBAL["global"]["gpuMachineSet"] == "create"


def test_imperative_jobs(cg):
    # The override restates rebuild-on-commit because a later list replaces the earlier one.
    # rebuild-on-commit first: the CronJob stops at the first failing job, and rebuild-on-commit exits 0 on errors.
    assert imperative_jobs(cg["default"]) == ["rebuild-on-commit", "create-gpu-machineset"]
    assert imperative_jobs(cg["none"]) == ["rebuild-on-commit"]
    assert imperative_jobs(cg["gcp"]) == ["rebuild-on-commit"]
    cron = one(cg["default"], "CronJob", "imperative-cronjob")
    init = {c["name"]: c for c in cron["spec"]["jobTemplate"]["spec"]["template"]["spec"]["initContainers"]}
    machine = init["create-gpu-machineset"]["command"]
    assert machine[-1] == "rhvp.cluster_utils.create_machineset"
    for extravar in ("machineset_instance_type=g6.2xlarge", "machineset_replicas=1",
                     'machineset_taints=[{"effect":"NoSchedule","key":"nvidia.com/gpu","value":"true"}]',
                     'machineset_node_labels={"node-role.kubernetes.io/nvidia-gpu":""}'):
        assert extravar in machine
    assert init["rebuild-on-commit"]["command"][-1] == "ansible/rebuild-on-commit.yaml"
    assert init["rebuild-on-commit"]["workingDir"] == "/git/repo"
    role = one(cg["default"], "ClusterRole", "imperative-cluster-role")
    assert any("machinesets" in r["resources"] for r in role["rules"])
    default_role = one(cg["none"], "ClusterRole", "imperative-cluster-role")
    assert not any("machinesets" in r["resources"] for r in default_role["rules"])


def test_operators(cg):
    subs = {d["metadata"]["name"]: d for d in cg["default"] if d["kind"] == "Subscription"}
    assert subs["nfd"]["metadata"]["namespace"] == "openshift-nfd"
    assert subs["nfd"]["spec"]["channel"] == "stable"
    gpu = subs["gpu-operator-certified"]
    assert gpu["metadata"]["namespace"] == "nvidia-gpu-operator"
    assert (gpu["spec"]["source"], gpu["spec"]["channel"]) == ("certified-operators", "stable")
    groups = {d["metadata"]["namespace"]: d.get("spec") for d in cg["default"] if d["kind"] == "OperatorGroup"}
    assert groups["openshift-nfd"] == {"targetNamespaces": ["openshift-nfd"]}
    assert groups["nvidia-gpu-operator"] == {"targetNamespaces": ["nvidia-gpu-operator"]}
    namespaces = {d["metadata"]["name"] for d in cg["default"] if d["kind"] == "Namespace"}
    assert {"vllm-inference", "vllm-notebook", "openshift-nfd", "nvidia-gpu-operator"} <= namespaces
    # The clustergroup chart creates no namespace named after the pattern itself.
    pattern = GLOBAL["global"]["pattern"]
    assert pattern == "vllm-ray-inference-server"
    assert not any(ns == pattern or ns.startswith(f"{pattern}-") for ns in namespaces), sorted(namespaces)


# ---- rebuild-on-commit decisions (no cluster: pure function) -------------------------------


def load_rebuild_module():
    spec = importlib.util.spec_from_file_location("rebuild_on_commit", REPO / "ansible/rebuild_on_commit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(name, phase, commit="", created="2026-01-01T00:00:00Z", via_spec=False):
    revision = {"git": {"commit": commit}} if commit else None
    b = {"metadata": {"name": name, "creationTimestamp": created}, "status": {"phase": phase}, "spec": {}}
    if revision:
        (b["spec"] if via_spec else b["status"])["revision"] = revision
    return b


HEAD = "b" * 40
OLD = "a" * 40
HEAD_TIME = datetime(2026, 1, 2, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "builds, changed, expect",
    [
        ([], True, "skip"),                                                        # trigger makes the first
        ([build("x-1", "Running", OLD)], True, "skip"),                            # build in progress
        ([build("x-1", "Complete", HEAD)], True, "skip"),                          # up to date
        ([build("x-1", "Complete", OLD)], False, "skip"),                          # folder unchanged
        ([build("x-1", "Complete", OLD)], True, "start"),                          # folder changed
        ([build("x-1", "Complete", OLD)], None, "start"),                          # cannot tell: rebuild
        ([build("x-1", "Complete", OLD), build("x-2", "Failed", HEAD, "2026-01-03T00:00:00Z", True)],
         True, "skip"),                                                            # failed at HEAD: no retry
        ([build("x-1", "Failed", OLD)], False, "skip"),                            # same folder as failure
        ([build("x-1", "Failed", OLD)], True, "start"),                            # new folder after failure
        ([build("x-1", "Failed", "", "2026-01-01T00:00:00Z")], True, "start"),     # unknown, commit is newer
        ([build("x-1", "Failed", "", "2026-01-03T00:00:00Z")], True, "skip"),      # unknown, commit is older
    ],
)
def test_rebuild_decisions(builds, changed, expect):
    module = load_rebuild_module()
    action, reason = module.decide(builds, HEAD, HEAD_TIME, lambda commit: changed)
    assert action == expect, reason


# ---- rebuild-on-commit playbook, end to end with a fake `oc` --------------------------------

FAKE_OC = r"""#!/bin/sh
# Fake oc: answers from files in $FAKE_OC_STATE and logs start-build calls.
state="${FAKE_OC_STATE}"
case "$1 $2" in
  "get buildconfig")
    if [ -f "${state}/$5.$3.bc" ]; then echo "buildconfig.build.openshift.io/$3"; fi ;;
  "get builds")
    ns="$4"; bc="${6#openshift.io/build-config.name=}"
    if [ -f "${state}/${ns}.${bc}.builds.json" ]; then cat "${state}/${ns}.${bc}.builds.json"; else echo '{"items": []}'; fi ;;
  "start-build "*)
    echo "$*" >> "${state}/started.log"; echo "build.build.openshift.io/$2-9" ;;
  *) echo "fake oc: unexpected: $*" >&2; exit 3 ;;
esac
"""


def git(*args, cwd):
    subprocess.run(["git", "-c", "user.email=ci@example.com", "-c", "user.name=ci", *args],
                   cwd=cwd, check=True, capture_output=True)


def rev(cwd) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


def test_one_target_failing_does_not_stop_the_others_or_fail_the_job(tmp_path):
    # The server's start-build is Forbidden (rebuildOnCommit off in that chart); the notebook must still be checked
    # and started, and the script must exit 0 so the imperative CronJob runs its next job.
    repo = tmp_path / "repo"
    for rel in ("inference-server/Containerfile", "notebooks/Containerfile"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("v1\n")
    git("init", "-q", "-b", "main", cwd=repo)
    git("add", ".", cwd=repo)
    git("commit", "-q", "-m", "first", cwd=repo)
    head = rev(repo)
    state = tmp_path / "state"
    state.mkdir()
    for ns, bc in (("vllm-inference", "ray-serve-llm"), ("vllm-notebook", "notebook")):
        (state / f"{ns}.{bc}.bc").write_text("")
        (state / f"{ns}.{bc}.builds.json").write_text(json.dumps({"items": [build(f"{bc}-1", "Complete", "0" * 40)]}))
    oc = tmp_path / "oc"
    oc.write_text(FAKE_OC.replace('  "start-build "*)', '  "start-build ray-serve-llm")\n'
                                  '    echo "Error from server (Forbidden): cannot create" >&2; exit 1 ;;\n'
                                  '  "start-build "*)'))
    oc.chmod(0o755)
    targets = [{"namespace": "vllm-inference", "buildconfig": "ray-serve-llm", "context_dir": "inference-server"},
               {"namespace": "vllm-notebook", "buildconfig": "notebook", "context_dir": "notebooks"}]
    proc = subprocess.run([sys.executable, str(REPO / "ansible/rebuild_on_commit.py"), "--repo", str(repo),
                           "--targets", json.dumps(targets), "--oc", str(oc)],
                          env={**os.environ, "FAKE_OC_STATE": str(state)}, capture_output=True, text=True,
                          timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "ERROR vllm-inference/ray-serve-llm" in proc.stdout and "Forbidden" in proc.stdout
    assert "STARTED vllm-notebook/notebook" in proc.stdout
    assert (state / "started.log").read_text().strip() == f"start-build notebook -n vllm-notebook --commit={head} -o name"


def test_rebuild_playbook_end_to_end(tmp_path):
    engine = container_engine()
    if engine is None:
        pytest.skip("needs docker or podman")
    origin = tmp_path / "origin"
    for rel in ("ansible/rebuild-on-commit.yaml", "ansible/rebuild_on_commit.py"):
        (origin / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, origin / rel)
    for rel in ("inference-server/Containerfile", "notebooks/Containerfile", "README.md"):
        (origin / rel).parent.mkdir(parents=True, exist_ok=True)
        (origin / rel).write_text("v1\n")
    git("init", "-q", "-b", "main", cwd=origin)
    git("config", "uploadpack.allowReachableSHA1InWant", "true", cwd=origin)  # as GitHub does
    git("add", ".", cwd=origin)
    git("commit", "-q", "-m", "first", cwd=origin)
    first = rev(origin)
    (origin / "README.md").write_text("v2\n")
    git("commit", "-q", "-am", "docs only", cwd=origin)
    docs_only = rev(origin)
    (origin / "inference-server/Containerfile").write_text("v2\n")
    git("commit", "-q", "-am", "server change", cwd=origin)
    server_change = rev(origin)

    state = tmp_path / "state"
    state.mkdir()
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "oc").write_text(FAKE_OC)
    (fakebin / "oc").chmod(0o755)
    (state / "vllm-inference.ray-serve-llm.bc").write_text("")
    (state / "vllm-notebook.notebook.bc").write_text("")

    def builds_file(ns, bc, items):
        (state / f"{ns}.{bc}.builds.json").write_text(json.dumps({"items": items}))

    def run_job(checkout_commit) -> str:
        clone = tmp_path / f"clone-{checkout_commit[:7]}"
        if not clone.exists():
            # Like the imperative job: a shallow, single-branch clone, then the commit to test.
            subprocess.run(["git", "clone", "-q", "--depth", "1", "--no-single-branch",
                            f"file://{origin}", str(clone)], check=True, capture_output=True)
            git("fetch", "-q", "--depth", "1", "origin", checkout_commit, cwd=clone)
            git("checkout", "-q", checkout_commit, cwd=clone)
        log = state / "started.log"
        log.unlink(missing_ok=True)
        cmd = [engine, "run", "--rm"]
        cmd += (["--userns=keep-id", "--security-opt", "label=disable"] if engine == "podman"
                else ["--user", f"{os.getuid()}:{os.getgid()}"])
        # HOME and Ansible's temp folders: writable for an arbitrary UID, as /git/home is
        # in the imperative job.
        # The clone is made on the host, not inside the container as in the imperative job, so git in the
        # container can see a different owner and refuse it ("dubious ownership", as on GitHub's runners).
        # Trust it through command-line-level config, which git honors for safe.directory.
        cmd += ["-e", "GIT_CONFIG_COUNT=1", "-e", "GIT_CONFIG_KEY_0=safe.directory", "-e", "GIT_CONFIG_VALUE_0=*"]
        cmd += ["-e", "HOME=/tmp", "-e", "ANSIBLE_REMOTE_TMP=/tmp/ansible-remote",
                "-e", "ANSIBLE_LOCAL_TEMP=/tmp/ansible-local",
                "-e", f"FAKE_OC_STATE={state}", "-e", f"PATH={fakebin}:/usr/local/bin:/usr/bin:/bin",
                "-v", f"{tmp_path}:{tmp_path}", "-w", str(clone), "--entrypoint", "ansible-playbook",
                UTILITY_IMAGE, "ansible/rebuild-on-commit.yaml"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
        # The job exits 0 even when its oc or git commands fail, so "nothing started" alone cannot tell "nothing
        # to do" from "every command failed". Its ERROR lines can.
        assert "ERROR" not in proc.stdout, proc.stdout[-3000:] + proc.stderr[-3000:]
        return log.read_text() if log.exists() else ""

    # Both images built from the first commit; the checkout is that commit: nothing to do.
    builds_file("vllm-inference", "ray-serve-llm", [build("ray-serve-llm-1", "Complete", first)])
    builds_file("vllm-notebook", "notebook", [build("notebook-1", "Complete", first)])
    assert run_job(first) == ""
    # A docs-only commit: build folders unchanged, nothing to do.
    assert run_job(docs_only) == ""
    # A commit that changes inference-server/: only the server rebuilds, at that commit.
    started = run_job(server_change)
    assert started.strip() == f"start-build ray-serve-llm -n vllm-inference --commit={server_change} -o name"
    # While that build runs, nothing more is started.
    builds_file("vllm-inference", "ray-serve-llm", [build("ray-serve-llm-1", "Complete", first),
                                                    build("ray-serve-llm-2", "Running", server_change,
                                                          "2026-01-03T00:00:00Z", True)])
    assert run_job(server_change) == ""
    # No BuildConfig (image.override set): skipped.
    (state / "vllm-inference.ray-serve-llm.bc").unlink()
    builds_file("vllm-inference", "ray-serve-llm", [build("ray-serve-llm-1", "Complete", first)])
    assert run_job(server_change) == ""
