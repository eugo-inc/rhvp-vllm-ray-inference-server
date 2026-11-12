# SPDX-License-Identifier: Apache-2.0
"""The OpenShift backend: each round's side is a pod in the server namespace; the runner is a Job in the notebook
namespace (charts/all/notebook/templates/compare-*.yaml) and talks to the Kubernetes API with its service account
token through plain httpx (no Kubernetes client package).

A side pod runs like the pattern's own server: the restricted SCC's arbitrary UID in group 0, one GPU (request =
limit) with a toleration for the GPU taint, the CPU type's nodeSelector, the model PVC at /models, fresh emptyDirs
at /cache and /tmp/r, a memory-backed /dev/shm, and its own time limit (activeDeadlineSeconds), so a side outlives a
runner that was killed outright by one round at most.

The live server. Before the first round the runner scales the pattern's own server Deployment to 0, so the GPU and
the ReadWriteOnce model PVC are free, and when the run ends (complete, partial, stopped or failed) it scales it back
(to RHVP_COMPARE_SERVER_REPLICAS, default 1). Argo CD ignores the Deployment's replicas (variants/prod), so a sync
during the run does not bring the server back onto the GPU. The runner also waits until no other GPU pod is left in
the namespace before each round.

Progress. After every round the runner writes ConfigMap compare-<run id> in its own namespace: status.json (see
summary.py) and result.json, the whole results file while it fits.

The runner reaches a side by its pod IP (no Service). GuideLLM runs as a child process of the runner (see
guidellm.py), so its report lands straight on the results PVC.

The Hugging Face token goes into a side only by reference (secretKeyRef), and only when the Secret holds a real
token (it starts with hf_): the pattern's Secret holds "none" when no token was loaded, and some images would send
that to the Hub as a token. The runner reads the Secret to decide and never prints or stores it.

GPU power comes from DCGM through the cluster's monitoring (energy.py). Settings come from environment variables the
chart sets; the defaults match the pattern's names (AGENTS.md, "Names that must stay in sync").
"""

from __future__ import annotations

import base64
import json
import os
import ssl
import time
from pathlib import Path
from collections.abc import Callable, Mapping

import httpx

from . import energy, guidellm, runner

SERVICE_ACCOUNT = Path("/var/run/secrets/kubernetes.io/serviceaccount")
SIDE_APP = "image-compare-side"
LABEL_RUN = "rhvp-compare/run"
PART_OF = "vllm-ray-inference-server"
#: Labels of the run's status ConfigMap.
STATUS_COMPONENT = "compare-run"
STATUS_RUN_LABEL = "compare-run-id"
#: Waiting reasons that never clear by themselves.
FATAL_WAITING = ("InvalidImageName", "CreateContainerConfigError", "CreateContainerError")
#: Waiting reasons that may clear (a slow or flaky registry), up to the pending limit.
PULL_WAITING = ("ErrImagePull", "ImagePullBackOff")


class Kube:
    """The few Kubernetes API calls the backend needs. Error texts carry the HTTP status, never a URL or a token."""

    def __init__(self, base_url: str | None = None, token: str | None = None, verify=None,
                 transport: httpx.BaseTransport | None = None) -> None:
        if base_url is None:
            host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
            port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
            base_url = f"https://[{host}]:{port}" if ":" in host else f"https://{host}:{port}"
        if token is None and (SERVICE_ACCOUNT / "token").is_file():
            token = (SERVICE_ACCOUNT / "token").read_text(encoding="utf-8").strip()
        if verify is None:
            ca = SERVICE_ACCOUNT / "ca.crt"
            verify = ssl.create_default_context(cafile=str(ca)) if ca.is_file() else True
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.http = httpx.Client(base_url=base_url, headers=headers, verify=verify, timeout=30.0, transport=transport)

    def _call(self, method: str, path: str, what: str, ok_missing: bool = False, **kwargs) -> httpx.Response | None:
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise runner.RoundError(f"the Kubernetes API did not answer while trying to {what} "
                                    f"({type(error).__name__})") from None
        if response.status_code == 404 and ok_missing:
            return None
        if response.status_code >= 400:
            try:
                reason = response.json().get("message", "")
            except ValueError:
                reason = ""
            raise runner.RoundError(f"could not {what}: HTTP {response.status_code} {reason}".strip())
        return response

    def pods(self, namespace: str, selector: str | None = None) -> list[dict]:
        params = {"labelSelector": selector} if selector else {}
        response = self._call("GET", f"/api/v1/namespaces/{namespace}/pods", f"list pods in {namespace}",
                              params=params)
        return list(response.json().get("items") or [])

    def pod(self, namespace: str, name: str) -> dict | None:
        response = self._call("GET", f"/api/v1/namespaces/{namespace}/pods/{name}", f"read pod {name}",
                              ok_missing=True)
        return None if response is None else response.json()

    def create_pod(self, namespace: str, body: Mapping) -> dict:
        return self._call("POST", f"/api/v1/namespaces/{namespace}/pods", f"create pod in {namespace}",
                          json=dict(body)).json()

    def delete_pod(self, namespace: str, name: str, grace_s: int = 30) -> None:
        self._call("DELETE", f"/api/v1/namespaces/{namespace}/pods/{name}", f"delete pod {name}", ok_missing=True,
                   params={"gracePeriodSeconds": str(grace_s)})

    def log(self, namespace: str, name: str, container: str) -> str:
        response = self._call("GET", f"/api/v1/namespaces/{namespace}/pods/{name}/log", f"read the log of {name}",
                              params={"container": container, "timestamps": "true"}, timeout=300.0)
        return response.text

    def node(self, name: str) -> dict | None:
        response = self._call("GET", f"/api/v1/nodes/{name}", f"read node {name}", ok_missing=True)
        return None if response is None else response.json()

    def secret_value(self, namespace: str, name: str, key: str) -> str | None:
        response = self._call("GET", f"/api/v1/namespaces/{namespace}/secrets/{name}", f"read secret {name}",
                              ok_missing=True)
        if response is None:
            return None
        raw = (response.json().get("data") or {}).get(key)
        try:
            return base64.b64decode(raw).decode("utf-8").strip() if raw else None
        except (ValueError, UnicodeDecodeError):
            return None

    def replicas(self, namespace: str, deployment: str) -> int | None:
        response = self._call("GET", f"/apis/apps/v1/namespaces/{namespace}/deployments/{deployment}/scale",
                              f"read the scale of {deployment}", ok_missing=True)
        return None if response is None else int((response.json().get("spec") or {}).get("replicas") or 0)

    def scale(self, namespace: str, deployment: str, replicas: int) -> None:
        self._call("PATCH", f"/apis/apps/v1/namespaces/{namespace}/deployments/{deployment}/scale",
                   f"scale {deployment} to {replicas}", content=json.dumps({"spec": {"replicas": replicas}}),
                   headers={"Content-Type": "application/merge-patch+json"})

    def put_configmap(self, namespace: str, name: str, labels: Mapping[str, str], data: Mapping[str, str | None]) -> None:
        """Create the ConfigMap, or merge `data` into it (a None value removes that key)."""
        patched = self._call("PATCH", f"/api/v1/namespaces/{namespace}/configmaps/{name}", f"update ConfigMap {name}",
                             ok_missing=True, content=json.dumps({"data": dict(data)}),
                             headers={"Content-Type": "application/merge-patch+json"})
        if patched is None:
            self._call("POST", f"/api/v1/namespaces/{namespace}/configmaps", f"create ConfigMap {name}",
                       json={"apiVersion": "v1", "kind": "ConfigMap",
                             "metadata": {"name": name, "labels": dict(labels)},
                             "data": {k: v for k, v in data.items() if v is not None}})

    def cluster_version(self) -> str | None:
        response = self._call("GET", "/apis/config.openshift.io/v1/clusterversions/version",
                              "read the cluster version", ok_missing=True)
        if response is None:
            return None
        return ((response.json().get("status") or {}).get("desired") or {}).get("version")


def _condition(pod: Mapping, kind: str) -> Mapping:
    return next((c for c in (pod.get("status") or {}).get("conditions") or [] if c.get("type") == kind), {})


def _container_status(pod: Mapping, name: str = "server") -> Mapping:
    return next((c for c in (pod.get("status") or {}).get("containerStatuses") or [] if c.get("name") == name), {})


def _int(value) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def gpu_facts(labels: Mapping[str, str]) -> dict | None:
    """The GPU, from the NVIDIA GPU feature discovery labels on its node."""
    product = labels.get("nvidia.com/gpu.product")
    if not product:
        return None
    driver = labels.get("nvidia.com/cuda.driver-version.full")
    if not driver and labels.get("nvidia.com/cuda.driver.major"):
        driver = ".".join(labels.get(f"nvidia.com/cuda.driver.{p}", "") for p in ("major", "minor", "rev")).strip(".")
    cuda = labels.get("nvidia.com/cuda.runtime-version.full")
    if not cuda and labels.get("nvidia.com/cuda.runtime.major"):
        cuda = f"{labels['nvidia.com/cuda.runtime.major']}.{labels.get('nvidia.com/cuda.runtime.minor', '0')}"
    return {"product": product, "driver": driver or None, "cuda_driver": cuda or None,
            "memory_mib": _int(labels.get("nvidia.com/gpu.memory")), "count": _int(labels.get("nvidia.com/gpu.count")),
            "source": "node labels (NVIDIA GPU feature discovery)"}


def node_facts(node: Mapping) -> dict:
    labels = (node.get("metadata") or {}).get("labels") or {}
    info = (node.get("status") or {}).get("nodeInfo") or {}
    return {"name": (node.get("metadata") or {}).get("name"), "arch": labels.get("kubernetes.io/arch"),
            "instance_type": labels.get("node.kubernetes.io/instance-type"), "os_image": info.get("osImage"),
            "kernel": info.get("kernelVersion"), "container_runtime": info.get("containerRuntimeVersion")}


def image_digest(pod: Mapping) -> str | None:
    """repository@sha256:... of the side's container, as the node pulled it."""
    image_id = _container_status(pod).get("imageID") or ""
    for prefix in ("docker-pullable://", "docker://"):
        image_id = image_id.removeprefix(prefix)
    return image_id if "@sha256:" in image_id else None


def _own_namespace() -> str | None:
    path = SERVICE_ACCOUNT / "namespace"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def side_deadline(plan: Mapping, round_: Mapping) -> int:
    """A side pod's own time limit: the start, GuideLLM twice (it is retried once on a hang), the energy settle time
    and ten minutes more. The runner takes every side down itself; this only bounds a side whose runner was killed."""
    server, load = plan["server"], plan["load"]
    start = server["warmup_timeout_s"] if round_["kind"] == "warmup" else server["ready_timeout_s"]
    return int(start + 2 * load["timeout_s"] + plan["energy"]["settle_s"] + 600)


class DcgmSampler:
    """GPU power from DCGM in the cluster's monitoring, read after the load (start and stop do nothing)."""

    source = "dcgm"

    def __init__(self, thanos: energy.Thanos, pod: str, namespace: str, node: str | None) -> None:
        self.thanos, self.pod, self.namespace, self.node = thanos, pod, namespace, node

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def samples(self, start: float, end: float):
        power, counter, matched, note = energy.dcgm_samples(self.thanos, self.pod, self.namespace, self.node, start,
                                                            end)
        return power, counter, note if note else (f"series matched by {matched}" if matched else None)


class KubeBackend:
    platform = "openshift"

    def __init__(self, kube: Kube | None = None, env: Mapping[str, str] | None = None,
                 results_root: Path = Path("/results"), clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, thanos_factory=energy.Thanos) -> None:
        env = os.environ if env is None else env
        self.kube = kube or Kube()
        self.namespace = env.get("RHVP_COMPARE_NAMESPACE", "vllm-inference")
        self.arch = env.get("RHVP_COMPARE_ARCH", "amd64")
        self.models_pvc = env.get("RHVP_COMPARE_MODELS_PVC", "ray-serve-llm-models")
        self.hf_secret = env.get("RHVP_COMPARE_HF_SECRET", "huggingface-secret")
        self.hf_key = env.get("RHVP_COMPARE_HF_KEY", "hftoken")
        self.llm_configmap = env.get("RHVP_COMPARE_LLM_CONFIGMAP", "ray-serve-llm-config")
        self.gpu_resource = env.get("RHVP_COMPARE_GPU_RESOURCE", "nvidia.com/gpu")
        self.gpu_taint = env.get("RHVP_COMPARE_GPU_TAINT_KEY", "nvidia.com/gpu")
        self.shm = env.get("RHVP_COMPARE_SHM", "4Gi")
        self.cache = env.get("RHVP_COMPARE_CACHE", "20Gi")
        self.pending_s = float(env.get("RHVP_COMPARE_PENDING_SECONDS", "900"))
        self.gpu_wait_s = float(env.get("RHVP_COMPARE_GPU_WAIT_SECONDS", "900"))
        self.server_deployment = env.get("RHVP_COMPARE_SERVER_DEPLOYMENT", "ray-serve-llm")
        self.server_replicas = int(env.get("RHVP_COMPARE_SERVER_REPLICAS", "1"))
        self.status_namespace = env.get("RHVP_COMPARE_STATUS_NAMESPACE") or _own_namespace()
        try:
            self.resources = json.loads(env.get("RHVP_COMPARE_SIDE_RESOURCES") or "{}")
        except ValueError:
            raise runner.RoundError("RHVP_COMPARE_SIDE_RESOURCES is not JSON") from None
        self.results_root = results_root
        self.clock, self.sleep = clock, sleep
        self.thanos_factory = thanos_factory
        self._thanos: energy.Thanos | None = None
        self.run_id = ""
        self.use_token = False
        self._token: str | None = None
        self._scaled_down = False

    # ---- before and after the run ----

    def prepare(self, plan: Mapping) -> None:
        self.run_id = plan["run_id"]
        self._delete_run_pods()  # pods left by an earlier attempt of this run; also fails early without access
        token = self.kube.secret_value(self.namespace, self.hf_secret, self.hf_key) if self.hf_secret else None
        self.use_token = bool(token and token.startswith("hf_"))
        self._token = token if self.use_token else None
        if self.server_deployment and self.kube.replicas(self.namespace, self.server_deployment) is not None:
            self._scaled_down = True  # set first: a scale that fails half-way is still undone
            self.kube.scale(self.namespace, self.server_deployment, 0)

    def describe(self) -> dict:
        version = self.kube.cluster_version()
        return {"cluster": {"openshift_version": version}} if version else {}

    def _delete_run_pods(self) -> None:
        for pod in self.kube.pods(self.namespace, f"{LABEL_RUN}={self.run_id}"):
            self.kube.delete_pod(self.namespace, pod["metadata"]["name"], grace_s=0)

    def cleanup(self) -> None:
        """Delete the run's side pods and bring the pattern's server back; both are tried, the first error raised."""
        errors: list[Exception] = []
        if self.run_id:
            try:
                self._delete_run_pods()
            except Exception as error:
                errors.append(error)
        if self._scaled_down:
            try:
                self.kube.scale(self.namespace, self.server_deployment, self.server_replicas)
                self._scaled_down = False
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]

    def publish(self, run_id: str, status_text: str, result_text: str | None) -> None:
        if not self.status_namespace:
            return
        self.kube.put_configmap(self.status_namespace, f"compare-{run_id}",
                                {"app.kubernetes.io/component": STATUS_COMPONENT, STATUS_RUN_LABEL: run_id,
                                 "app.kubernetes.io/part-of": PART_OF},
                                {"status.json": status_text, "result.json": result_text})

    # ---- one round ----

    def _requests_gpu(self, pod: Mapping) -> bool:
        for container in (pod.get("spec") or {}).get("containers") or []:
            resources = container.get("resources") or {}
            for part in ("limits", "requests"):
                if str((resources.get(part) or {}).get(self.gpu_resource, "0")) not in ("0", ""):
                    return True
        return False

    def gpu_free(self, plan: Mapping, side: Mapping) -> None:
        """Wait until no other pod in the namespace holds or waits for a GPU (for example the pattern's own server
        while its scale-down to 0 goes through)."""
        if not side.get("gpu", True):
            return
        deadline = self.clock() + self.gpu_wait_s
        while True:
            holders = [p["metadata"]["name"] for p in self.kube.pods(self.namespace)
                       if (p.get("status") or {}).get("phase") in ("Pending", "Running") and self._requests_gpu(p)]
            if not holders:
                return
            if self.clock() > deadline:
                raise runner.RoundError(f"other pods in {self.namespace} still hold the GPU: {', '.join(holders)}",
                                        runner.GPU_NOT_FREE)
            self.sleep(10)

    def pod_body(self, plan: Mapping, round_: Mapping, side: Mapping, spec: Mapping) -> dict:
        name = f"cmp-{self.run_id[-8:]}-r{round_['index']:02d}-{round_['side'].lower()}"
        env = [{"name": k, "value": v} for k, v in spec["env"].items()]
        if self.use_token:
            env.append({"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {"name": self.hf_secret, "key": self.hf_key,
                                                                          "optional": True}}})
        resources = json.loads(json.dumps(self.resources or {}))
        tolerations = []
        if spec.get("gpu"):
            for part in ("requests", "limits"):
                resources.setdefault(part, {})[self.gpu_resource] = 1
            tolerations = [{"key": self.gpu_taint, "operator": "Exists", "effect": "NoSchedule"}]
        mounts = [{"name": "models", "mountPath": "/models"}, {"name": "cache", "mountPath": "/cache"},
                  {"name": "ray-tmp", "mountPath": "/tmp/r"}, {"name": "dshm", "mountPath": "/dev/shm"}]
        volumes = [{"name": "models", "persistentVolumeClaim": {"claimName": self.models_pvc}},
                   {"name": "cache", "emptyDir": {"sizeLimit": self.cache}},
                   {"name": "ray-tmp", "emptyDir": {}},
                   {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": self.shm}}]
        if spec.get("llm_config"):
            mounts.append({"name": "config", "mountPath": "/config", "readOnly": True})
            volumes.append({"name": "config", "configMap": {"name": self.llm_configmap}})
        container: dict = {
            "name": "server", "image": side["image"], "imagePullPolicy": "IfNotPresent", "args": list(spec["args"]),
            "env": env, "ports": [{"name": "http", "containerPort": plan["server"]["port"]}],
            "resources": resources, "volumeMounts": mounts,
            "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
        }
        if spec.get("command"):
            container["command"] = list(spec["command"])
        return {
            "apiVersion": "v1", "kind": "Pod",
            "metadata": {"name": name, "labels": {
                "app.kubernetes.io/name": SIDE_APP, "app.kubernetes.io/part-of": PART_OF, LABEL_RUN: self.run_id,
                "rhvp-compare/side": round_["side"], "rhvp-compare/round": str(round_["index"])}},
            "spec": {
                "restartPolicy": "Never", "terminationGracePeriodSeconds": 30, "automountServiceAccountToken": False,
                "activeDeadlineSeconds": side_deadline(plan, round_),
                "enableServiceLinks": False,
                "securityContext": {"runAsNonRoot": True, "seccompProfile": {"type": "RuntimeDefault"}},
                "nodeSelector": {"kubernetes.io/arch": self.arch}, "tolerations": tolerations,
                "containers": [container], "volumes": volumes,
            },
        }

    def start(self, plan: Mapping, round_: Mapping, side: Mapping, spec: Mapping) -> dict:
        body = self.pod_body(plan, round_, side, spec)
        name = body["metadata"]["name"]
        if self.kube.pod(self.namespace, name) is not None:  # left from an earlier attempt
            self._delete_and_wait(name)
        self.kube.create_pod(self.namespace, body)
        return {"name": name, "image": side["image"], "port": plan["server"]["port"], "gpu": bool(spec.get("gpu")),
                "created": self.clock(), "pull_since": None, "last": None, "last_at": -1e9}

    def _pod(self, handle: dict, fresh: bool = False) -> dict | None:
        """The side's pod, read at most every 2 seconds (the probes ask twice a second)."""
        if fresh or self.clock() - handle["last_at"] >= 2.0:
            handle["last"] = self.kube.pod(self.namespace, handle["name"])
            handle["last_at"] = self.clock()
        return handle["last"]

    def _problem(self, handle: dict, pod: Mapping | None) -> str | None:
        if pod is None:
            return "the side's pod is gone"
        status = pod.get("status") or {}
        phase = status.get("phase")
        state = _container_status(pod).get("state") or {}
        if "terminated" in state:
            ended = state["terminated"]
            return f"the server exited: {ended.get('reason') or 'Error'}, exit code {ended.get('exitCode')}"
        if phase in ("Failed", "Succeeded"):
            return f"the side's pod ended ({phase}: {status.get('reason') or status.get('message') or 'no reason'})"
        waiting = state.get("waiting") or {}
        reason = waiting.get("reason")
        if reason in FATAL_WAITING:
            return f"the server could not start: {reason}: {waiting.get('message', '')}".strip()
        if reason in PULL_WAITING:
            handle["pull_since"] = handle["pull_since"] or self.clock()
            if self.clock() - handle["pull_since"] > self.pending_s:
                return f"the image could not be pulled: {waiting.get('message') or reason}"
        else:
            handle["pull_since"] = None
        scheduled = _condition(pod, "PodScheduled")
        if scheduled.get("status") == "False" and self.clock() - handle["created"] > self.pending_s:
            return f"the side never got a node in {self.pending_s:g} s: {scheduled.get('message') or 'unschedulable'}"
        return None

    def endpoint(self, handle: dict, deadline: float) -> str:
        while True:
            pod = self._pod(handle, fresh=True)
            problem = self._problem(handle, pod)
            if problem:
                raise runner.RoundError(problem, runner.SIDE_FAILED)
            ip = (pod.get("status") or {}).get("podIP")
            if ip:
                return f"http://[{ip}]:{handle['port']}" if ":" in ip else f"http://{ip}:{handle['port']}"
            if self.clock() > deadline:
                raise runner.RoundError("the side's pod got no IP address before the time limit",
                                        runner.SIDE_NEVER_READY)
            self.sleep(1.0)

    def alive(self, handle: dict) -> str | None:
        return self._problem(handle, self._pod(handle))

    def run_load(self, plan: Mapping, round_: Mapping, target: str, out_dir: Path) -> Path:
        # GuideLLM downloads the model's tokenizer, which needs the token for a gated model (never "none")
        return guidellm.run(plan, target, out_dir, env={"HF_TOKEN": self._token} if self._token else None)

    def sampler(self, plan: Mapping, handle: dict):
        if plan["energy"]["source"] != "dcgm" or not handle.get("gpu"):
            return None
        if self._thanos is None:
            self._thanos = self.thanos_factory(plan["energy"]["thanos_url"])
        pod = self._pod(handle, fresh=True) or {}
        return DcgmSampler(self._thanos, handle["name"], self.namespace, (pod.get("spec") or {}).get("nodeName"))

    def logs(self, handle: dict) -> str:
        return self.kube.log(self.namespace, handle["name"], "server")

    def facts(self, handle: dict) -> dict:
        pod = self._pod(handle, fresh=True) or {}
        node_name = (pod.get("spec") or {}).get("nodeName")
        out: dict = {"node_name": node_name, "image_digest": image_digest(pod)}
        if node_name:
            node = self.kube.node(node_name)
            if node:
                out["node"] = node_facts(node)
                if handle.get("gpu"):
                    out["gpu"] = gpu_facts((node.get("metadata") or {}).get("labels") or {})
        return out

    def _delete_and_wait(self, name: str, wait_s: float = 180.0) -> None:
        self.kube.delete_pod(self.namespace, name)
        deadline = self.clock() + wait_s
        while self.kube.pod(self.namespace, name) is not None:
            if self.clock() > deadline:
                raise runner.RoundError(f"pod {name} was not gone {wait_s:g} s after it was deleted")
            self.sleep(2.0)

    def stop(self, handle: dict) -> None:
        """Delete the side and wait until it is gone, so the GPU and the model PVC are free for the next round."""
        self._delete_and_wait(handle["name"])
