# SPDX-License-Identifier: Apache-2.0
"""The docker backend: one GPU machine, no cluster. For trying a comparison before it goes to OpenShift.

The runner itself runs in a container of the GuideLLM image, with this package mounted from the repo
(scripts/compare-local.sh starts it), and talks to the Docker Engine API over the mounted socket with httpx; no
docker client is needed. Every side joins the runner's user-defined network, so the runner and GuideLLM (a child
process of the runner) reach it by its container name.

Like OpenShift, each side runs as an arbitrary UID in group 0 by default (RHVP_COMPARE_USER), with its caches on a
fresh tmpfs at /cache and the model weights on a named volume at /models, kept between rounds. GPU power comes from
nvidia-smi in a small container that sees the same GPU (DCGM is not there).

Settings (environment variables):

- RHVP_COMPARE_NETWORK: the docker network the runner is on (required)
- RHVP_COMPARE_HOST_RESULTS: the host folder mounted at the runner's results folder (required: a side of the
  ray-serve-llm preset mounts its llm.yaml from there)
- RHVP_COMPARE_MODELS_VOLUME: the volume for /models (default rhvp-compare-models)
- RHVP_COMPARE_GPUS: "all" (default) or device ids such as "0"; give one id on a machine with several GPUs
- RHVP_COMPARE_USER: the user the sides run as (default 1000660000:0)
- RHVP_COMPARE_SHM: /dev/shm size in bytes (default 4 GiB)
- DOCKER_SOCKET: the Engine API socket (default /var/run/docker.sock)
"""

from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path
from collections.abc import Mapping

import httpx

from . import energy, guidellm, logs, runner

LABEL = "rhvp-compare.run"


def demux(raw: bytes) -> str:
    """The text of a non-TTY container log stream (8-byte frame headers: stream, 3 zero bytes, payload size)."""
    out, at = [], 0
    while at + 8 <= len(raw) and raw[at] in (0, 1, 2) and raw[at + 1:at + 4] == b"\0\0\0":
        size = struct.unpack(">I", raw[at + 4:at + 8])[0]
        out.append(raw[at + 8:at + 8 + size])
        at += 8 + size
    if not out:  # not framed (a TTY container): the bytes are the text
        return raw.decode("utf-8", "replace")
    return b"".join(out).decode("utf-8", "replace")


class Docker:
    """The few Engine API calls the backend needs."""

    def __init__(self, socket: str | None = None, transport: httpx.BaseTransport | None = None) -> None:
        transport = transport or httpx.HTTPTransport(uds=socket or os.environ.get("DOCKER_SOCKET",
                                                                                   "/var/run/docker.sock"))
        self.http = httpx.Client(base_url="http://docker", transport=transport, timeout=httpx.Timeout(120.0))

    def _ok(self, response: httpx.Response, what: str) -> httpx.Response:
        if response.status_code >= 400:
            try:
                message = response.json().get("message", "")
            except ValueError:
                message = response.text[:200]
            raise runner.RoundError(f"docker could not {what}: HTTP {response.status_code} {message}".strip())
        return response

    def info(self) -> dict:
        return self._ok(self.http.get("/info"), "describe itself").json()

    def image(self, ref: str) -> dict | None:
        response = self.http.get(f"/images/{ref}/json")
        return None if response.status_code == 404 else self._ok(response, f"inspect {ref}").json()

    def pull(self, ref: str) -> None:
        if self.image(ref) is not None:
            return
        with self.http.stream("POST", "/images/create", params={"fromImage": ref},
                              timeout=httpx.Timeout(3600.0, connect=30.0)) as response:
            self._ok(response, f"pull {ref}")
            for line in response.iter_lines():
                if '"error"' in line:
                    raise runner.RoundError(f"docker could not pull {ref}: {json.loads(line).get('error')}")
        if self.image(ref) is None:
            raise runner.RoundError(f"docker did not pull {ref}")

    def create(self, name: str, body: Mapping) -> str:
        self.http.delete(f"/containers/{name}", params={"force": "1"})
        response = self.http.post("/containers/create", params={"name": name}, json=dict(body))
        return self._ok(response, f"create {name}").json()["Id"]

    def start(self, cid: str) -> None:
        self._ok(self.http.post(f"/containers/{cid}/start"), "start a container")

    def inspect(self, cid: str) -> dict | None:
        response = self.http.get(f"/containers/{cid}/json")
        return None if response.status_code == 404 else self._ok(response, "inspect a container").json()

    def logs(self, cid: str) -> str:
        response = self.http.get(f"/containers/{cid}/logs", params={"stdout": "1", "stderr": "1",
                                                                    "timestamps": "1"},
                                 timeout=httpx.Timeout(300.0))
        return demux(self._ok(response, "read a log").content)

    def wait(self, cid: str, timeout_s: float) -> int:
        try:
            response = self.http.post(f"/containers/{cid}/wait", timeout=httpx.Timeout(timeout_s, connect=30.0))
        except httpx.TimeoutException:
            raise runner.RoundError(f"a container did not finish within {timeout_s:g} s") from None
        return int(self._ok(response, "wait for a container").json().get("StatusCode", -1))

    def stop(self, cid: str, grace_s: int = 30) -> None:
        self.http.post(f"/containers/{cid}/stop", params={"t": str(grace_s)}, timeout=httpx.Timeout(grace_s + 60))

    def remove(self, cid: str) -> None:
        self.http.delete(f"/containers/{cid}", params={"force": "1", "v": "1"})

    def by_label(self, value: str) -> list[str]:
        filters = json.dumps({"label": [f"{LABEL}={value}"]})
        response = self.http.get("/containers/json", params={"all": "1", "filters": filters})
        return [c["Id"] for c in self._ok(response, "list containers").json()]

    def volume(self, name: str) -> None:
        if self.http.get(f"/volumes/{name}").status_code == 404:
            self._ok(self.http.post("/volumes/create", json={"Name": name}), f"create volume {name}")

    def network(self, name: str) -> None:
        self._ok(self.http.get(f"/networks/{name}"), f"find network {name}")

    def run(self, name: str, body: Mapping, timeout_s: float = 300.0,
            log_to: Path | None = None) -> tuple[int, str]:
        """Create, start, wait, read the log, remove: (exit code, log). With `log_to`, the log is also written
        there, even when the container did not finish in time."""
        cid = self.create(name, body)
        log = ""
        try:
            self.start(cid)
            code = self.wait(cid, timeout_s)
            log = self.logs(cid)
            return code, log
        finally:
            if log_to is not None:
                try:
                    log_to.write_text(log or self.logs(cid), encoding="utf-8")
                except Exception:  # the log is a help, not a result
                    pass
            self.remove(cid)


def repository(ref: str) -> str:
    """An image reference without its tag and digest: registry.example/ns/name:tag@sha256:... -> registry.example/ns/name."""
    name = ref.split("@", 1)[0]
    return name.rsplit(":", 1)[0] if ":" in name.rsplit("/", 1)[-1] else name


def _gpu_request(gpus: str) -> list[dict]:
    if gpus.strip().lower() == "all":
        return [{"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}]
    return [{"Driver": "nvidia", "DeviceIDs": [g.strip() for g in gpus.split(",") if g.strip()],
             "Capabilities": [["gpu"]]}]


class NvidiaSmi:
    """Power samples from nvidia-smi in a container that sees the side's GPU, every 500 ms."""

    source = "nvidia-smi"

    def __init__(self, docker: Docker, image: str, gpus: str, name: str, run_id: str) -> None:
        self.docker, self.image, self.gpus, self.name, self.run_id = docker, image, gpus, name, run_id
        self.cid: str | None = None
        self.log = ""

    def start(self) -> None:
        self.cid = self.docker.create(self.name, {
            "Image": self.image, "Entrypoint": ["nvidia-smi"],
            "Cmd": ["-i", "0", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-lms", "500"],
            "Labels": {LABEL: self.run_id},
            "HostConfig": {"NetworkMode": "none", "DeviceRequests": _gpu_request(self.gpus)}})
        self.docker.start(self.cid)

    def stop(self) -> None:
        if self.cid is None:
            return
        try:
            self.docker.stop(self.cid, grace_s=2)
            self.log = self.docker.logs(self.cid)
        finally:
            self.docker.remove(self.cid)
            self.cid = None

    def samples(self, start: float, end: float):
        return energy.parse_nvidia_smi(self.log), [], None if self.log else "nvidia-smi wrote nothing"


class DockerBackend:
    platform = "docker"

    def __init__(self, docker: Docker | None = None, env: Mapping[str, str] | None = None,
                 results_root: Path = Path("/results")) -> None:
        env = os.environ if env is None else env
        self.docker = docker or Docker()
        self.network = env.get("RHVP_COMPARE_NETWORK", "")
        self.host_results = env.get("RHVP_COMPARE_HOST_RESULTS", "")
        if not self.network or not self.host_results:
            raise runner.RoundError("set RHVP_COMPARE_NETWORK and RHVP_COMPARE_HOST_RESULTS (see "
                                    "scripts/compare-local.sh)")
        self.models_volume = env.get("RHVP_COMPARE_MODELS_VOLUME", "rhvp-compare-models")
        self.gpus = env.get("RHVP_COMPARE_GPUS", "all")
        self.user = env.get("RHVP_COMPARE_USER", "1000660000:0")
        self.shm = int(env.get("RHVP_COMPARE_SHM", str(4 * 2**30)))
        self.hf_token = env.get("HF_TOKEN", "") if env.get("HF_TOKEN", "").startswith("hf_") else ""
        self.results_root = results_root
        self.run_id = ""
        self._gpu_facts: dict | None = None

    # ---- before and after the run ----

    def prepare(self, plan: Mapping) -> None:
        self.run_id = plan["run_id"]
        self.docker.network(self.network)
        self.docker.volume(self.models_volume)
        for ref in [s["image"] for s in plan["sides"]]:
            self.docker.pull(ref)
        # A new volume belongs to root; make it writable for the sides (an arbitrary UID in group 0).
        code, _ = self.docker.run(self._name("prepare"), {
            "Image": plan["sides"][0]["image"], "User": "0:0", "Entrypoint": ["sh", "-c"],
            "Cmd": ["mkdir -p /models/hf && chgrp 0 /models /models/hf && chmod g+rwx /models /models/hf"],
            "Labels": {LABEL: self.run_id}, "HostConfig": {"Binds": [f"{self.models_volume}:/models"],
                                                          "NetworkMode": "none"}})
        if code != 0:
            raise runner.RoundError(f"could not make the models volume writable (exit {code})")
        llm = self.results_root / self.run_id / "llm.yaml"
        if any(s["preset"] == "ray-serve-llm" for s in plan["sides"]):
            from . import plan as plans

            llm.parent.mkdir(parents=True, exist_ok=True)
            llm.write_text(plans.llm_yaml(plan), encoding="utf-8")

    def describe(self) -> dict:
        info = self.docker.info()
        return {"node": {"arch": info.get("Architecture"), "os_image": info.get("OperatingSystem"),
                         "kernel": info.get("KernelVersion"), "container_runtime": f"docker {info.get('ServerVersion')}",
                         "cpus": info.get("NCPU"), "memory_bytes": info.get("MemTotal"), "instance_type": None}}

    def cleanup(self) -> None:
        for cid in self.docker.by_label(self.run_id) if self.run_id else []:
            self.docker.remove(cid)

    # ---- one round ----

    def _name(self, what: str) -> str:
        return f"rhvp-cmp-{self.run_id[-8:]}-{what}".lower()

    def gpu_free(self, plan: Mapping, side: Mapping, wait_s: float = 600.0) -> None:
        if not side.get("gpu", True):
            return
        deadline = time.monotonic() + wait_s
        while True:
            code, log = self.docker.run(self._name("gpu-check"), {
                "Image": side["image"], "Entrypoint": ["nvidia-smi"],
                "Cmd": ["--query-compute-apps=pid", "--format=csv,noheader"], "Labels": {LABEL: self.run_id},
                "HostConfig": {"NetworkMode": "none", "DeviceRequests": _gpu_request(self.gpus)}}, timeout_s=60)
            busy = [text for _, text in logs.lines(log) if text.strip()]
            if code == 0 and not busy:
                return
            if time.monotonic() > deadline:
                raise runner.RoundError("another program holds the GPU" if code == 0 else
                                        f"nvidia-smi could not read the GPU (exit {code})")
            time.sleep(10)

    def start(self, plan: Mapping, round_: Mapping, side: Mapping, spec: Mapping) -> dict:
        name = self._name(f"r{round_['index']:02d}-{round_['side'].lower()}")
        env = dict(spec["env"])
        if self.hf_token:
            env["HF_TOKEN"] = self.hf_token
        binds = [f"{self.models_volume}:/models"]
        if spec.get("llm_config"):
            binds.append(f"{self.host_results}/{self.run_id}/llm.yaml:/config/llm.yaml:ro")
        host = {"NetworkMode": self.network, "Binds": binds, "ShmSize": self.shm,
                "Tmpfs": {"/cache": "rw,exec,size=20g,mode=1777", "/tmp/r": "rw,exec,mode=1777"}}
        if spec.get("gpu"):
            host["DeviceRequests"] = _gpu_request(self.gpus)
        body: dict = {"Image": side["image"], "Cmd": list(spec["args"]), "User": self.user,
                      "Env": [f"{k}={v}" for k, v in env.items()], "Labels": {LABEL: self.run_id},
                      "ExposedPorts": {f"{plan['server']['port']}/tcp": {}}, "HostConfig": host}
        if spec.get("command"):
            body["Entrypoint"] = list(spec["command"])
        cid = self.docker.create(name, body)
        self.docker.start(cid)
        return {"id": cid, "name": name, "image": side["image"], "port": plan["server"]["port"],
                "gpu": spec.get("gpu", True)}

    def endpoint(self, handle: Mapping, deadline: float) -> str:
        problem = self.alive(handle)
        if problem:
            raise runner.RoundError(problem)
        return f"http://{handle['name']}:{handle['port']}"

    def alive(self, handle: Mapping) -> str | None:
        state = (self.docker.inspect(handle["id"]) or {}).get("State") or {}
        if state.get("Running"):
            return None
        if state.get("OOMKilled"):
            return "the server was killed: out of memory"
        return f"the server exited with code {state.get('ExitCode')}" if state else "the server container is gone"

    def run_load(self, plan: Mapping, round_: Mapping, target: str, out_dir: Path) -> Path:
        return guidellm.run(plan, target, out_dir)

    def sampler(self, plan: Mapping, handle: Mapping):
        if plan["energy"]["source"] != "nvidia-smi" or not handle.get("gpu"):
            return None
        return NvidiaSmi(self.docker, handle["image"], self.gpus, f"{handle['name']}-power", self.run_id)

    def logs(self, handle: Mapping) -> str:
        return self.docker.logs(handle["id"])

    def facts(self, handle: Mapping) -> dict:
        ref = handle["image"]
        if "@sha256:" in ref:  # pinned by digest: that is the digest
            digest = f"{repository(ref)}@{ref.split('@', 1)[1]}"
        else:
            digests = (self.docker.image(ref) or {}).get("RepoDigests") or []
            digest = next((d for d in digests if repository(ref).endswith(d.split("@", 1)[0])),
                          digests[0] if digests else None)
        out = {"node_name": None, "image_digest": digest}
        if handle.get("gpu"):
            out["gpu"] = self._gpu(handle["image"])
        return out

    def _gpu(self, image: str) -> dict | None:
        if self._gpu_facts is None:
            code, log = self.docker.run(self._name("gpu-facts"), {
                "Image": image, "Entrypoint": ["nvidia-smi"],
                "Cmd": ["--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
                "Labels": {LABEL: self.run_id},
                "HostConfig": {"NetworkMode": "none", "DeviceRequests": _gpu_request(self.gpus)}}, timeout_s=60)
            rows = [text for _, text in logs.lines(log) if text.strip()]
            if code == 0 and rows:
                first = [x.strip() for x in rows[0].split(",")]
                memory = int(first[2]) if len(first) > 2 and first[2].isdigit() else None
                self._gpu_facts = {"product": first[0], "driver": first[1] if len(first) > 1 else None,
                                   "memory_mib": memory, "count": len(rows), "source": "nvidia-smi"}
            else:
                self._gpu_facts = {}
        return self._gpu_facts or None

    def stop(self, handle: Mapping) -> None:
        self.docker.stop(handle["id"], grace_s=30)
        self.docker.remove(handle["id"])
