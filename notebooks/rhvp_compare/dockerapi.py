# SPDX-License-Identifier: Apache-2.0
"""The docker backend: one GPU machine, no cluster. For trying a comparison before it goes to OpenShift.

The runner itself runs in a container of the GuideLLM image, with this package mounted from the repo
(scripts/compare-local.sh starts it), and talks to the Docker Engine API over the mounted socket with httpx; no
docker client is needed. Every side joins the runner's user-defined network, so the runner and GuideLLM (a child
process of the runner) reach it by its container name.

Like OpenShift, each side runs as an arbitrary UID in group 0 by default (RHVP_COMPARE_USER), with its caches on a
fresh tmpfs at /cache and the model weights on a named volume at /models, kept between launches. GPU power comes from
nvidia-smi in a small container that sees the same GPU (DCGM is not there).

One run at a time: the runner container carries the label rhvp-compare.runner (compare-local.sh sets it); a run
refuses to start while an older runner container is running. A server container that normally holds the GPU can be
named in RHVP_COMPARE_SERVER_CONTAINER: the runner stops it before the first launch, once both images are pulled, and
starts it again when the run ends, however it ends (the docker counterpart of scaling the pattern's server on
OpenShift). A stop during the pulls leaves it running. One that was running when the run began but has stopped by
itself by then (exited, or removed) is left as it is; the run records that (run.server_found_down) and does not start
it.

Settings (environment variables):

- RHVP_COMPARE_NETWORK: the docker network the runner is on (required)
- RHVP_COMPARE_HOST_RESULTS: the host folder mounted at the runner's results folder (required: a side of the
  ray-serve-llm preset mounts its llm.yaml from there)
- RHVP_COMPARE_MODELS_VOLUME: the volume for /models (default rhvp-compare-models)
- RHVP_COMPARE_GPUS: "all" (default) or device ids such as "0"; give one id on a machine with several GPUs
- RHVP_COMPARE_USER: the user the sides run as (default 1000660000:0)
- RHVP_COMPARE_SHM: /dev/shm size in bytes (default 4 GiB)
- RHVP_COMPARE_MEMORY: memory limit of each side, as docker writes it (for example 32g; default none)
- RHVP_COMPARE_SERVER_CONTAINER: a container to stop for the run and start again after it (default none)
- RHVP_COMPARE_RUNNER_NAME: the runner container's own name (default: its hostname, the container id)
- DOCKER_SOCKET: the Engine API socket (default /var/run/docker.sock)
"""

from __future__ import annotations

import json
import os
import socket
import struct
import time
from pathlib import Path
from collections.abc import Mapping

import httpx

from . import energy, guidellm, logs, runner

LABEL = "rhvp-compare.run"
RUNNER_LABEL = "rhvp-compare.runner"
#: The memory limit of the small helper containers (the volume's permissions, the GPU checks, the power sampler).
HELPER_MEMORY = 256 * 2**20
#: Waits between the tries to start the server container again.
RESTORE_BACKOFF_S = (2.0, 4.0, 8.0)


def memory_bytes(text: str | None) -> int | None:
    """A memory size as docker writes it (a number, then b, k, m or g) in bytes; None for an empty text."""
    value = (text or "").strip().lower()
    if not value:
        return None
    scale = {"b": 1, "k": 2**10, "m": 2**20, "g": 2**30}
    number, unit = (value[:-1], scale[value[-1]]) if value[-1] in scale else (value, 1)
    try:
        out = int(float(number) * unit)
    except ValueError:
        out = 0
    if out < 6 * 2**20:
        raise runner.RoundError(f"RHVP_COMPARE_MEMORY is not a memory size docker takes (at least 6m): {text!r}")
    return out


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

    def logs(self, cid: str, timeout_s: float = 300.0) -> str:
        response = self.http.get(f"/containers/{cid}/logs", params={"stdout": "1", "stderr": "1",
                                                                    "timestamps": "1"},
                                 timeout=httpx.Timeout(timeout_s))
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

    def running(self, label: str) -> list[dict]:
        """The running containers that carry `label` (any value)."""
        filters = json.dumps({"label": [label], "status": ["running"]})
        return list(self._ok(self.http.get("/containers/json", params={"filters": filters}),
                             "list containers").json())

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
    """An image reference without its tag and digest: registry.example/ns/name:tag@sha256:... becomes
    registry.example/ns/name."""
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
            "HostConfig": {"NetworkMode": "none", "DeviceRequests": _gpu_request(self.gpus),
                           "Memory": HELPER_MEMORY}})
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
        self.memory = memory_bytes(env.get("RHVP_COMPARE_MEMORY"))
        self.server_container = env.get("RHVP_COMPARE_SERVER_CONTAINER", "").strip()
        self.runner_name = env.get("RHVP_COMPARE_RUNNER_NAME", "").strip() or socket.gethostname()
        self.results_root = results_root
        self.run_id = ""
        self._gpu_facts: dict | None = None
        self._server_stopped = False
        self._restored = False
        self._stop_requested = False
        #: The server container's status when prepare() came to stop it, when it was running at the first read but
        #: not any more ("exited", or "gone" once removed); the runner records it in run.server_found_down.
        self.server_found_down: str | None = None
        self.log_timeout_s, self.stop_grace_s = 300.0, 30

    restore_before_teardown = False  # no scheduler: the server would take the GPU from the side still on it

    def check_alone(self) -> None:
        """Refuse to start while an older runner container is running (the oldest wins; ties by name)."""
        def order(c: Mapping) -> tuple[int, str]:
            return int(c.get("Created") or 0), (c.get("Names") or [""])[0]

        def is_me(c: Mapping) -> bool:
            return f"/{self.runner_name}" in (c.get("Names") or []) or str(c.get("Id", "")).startswith(
                self.runner_name)

        found = self.docker.running(RUNNER_LABEL)
        mine = next((c for c in found if is_me(c)), None)
        for c in sorted(found, key=order):
            if not is_me(c) and (mine is None or order(c) < order(mine)):
                name = (c.get("Names") or ["?"])[0].lstrip("/")
                raise runner.RunActive(f"another comparison is running: runner container {name}. This run changed "
                                       f"nothing; start it again once that one has ended.")

    def on_stop(self) -> None:
        """A stop has begun: prepare() no longer stops the server, and shorter waits from now on. Flags only: the
        runner calls this from its signal handler, also while prepare() runs. The flag prepare() reads is set first,
        because the runner ignores an error raised here, and an error before that flag would let prepare() go on to
        stop the server."""
        self._stop_requested = True
        self.log_timeout_s, self.stop_grace_s = 20.0, 10

    def restore_server(self) -> bool | None:
        """Start the server container stopped in prepare() again, and check that it runs. None when none was
        stopped."""
        if not self._server_stopped:
            return True if self._restored else None
        waits, last = iter(RESTORE_BACKOFF_S), ""
        while True:
            try:
                self.docker.start(self.server_container)
                state = (self.docker.inspect(self.server_container) or {}).get("State") or {}
                if state.get("Running"):
                    self._server_stopped, self._restored = False, True
                    return True
                last = f"it is {state.get('Status') or 'gone'} after the start"
            except runner.RoundError as error:
                last = str(error)
            wait = next(waits, None)
            if wait is None:
                break
            time.sleep(wait)
        raise runner.RoundError(f"could not start container {self.server_container} again ({last}). Start it by "
                                f"hand: docker start {self.server_container}")

    # ---- before and after the run ----

    def prepare(self, plan: Mapping) -> None:
        self.run_id = plan["run_id"]
        self.docker.network(self.network)
        state = None
        if self.server_container:  # checked first, so a wrong name fails before minutes of pulls
            state = (self.docker.inspect(self.server_container) or {}).get("State")
            if state is None:
                raise runner.RoundError(f"RHVP_COMPARE_SERVER_CONTAINER names no container: {self.server_container}")
        self.docker.volume(self.models_volume)
        for ref in [s["image"] for s in plan["sides"]]:
            self.docker.pull(ref)
        # A new volume belongs to root; make it writable for the sides (an arbitrary UID in group 0).
        code, _ = self.docker.run(self._name("prepare"), {
            "Image": plan["sides"][0]["image"], "User": "0:0", "Entrypoint": ["sh", "-c"],
            "Cmd": ["mkdir -p /models/hf && chgrp 0 /models /models/hf && chmod g+rwx /models /models/hf"],
            "Labels": {LABEL: self.run_id}, "HostConfig": {"Binds": [f"{self.models_volume}:/models"],
                                                          "NetworkMode": "none", "Memory": HELPER_MEMORY}})
        if code != 0:
            raise runner.RoundError(f"could not make the models volume writable (exit {code})")
        llm = self.results_root / self.run_id / "llm.yaml"
        if any(s["preset"] == "ray-serve-llm" for s in plan["sides"]):
            from . import plan as plans

            llm.parent.mkdir(parents=True, exist_ok=True)
            llm.write_text(plans.llm_yaml(plan), encoding="utf-8")
        # The live server stops last. The pulls above can take minutes and cannot be interrupted. A stop that lands
        # during them (SIGTERM or Ctrl-C) only sets flags, so it is checked here: the server is then left running,
        # and a SIGKILL after the stop timeout cannot catch it half stopped. Its state is read again just before. A
        # server that stopped by itself during the pulls is left as it is, and recorded (server_found_down).
        if state is None or self._stop_requested:
            return
        was_running = bool(state.get("Running"))
        state = (self.docker.inspect(self.server_container) or {}).get("State") or {}
        if state.get("Running") and not self._stop_requested:
            self._server_stopped = True  # set first: a stop that fails half-way is still undone
            self.docker.stop(self.server_container, grace_s=self.stop_grace_s)
        elif was_running and not state.get("Running"):
            self.server_found_down = state.get("Status") or "gone"

    def describe(self) -> dict:
        info = self.docker.info()
        return {"node": {"arch": info.get("Architecture"), "os_image": info.get("OperatingSystem"),
                         "kernel": info.get("KernelVersion"),
                         "container_runtime": f"docker {info.get('ServerVersion')}", "cpus": info.get("NCPU"),
                         "memory_bytes": info.get("MemTotal"), "instance_type": None}}

    def cleanup(self) -> None:
        for cid in self.docker.by_label(self.run_id) if self.run_id else []:
            self.docker.remove(cid)

    # ---- one launch ----

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
                # the host's process namespace: without it nvidia-smi in a container lists no other program's GPU
                # processes (seen with the NVIDIA Container Toolkit), and the check would always pass
                "HostConfig": {"NetworkMode": "none", "PidMode": "host", "DeviceRequests": _gpu_request(self.gpus),
                               "Memory": HELPER_MEMORY}}, timeout_s=60)
            busy = [text for _, text in logs.lines(log) if text.strip()]
            if code == 0 and not busy:
                return
            if time.monotonic() > deadline:
                raise runner.RoundError("another program holds the GPU" if code == 0 else
                                        f"nvidia-smi could not read the GPU (exit {code})", runner.GPU_NOT_FREE)
            time.sleep(10)

    def start(self, plan: Mapping, launch: Mapping, side: Mapping, spec: Mapping) -> dict:
        name = self._name(f"{launch['index']:02d}-{launch['side'].replace('~', '-').lower()}")  # a twin: a-same
        env = dict(spec["env"])
        if self.hf_token:
            env["HF_TOKEN"] = self.hf_token
        binds = [f"{self.models_volume}:/models"]
        if spec.get("llm_config"):
            binds.append(f"{self.host_results}/{self.run_id}/llm.yaml:/config/llm.yaml:ro")
        host = {"NetworkMode": self.network, "Binds": binds, "ShmSize": self.shm,
                "Tmpfs": {"/cache": "rw,exec,size=20g,mode=1777", "/tmp/r": "rw,exec,mode=1777"}}
        if self.memory:
            host.update({"Memory": self.memory, "MemorySwap": self.memory})  # no swap on top
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
            raise runner.RoundError(problem, runner.SIDE_FAILED)
        return f"http://{handle['name']}:{handle['port']}"

    def alive(self, handle: Mapping) -> str | None:
        state = (self.docker.inspect(handle["id"]) or {}).get("State") or {}
        if state.get("Running"):
            return None
        if state.get("OOMKilled"):
            return "the server was killed: out of memory"
        return f"the server exited with code {state.get('ExitCode')}" if state else "the server container is gone"

    def run_load(self, plan: Mapping, launch: Mapping, target: str, out_dir: Path) -> Path:
        return guidellm.run(plan, target, out_dir)

    def sampler(self, plan: Mapping, handle: Mapping):
        if plan["energy"]["source"] != "nvidia-smi" or not handle.get("gpu"):
            return None
        return NvidiaSmi(self.docker, handle["image"], self.gpus, f"{handle['name']}-power", self.run_id)

    def logs(self, handle: Mapping) -> str:
        return self.docker.logs(handle["id"], timeout_s=self.log_timeout_s)

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
                "HostConfig": {"NetworkMode": "none", "DeviceRequests": _gpu_request(self.gpus),
                               "Memory": HELPER_MEMORY}}, timeout_s=60)
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
        self.docker.stop(handle["id"], grace_s=self.stop_grace_s)
        self.docker.remove(handle["id"])
