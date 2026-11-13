# SPDX-License-Identifier: Apache-2.0
"""The docker backend: one GPU machine, no cluster. For trying a comparison before it goes to OpenShift.

The runner itself runs in a container of the GuideLLM image, with this package mounted from the repo
(scripts/compare-local.sh starts it), and talks to the Docker Engine API over the mounted socket with httpx; no
docker client is needed. Every side joins the runner's user-defined network, so the runner and GuideLLM (a child
process of the runner) reach it by its container name.

Like OpenShift, each side runs as an arbitrary UID in group 0 by default (RHVP_COMPARE_USER), with its caches on a
fresh tmpfs at /cache and the model weights on a named volume at /models, kept between rounds. GPU power comes from
nvidia-smi in a small container that sees the same GPU (DCGM is not there).

One run at a time: the runner container carries the label rhvp-compare.runner (compare-local.sh sets it); a run
refuses to start while an older runner container is running. A server container that normally holds the GPU can be
named in RHVP_COMPARE_SERVER_CONTAINER: the runner stops it before the first round, once both images are pulled, and
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
- RHVP_COMPARE_REPLICA_GPUS: in mode scaleout, one device id per replica, comma-separated ("0,1": the first
  replica on GPU 0, the second on GPU 1); empty (the default): every replica gets RHVP_COMPARE_GPUS
- RHVP_COMPARE_CPUSET: CPUs every container of the run may use, as docker writes them (for example 0-3; default
  all)
- RHVP_COMPARE_CACHE_VOLUME: the name prefix of the compile-cache volumes (default rhvp-compare-cache)
- DOCKER_SOCKET: the Engine API socket (default /var/run/docker.sock)

A side with `compileCache: warm` gets its own volume at /cache for the whole run (`<prefix>-<last 8 characters of
the run id>-<a or b>`) in place of the tmpfs, so its warm-up round fills the compile caches its scored rounds find;
the volumes are removed when the run ends. In mode scaleout both replicas of a round share it.

The scale round (mode scaleout): the second replica is a container named like the first plus `-2`; the router
(router.py) and GuideLLM run as child processes of the runner, GuideLLM against the router. Where each replica runs is
set by RHVP_COMPARE_REPLICA_GPUS (else RHVP_COMPARE_GPUS), and it must agree with the plan's scaleOut.sharedGpu
(`check_plan`, before anything changes): sharedGpu true needs both replicas on the same GPUs, false needs a GPU id of
its own for each. prepare() then reads, with nvidia-smi, the GPUs each replica's device request reaches (a request for
a GPU this machine lacks fails there, before the first round, and so do two requests that reach the same GPU when
sharedGpu is false, such as an index and that GPU's UUID), and every launch record keeps the request (`gpus`) and
those GPUs' UUIDs (`gpu_uuids`), which results.md and the notebook show.

Launch records (`launch_facts`): the environment of the side's largest process (by resident memory, which is the
server even when an entrypoint script or an init process is PID 1), read with docker exec once the server has
answered; "container" (the configured environment: the image's ENV plus the runner's) when that cannot be read.
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

    def exec(self, cid: str, cmd: list[str], timeout_s: float = 30.0) -> tuple[int, str]:
        """Run `cmd` in a running container: (exit code, stdout and stderr)."""
        created = self._ok(self.http.post(f"/containers/{cid}/exec", json={
            "Cmd": cmd, "AttachStdout": True, "AttachStderr": True, "Tty": False}), "start an exec").json()
        response = self.http.post(f"/exec/{created['Id']}/start", json={"Detach": False, "Tty": False},
                                  timeout=httpx.Timeout(timeout_s, connect=30.0))
        text = demux(self._ok(response, "run an exec").content)
        code = self._ok(self.http.get(f"/exec/{created['Id']}/json"), "inspect an exec").json().get("ExitCode")
        return (int(code) if code is not None else -1), text

    def remove_volume(self, name: str) -> None:
        response = self.http.delete(f"/volumes/{name}", params={"force": "1"})
        if response.status_code not in (204, 404):
            self._ok(response, f"remove volume {name}")

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


#: Prints the environment (NUL-separated) of the container's process with the most resident memory, among those
#: whose environment it may read.
SERVER_ENVIRON = (
    'best=1; most=0; for d in /proc/[0-9]*; do [ -r "$d/environ" ] || continue; rss=0; '
    'while read -r key value rest; do [ "$key" = "VmRSS:" ] && rss=$value; done 2>/dev/null < "$d/status"; '
    'if [ "$rss" -gt "$most" ] 2>/dev/null; then most=$rss; best=${d#/proc/}; fi; done; '
    'cat "/proc/$best/environ"')


def environ(text: str) -> dict[str, str]:
    """NAME=value entries, NUL-separated (/proc/<pid>/environ), as a map."""
    out = {}
    for item in text.split("\0"):
        name, sep, value = item.partition("=")
        if sep and name:
            out[name] = value
    return out


def _gpu_request(gpus: str) -> list[dict]:
    if gpus.strip().lower() == "all":
        return [{"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}]
    return [{"Driver": "nvidia", "DeviceIDs": [g.strip() for g in gpus.split(",") if g.strip()],
             "Capabilities": [["gpu"]]}]


class NvidiaSmi:
    """Power samples from nvidia-smi in a container that sees the side's GPU, every 500 ms."""

    source = "nvidia-smi"

    def __init__(self, docker: Docker, image: str, gpus: str, name: str, run_id: str, cpuset: str = "") -> None:
        self.docker, self.image, self.gpus, self.name, self.run_id = docker, image, gpus, name, run_id
        self.cpuset = cpuset
        self.cid: str | None = None
        self.log = ""

    def start(self) -> None:
        self.cid = self.docker.create(self.name, {
            "Image": self.image, "Entrypoint": ["nvidia-smi"],
            "Cmd": ["-i", "0", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-lms", "500"],
            "Labels": {LABEL: self.run_id},
            "HostConfig": {"NetworkMode": "none", "DeviceRequests": _gpu_request(self.gpus),
                           "Memory": HELPER_MEMORY, **({"CpusetCpus": self.cpuset} if self.cpuset else {})}})
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
        self.replica_gpus = [g.strip() for g in env.get("RHVP_COMPARE_REPLICA_GPUS", "").split(",") if g.strip()]
        self.cpuset = env.get("RHVP_COMPARE_CPUSET", "").strip()
        self.cache_prefix = env.get("RHVP_COMPARE_CACHE_VOLUME", "").strip() or "rhvp-compare-cache"
        self.cache_volumes: dict[str, str] = {}
        self.results_root = results_root
        self.run_id = ""
        self._gpu_facts: dict | None = None
        self._gpu_uuids: dict[str, list[str]] = {}  # device request -> the UUIDs nvidia-smi saw through it
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

    def check_plan(self, plan: Mapping) -> None:
        """Refuse (PlanError, before anything changes) a scaleout plan whose scaleOut.sharedGpu does not match where
        the replicas would run: sharedGpu true needs both replicas on the same GPUs, false a GPU of its own for each
        (RHVP_COMPARE_REPLICA_GPUS)."""
        from . import plan as plans

        scale = plan.get("scale_out") if plan.get("mode") == plans.SCALEOUT else None
        if not scale or not any(s.get("gpu", True) for s in plan["sides"]):
            return
        if self.replica_gpus and len(self.replica_gpus) < 2:
            raise plans.PlanError(f"RHVP_COMPARE_REPLICA_GPUS names {len(self.replica_gpus)} GPU; mode scaleout "
                                  "starts two replicas, so it needs one GPU id per replica (such as 0,1), or none")
        first, second = self._gpus(1), self._gpus(2)
        if scale["shared_gpu"] and first != second:
            raise plans.PlanError(
                f"scaleOut.sharedGpu is true (both replicas of a side on one GPU), but RHVP_COMPARE_REPLICA_GPUS "
                f"gives replica 1 GPU {first} and replica 2 GPU {second}. Set scaleOut.sharedGpu: false (each "
                "replica then gets the full server.gpuMemoryUtilization of its own GPU), or leave REPLICA_GPUS empty")
        if not scale["shared_gpu"] and (first == second or "all" in (first.lower(), second.lower())):
            raise plans.PlanError(
                f"scaleOut.sharedGpu is false (each replica on its own GPU), but both replicas would get GPUs "
                f"{first if first == second else first + ' and ' + second}, and a vLLM server uses the first GPU it "
                "sees: set REPLICA_GPUS (scripts/compare-local.sh) to a different GPU id for each replica, such as "
                "0,1, or set scaleOut.sharedGpu: true")

    def _probe_replica_gpus(self, plan: Mapping) -> None:
        """Mode scaleout: read the GPUs each replica's device request reaches (index and UUID, with nvidia-smi), so
        a request this machine cannot meet fails before the first round, and the launch records name the GPUs."""
        from . import plan as plans

        gpu_side = next((s for s in plan["sides"] if s.get("gpu", True)), None)
        if plan.get("mode") != plans.SCALEOUT or gpu_side is None:
            return
        for replica in (1, 2):
            request = self._gpus(replica)
            if request in self._gpu_uuids:
                continue
            what = f"the GPUs of replica {replica} ({'REPLICA_GPUS' if self.replica_gpus else 'GPUS'} {request})"
            try:
                code, log = self.docker.run(self._name(f"gpu-probe-{replica}"), {
                    "Image": gpu_side["image"], "Entrypoint": ["nvidia-smi"],
                    "Cmd": ["--query-gpu=index,uuid", "--format=csv,noheader"], "Labels": {LABEL: self.run_id},
                    "HostConfig": self._host({"NetworkMode": "none", "DeviceRequests": _gpu_request(request),
                                              "Memory": HELPER_MEMORY})}, timeout_s=60)
            except runner.RoundError as error:
                raise runner.RoundError(f"{what} could not be reached: {error}. Check GPUS and REPLICA_GPUS against "
                                        "this machine's GPUs") from None
            rows = [text.split(",") for _, text in logs.lines(log) if text.strip()]
            if code != 0 or not rows:
                raise runner.RoundError(f"{what} could not be read with nvidia-smi (exit {code}). Check GPUS and "
                                        "REPLICA_GPUS against this machine's GPUs")
            self._gpu_uuids[request] = [row[1].strip() for row in rows if len(row) > 1 and row[1].strip()]
        same = set(self._gpu_uuids[self._gpus(1)]) & set(self._gpu_uuids[self._gpus(2)])
        if not plan["scale_out"]["shared_gpu"] and same:  # two strings for one GPU, such as an index and its UUID
            raise runner.RoundError(
                f"scaleOut.sharedGpu is false (each replica on its own GPU), but REPLICA_GPUS {self._gpus(1)} and "
                f"{self._gpus(2)} both reach GPU {', '.join(sorted(same))}: give each replica a different GPU, or set "
                "scaleOut.sharedGpu: true")

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
        self.check_plan(plan)  # main_run asks first (PLAN_INVALID); this keeps a direct caller from skipping it
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
            "Labels": {LABEL: self.run_id}, "HostConfig": self._host({"Binds": [f"{self.models_volume}:/models"],
                                                                     "NetworkMode": "none",
                                                                     "Memory": HELPER_MEMORY})})
        if code != 0:
            raise runner.RoundError(f"could not make the models volume writable (exit {code})")
        for side in plan["sides"]:
            if side.get("compile_cache") != "warm":
                continue
            volume = self.cache_volumes[side["key"]] = f"{self.cache_prefix}-{self.run_id[-8:]}-{side['key'].lower()}"
            self.docker.remove_volume(volume)  # an earlier attempt's caches would make the first round warm
            self.docker.volume(volume)
            code, _ = self.docker.run(self._name("prepare-cache"), {
                "Image": side["image"], "User": "0:0", "Entrypoint": ["sh", "-c"], "Cmd": ["chmod 1777 /cache"],
                "Labels": {LABEL: self.run_id}, "HostConfig": self._host({"Binds": [f"{volume}:/cache"],
                                                                         "NetworkMode": "none",
                                                                         "Memory": HELPER_MEMORY})})
            if code != 0:
                raise runner.RoundError(f"could not make the compile-cache volume {volume} writable (exit {code})")
        llm = self.results_root / self.run_id / "llm.yaml"
        if any(s["preset"] == "ray-serve-llm" for s in plan["sides"]):
            from . import plan as plans

            llm.parent.mkdir(parents=True, exist_ok=True)
            llm.write_text(plans.llm_yaml(plan), encoding="utf-8")
        self._probe_replica_gpus(plan)
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
        for volume in self.cache_volumes.values():
            self.docker.remove_volume(volume)

    # ---- one round ----

    def _name(self, what: str) -> str:
        return f"rhvp-cmp-{self.run_id[-8:]}-{what}".lower()

    def _host(self, host: dict) -> dict:
        """A container's HostConfig, with the run's CPU set (RHVP_COMPARE_CPUSET) when there is one."""
        if self.cpuset:
            host["CpusetCpus"] = self.cpuset
        return host

    def _gpus(self, replica: int) -> str:
        """The GPUs replica `replica` gets: its own id from RHVP_COMPARE_REPLICA_GPUS, else RHVP_COMPARE_GPUS."""
        if not self.replica_gpus:
            return self.gpus
        if replica > len(self.replica_gpus):
            raise runner.RoundError(f"RHVP_COMPARE_REPLICA_GPUS names {len(self.replica_gpus)} GPUs; replica "
                                    f"{replica} needs one more")
        return self.replica_gpus[replica - 1]

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
                "HostConfig": self._host({"NetworkMode": "none", "PidMode": "host",
                                          "DeviceRequests": _gpu_request(self._gpus(1)), "Memory": HELPER_MEMORY})},
                timeout_s=60)
            busy = [text for _, text in logs.lines(log) if text.strip()]
            if code == 0 and not busy:
                return
            if time.monotonic() > deadline:
                raise runner.RoundError("another program holds the GPU" if code == 0 else
                                        f"nvidia-smi could not read the GPU (exit {code})", runner.GPU_NOT_FREE)
            time.sleep(10)

    def start(self, plan: Mapping, round_: Mapping, side: Mapping, spec: Mapping, replica: int = 1) -> dict:
        name = self._name(f"r{round_['index']:02d}-{round_['side'].lower()}" + (f"-{replica}" if replica > 1 else ""))
        env = dict(spec["env"])
        if self.hf_token:
            env["HF_TOKEN"] = self.hf_token
        binds = [f"{self.models_volume}:/models"]
        if spec.get("llm_config"):
            binds.append(f"{self.host_results}/{self.run_id}/llm.yaml:/config/llm.yaml:ro")
        tmpfs = {"/cache": "rw,exec,size=20g,mode=1777", "/tmp/r": "rw,exec,mode=1777"}
        cache = self.cache_volumes.get(side["key"])
        if cache:  # compileCache warm: the side's volume, kept for the run, in place of a fresh tmpfs
            binds.append(f"{cache}:/cache")
            del tmpfs["/cache"]
        host = self._host({"NetworkMode": self.network, "Binds": binds, "ShmSize": self.shm, "Tmpfs": tmpfs})
        if self.memory:
            host.update({"Memory": self.memory, "MemorySwap": self.memory})  # no swap on top
        if spec.get("gpu"):
            host["DeviceRequests"] = _gpu_request(self._gpus(replica))
        body: dict = {"Image": side["image"], "Cmd": list(spec["args"]), "User": self.user,
                      "Env": [f"{k}={v}" for k, v in env.items()], "Labels": {LABEL: self.run_id},
                      "ExposedPorts": {f"{plan['server']['port']}/tcp": {}}, "HostConfig": host}
        if spec.get("command"):
            body["Entrypoint"] = list(spec["command"])
        cid = self.docker.create(name, body)
        self.docker.start(cid)
        return {"id": cid, "name": name, "image": side["image"], "port": plan["server"]["port"],
                "gpu": spec.get("gpu", True), "replica": replica}

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

    def run_load(self, plan: Mapping, round_: Mapping, target: str, out_dir: Path) -> Path:
        return guidellm.run(plan, target, out_dir)

    def start_load(self, plan: Mapping, round_: Mapping, target: str, out_dir: Path) -> guidellm.Running:
        return guidellm.start(plan, target, out_dir)

    def router(self, plan: Mapping, targets: list[str], out_dir: Path):
        from . import router as routers  # its server needs aiohttp, which the runner's image has

        try:
            return routers.spawn(targets, out_dir / "router.log")
        except routers.RouterError as error:
            raise runner.RoundError(str(error)) from None

    def launch_facts(self, handle: Mapping) -> dict:
        """The environment the server process saw (see the module's docstring), the image's own entrypoint, and for
        a side on a GPU the device request its container got (`gpus`) with the UUIDs prepare() read through it."""
        config = (self.docker.image(handle["image"]) or {}).get("Config") or {}
        out = {"image_entrypoint": config.get("Entrypoint"), "image_cmd": config.get("Cmd"), "env": None,
               "env_source": "none"}
        if handle.get("gpu"):
            request = self._gpus(handle.get("replica", 1))
            out.update(gpus=request, gpu_uuids=self._gpu_uuids.get(request))
        try:
            code, text = self.docker.exec(handle["id"], ["sh", "-c", SERVER_ENVIRON])
            env = environ(text) if code == 0 else {}
        except (runner.RoundError, httpx.HTTPError):
            env = {}
        if env:
            out.update(env=env, env_source="process")
            return out
        inspected = (self.docker.inspect(handle["id"]) or {}).get("Config") or {}
        if inspected.get("Env") is not None:
            out.update(env=environ("\0".join(inspected["Env"])), env_source="container")
        return out

    def sampler(self, plan: Mapping, handle: Mapping):
        if plan["energy"]["source"] != "nvidia-smi" or not handle.get("gpu"):
            return None
        return NvidiaSmi(self.docker, handle["image"], self._gpus(handle.get("replica", 1)), f"{handle['name']}-power",
                         self.run_id, self.cpuset)

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
                "HostConfig": self._host({"NetworkMode": "none", "DeviceRequests": _gpu_request(self._gpus(1)),
                                          "Memory": HELPER_MEMORY})}, timeout_s=60)
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
