# SPDX-License-Identifier: Apache-2.0
"""rhvp_compare's readers: the GuideLLM report and command line, the server log, GPU power (nvidia-smi and DCGM
through the Thanos Querier), and the platform helpers of the two backends.

The GuideLLM fixture is a trimmed report from GuideLLM v0.8.1 against its own mock server (two levels, 8 s each);
the log lines follow vLLM's own wording.

Run with: python -m pytest -q tests/test_compare_parsers.py
"""

import base64
import json
import struct
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

from rhvp_compare import dockerapi, energy, guidellm, kube, logs, plan as plans  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
REPORT = json.loads((FIXTURES / "guidellm-report-v0.8.1.json").read_text())


def _plan(**load) -> dict:
    config = {"runName": "t", "model": {"id": "m", "source": "org/m", "revision": "abc"},
              "sides": [{"name": "a", "image": "x/a:1"}, {"name": "b", "image": "x/b:1"}],
              "load": {"concurrencies": [1, 4], "secondsPerLevel": 8, **load}}
    return plans.build(config, "docker")


# ---- GuideLLM ----

def test_levels_from_the_report():
    levels = guidellm.levels(REPORT)
    assert [lv["concurrency"] for lv in levels] == [1, 4]
    first = levels[0]
    assert first["requests"] == {"successful": 26, "errored": 0, "incomplete": 1}
    assert first["usable"] and first["note"] is None
    assert first["ttft_ms"]["median"] == pytest.approx(43.40648651123047)
    assert first["itl_ms"]["median"] == pytest.approx(8.25753519611974)
    assert first["output_tokens_per_s"] == pytest.approx(105.125)
    assert first["duration_s"] == pytest.approx(first["end_epoch"] - first["start_epoch"], abs=0.5)
    assert first["started_at"].endswith("Z")
    assert guidellm.version(REPORT) == "0.8.1"


def test_a_level_with_errors_stays_out_of_the_comparison():
    report = json.loads(json.dumps(REPORT))
    report["benchmarks"][1]["metrics"]["request_totals"]["errored"] = 2
    levels = guidellm.levels(report)
    assert not levels[1]["usable"] and "2 requests failed" in levels[1]["note"]
    flat = guidellm.flat_metrics(levels)
    assert "c1.itl_ms_median" in flat and not any(k.startswith("c4.") for k in flat)


def test_flat_metrics_take_tokens_per_joule_from_the_energy_entry():
    levels = guidellm.levels(REPORT)
    levels[0]["energy"] = {"tokens_per_joule": 2.5}
    flat = guidellm.flat_metrics(levels)
    assert flat["c1.tokens_per_joule"] == 2.5 and "c4.tokens_per_joule" not in flat


@pytest.mark.parametrize("report", [{}, {"benchmarks": []}, {"benchmarks": [{"metrics": {}}]}])
def test_a_broken_report_is_a_report_error(report):
    with pytest.raises(guidellm.ReportError):
        guidellm.levels(report)


def test_command_line():
    args = guidellm.command(_plan(), "http://side:8000", "/results/r/guidellm.json")
    text = " ".join(args)
    assert args[0] == "run"
    assert "kind=openai_http,target=http://side:8000,model=m,request_format=/v1/completions" in args
    assert "--override profile.streams 1,4" in text
    assert "kind=max_duration,seconds=8" in args
    assert "kind=huggingface_auto,model=org/m,load_kwargs.revision=abc" in args
    assert "kind=static,value=42" in args
    assert "kind=json,path=/results/r/guidellm.json" in args
    one = guidellm.command(plans.build({**_plan_config(), "load": {"concurrencies": [8]}}, "docker"), "t", "o")
    assert "--override" not in one and "kind=concurrent,streams=8" in one


def _plan_config() -> dict:
    return {"runName": "t", "model": {"id": "m", "source": "org/m"},
            "sides": [{"name": "a", "image": "x/a:1"}, {"name": "b", "image": "x/b:1"}]}


def test_run_stops_a_hung_load_and_reports_a_bad_exit(tmp_path):
    plan = _plan(timeoutSeconds=30)
    plan["load"]["timeout_s"] = 1
    hang = tmp_path / "hang.sh"
    hang.write_text("#!/bin/sh\nsleep 30\n")
    hang.chmod(0o755)
    with pytest.raises(guidellm.LoadError) as caught:
        guidellm.run(plan, "http://side:8000", tmp_path / "out", executable=str(hang))
    assert caught.value.timed_out
    fail = tmp_path / "fail.sh"
    fail.write_text("#!/bin/sh\necho no\nexit 3\n")
    fail.chmod(0o755)
    with pytest.raises(guidellm.LoadError) as caught:
        guidellm.run(plan, "http://side:8000", tmp_path / "out", executable=str(fail))
    assert not caught.value.timed_out and "code 3" in str(caught.value)
    assert (tmp_path / "out" / "guidellm.log").read_text() == "no\n"


def test_run_returns_the_report(tmp_path):
    fake = tmp_path / "fake.sh"
    fake.write_text('#!/bin/sh\nfor a in "$@"; do case "$a" in kind=json,path=*) echo "{}" > "${a#kind=json,path=}";; '
                    'esac; done\n')
    fake.chmod(0o755)
    path = guidellm.run(_plan(), "http://side:8000", tmp_path / "out", executable=str(fake))
    assert path == tmp_path / "out" / "guidellm.json" and path.read_text() == "{}\n"


# ---- the server log ----

VLLM_LOG = """\
2026-01-01T00:00:03.500000000Z (APIServer pid=1) INFO 01-01 00:00:03 [api_server.py:1] vLLM API server version 0.33.0
2026-01-01T00:00:05.000000000Z (EngineCore pid=7) INFO 01-01 00:00:05 [core.py:1] Initializing a V1 LLM engine (v0.33.0) with config: model='org/m'
2026-01-01T00:00:20.250000Z (EngineCore pid=7) INFO 01-01 00:00:20 [loader.py:1] Loading weights took 12.25 seconds
2026-01-01T00:00:21Z (EngineCore pid=7) INFO 01-01 00:00:21 [runner.py:1] Model loading took 0.93 GiB memory and 13.10 seconds
2026-01-01T00:00:40.000000000Z (EngineCore pid=7) INFO 01-01 00:00:40 [backends.py:1] torch.compile took 15.20 s in total
2026-01-01T00:00:50.000000000Z (EngineCore pid=7) INFO 01-01 00:00:50 [runner.py:2] Graph capturing finished in 4 secs, took 0.10 GiB
2026-01-01T00:00:55.000000000Z (EngineCore pid=7) INFO 01-01 00:00:55 [runner.py:2] Graph capturing finished in 3 secs, took 0.05 GiB
2026-01-01T00:00:56.000000000Z (EngineCore pid=7) INFO 01-01 00:00:56 [core.py:2] init engine (profile, create kv cache, warmup model) took 30.40 seconds
2026-01-01T00:01:00.000000000Z (APIServer pid=1) INFO:     Application startup complete.
"""


def test_breakdown_of_a_vllm_start():
    started = logs.parse_time("2026-01-01T00:00:00Z")
    steps = {entry["phase"]: entry for entry in logs.breakdown(VLLM_LOG, started)}
    assert steps["first_log"]["at_s"] == pytest.approx(3.5)
    assert steps["api_server_start"]["at_s"] == pytest.approx(3.5)
    assert steps["engine_start"]["at_s"] == pytest.approx(5.0)
    assert steps["weights_load"]["seconds"] == pytest.approx(12.25)
    assert steps["model_load"]["seconds"] == pytest.approx(13.10)
    assert steps["torch_compile"]["seconds"] == pytest.approx(15.2)
    assert steps["cuda_graphs"]["seconds"] == pytest.approx(7.0) and steps["cuda_graphs"]["count"] == 2
    assert steps["engine_init"]["seconds"] == pytest.approx(30.4)
    assert steps["app_startup_complete"]["at_s"] == pytest.approx(60.0)
    assert all(entry["source"] == "log" for entry in steps.values())
    assert logs.server_version(logs.lines(VLLM_LOG)) == "0.33.0"


def test_log_lines_without_time_and_with_colour():
    parsed = logs.lines("plain line\n\x1b[32m2026-01-01T00:00:00.123456789+00:00 green\x1b[0m\n")
    assert parsed[0] == (None, "plain line")
    assert parsed[1][1] == "green" and parsed[1][0] == pytest.approx(logs.parse_time("2026-01-01T00:00:00.123456Z"))
    assert logs.breakdown("", 0.0) == []
    assert logs.server_version([(None, "nothing here")]) is None


# ---- GPU power ----

def test_nvidia_smi_samples():
    log = ("2026-01-01T00:00:00.000Z 41.20\n2026-01-01T00:00:00.500Z [N/A]\n2026-01-01T00:00:01.000Z 55.5, 60\n"
           "no time 99\n")
    samples = energy.parse_nvidia_smi(log)
    assert [s.value for s in samples] == [41.2, 55.5]
    assert samples[1].t - samples[0].t == pytest.approx(1.0)


def test_window_power_prefers_the_energy_counter():
    power = [energy.Sample(t, 100.0) for t in range(0, 11)]
    counter = [energy.Sample(2, 1_000_000), energy.Sample(8, 1_000_000 + 6 * 150_000)]  # mJ: 150 W over 6 s
    by_counter = energy.window_power(power, 1, 9, 2, counter)
    assert by_counter.method == "energy_counter" and by_counter.mean_w == pytest.approx(150.0)
    by_power = energy.window_power(power, 1, 9, 2)
    assert by_power.method == "power_mean" and by_power.mean_w == pytest.approx(100.0) and by_power.samples == 9
    too_few = energy.window_power(power[:1], 0, 9, 2)
    assert too_few.mean_w is None and "at least 2" in too_few.note


def test_level_energy():
    level = {"output_tokens_per_s": 300.0, "duration_s": 10.0}
    entry = energy.level_energy(level, energy.Power(150.0, 20, "power_mean", None), "nvidia-smi")
    assert entry["tokens_per_joule"] == pytest.approx(2.0) and entry["joules_per_token"] == pytest.approx(0.5)
    assert entry["energy_j"] == pytest.approx(1500.0) and entry["label"] == "GPU-only power"
    empty = energy.level_energy(level, energy.Power(None, 0, None, "none"), "dcgm")
    assert empty["tokens_per_joule"] is None and empty["note"] == "none"


def _matrix(*series) -> dict:
    return {"status": "success", "data": {"resultType": "matrix", "result": [
        {"metric": labels, "values": [[t, str(v)] for t, v in values]} for labels, values in series]}}


def test_parse_matrix():
    parsed = energy.parse_matrix(_matrix(({"pod": "p"}, [(1, 10), (2, "NaN"), (3, 30)])))
    assert parsed[0][0] == {"pod": "p"} and [s.value for s in parsed[0][1]] == [10.0, 30.0]
    with pytest.raises(ValueError):
        energy.parse_matrix({"status": "error", "error": "bad"})
    with pytest.raises(ValueError):
        energy.parse_matrix({"status": "success", "data": {"resultType": "vector", "result": []}})


def _thanos(answers: dict) -> tuple[energy.Thanos, list]:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params["query"]
        seen.append((query, request.headers.get("authorization")))
        for needle, answer in answers.items():
            if needle in query:
                return httpx.Response(200, json=answer)
        return httpx.Response(200, json=_matrix())

    return energy.Thanos("https://thanos.example:9091", transport=httpx.MockTransport(handler), token="t0k",
                         verify=False), seen


def test_dcgm_samples_by_exported_pod_labels():
    thanos, seen = _thanos({
        f'{energy.POWER}{{exported_pod="side",exported_namespace="ns"}}': _matrix(({}, [(10, 200), (11, 210)])),
        f'{energy.ENERGY_COUNTER}{{exported_pod="side"': _matrix(({}, [(10, 5000), (11, 5205)])),
    })
    power, counter, matched, note = energy.dcgm_samples(thanos, "side", "ns", "node-1", 10, 11)
    assert [s.value for s in power] == [200, 210] and len(counter) == 2
    assert matched == "pod (exported labels)" and note is None
    assert seen[0][0].endswith("[2s]") and seen[0][1] == "Bearer t0k"


def test_dcgm_node_series_only_when_there_is_one_gpu():
    one = _matrix(({"gpu": "0"}, [(10, 90), (11, 95)]))
    two = _matrix(({"gpu": "0"}, [(10, 90)]), ({"gpu": "1"}, [(10, 300)]))
    thanos, _ = _thanos({'Hostname="node-1"': one})
    assert energy.dcgm_samples(thanos, "side", "ns", "node-1", 10, 11)[2] == "node"
    thanos, _ = _thanos({'Hostname="node-1"': two})
    power, _, matched, note = energy.dcgm_samples(thanos, "side", "ns", "node-1", 10, 11)
    assert power == [] and matched is None and "2 GPU series" in note
    thanos, _ = _thanos({})
    assert "no DCGM power series" in energy.dcgm_samples(thanos, "side", "ns", None, 10, 11)[3]


def test_dcgm_query_errors_carry_no_url():
    thanos = energy.Thanos("https://thanos.example:9091", verify=False, token="t",
                           transport=httpx.MockTransport(lambda r: httpx.Response(403)))
    power, _, _, note = energy.dcgm_samples(thanos, "side", "ns", None, 1, 2)
    assert power == [] and "HTTP 403" in note and "thanos.example" not in note


# ---- the OpenShift backend's helpers ----

GFD_LABELS = {"nvidia.com/gpu.product": "NVIDIA-L4", "nvidia.com/gpu.memory": "23034", "nvidia.com/gpu.count": "1",
              "nvidia.com/cuda.driver-version.full": "580.95.05", "nvidia.com/cuda.runtime-version.full": "13.0"}


def test_gpu_and_node_facts_from_labels():
    gpu = kube.gpu_facts(GFD_LABELS)
    assert gpu["product"] == "NVIDIA-L4" and gpu["memory_mib"] == 23034 and gpu["driver"] == "580.95.05"
    older = kube.gpu_facts({"nvidia.com/gpu.product": "X", "nvidia.com/cuda.driver.major": "570",
                            "nvidia.com/cuda.driver.minor": "1", "nvidia.com/cuda.driver.rev": "2"})
    assert older["driver"] == "570.1.2" and older["memory_mib"] is None
    assert kube.gpu_facts({}) is None
    node = kube.node_facts({"metadata": {"name": "n1", "labels": {"kubernetes.io/arch": "amd64",
                                                                  "node.kubernetes.io/instance-type": "g6.2xlarge"}},
                            "status": {"nodeInfo": {"osImage": "RHCOS", "kernelVersion": "5.14",
                                                    "containerRuntimeVersion": "cri-o://1.33"}}})
    assert node == {"name": "n1", "arch": "amd64", "instance_type": "g6.2xlarge", "os_image": "RHCOS",
                    "kernel": "5.14", "container_runtime": "cri-o://1.33"}


def test_image_digest_from_the_pod():
    pod = {"status": {"containerStatuses": [{"name": "server", "imageID": "docker.io/x/a@sha256:abc"}]}}
    assert kube.image_digest(pod) == "docker.io/x/a@sha256:abc"
    pod["status"]["containerStatuses"][0]["imageID"] = "docker-pullable://x/a@sha256:def"
    assert kube.image_digest(pod) == "x/a@sha256:def"
    assert kube.image_digest({}) is None


def _side_backend(secret: str | None = None) -> tuple[kube.KubeBackend, list]:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path.endswith("/secrets/huggingface-secret"):
            if secret is None:
                return httpx.Response(404, json={"message": "not found"})
            return httpx.Response(200, json={"data": {"hftoken": base64.b64encode(secret.encode()).decode()}})
        if request.url.path.endswith("/pods") and request.method == "GET":
            return httpx.Response(200, json={"items": []})
        return httpx.Response(404, json={})

    client = kube.Kube(base_url="https://api.example:6443", token="t", verify=False,
                       transport=httpx.MockTransport(handler))
    env = {"RHVP_COMPARE_ARCH": "arm64", "RHVP_COMPARE_SIDE_RESOURCES": json.dumps(
        {"requests": {"cpu": "2", "memory": "12Gi"}, "limits": {"memory": "24Gi"}})}
    return kube.KubeBackend(kube=client, env=env, sleep=lambda s: None), calls


def test_side_pod_like_the_patterns_server():
    backend, _ = _side_backend()
    plan = _plan()
    backend.prepare(plan)
    side = plan["sides"][1]
    body = backend.pod_body(plan, plan["rounds"][3], side, plans.side_spec(plan, side))
    spec = body["spec"]
    [container] = spec["containers"]
    assert body["metadata"]["name"].startswith("cmp-") and body["metadata"]["name"].endswith("-r04-b")
    assert body["metadata"]["labels"][kube.LABEL_RUN] == plan["run_id"]
    assert spec["restartPolicy"] == "Never" and spec["nodeSelector"] == {"kubernetes.io/arch": "arm64"}
    assert spec["tolerations"] == [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
    assert container["resources"]["requests"]["nvidia.com/gpu"] == container["resources"]["limits"]["nvidia.com/gpu"]
    assert container["resources"]["requests"]["memory"] == "12Gi"
    assert container["securityContext"] == {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}}
    assert "runAsUser" not in json.dumps(body)
    assert {m["mountPath"] for m in container["volumeMounts"]} == {"/models", "/cache", "/tmp/r", "/dev/shm"}
    dshm = next(v for v in spec["volumes"] if v["name"] == "dshm")
    assert dshm["emptyDir"] == {"medium": "Memory", "sizeLimit": "4Gi"}
    assert next(v for v in spec["volumes"] if v["name"] == "models")["persistentVolumeClaim"] == {
        "claimName": "ray-serve-llm-models"}
    assert "HF_TOKEN" not in {e["name"] for e in container["env"]}
    assert container["args"][:2] == ["--model", "org/m"] and "command" not in container


@pytest.mark.parametrize("secret, referenced", [("hf_abc", True), ("none", False), (None, False)])
def test_the_token_goes_in_by_reference_and_only_when_real(secret, referenced):
    backend, _ = _side_backend(secret)
    plan = _plan()
    backend.prepare(plan)
    side = plan["sides"][0]
    body = backend.pod_body(plan, plan["rounds"][0], side, plans.side_spec(plan, side))
    env = {e["name"]: e for e in body["spec"]["containers"][0]["env"]}
    assert ("HF_TOKEN" in env) is referenced
    if referenced:
        assert env["HF_TOKEN"]["valueFrom"]["secretKeyRef"] == {"name": "huggingface-secret", "key": "hftoken",
                                                                "optional": True}
        assert "hf_abc" not in json.dumps(body)


def test_a_side_without_a_gpu_and_the_ray_preset():
    backend, _ = _side_backend()
    config = _plan_config()
    config["sides"] = [{"name": "a", "image": "x/a:1", "preset": "guidellm-mock"},
                       {"name": "b", "image": "x/b:1", "preset": "ray-serve-llm", "gpu": False}]
    plan = plans.build(config, "openshift")
    backend.prepare(plan)
    mock = backend.pod_body(plan, plan["rounds"][0], plan["sides"][0], plans.side_spec(plan, plan["sides"][0]))
    assert mock["spec"]["tolerations"] == []
    assert "nvidia.com/gpu" not in json.dumps(mock["spec"]["containers"][0]["resources"])
    assert mock["spec"]["containers"][0]["command"] == ["guidellm", "mock-server"]
    ray = backend.pod_body(plan, plan["rounds"][1], plan["sides"][1], plans.side_spec(plan, plan["sides"][1]))
    assert {"name": "config", "configMap": {"name": "ray-serve-llm-config"}} in ray["spec"]["volumes"]
    assert plan["energy"]["source"] == "none"


def _pod(phase="Pending", waiting=None, terminated=None, scheduled=None, ip=None) -> dict:
    state = {}
    if waiting:
        state["waiting"] = {"reason": waiting, "message": "detail"}
    if terminated:
        state["terminated"] = {"reason": terminated, "exitCode": 1}
    status = {"phase": phase, "containerStatuses": [{"name": "server", "state": state}] if state else [],
              "conditions": [{"type": "PodScheduled", "status": scheduled[0], "message": scheduled[1]}]
              if scheduled else []}
    if ip:
        status["podIP"] = ip
    return {"status": status, "spec": {}}


def test_problems_a_side_can_have():
    backend, _ = _side_backend()
    now = [1000.0]
    backend.clock = lambda: now[0]
    handle = {"name": "p", "created": 1000.0, "pull_since": None}
    assert backend._problem(handle, _pod("Running")) is None
    assert "exit code 1" in backend._problem(handle, _pod("Failed", terminated="OOMKilled"))
    assert "InvalidImageName" in backend._problem(handle, _pod(waiting="InvalidImageName"))
    assert backend._problem(handle, _pod(waiting="ImagePullBackOff")) is None
    now[0] += 901
    assert "could not be pulled" in backend._problem(handle, _pod(waiting="ImagePullBackOff"))
    late = {"name": "p", "created": 0.0, "pull_since": None}
    assert "Insufficient nvidia.com/gpu" in backend._problem(late, _pod(scheduled=("False",
                                                                                   "0/3 nodes: Insufficient nvidia.com/gpu")))
    assert backend._problem(handle, None) == "the side's pod is gone"


def test_endpoint_is_the_pod_ip():
    backend, _ = _side_backend()
    handle = {"name": "p", "port": 8000, "created": backend.clock(), "pull_since": None, "last": None,
              "last_at": -1e9}
    backend.kube.pod = lambda ns, name: _pod("Running", ip="10.0.0.5")
    assert backend.endpoint(handle, backend.clock() + 10) == "http://10.0.0.5:8000"
    backend.kube.pod = lambda ns, name: _pod("Running", ip="fd00::5")
    assert backend.endpoint(handle, backend.clock() + 10) == "http://[fd00::5]:8000"


def test_kube_errors_carry_no_url_or_token():
    client = kube.Kube(base_url="https://api.example:6443", token="secret-token", verify=False,
                       transport=httpx.MockTransport(lambda r: httpx.Response(403, json={"message": "forbidden"})))
    with pytest.raises(Exception) as caught:
        client.pods("ns")
    text = str(caught.value)
    assert "HTTP 403 forbidden" in text and "api.example" not in text and "secret-token" not in text


# ---- the docker backend's helpers ----

def test_docker_log_frames():
    frame = lambda stream, text: struct.pack(">BxxxI", stream, len(text)) + text  # noqa: E731
    raw = frame(1, b"2026-01-01T00:00:00Z out\n") + frame(2, b"2026-01-01T00:00:01Z err\n")
    assert dockerapi.demux(raw) == "2026-01-01T00:00:00Z out\n2026-01-01T00:00:01Z err\n"
    assert dockerapi.demux(b"plain tty text\n") == "plain tty text\n"


def test_repository_of_an_image_reference():
    assert dockerapi.repository("docker.io/vllm/vllm-openai:v0.33.0@sha256:ab") == "docker.io/vllm/vllm-openai"
    assert dockerapi.repository("localhost:5000/x") == "localhost:5000/x"
    assert dockerapi.repository("x:1") == "x"
