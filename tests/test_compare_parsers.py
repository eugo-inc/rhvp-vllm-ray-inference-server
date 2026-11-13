# SPDX-License-Identifier: Apache-2.0
"""rhvp_compare's readers: the GuideLLM report and command line, the server log, GPU power (nvidia-smi and DCGM
through the Thanos Querier), and the platform helpers of the two backends.

The GuideLLM fixture is a trimmed report from GuideLLM v0.8.1 against its own mock server (two levels, 8 s each);
the log lines follow vLLM's own wording.

Run with: python -m pytest -q tests/test_compare_parsers.py
"""

import base64
import json
import os
import signal
import struct
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

from rhvp_compare import dockerapi, energy, guidellm, kube, logs, plan as plans, runner  # noqa: E402

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
    flat = guidellm.flat_values(levels)
    assert "c1/itl_ms_median" in flat and not any(k.startswith("c4/") for k in flat)


def test_flat_values_take_tokens_per_joule_from_the_energy_entry():
    levels = guidellm.levels(REPORT)
    levels[0]["energy"] = {"tokens_per_joule": 2.5}
    flat = guidellm.flat_values(levels)
    assert flat["c1/tokens_per_joule[gpu]"] == 2.5 and "c4/tokens_per_joule[gpu]" not in flat
    ids = {c["id"] for c in _plan()["cases"]}
    assert set(guidellm.flat_values(guidellm.levels(REPORT))) <= ids  # the ids of the plan's cases


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


def test_flat_values_keep_a_zero():
    levels = guidellm.levels(REPORT)
    levels[0]["output_tokens_per_s"] = 0.0
    levels[0]["energy"] = energy.level_energy(levels[0], energy.Power(150.0, 20, "power_mean", None), "nvidia-smi")
    flat = guidellm.flat_values(levels)
    c = levels[0]["concurrency"]
    assert flat[f"c{c}/output_tokens_per_s"] == 0.0 and flat[f"c{c}/tokens_per_joule[gpu]"] == 0.0
    assert levels[0]["energy"]["joules_per_token"] is None


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
        'DCGM_FI_DEV_POWER_USAGE{exported_pod="side",exported_namespace="ns"}': _matrix(({}, [(10, 200), (11, 210)])),
        'DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{exported_pod="side"': _matrix(({}, [(10, 5000), (11, 5205)])),
    })
    power, counter, matched, note = energy.dcgm_samples(thanos, "side", "ns", "node-1", 10, 11)
    assert [s.value for s in power] == [200, 210] and [s.value for s in counter] == [5000, 5205]
    assert matched == "pod (exported labels), DCGM_FI_DEV_POWER_USAGE" and note is None
    assert seen[0][0].endswith("[2s]") and seen[0][1] == "Bearer t0k"


def test_dcgm_samples_under_the_exporters_new_names():
    # DCGM exporter 4.9.0 renames both series; its energy counter is in joules, not millijoules.
    thanos, seen = _thanos({
        'DCGM_FI_DEV_BOARD_POWER_WATTS{exported_pod="side"': _matrix(({}, [(10, 200), (11, 210)])),
        'DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL{exported_pod="side"': _matrix(({}, [(10, 1000.0), (11, 1205.0)])),
    })
    power, counter, matched, note = energy.dcgm_samples(thanos, "side", "ns", None, 10, 11)
    assert [s.value for s in power] == [200, 210] and matched.endswith("DCGM_FI_DEV_BOARD_POWER_WATTS")
    assert [s.value for s in counter] == pytest.approx([1_000_000, 1_205_000])  # millijoules
    assert energy.window_power(power, 10, 11, 2, counter).mean_w == pytest.approx(205.0)
    assert seen[0][0].startswith("DCGM_FI_DEV_POWER_USAGE{")  # the older name is tried first


def test_repeated_counter_values_are_dropped_before_the_division():
    # A scrape faster than the exporter's update repeats the last value: 1000 J from t=10 to t=12, not to t=13.
    counter = [energy.Sample(10, 0), energy.Sample(11, 500_000), energy.Sample(12, 1_000_000),
               energy.Sample(13, 1_000_000)]
    assert energy.collapse(counter) == counter[:3]
    assert energy.window_power([], 10, 13, 2, counter).mean_w == pytest.approx(500.0)
    flat = [energy.Sample(10, 5000), energy.Sample(11, 5000)]
    power = [energy.Sample(10, 90), energy.Sample(11, 110)]
    result = energy.window_power(power, 10, 11, 2, flat)  # one distinct value: fall back to the power samples
    assert result.method == "power_mean" and result.mean_w == pytest.approx(100.0)


def test_dcgm_node_series_only_when_there_is_one_gpu():
    one = _matrix(({"gpu": "0"}, [(10, 90), (11, 95)]))
    two = _matrix(({"gpu": "0"}, [(10, 90)]), ({"gpu": "1"}, [(10, 300)]))
    thanos, _ = _thanos({'Hostname="node-1"': one})
    assert energy.dcgm_samples(thanos, "side", "ns", "node-1", 10, 11)[2].startswith("node, ")
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
    # A runtime that reports only the image's own id: the image the pod asked for, when it is pinned by digest.
    pod["status"]["containerStatuses"][0]["imageID"] = "sha256:0123"
    assert kube.image_digest(pod) is None
    pod["spec"] = {"containers": [{"name": "server", "image": "quay.io/x/a:1@sha256:feed"}]}
    assert kube.image_digest(pod) == "quay.io/x/a:1@sha256:feed"
    pod["spec"]["containers"][0]["image"] = "quay.io/x/a:1"
    assert kube.image_digest(pod) is None
    assert kube.image_digest({}) is None


def _side_backend(secret: str | None = None, server_replicas: int | None = None,
                  configmaps: dict | None = None, scale_fails: int = 0, runner_pods: list | None = None,
                  clock=None) -> tuple[kube.KubeBackend, list]:
    """A KubeBackend on a fake API. `scale_fails`: how many scale-ups answer HTTP 503 before one goes through."""
    calls = []
    state = {"replicas": server_replicas, "fails": scale_fails}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.url.path.endswith("/deployments/ray-serve-llm/scale"):
            if state["replicas"] is None:
                return httpx.Response(404, json={"message": "not found"})
            if request.method == "PATCH":
                assert request.headers["content-type"] == "application/merge-patch+json"
                if body["spec"]["replicas"] > 0 and state["fails"] > 0:
                    state["fails"] -= 1
                    return httpx.Response(503, json={"message": "unavailable"})
                state["replicas"] = body["spec"]["replicas"]
            return httpx.Response(200, json={"spec": {"replicas": state["replicas"]}})
        if configmaps is not None and "/configmaps" in request.url.path:
            name = request.url.path.rsplit("/", 1)[1]
            if request.method == "PATCH":
                if name not in configmaps:
                    return httpx.Response(404, json={"message": "not found"})
                configmaps[name]["data"].update(body["data"])
                configmaps[name]["data"] = {k: v for k, v in configmaps[name]["data"].items() if v is not None}
                return httpx.Response(200, json=configmaps[name])
            configmaps[body["metadata"]["name"]] = body
            return httpx.Response(201, json=body)
        if request.url.path.endswith("/secrets/huggingface-secret"):
            if secret is None:
                return httpx.Response(404, json={"message": "not found"})
            return httpx.Response(200, json={"data": {"hftoken": base64.b64encode(secret.encode()).decode()}})
        if request.url.path == "/api/v1/namespaces/nb/pods" and request.method == "GET":
            assert request.url.params["labelSelector"] == "app.kubernetes.io/name=image-compare"
            return httpx.Response(200, json={"items": runner_pods or []})
        if request.url.path.endswith("/pods") and request.method == "GET":
            return httpx.Response(200, json={"items": []})
        return httpx.Response(404, json={})

    client = kube.Kube(base_url="https://api.example:6443", token="t", verify=False,
                       transport=httpx.MockTransport(handler))
    env = {"RHVP_COMPARE_ARCH": "arm64", "RHVP_COMPARE_STATUS_NAMESPACE": "nb", "RHVP_COMPARE_POD_NAME": "me",
           "RHVP_COMPARE_SIDE_RESOURCES": json.dumps({"requests": {"cpu": "2", "memory": "12Gi"},
                                                      "limits": {"memory": "24Gi"}})}
    extra = {"clock": clock} if clock else {}
    return kube.KubeBackend(kube=client, env=env, sleep=lambda s: None, **extra), calls


def test_the_live_server_is_scaled_down_for_the_run_and_back_after():
    backend, calls = _side_backend(server_replicas=1)
    backend.prepare(_plan())
    scale = [(method, body) for method, path, body in calls if path.endswith("/scale")]
    assert scale == [("GET", None), ("PATCH", {"spec": {"replicas": 0}})]
    calls.clear()
    assert backend.restore_server() is True
    # scaled back, then read back to check
    assert [(m, b) for m, p, b in calls if p.endswith("/scale")] == [("PATCH", {"spec": {"replicas": 1}}),
                                                                   ("GET", None)]
    calls.clear()
    assert backend.restore_server() is True  # once only
    assert not [c for c in calls if c[1].endswith("/scale")]
    absent, calls = _side_backend(server_replicas=None)  # no server Deployment: nothing to scale
    absent.prepare(_plan())
    assert absent.restore_server() is None
    assert [m for m, p, b in calls if p.endswith("/scale")] == ["GET"]


def test_a_scale_back_that_fails_is_tried_again():
    backend, calls = _side_backend(server_replicas=1, scale_fails=3)
    backend.prepare(_plan())
    assert backend.restore_server() is True
    assert [b for m, p, b in calls if p.endswith("/scale") and m == "PATCH"][-4:] == [{"spec": {"replicas": 1}}] * 4


def test_a_scale_back_that_keeps_failing_says_how_to_bring_the_server_back():
    now = [0.0]

    def clock():
        now[0] += 5.0  # each look at the clock is 5 s later: the minute of tries runs out
        return now[0]
    backend, calls = _side_backend(server_replicas=1, scale_fails=99, clock=clock)
    backend.prepare(_plan())
    with pytest.raises(runner.RoundError) as caught:
        backend.restore_server()
    text = str(caught.value)
    assert "HTTP 503" in text and "oc scale deployment/ray-serve-llm --replicas=1 -n vllm-inference" in text
    assert "oc delete pod -n vllm-inference -l app.kubernetes.io/name=image-compare-side" in text
    tries = len([1 for m, p, b in calls if p.endswith("/scale") and m == "PATCH" and b["spec"]["replicas"] == 1])
    assert 2 <= tries <= 5
    with pytest.raises(runner.RoundError):  # asked again at the end: one more try
        backend.restore_server()


def test_cleanup_deletes_the_run_pods_only():
    backend, calls = _side_backend(server_replicas=1)
    backend.prepare(_plan())
    calls.clear()
    backend.cleanup()
    assert not [c for c in calls if c[1].endswith("/scale")]
    assert [(m, p) for m, p, b in calls] == [("GET", "/api/v1/namespaces/vllm-inference/pods")]


def _runner_pod(name: str, created: str, phase: str = "Running") -> dict:
    return {"metadata": {"name": name, "creationTimestamp": created, "labels": {"job-name": f"job-{name}"}},
            "status": {"phase": phase}}


@pytest.mark.parametrize("others, refused", [
    ([], False),
    ([_runner_pod("old", "2026-01-01T00:00:00Z")], True),
    ([_runner_pod("old", "2026-01-01T00:00:00Z", "Pending")], True),
    ([_runner_pod("old", "2026-01-01T00:00:00Z", "Succeeded")], False),  # ended: no longer a run
    ([_runner_pod("old", "2026-01-01T00:00:00Z", "Failed")], False),
    ([_runner_pod("young", "2026-01-03T00:00:00Z")], False),  # the older one (this one) goes on; the other refuses
    ([_runner_pod("a-same-age", "2026-01-02T00:00:00Z")], True),  # same age: the name that sorts first wins
])
def test_one_run_at_a_time(others, refused):
    pods = [_runner_pod("me", "2026-01-02T00:00:00Z")] + others
    backend, calls = _side_backend(runner_pods=pods)
    if refused:
        with pytest.raises(runner.RunActive, match="another comparison is running: runner pod"):
            backend.check_alone()
    else:
        backend.check_alone()
    assert calls == [("GET", "/api/v1/namespaces/nb/pods", None)]  # only looked


def test_later_sides_run_on_the_first_sides_node():
    backend, _ = _side_backend()
    plan = _plan()
    backend.prepare(plan)
    side = plan["sides"][0]
    first = backend.pod_body(plan, plan["launches"][0], side, plans.side_spec(plan, side))
    assert "kubernetes.io/hostname" not in first["spec"]["nodeSelector"]
    backend.kube.node = lambda name: {"metadata": {"name": name, "labels": {"kubernetes.io/hostname": "gpu-1"}}}
    backend._pod = lambda handle, fresh=False: {"spec": {"nodeName": "gpu-1.example"}}
    assert backend.facts({"name": "p", "gpu": True})["node_name"] == "gpu-1.example"
    later = backend.pod_body(plan, plan["launches"][2], side, plans.side_spec(plan, side))
    assert later["spec"]["nodeSelector"] == {"kubernetes.io/arch": "arm64", "kubernetes.io/hostname": "gpu-1"}


def test_a_stop_shortens_every_wait():
    backend, _ = _side_backend()
    assert (backend.kube.timeout_s, backend.kube.log_timeout_s, backend.delete_wait_s) == (30.0, 60.0, 180.0)
    backend.on_stop()
    assert (backend.kube.timeout_s, backend.kube.log_timeout_s) == (kube.HURRY_TIMEOUT_S, kube.HURRY_TIMEOUT_S)
    assert backend.delete_wait_s == kube.HURRY_DELETE_WAIT_S
    assert kube.stop_budget_seconds() < 300  # the chart's grace period (tests/test_charts.py checks the chart)


def test_the_token_is_read_again_for_every_request(tmp_path):
    token = tmp_path / "token"
    token.write_text("first\n")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"items": []})
    client = kube.Kube(base_url="https://api.example:6443", verify=False, token_path=token,
                       transport=httpx.MockTransport(handler))
    client.pods("ns")
    token.write_text("second\n")  # the kubelet rotated it
    client.pods("ns")
    assert seen == ["Bearer first", "Bearer second"]
    thanos = energy.Thanos("https://thanos.example:9091", verify=False, token_path=token,
                           transport=httpx.MockTransport(handler))
    thanos._client.get("/x")
    assert seen[-1] == "Bearer second"


def test_the_status_configmap_is_created_then_merged():
    store: dict = {}
    backend, _ = _side_backend(configmaps=store)
    backend.publish("run-1", '{"phase":"running"}', '{"big":1}')
    made = store["compare-run-1"]
    assert made["metadata"]["labels"] == {"app.kubernetes.io/component": "compare-run", "compare-run-id": "run-1",
                                          "app.kubernetes.io/part-of": "vllm-ray-inference-server"}
    assert made["data"] == {"status.json": '{"phase":"running"}', "result.json": '{"big":1}'}
    backend.publish("run-1", '{"phase":"complete"}', None)  # too big this time: result.json is dropped
    assert store["compare-run-1"]["data"] == {"status.json": '{"phase":"complete"}'}


def test_a_side_pod_has_its_own_time_limit():
    backend, _ = _side_backend()
    plan = _plan()
    backend.prepare(plan)
    side = plan["sides"][0]
    warm = backend.pod_body(plan, plan["launches"][0], side, plans.side_spec(plan, side))
    scored = backend.pod_body(plan, plan["launches"][2], side, plans.side_spec(plan, side))
    load = 2 * plan["load"]["timeout_s"] + plan["energy"]["settle_s"] + 600
    assert warm["spec"]["activeDeadlineSeconds"] == int(plan["server"]["warmup_timeout_s"] + load)
    assert scored["spec"]["activeDeadlineSeconds"] == int(plan["server"]["ready_timeout_s"] + load)


@pytest.mark.parametrize("secret, passed", [("hf_abc", True), ("none", False)])
def test_guidellm_gets_the_token_only_when_real(secret, passed, monkeypatch, tmp_path):
    backend, _ = _side_backend(secret)
    plan = _plan()
    backend.prepare(plan)
    seen = {}
    monkeypatch.setattr(guidellm, "run", lambda plan, target, out_dir, env=None: seen.setdefault("env", env))
    backend.run_load(plan, plan["launches"][2], "http://side:8000", tmp_path)
    assert seen["env"] == ({"HF_TOKEN": "hf_abc"} if passed else None)


def test_side_pod_like_the_patterns_server():
    backend, _ = _side_backend()
    plan = _plan()
    backend.prepare(plan)
    side = plan["sides"][1]
    twin = plan["launches"][5]  # launch 6: side B's twin in round 1, which launches side B
    assert (twin["index"], twin["side"], twin["of"], twin["round"]) == (6, "B~same", "B", 1)
    body = backend.pod_body(plan, twin, side, plans.side_spec(plan, side))
    spec = body["spec"]
    [container] = spec["containers"]
    assert body["metadata"]["name"] == f"cmp-{plan['run_id'][-8:]}-06-b-same"
    labels = body["metadata"]["labels"]
    assert labels[kube.LABEL_RUN] == plan["run_id"]
    assert (labels["rhvp-compare/side"], labels["rhvp-compare/launch"], labels["rhvp-compare/round"]) == (
        "B-same", "6", "1")
    warm = backend.pod_body(plan, plan["launches"][1], side, plans.side_spec(plan, side))
    assert warm["metadata"]["name"].endswith("-02-b") and warm["metadata"]["labels"]["rhvp-compare/round"] == "warmup"
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
    body = backend.pod_body(plan, plan["launches"][0], side, plans.side_spec(plan, side))
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
    mock = backend.pod_body(plan, plan["launches"][0], plan["sides"][0], plans.side_spec(plan, plan["sides"][0]))
    assert mock["spec"]["tolerations"] == []
    assert "nvidia.com/gpu" not in json.dumps(mock["spec"]["containers"][0]["resources"])
    assert mock["spec"]["containers"][0]["command"] == ["guidellm", "mock-server"]
    ray = backend.pod_body(plan, plan["launches"][1], plan["sides"][1], plans.side_spec(plan, plan["sides"][1]))
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


class FakeDocker:
    """The Engine API calls DockerBackend makes for the run lock and the server container."""

    def __init__(self, running=None, server_state=None, start_fails: int = 0):
        self.running_list = running or []
        self.server_state = server_state
        self.start_fails = start_fails
        self.calls = []

    def running(self, label):
        self.calls.append(("running", label))
        return self.running_list

    def inspect(self, cid):
        self.calls.append(("inspect", cid))
        return None if self.server_state is None else {"State": dict(self.server_state)}

    def stop(self, cid, grace_s=30):
        self.calls.append(("stop", cid, grace_s))
        self.server_state = {"Running": False, "Status": "exited"}

    def start(self, cid):
        self.calls.append(("start", cid))
        if self.start_fails:
            self.start_fails -= 1
            raise runner.RoundError("docker could not start a container: HTTP 500")
        self.server_state = {"Running": True, "Status": "running"}

    def network(self, name):
        pass

    def volume(self, name):
        pass

    def pull(self, ref):
        self.calls.append(("pull", ref))

    def run(self, name, body, timeout_s=300.0, log_to=None):
        self.calls.append(("run", name, body))
        return 0, getattr(self, "run_log", "")

    def info(self):
        return {}

    def by_label(self, value):
        return []


def _stops(docker) -> list[tuple]:
    """(container, grace in seconds) of every stop call."""
    return [c[1:] for c in docker.calls if c[0] == "stop"]


def _docker_backend(docker, results_root: Path = Path("/results"), **env) -> dockerapi.DockerBackend:
    base = {"RHVP_COMPARE_NETWORK": "n", "RHVP_COMPARE_HOST_RESULTS": "/tmp/r", "RHVP_COMPARE_RUNNER_NAME": "me"}
    return dockerapi.DockerBackend(docker=docker, env={**base, **env}, results_root=results_root)


@pytest.mark.parametrize("text, expected", [("", None), ("32g", 32 * 2**30), ("512m", 512 * 2**20),
                                            ("1073741824", 2**30)])
def test_side_memory_limit(text, expected):
    assert dockerapi.memory_bytes(text) == expected
    with pytest.raises(runner.RoundError):
        dockerapi.memory_bytes("lots")
    backend = _docker_backend(FakeDocker(), RHVP_COMPARE_MEMORY=text)
    assert backend.memory == expected


def _container(name: str, created: int) -> dict:
    return {"Id": f"{name}-id", "Names": [f"/{name}"], "Created": created}


@pytest.mark.parametrize("others, refused", [
    ([], False),
    ([_container("old", 100)], True),
    ([_container("young", 300)], False),
])
def test_one_local_run_at_a_time(others, refused):
    docker = FakeDocker(running=[_container("me", 200)] + others)
    backend = _docker_backend(docker)
    if refused:
        with pytest.raises(runner.RunActive, match="runner container old"):
            backend.check_alone()
    else:
        backend.check_alone()
    assert docker.calls == [("running", dockerapi.RUNNER_LABEL)]


def test_the_server_container_is_stopped_for_the_run_and_started_after(monkeypatch):
    monkeypatch.setattr(dockerapi.time, "sleep", lambda s: None)
    docker = FakeDocker(server_state={"Running": True}, start_fails=1)
    backend = _docker_backend(docker, RHVP_COMPARE_SERVER_CONTAINER="live")
    backend.stop_grace_s = 17  # not docker's default of 30, so the stop is seen to pass the backend's own grace
    backend.prepare(_plan())
    assert _stops(docker) == [("live", 17)] and backend.server_found_down is None
    kinds = [c[0] for c in docker.calls]
    # The server stops only after every pull and the volume helper: a stop during a long pull leaves it running.
    assert kinds.index("stop") > max(i for i, k in enumerate(kinds) if k in ("pull", "run"))
    assert backend.restore_server() is True  # the first start failed, the second went through
    assert [c for c in docker.calls if c[0] == "start"] == [("start", "live")] * 2
    assert backend.restore_server() is True and len([c for c in docker.calls if c[0] == "start"]) == 2
    stuck = FakeDocker(server_state={"Running": True}, start_fails=99)
    backend = _docker_backend(stuck, RHVP_COMPARE_SERVER_CONTAINER="live")
    backend.prepare(_plan())
    with pytest.raises(runner.RoundError, match="docker start live"):
        backend.restore_server()
    idle = _docker_backend(FakeDocker(server_state={"Running": False, "Status": "exited"}),
                           RHVP_COMPARE_SERVER_CONTAINER="live")
    idle.prepare(_plan())
    assert idle.restore_server() is None  # it was not running: left alone
    assert idle.server_found_down is None  # and it was not running when the run began either: nothing to record
    missing = FakeDocker()
    with pytest.raises(runner.RoundError, match="names no container"):
        _docker_backend(missing, RHVP_COMPARE_SERVER_CONTAINER="missing").prepare(_plan())
    assert not [c for c in missing.calls if c[0] == "pull"]  # a wrong name fails before any pull


class HookDocker(FakeDocker):
    """A FakeDocker that runs a callback once: during the first pull (`at="pull"`), or during the second read of the
    server's state, the one just before its stop (`at="re-read"`). It stands in for a signal or a state change that
    lands then."""

    def __init__(self, hook, at: str = "pull", **kwargs):
        super().__init__(**kwargs)
        self.hook, self.at, self.inspects = hook, at, 0

    def _fire(self):
        hook, self.hook = self.hook, None
        if hook:
            hook()

    def pull(self, ref):
        super().pull(ref)
        if self.at == "pull":
            self._fire()

    def inspect(self, cid):
        self.inspects += 1
        if self.at == "re-read" and self.inspects == 2:
            self._fire()
        return super().inspect(cid)


def test_a_stop_during_the_pulls_leaves_the_server_running():
    holder = {}
    docker = HookDocker(lambda: holder["backend"].on_stop(), server_state={"Running": True})
    backend = holder["backend"] = _docker_backend(docker, RHVP_COMPARE_SERVER_CONTAINER="live")
    backend.prepare(_plan())
    assert _stops(docker) == [] and docker.server_state["Running"]
    assert backend.restore_server() is None  # nothing was stopped, so nothing to bring back
    assert backend.server_found_down is None  # it is still running: nothing to record


@pytest.mark.parametrize("now, found", [({"Running": False, "Status": "exited"}, "exited"), (None, "gone")])
def test_the_server_state_is_read_again_just_before_the_stop(now, found):
    docker = HookDocker(lambda: setattr(docker, "server_state", now), server_state={"Running": True})
    backend = _docker_backend(docker, RHVP_COMPARE_SERVER_CONTAINER="live")
    backend.prepare(_plan())
    assert _stops(docker) == []  # it had stopped by itself (or been removed) during the pulls: left alone
    assert backend.restore_server() is None
    assert backend.server_found_down == found  # and the runner records it (run.server_found_down)


@pytest.mark.parametrize("at", ["pull", "re-read"])
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_a_signal_before_the_server_stop_leaves_it_running(tmp_path, sig, at):
    """The whole path, with a real signal sent during the image pulls or during the server's second state read:
    Runner._on_term hands the stop to the backend at once (on_stop), so prepare() does not stop the server, and the
    run's end has nothing to start again. SIGINT is Ctrl-C on scripts/compare-local.sh."""
    held = {}

    def send():
        # Only once the runner's handler is in place; otherwise the signal would end the test process itself.
        assert signal.getsignal(sig) == held["runner"]._on_term
        os.kill(os.getpid(), sig)

    docker = HookDocker(send, at=at, server_state={"Running": True, "Status": "running"})
    backend = _docker_backend(docker, results_root=tmp_path, RHVP_COMPARE_SERVER_CONTAINER="live")
    the_runner = held["runner"] = runner.Runner(_plan(), backend, tmp_path, sleep=lambda s: None)
    results = the_runner.run()
    assert docker.hook is None  # the signal was sent
    assert _stops(docker) == [] and not [c for c in docker.calls if c[0] == "start"]
    assert docker.server_state["Running"]
    run = results["run"]
    assert (run["status"], run["error_code"], run["server_restored"]) == ("partial", runner.STOPPED, None)
    assert results["launches"] == [] and run["server_found_down"] is None
    assert run["errors"] == ["stopped by SIGTERM"]  # the runner's words for any stop; no cleanup error
    assert signal.getsignal(sig) != the_runner._on_term  # the earlier handler is back


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_a_server_that_stopped_by_itself_is_recorded_also_in_a_run_stopped_during_the_pulls(tmp_path, sig):
    """The live server exits by itself during a pull, and a stop lands in the same pull: the run still records it
    (run.server_found_down, a note, results.md), and neither stops nor starts it."""
    held = {}

    def exit_then_signal():
        docker.server_state = {"Running": False, "Status": "exited"}
        assert signal.getsignal(sig) == held["runner"]._on_term
        os.kill(os.getpid(), sig)

    docker = HookDocker(exit_then_signal, at="pull", server_state={"Running": True, "Status": "running"})
    backend = _docker_backend(docker, results_root=tmp_path, RHVP_COMPARE_SERVER_CONTAINER="live")
    the_runner = held["runner"] = runner.Runner(_plan(), backend, tmp_path, sleep=lambda s: None)
    results = the_runner.run()
    assert docker.hook is None  # the signal was sent
    assert _stops(docker) == [] and not [c for c in docker.calls if c[0] == "start"]
    run = results["run"]
    assert (run["status"], run["error_code"], run["server_restored"]) == ("partial", runner.STOPPED, None)
    assert run["server_found_down"] == "exited"
    assert any(note.startswith(runner.FOUND_DOWN) for note in run["notes"])
    assert "**The live server had stopped by itself**" in (the_runner.dir / "results.md").read_text()


class FailingReread(HookDocker):
    """The second read of the server's state (after the pulls) fails, as a Docker API error would."""

    def inspect(self, cid):
        if self.inspects == 1:
            self.inspects += 1
            raise runner.RoundError("docker could not inspect a container: HTTP 500")
        return super().inspect(cid)


def test_a_state_read_that_fails_during_a_stop_leaves_the_run_partial(tmp_path):
    """A stop lands during the pulls, and the state read after them fails: the server is left as it is either way,
    so the run is stopped (partial, exit 1) with a note, not failed (exit 3). Without a stop the same error fails
    the run before it has changed anything."""
    held = {}
    docker = FailingReread(lambda: os.kill(os.getpid(), signal.SIGTERM), at="pull",
                           server_state={"Running": True, "Status": "running"})
    backend = _docker_backend(docker, results_root=tmp_path, RHVP_COMPARE_SERVER_CONTAINER="live")
    the_runner = held["runner"] = runner.Runner(_plan(), backend, tmp_path, sleep=lambda s: None)
    results = the_runner.run()
    run = results["run"]
    assert (run["status"], run["error_code"], run["server_found_down"]) == ("partial", runner.STOPPED, None)
    assert runner.exit_code(results) == runner.EXIT_PARTIAL and _stops(docker) == []
    assert "the live server's state was not read again after the stop: RoundError" in run["notes"]
    quiet = FailingReread(None, server_state={"Running": True, "Status": "running"})
    with pytest.raises(runner.RoundError, match="inspect"):
        _docker_backend(quiet, results_root=tmp_path, RHVP_COMPARE_SERVER_CONTAINER="live").prepare(_plan())
    assert _stops(quiet) == []


def test_the_local_gpu_check_sees_the_hosts_gpu_programs(monkeypatch):
    monkeypatch.setattr(dockerapi.time, "sleep", lambda s: None)
    docker = FakeDocker()
    backend = _docker_backend(docker)
    backend.run_id = "t-12345678"
    side = {"image": "x/a:1", "gpu": True}
    backend.gpu_free({}, side)
    [(_, _, body)] = [c for c in docker.calls if c[0] == "run"]
    assert body["HostConfig"]["PidMode"] == "host" and body["HostConfig"]["Memory"] == dockerapi.HELPER_MEMORY
    docker.run_log = "2026-01-01T00:00:00Z 4242\n"  # another program on the GPU
    with pytest.raises(runner.RoundError, match="another program holds the GPU") as caught:
        backend.gpu_free({}, side, wait_s=0)
    assert caught.value.code == runner.GPU_NOT_FREE
