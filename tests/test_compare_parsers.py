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
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

from rhvp_compare import dockerapi, energy, guidellm, kube, launch, logs, plan as plans, runner  # noqa: E402

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


def test_flat_metrics_keep_a_zero():
    levels = guidellm.levels(REPORT)
    levels[0]["output_tokens_per_s"] = 0.0
    levels[0]["energy"] = energy.level_energy(levels[0], energy.Power(150.0, 20, "power_mean", None), "nvidia-smi")
    flat = guidellm.flat_metrics(levels)
    c = levels[0]["concurrency"]
    assert flat[f"c{c}.output_tokens_per_s"] == 0.0 and flat[f"c{c}.tokens_per_joule"] == 0.0
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
    assert kube.image_digest({}) is None


def _side_backend(secret: str | None = None, server_replicas: int | None = None,
                  configmaps: dict | None = None, scale_fails: int = 0, runner_pods: list | None = None,
                  clock=None, scaled_object: dict | None = None, env_extra: dict | None = None,
                  autoscaler_replicas: int | None = None) -> tuple[kube.KubeBackend, list]:
    """A KubeBackend on a fake API. `scale_fails`: how many scale-ups answer HTTP 503 before one goes through.
    `scaled_object`: the server's ScaledObject (its metadata.annotations are patched in place), None for none;
    `autoscaler_replicas`: once it is resumed, the autoscaler sets the server to this many replicas."""
    calls = []
    state = {"replicas": server_replicas, "fails": scale_fails}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.url.path == "/apis/keda.sh/v1alpha1/namespaces/vllm-inference/scaledobjects/ray-serve-llm":
            if scaled_object is None:
                return httpx.Response(404, json={"message": "not found"})
            assert request.headers["content-type"] == "application/merge-patch+json"
            notes = scaled_object.setdefault("metadata", {}).setdefault("annotations", {})
            notes.update(body["metadata"]["annotations"])
            for key in [k for k, v in notes.items() if v is None]:
                del notes[key]
            if not notes and autoscaler_replicas is not None:
                state["autoscaler"] = autoscaler_replicas
            return httpx.Response(200, json=scaled_object)
        if request.url.path.endswith("/deployments/ray-serve-llm/scale"):
            if state["replicas"] is None:
                return httpx.Response(404, json={"message": "not found"})
            if request.method == "PATCH":
                assert request.headers["content-type"] == "application/merge-patch+json"
                if body["spec"]["replicas"] > 0 and state["fails"] > 0:
                    state["fails"] -= 1
                    return httpx.Response(503, json={"message": "unavailable"})
                state["replicas"] = body["spec"]["replicas"]
            if request.method == "GET" and state.get("autoscaler") is not None and state["replicas"]:
                state["replicas"] = state["autoscaler"]
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
    env.update(env_extra or {})
    extra = {"clock": clock} if clock else {}
    return kube.KubeBackend(kube=client, env=env, sleep=lambda s: None, **extra), calls


SCALED = {"RHVP_COMPARE_SERVER_SCALED_OBJECT": "ray-serve-llm"}
KEDA_PATH = "/apis/keda.sh/v1alpha1/namespaces/vllm-inference/scaledobjects/ray-serve-llm"


def test_the_servers_autoscaler_is_paused_for_the_run_and_resumed_first():
    scaled = {"metadata": {"name": "ray-serve-llm", "annotations": {"other": "x"}}}
    backend, calls = _side_backend(server_replicas=1, scaled_object=scaled, env_extra=SCALED, autoscaler_replicas=2)
    backend.prepare(_plan())
    steps = [(m, p.rsplit("/", 1)[1], b) for m, p, b in calls if p == KEDA_PATH or p.endswith("/scale")]
    assert steps == [("GET", "scale", None),
                     ("PATCH", "ray-serve-llm", {"metadata": {"annotations": {kube.KEDA_PAUSED: "0"}}}),
                     ("PATCH", "scale", {"spec": {"replicas": 0}})]
    assert scaled["metadata"]["annotations"] == {"other": "x", kube.KEDA_PAUSED: "0"}
    calls.clear()
    assert backend.restore_server() is True  # the autoscaler chose 2: at least the 1 asked for, so back
    steps = [(m, p.rsplit("/", 1)[1], b) for m, p, b in calls if p == KEDA_PATH or p.endswith("/scale")]
    assert steps == [("PATCH", "ray-serve-llm", {"metadata": {"annotations": {kube.KEDA_PAUSED: None}}}),
                     ("PATCH", "scale", {"spec": {"replicas": 1}}), ("GET", "scale", None)]
    assert scaled["metadata"]["annotations"] == {"other": "x"}


def test_a_missing_autoscaler_is_left_alone():
    backend, calls = _side_backend(server_replicas=1, scaled_object=None, env_extra=SCALED)
    backend.prepare(_plan())
    assert backend.restore_server() is True
    assert [m for m, p, b in calls if p == KEDA_PATH] == ["PATCH"]  # one look at prepare, none at the restore
    plain, calls = _side_backend(server_replicas=1)
    plain.prepare(_plan())
    assert plain.restore_server() is True and not [c for c in calls if "keda" in c[1]]


def test_a_paused_autoscaler_that_cannot_be_resumed_says_how():
    now = [0.0]

    def clock():
        now[0] += 5.0
        return now[0]
    scaled = {"metadata": {"name": "ray-serve-llm"}}
    backend, calls = _side_backend(server_replicas=1, scaled_object=scaled, env_extra=SCALED, clock=clock)
    backend.prepare(_plan())
    real = backend.kube.pause_scaled_object

    def fails(namespace, name, paused):
        if not paused:
            raise runner.RoundError("the Kubernetes API answered HTTP 403 while trying to resume the autoscaler")
        return real(namespace, name, paused)
    backend.kube.pause_scaled_object = fails
    with pytest.raises(runner.RoundError) as caught:
        backend.restore_server()
    assert ("oc annotate scaledobject/ray-serve-llm -n vllm-inference autoscaling.keda.sh/paused-replicas- && "
            "oc scale deployment/ray-serve-llm --replicas=1 -n vllm-inference") in str(caught.value)


def test_the_openshift_job_refuses_the_scale_out_mode_before_it_changes_anything(tmp_path, capsys):
    config = _plan_config()
    config.update(mode="scaleout", rounds={"pairs": 7, "warmup": False})
    path = tmp_path / "compare.json"
    path.write_text(json.dumps(config))
    backend, calls = _side_backend(server_replicas=1, scaled_object={}, env_extra=SCALED)
    with pytest.raises(plans.PlanError, match="mode scaleout runs on the docker backend only"):
        runner.main_run(str(path), str(tmp_path / "results"), backend,
                        env={"RHVP_COMPARE_TERMINATION_LOG": str(tmp_path / "term")})
    assert calls == [] and json.loads((tmp_path / "term").read_text())["error_code"] == "PLAN_INVALID"


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
    first = backend.pod_body(plan, plan["rounds"][0], side, plans.side_spec(plan, side))
    assert "kubernetes.io/hostname" not in first["spec"]["nodeSelector"]
    backend.kube.node = lambda name: {"metadata": {"name": name, "labels": {"kubernetes.io/hostname": "gpu-1"}}}
    backend._pod = lambda handle, fresh=False: {"spec": {"nodeName": "gpu-1.example"}}
    assert backend.facts({"name": "p", "gpu": True})["node_name"] == "gpu-1.example"
    later = backend.pod_body(plan, plan["rounds"][2], side, plans.side_spec(plan, side))
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
    warm = backend.pod_body(plan, plan["rounds"][0], side, plans.side_spec(plan, side))
    scored = backend.pod_body(plan, plan["rounds"][2], side, plans.side_spec(plan, side))
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
    backend.run_load(plan, plan["rounds"][2], "http://side:8000", tmp_path)
    assert seen["env"] == ({"HF_TOKEN": "hf_abc"} if passed else None)


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
    assert results["rounds"] == [] and run["server_found_down"] is None
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


# ---- mode scaleout: the surge, the scale-ups the server logged, the docker replicas ----

SLO_REPORT = json.loads((FIXTURES / "guidellm-report-v0.8.1-slo.json").read_text())


def _scale_plan(**scale) -> dict:
    config = {**_plan_config(), "mode": "scaleout", "rounds": {"pairs": 7, "warmup": True},
              "scaleOut": {"requestRate": 2.5, "surgeSeconds": 60, "stepAfterSeconds": 10, "ttftMs": 300,
                           "tpotMs": 25, **scale}}
    return plans.build(config, "docker")


def _constant(report: dict, rate: float = 2.5) -> dict:
    """The fixture is a concurrent run with latency goals; a surge's report differs in its strategy block only,
    written here by hand as GuideLLM v0.8.1 writes it for the constant profile."""
    out = json.loads(json.dumps(report))
    out["benchmarks"][0]["config"]["strategy"] = {"type_": "constant", "rate": rate, "max_concurrency": None,
                                                  "rampup_duration": 0.0}
    return out


def test_the_surge_command_runs_a_constant_rate_with_the_latency_goals():
    args = guidellm.command(_scale_plan(), "http://127.0.0.1:8787", "/results/r/guidellm.json")
    assert args[args.index("--profile") + 1] == "kind=constant,rate=2.5"
    assert args[args.index("--constraint") + 1] == "kind=max_duration,seconds=60"
    assert "--override" not in args and "kind=json,path=/results/r/guidellm.json" in args
    metrics = json.loads(args[args.index("--metrics") + 1])
    assert metrics == {"kind": "generative", "sample_size": 0, "slo": {"ttft_ms": 300, "tpot_ms": 25}}
    # the shape GuideLLM v0.8.1 records for the goals it was given (the fixture's own config)
    recorded = SLO_REPORT["config"]["spec"]["metrics"]
    assert set(metrics) <= set(recorded) and set(metrics["slo"]) <= set(recorded["slo"])
    one_goal = guidellm.command(_scale_plan(tpotMs=None), "t", "o")
    assert json.loads(one_goal[one_goal.index("--metrics") + 1])["slo"] == {"ttft_ms": 300}
    assert guidellm.command(_plan(), "t", "o")[guidellm.command(_plan(), "t", "o").index("--metrics") + 1] == (
        "kind=generative,sample_size=0")  # mode pairs: unchanged


def test_the_surge_level_from_a_report_with_latency_goals():
    level = guidellm.surge_level(_constant(SLO_REPORT))
    assert level["concurrency"] is None and level["rate"] == 2.5 and level["usable"] and level["note"] is None
    assert level["slo_attainment"] == pytest.approx(0.9574468085106383)
    assert level["slo_determined_requests"] == 47 and level["goodput_rps"] == pytest.approx(11.0)
    assert level["requests"] == {"successful": 47, "errored": 0, "incomplete": 1}
    flat = guidellm.surge_metrics(level)
    assert set(flat) == {"surge.slo_attainment", "surge.goodput_rps", "surge.ttft_ms_median", "surge.itl_ms_median",
                         "surge.output_tokens_per_s"}
    assert flat["surge.ttft_ms_median"] == level["ttft_ms"]["median"]


def test_a_surge_with_failed_requests_keeps_its_attainment_only():
    report = _constant(SLO_REPORT)
    metrics = report["benchmarks"][0]["metrics"]
    metrics["request_totals"]["errored"] = 5
    metrics["slo_attainment"] = 0.5
    level = guidellm.surge_level(report)
    assert not level["usable"] and "count as requests that missed the latency goals" in level["note"]
    assert set(guidellm.surge_metrics(level)) == {"surge.slo_attainment", "surge.goodput_rps"}


@pytest.mark.parametrize("successful, errored, attainment, goodput, expected", [
    (0, 0, None, None, {"surge.slo_attainment": 0.0, "surge.goodput_rps": 0.0}),  # nothing finished: a 0
    (0, 6, 0.0, None, {"surge.slo_attainment": 0.0, "surge.goodput_rps": 0.0}),   # all failed: GuideLLM's 0
    (10, 0, None, None, {}),  # judged none (a goal it cannot measure): missing, not 0
])
def test_a_surge_that_gives_no_attainment(successful, errored, attainment, goodput, expected):
    report = _constant(SLO_REPORT)
    metrics = report["benchmarks"][0]["metrics"]
    metrics["request_totals"].update(successful=successful, errored=errored)
    metrics["slo_attainment"] = attainment
    if goodput is None:
        del metrics["request_goodput"]
    level = guidellm.surge_level(report)
    got = {k: v for k, v in guidellm.surge_metrics(level).items() if k in ("surge.slo_attainment",
                                                                        "surge.goodput_rps")}
    assert got == expected
    if successful and attainment is None:
        assert "judged no request" in level["note"]


def test_start_runs_guidellm_in_the_background(tmp_path):
    fake = tmp_path / "guidellm"
    fake.write_text("#!/bin/sh\nsleep 1\nfor a in \"$@\"; do case \"$a\" in kind=json,path=*) "
                    "echo '{}' > \"${a#kind=json,path=}\";; esac; done\n")
    fake.chmod(0o755)
    running = guidellm.start(_scale_plan(), "http://127.0.0.1:8787", tmp_path / "out", executable=str(fake))
    assert running.poll() is None  # still running: start returned at once
    assert running.finish() == tmp_path / "out" / "guidellm.json"
    hang = tmp_path / "hang"
    hang.write_text("#!/bin/sh\nsleep 30\n")
    hang.chmod(0o755)
    started = guidellm.start(_scale_plan(), "t", tmp_path / "out2", executable=str(hang))
    started.stop()
    assert started.poll() is not None


def test_the_first_scale_up_the_server_logged():
    events = [{"ts": 90.0, "from": 1, "to": 2, "reason": "target"}, {"ts": 112.5, "from": 1, "to": 2, "reason": "running"},
              {"ts": 105.0, "from": 1, "to": 2, "reason": "target"}, {"ts": 130.0, "from": 2, "to": 1,
                                                                    "reason": "target"}]
    assert logs.first_scale_up(events, 100.0) == 5.0  # a decision before the surge and a scale-down do not count
    assert logs.first_scale_up(events[3:], 100.0) is None


class ReplicaDocker(FakeDocker):
    """FakeDocker with the calls a side's start, its launch facts and the cache volumes make."""

    def __init__(self, exec_result=(0, "VLLM_X=1\0HF_TOKEN=hf_s\0PATH=/bin\0")):
        super().__init__()
        self.created, self.volumes_removed, self.exec_result = {}, [], exec_result

    def create(self, name, body):
        self.created[name] = body
        return f"{name}-id"

    def image(self, ref):
        return {"Config": {"Entrypoint": ["/usr/bin/tini", "--"], "Cmd": ["serve"]}}

    def exec(self, cid, cmd, timeout_s=30.0):
        self.calls.append(("exec", cid, cmd))
        if isinstance(self.exec_result, Exception):
            raise self.exec_result
        return self.exec_result

    def inspect(self, cid):
        return {"State": {"Running": True}, "Config": {"Env": ["VLLM_Y=2", "PATH=/usr/bin"]}}

    def by_label(self, value):
        return []

    def remove_volume(self, name):
        self.volumes_removed.append(name)


class ProbedDocker(ReplicaDocker):
    """nvidia-smi's answer per GPU probe: `answers` maps the probe's replica (1 or 2) to its index, UUID line."""

    def __init__(self, answers: dict):
        super().__init__()
        self.answers = answers

    def run(self, name, body, timeout_s=300.0, log_to=None):
        self.calls.append(("run", name, body))
        for replica, answer in self.answers.items():
            if f"gpu-probe-{replica}" in name:
                return 0, f"2026-01-01T00:00:00Z {answer}\n"
        return 0, ""


def test_docker_replicas_get_their_own_gpu_the_cpu_set_and_the_warm_cache():
    docker = ProbedDocker({1: "0, GPU-0000", 2: "1, GPU-1111"})  # what nvidia-smi lists for each replica's GPU
    backend = _docker_backend(docker, RHVP_COMPARE_REPLICA_GPUS="0,1", RHVP_COMPARE_CPUSET="0-3",
                              RHVP_COMPARE_CACHE_VOLUME="cc")
    config = {**_plan_config(), "mode": "scaleout", "rounds": {"pairs": 7, "warmup": True},
              "scaleOut": {"sharedGpu": False},
              "sides": [{"name": "a", "image": "x/a:1", "compileCache": "warm"}, {"name": "b", "image": "x/b:1"}]}
    plan = plans.build(config, "docker")
    backend.prepare(plan)
    volume = f"cc-{plan['run_id'][-8:]}-a"
    assert backend.cache_volumes == {"A": volume} and docker.volumes_removed == [volume]  # a fresh one
    chmod = [body for kind, name, body in [c for c in docker.calls if c[0] == "run"] if "cache" in name]
    assert chmod[0]["Cmd"] == ["chmod 1777 /cache"] and chmod[0]["HostConfig"]["CpusetCpus"] == "0-3"
    side_a, side_b = plan["sides"]
    round_ = plan["rounds"][2]
    first = backend.start(plan, round_, side_a, plans.side_spec(plan, side_a))
    second = backend.start(plan, round_, side_a, plans.side_spec(plan, side_a), replica=2)
    assert second["name"] == first["name"] + "-2" and second["replica"] == 2
    host1, host2 = docker.created[first["name"]]["HostConfig"], docker.created[second["name"]]["HostConfig"]
    assert host1["DeviceRequests"][0]["DeviceIDs"] == ["0"] and host2["DeviceRequests"][0]["DeviceIDs"] == ["1"]
    assert f"{volume}:/cache" in host2["Binds"] and "/cache" not in host2["Tmpfs"] and host2["CpusetCpus"] == "0-3"
    cold = backend.start(plan, plan["rounds"][3], side_b, plans.side_spec(plan, side_b))
    cold_host = docker.created[cold["name"]]["HostConfig"]
    assert "/cache" in cold_host["Tmpfs"] and not any(b.endswith(":/cache") for b in cold_host["Binds"])
    with pytest.raises(runner.RoundError, match="replica 3 needs one more"):
        backend.start(plan, round_, side_a, plans.side_spec(plan, side_a), replica=3)
    backend.cleanup()
    assert docker.volumes_removed == [volume, volume]


def _scale_plan_for(shared: bool) -> dict:
    config = {**_plan_config(), "mode": "scaleout", "rounds": {"pairs": 7, "warmup": True},
              "scaleOut": {"sharedGpu": shared}}
    return plans.build(config, "docker")


@pytest.mark.parametrize("shared, replica_gpus, refused", [
    (True, "", None),
    (True, "0,0", None),
    (False, "0,1", None),
    (True, "0,1", "gives replica 1 GPU 0 and replica 2 GPU 1"),
    (False, "", "both replicas would get GPUs all"),
    (False, "1,1", "both replicas would get GPUs 1"),
    (False, "0", "needs one GPU id per replica"),
])
def test_docker_refuses_a_shared_gpu_setting_that_does_not_match_where_the_replicas_run(shared, replica_gpus,
                                                                                       refused):
    # scaleOut.sharedGpu sets the memory share and the notebook's sentence; REPLICA_GPUS (else GPUS) places the
    # replicas. A run whose two disagree is refused before anything changes, and prepare() refuses it too.
    plan = _scale_plan_for(shared)
    backend = _docker_backend(FakeDocker(), RHVP_COMPARE_REPLICA_GPUS=replica_gpus)
    if refused is None:
        backend.check_plan(plan)
        return
    with pytest.raises(plans.PlanError, match=refused):
        backend.check_plan(plan)
    docker = FakeDocker(server_state={"Running": True})
    backend = _docker_backend(docker, RHVP_COMPARE_REPLICA_GPUS=replica_gpus, RHVP_COMPARE_SERVER_CONTAINER="live")
    with pytest.raises(plans.PlanError, match=refused):
        backend.prepare(plan)
    assert docker.calls == []  # nothing pulled, nothing stopped


def test_a_mismatched_shared_gpu_setting_ends_the_run_plan_invalid_before_anything_changes(tmp_path):
    config_path = tmp_path / "compare.json"
    config_path.write_text(json.dumps({**_plan_config(), "mode": "scaleout", "rounds": {"pairs": 7, "warmup": True}}))
    docker = FakeDocker()
    backend = _docker_backend(docker, RHVP_COMPARE_REPLICA_GPUS="0,1")
    termination = tmp_path / "termination"
    with pytest.raises(plans.PlanError, match="scaleOut.sharedGpu is true"):
        runner.main_run(str(config_path), str(tmp_path / "results"), backend,
                        env={"RHVP_COMPARE_TERMINATION_LOG": str(termination)})
    assert "PLAN_INVALID" in termination.read_text() and docker.calls == []
    assert not (tmp_path / "results").exists()


def test_docker_reads_each_replicas_gpus_before_the_first_round_and_records_them():
    docker = ProbedDocker({1: "0, GPU-0000", 2: "1, GPU-1111"})
    backend = _docker_backend(docker, RHVP_COMPARE_REPLICA_GPUS="0,1")
    plan = _scale_plan_for(False)
    backend.prepare(plan)
    probes = [body for kind, name, body in [c for c in docker.calls if c[0] == "run"] if "gpu-probe" in name]
    assert [b["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] for b in probes] == [["0"], ["1"]]
    assert probes[0]["Cmd"] == ["--query-gpu=index,uuid", "--format=csv,noheader"]
    side_a = plan["sides"][0]
    second = backend.start(plan, plan["rounds"][2], side_a, plans.side_spec(plan, side_a), replica=2)
    facts = backend.launch_facts(second)
    assert facts["gpus"] == "1" and facts["gpu_uuids"] == ["GPU-1111"]
    record = launch.record(2, None, [], facts["env"], facts["env_source"], gpus=facts["gpus"],
                           gpu_uuids=facts["gpu_uuids"])
    assert (record["gpus"], record["gpu_uuids"]) == ("1", ["GPU-1111"])
    shared = _docker_backend(ReplicaDocker())  # sharedGpu true, no REPLICA_GPUS: one request, read once
    shared.docker.run_log = "2026-01-01T00:00:00Z 0, GPU-0000\n"
    shared.prepare(_scale_plan_for(True))
    assert len([c for c in shared.docker.calls if c[0] == "run" and "gpu-probe" in c[1]]) == 1


def test_two_requests_that_reach_one_gpu_fail_in_prepare_when_each_replica_needs_its_own():
    # An index and that GPU's UUID are different strings, so check_plan lets them through; nvidia-smi's UUIDs do not.
    docker = ProbedDocker({1: "0, GPU-0000", 2: "0, GPU-0000"})
    docker.server_state = {"Running": True}
    backend = _docker_backend(docker, RHVP_COMPARE_REPLICA_GPUS="0,GPU-0000", RHVP_COMPARE_SERVER_CONTAINER="live")
    plan = _scale_plan_for(False)
    backend.check_plan(plan)
    with pytest.raises(runner.RoundError, match="REPLICA_GPUS 0 and GPU-0000 both reach GPU GPU-0000"):
        backend.prepare(plan)
    assert ("stop", "live") not in docker.calls
    shared = _docker_backend(ProbedDocker({1: "0, GPU-0000", 2: "0, GPU-0000"}),
                             RHVP_COMPARE_REPLICA_GPUS="0,GPU-0000")
    with pytest.raises(plans.PlanError, match="gives replica 1 GPU 0 and replica 2 GPU GPU-0000"):
        shared.check_plan(_scale_plan_for(True))  # sharedGpu true still compares the strings, before any probe


@pytest.mark.parametrize("request_, mig", [
    ("0:0", True), ("12:3", True), ("MIG-4f2a9c1e-0000-5000-8000-000000000001", True),
    ("MIG-GPU-4f2a9c1e-0000-5000-8000-000000000000/1/0", True), ("0:0,0:1", True),
    ("0", False), ("GPU-4f2a9c1e-0000-5000-8000-000000000000", False), ("all", False), ("0:0,1", False), ("", False),
    ("0:", False),
])
def test_mig_requests_are_told_from_whole_gpus(request_, mig):
    assert dockerapi.is_mig_request(request_) is mig


@pytest.mark.parametrize("replica_gpus", ["0:0,0:1", "MIG-aaaa,MIG-bbbb"])
def test_two_mig_devices_of_one_gpu_run_when_each_replica_needs_its_own(replica_gpus):
    # nvidia-smi in a container given a MIG device lists the parent GPU's UUID, so both probes show GPU-0000: two MIG
    # requests are not compared, and the run goes on. Their launch records name the parent GPU.
    docker = ProbedDocker({1: "0, GPU-0000", 2: "0, GPU-0000"})
    backend = _docker_backend(docker, RHVP_COMPARE_REPLICA_GPUS=replica_gpus)
    plan = _scale_plan_for(False)
    backend.check_plan(plan)
    backend.prepare(plan)
    side_a = plan["sides"][0]
    second = backend.start(plan, plan["rounds"][2], side_a, plans.side_spec(plan, side_a), replica=2)
    facts = backend.launch_facts(second)
    assert facts["gpus"] == replica_gpus.split(",")[1] and facts["gpu_uuids"] == ["GPU-0000"]


def test_a_mig_device_and_its_whole_gpu_fail_in_prepare_when_each_replica_needs_its_own():
    docker = ProbedDocker({1: "0, GPU-0000", 2: "0, GPU-0000"})
    docker.server_state = {"Running": True}
    backend = _docker_backend(docker, RHVP_COMPARE_REPLICA_GPUS="0,0:1", RHVP_COMPARE_SERVER_CONTAINER="live")
    plan = _scale_plan_for(False)
    backend.check_plan(plan)
    with pytest.raises(runner.RoundError, match="REPLICA_GPUS 0 and 0:1 both reach GPU GPU-0000"):
        backend.prepare(plan)
    assert ("stop", "live") not in docker.calls


def test_a_replica_gpu_this_machine_lacks_fails_in_prepare_before_the_server_stops():
    class NoSecondGpu(ReplicaDocker):
        def run(self, name, body, timeout_s=300.0, log_to=None):
            self.calls.append(("run", name, body))
            if "gpu-probe-2" in name:
                raise runner.RoundError("docker could not start a container: HTTP 500 unknown device id: 1")
            return 0, "2026-01-01T00:00:00Z 0, GPU-0000\n"

    docker = NoSecondGpu()
    docker.server_state = {"Running": True}
    backend = _docker_backend(docker, RHVP_COMPARE_REPLICA_GPUS="0,1", RHVP_COMPARE_SERVER_CONTAINER="live")
    with pytest.raises(runner.RoundError, match="the GPUs of replica 2 .REPLICA_GPUS 1. could not be reached"):
        backend.prepare(_scale_plan_for(False))
    assert ("stop", "live") not in docker.calls


def test_docker_launch_facts_read_the_server_process_then_the_container():
    docker = ReplicaDocker()
    backend = _docker_backend(docker)
    facts = backend.launch_facts({"id": "c1", "image": "x/a:1"})
    assert facts == {"image_entrypoint": ["/usr/bin/tini", "--"], "image_cmd": ["serve"], "env_source": "process",
                     "env": {"VLLM_X": "1", "HF_TOKEN": "hf_s", "PATH": "/bin"}}
    assert docker.calls[-1] == ("exec", "c1", ["sh", "-c", dockerapi.SERVER_ENVIRON])
    no_shell = _docker_backend(ReplicaDocker(exec_result=runner.RoundError("docker could not start an exec")))
    facts = no_shell.launch_facts({"id": "c1", "image": "x/a:1"})
    assert facts["env_source"] == "container" and facts["env"] == {"VLLM_Y": "2", "PATH": "/usr/bin"}
    assert dockerapi.environ("A=1\0B=x=y\0\0junk\0") == {"A": "1", "B": "x=y"}


def test_the_server_environ_script_picks_the_largest_process(tmp_path):
    # Run the script against this machine's own /proc: it must print some process's environment, NUL-separated.
    out = subprocess.run(["sh", "-c", dockerapi.SERVER_ENVIRON], capture_output=True, timeout=30)
    assert out.returncode == 0 and b"=" in out.stdout and b"\0" in out.stdout
