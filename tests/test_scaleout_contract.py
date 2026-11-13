"""The scale-out settings contract: the keys and their defaults, the rules both the values check and the charts
apply, the rhvp-scale log line, the per-start launch record and the comparison's mode. None of them need Ray, a GPU
or a cluster; the render tests use the one render of tests/conftest.py and are skipped the same way."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
from datetime import datetime, timezone
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
import yaml

from chart_helpers import GLOBAL, REPO, docs, one

sys.path.insert(0, str(REPO / "notebooks"))
sys.path.insert(0, str(REPO / "scripts"))

import pattern_settings  # noqa: E402
from rhvp_compare import launch, logs, plan as plans  # noqa: E402

SERVER_VALUES = yaml.safe_load((REPO / "charts/all/ray-serve-llm/values.yaml").read_text())
SCHEMA = json.loads((REPO / "notebooks/rhvp_compare/results.schema.json").read_text())


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rhvp_scale = load_module("rhvp_scale", REPO / "inference-server/rhvp_scale.py")


def merge(base, over):
    if isinstance(base, dict) and isinstance(over, dict):
        return {**base, **{k: merge(base[k], v) if k in base else v for k, v in over.items()}}
    return over


def chart_values(**over) -> dict:
    """The server chart's values as Helm sees them with values-global.yaml, plus `over`."""
    return merge(merge(copy.deepcopy(SERVER_VALUES), GLOBAL), over)


# ---- keys and defaults ------------------------------------------------------------------------


def test_defaults_change_nothing():
    server = SERVER_VALUES["server"]
    assert server["autoscaling"] == {"minReplicas": 1, "maxReplicas": 1, "targetOngoingRequests": None}
    assert "minReplicas" not in server and "maxReplicas" not in server
    assert server["gpuPerReplica"] == 1 and server["gpuCount"] == 1
    assert server["metricsPort"] == 8080
    assert SERVER_VALUES["compileCache"] == {"mode": "none"}
    assert SERVER_VALUES["modelStorage"]["accessMode"] == "ReadWriteOnce"
    assert set(SERVER_VALUES["scaleOut"]) >= {"minPods", "maxPods", "waitingRequestsPerPod", "pollingIntervalSeconds",
                                              "cooldownSeconds", "thanosUrl"}
    assert SERVER_VALUES["global"]["scaleOut"] == {"mode": "none"}
    assert GLOBAL["global"]["scaleOut"] == {"mode": "none"}
    assert GLOBAL["global"]["compare"]["mode"] == "pairs"
    assert pattern_settings.check(chart_values()) == []


def test_values_check_names_every_mode():
    assert pattern_settings.SCALE_OUT_MODES == ("none", "keda")
    assert pattern_settings.SCALE_OUT_DESIGNED == ("kuberay",)
    assert pattern_settings.COMPARE_MODES == plans.MODES == ("pairs", "scaleout")
    assert pattern_settings.COMPILE_CACHE_MODES == ("none", "node", "shared")


# ---- the rules ------------------------------------------------------------------------------


@pytest.mark.parametrize("over, message", [
    ({"global": {"scaleOut": {"mode": "kuberay"}}}, pattern_settings.KUBERAY_NOT_BUILT),
    ({"global": {"scaleOut": {"mode": "hpa"}}}, "global.scaleOut.mode must be none or keda"),
    ({"global": {"scaleOut": {"mode": False}}}, "global.scaleOut.mode must be none or keda"),
    ({"global": {"compare": {"mode": "surge"}}}, "global.compare.mode must be pairs or scaleout; got surge"),
    ({"server": {"minReplicas": 2}}, pattern_settings.MOVED),
    ({"server": {"autoscaling": {"minReplicas": 0}}}, "server.autoscaling.minReplicas must be a whole number"),
    ({"server": {"autoscaling": {"maxReplicas": 1.5}}}, "server.autoscaling.maxReplicas must be a whole number"),
    ({"server": {"autoscaling": {"minReplicas": 2}}}, "server.autoscaling.maxReplicas must be a whole number"),
    ({"server": {"autoscaling": {"maxReplicas": 2}}}, pattern_settings.NO_TARGET),
    ({"server": {"autoscaling": {"targetOngoingRequests": 0}}}, "targetOngoingRequests must be a number above 0"),
    ({"server": {"autoscaling": {"maxReplicas": 2, "targetOngoingRequests": 4}}},
     "server.autoscaling.maxReplicas (2) times server.gpuPerReplica (1) is more than server.gpuCount (1)"),
    ({"server": {"gpuPerReplica": 0}}, "server.gpuPerReplica must be a number above 0"),
    ({"server": {"gpuPerReplica": "0.5"}}, "server.gpuPerReplica must be a number above 0; got 0.5 (string)"),
    ({"server": {"metricsPort": 8000}}, "server.metricsPort must be a port number"),
    ({"compileCache": {"mode": "disk"}}, "compileCache.mode must be none, node or shared"),
    ({"modelStorage": {"accessMode": "ReadOnlyMany"}}, "modelStorage.accessMode must be ReadWriteOnce or ReadWriteMany"),
    ({"global": {"scaleOut": {"mode": "keda"}}, "scaleOut": {"maxPods": 0}}, "scaleOut.maxPods must be a whole number"),
])
def test_values_check_refuses(over, message):
    problems = pattern_settings.check(chart_values(**over))
    assert len(problems) == 1 and message in problems[0], problems


def test_kuberay_message_is_plain():
    assert "kuberay mode is designed but not built yet" in pattern_settings.KUBERAY_NOT_BUILT


@pytest.mark.parametrize("over", [
    {"server": {"autoscaling": {"maxReplicas": 2, "targetOngoingRequests": 4}, "gpuPerReplica": 0.5}},
    {"server": {"autoscaling": {"maxReplicas": 4, "targetOngoingRequests": 8}, "gpuCount": 4}},
    {"server": {"autoscaling": {"maxReplicas": 3, "targetOngoingRequests": 2}, "gpuPerReplica": 1 / 3}},
    {"global": {"scaleOut": {"mode": "keda"}}, "modelStorage": {"accessMode": "ReadWriteMany"}},
    {"compileCache": {"mode": "shared"}},
    {"global": {"compare": {"mode": "scaleout"}}},
])
def test_values_check_accepts(over):
    assert pattern_settings.check(chart_values(**over)) == []


def test_values_check_runs_in_validate_values(tmp_path):
    """validate-values.sh refuses kuberay in values-global.yaml (needs its pinned schemas from the network)."""
    copy_root = tmp_path / "repo"
    for rel in subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.split():
        if (REPO / rel).is_file():
            (copy_root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO / rel, copy_root / rel)
    for rel in ("scripts/pattern_settings.py", "scripts/validate-values.sh", "values-global.yaml",
                "charts/all/ray-serve-llm/values.yaml"):
        shutil.copy(REPO / rel, copy_root / rel)
    text = (copy_root / "values-global.yaml").read_text()
    assert text.count("  scaleOut:\n    mode: none\n") == 1
    (copy_root / "values-global.yaml").write_text(text.replace("  scaleOut:\n    mode: none\n",
                                                               "  scaleOut:\n    mode: kuberay\n"))
    proc = subprocess.run([str(copy_root / "scripts/validate-values.sh")], capture_output=True, text=True,
                          env={**os.environ, "PYTHON": sys.executable}, timeout=300)
    if "sha256" in proc.stderr or "urlopen" in proc.stderr or "URLError" in proc.stderr:
        pytest.skip("validate-values.sh could not download its schemas")
    assert proc.returncode == 1
    assert f"scale-out settings, AWS: {pattern_settings.KUBERAY_NOT_BUILT}" in proc.stderr


# ---- the charts refuse the same settings --------------------------------------------------------


@pytest.mark.parametrize("case, message", [
    ("scaleout-kuberay", pattern_settings.KUBERAY_NOT_BUILT),
    ("scaleout-unknown", "global.scaleOut.mode must be none or keda (kuberay is designed but not built yet); got hpa"),
    ("autoscaling-no-target", pattern_settings.NO_TARGET),
    ("autoscaling-over-gpus",
     "server.autoscaling.maxReplicas (2) times server.gpuPerReplica (1) is more than server.gpuCount (1)"),
    ("replicas-moved", pattern_settings.MOVED),
    ("compare-mode-unknown", "global.compare.mode must be pairs or scaleout; got surge"),
])
def test_charts_refuse_with_the_values_check_message(render, case, message):
    assert message in (render / f"refused/{case}.txt").read_text()


def test_replica_counts_come_from_server_autoscaling(server, render):
    llm = yaml.safe_load(one(server, "ConfigMap", "ray-serve-llm-config")["data"]["llm.yaml"])
    assert (llm["min_replicas"], llm["max_replicas"]) == (1, 1)
    scaling = docs(render / "charts/ray-serve-llm-scaling.yaml")
    llm = yaml.safe_load(one(scaling, "ConfigMap", "ray-serve-llm-config")["data"]["llm.yaml"])
    assert (llm["min_replicas"], llm["max_replicas"]) == (1, 2)


# ---- the rhvp-scale line ---------------------------------------------------------------------


def test_scale_line_shape_is_shared():
    assert logs.SCALE_EVENT == rhvp_scale.EVENT == "rhvp-scale"
    assert logs.SCALE_FIELDS == rhvp_scale.FIELDS == ("event", "ts", "app", "deployment", "from", "to", "reason")
    assert logs.SCALE_REASONS == rhvp_scale.REASONS == ("target", "running")
    assert logs.SCALE_MARKER == rhvp_scale.MARKER


def test_scale_line_round_trip():
    line = rhvp_scale.scale_event("llm", "LLMServer:qwen", 1, 2, "target", ts=1794551388.12345)
    assert line == ('{"event":"rhvp-scale","ts":1794551388.123,"app":"llm","deployment":"LLMServer:qwen",'
                    '"from":1,"to":2,"reason":"target"}')
    assert list(json.loads(line)) == list(rhvp_scale.FIELDS)
    log = "\n".join([
        "2026-11-13T06:10:00.000000000Z INFO starting",
        f"2026-11-13T06:10:01.500000000Z 2026-11-13 06:10:01,500 INFO rhvp: {line}",
        "2026-11-13T06:10:30.250Z (ServeController pid=12) "
        + rhvp_scale.scale_event("llm", "LLMServer:qwen", 1, 2, "running", ts=1794551430.25) + " trailing text",
        rhvp_scale.scale_event("llm", "LLMServer:qwen", 2, 1, "target", ts=1794551500),
    ])
    events = logs.scale_events(log)
    assert [(e["from"], e["to"], e["reason"]) for e in events] == [(1, 2, "target"), (1, 2, "running"),
                                                                   (2, 1, "target")]
    def epoch(text: str) -> float:
        return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()

    assert events[0]["ts"] == 1794551388.123 and events[0]["log_ts"] == pytest.approx(epoch("2026-11-13T06:10:01.5"))
    assert events[1]["log_ts"] == pytest.approx(epoch("2026-11-13T06:10:30.25"))
    assert events[2]["log_ts"] is None and isinstance(events[2]["ts"], float)
    assert set(events[0]) == set(rhvp_scale.FIELDS) | {"log_ts"}


@pytest.mark.parametrize("text", [
    '{"event":"rhvp-scale","ts":1.0,"app":"llm","deployment":"d","from":1,"to":2}',
    '{"event":"rhvp-scale","ts":1.0,"app":"llm","deployment":"d","from":1,"to":2,"reason":"load"}',
    '{"event":"rhvp-scale","ts":1.0,"app":"llm","deployment":"d","from":true,"to":2,"reason":"target"}',
    '{"event":"rhvp-scale","ts":"now","app":"llm","deployment":"d","from":1,"to":2,"reason":"target"}',
    '{"event":"rhvp-scale","ts":1.0,"app":"","deployment":"d","from":1,"to":2,"reason":"target"}',
    '{"event":"rhvp-scale","ts":1.0,"app":"llm","deployment":"d","from":-1,"to":2,"reason":"target"}',
    '{"event":"rhvp-scale","ts":1.0,"app":"llm","deployment":"d","fr',
    '{"event": "rhvp-scale", "ts": 1.0, "app": "llm", "deployment": "d", "from": 1, "to": 2, "reason": "target"}',
])
def test_scale_parser_leaves_out_lines_outside_the_shape(text):
    assert logs.scale_events(f"2026-11-13T06:10:00Z {text}") == []


@pytest.mark.parametrize("args", [
    ("", "d", 1, 2, "target"), ("llm", "", 1, 2, "target"), ("llm", "d", -1, 2, "target"),
    ("llm", "d", 1, True, "target"), ("llm", "d", 1, 2, "load"), ("llm", "d", 1.0, 2, "target"),
])
def test_scale_line_refuses_values_outside_the_shape(args):
    with pytest.raises(ValueError):
        rhvp_scale.scale_event(*args, ts=1.0)
    with pytest.raises(ValueError):
        rhvp_scale.scale_event("llm", "d", 1, 2, "target", ts=float("nan"))


# ---- the per-start launch record --------------------------------------------------------------


def test_launch_env_keeps_the_allow_list_only():
    env = {"GLIBC_TUNABLES": "glibc.malloc.hugetlb=0", "CUDA_VISIBLE_DEVICES": "0", "VLLM_CACHE_ROOT": "/cache/vllm",
           "TORCH_CUDA_ARCH_LIST": "8.9", "HF_HOME": "/models/hf", "HF_TOKEN": "hf_x", "HF_TOKEN_PATH": "/x",
           "VLLM_API_KEY": "k", "VLLM_TOKENIZER_POOL_SIZE": "0", "HUGGING_FACE_HUB_TOKEN": "hf_y", "PATH": "/usr/bin",
           "LD_PRELOAD": "x.so", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HOME": "/cache/home"}
    kept, dropped = launch.recorded_env(env)
    assert kept == {"CUDA_VISIBLE_DEVICES": "0", "GLIBC_TUNABLES": "glibc.malloc.hugetlb=0", "HF_HOME": "/models/hf",
                    "TORCH_CUDA_ARCH_LIST": "8.9", "VLLM_CACHE_ROOT": "/cache/vllm",
                    "VLLM_TOKENIZER_POOL_SIZE": "0"}
    assert dropped == ["HF_TOKEN", "HF_TOKEN_PATH", "VLLM_API_KEY"]
    assert "hf_x" not in json.dumps(launch.record(1, None, [], env, "process"))


def test_launch_record_matches_the_schema():
    override = launch.record(1, ["/opt/app-root/src/entrypoint.sh"], ["--x"], {"VLLM_FOO": "1"}, "container",
                             started_at="2026-11-13T06:10:00Z", image_entrypoint=["/usr/bin/tini"],
                             image_cmd=None)
    assert override["entrypoint"] == "override" and override["env"] == {"VLLM_FOO": "1"}
    image = launch.record(2, None, ["--model", "m"], None, "process")
    assert image["entrypoint"] == "image" and image["env_source"] == "none" and image["env"] == {}
    validator = jsonschema.Draft202012Validator({"$defs": SCHEMA["$defs"], "$ref": "#/$defs/launch"})
    for item in (override, image):
        validator.validate(item)
    assert set(SCHEMA["$defs"]["launch"]["required"]) <= set(override)
    assert set(SCHEMA["$defs"]["launch"]["properties"]) == set(override)
    assert "launches" in SCHEMA["$defs"]["round"]["properties"]
    assert "launches" not in SCHEMA["$defs"]["round"]["required"]
    with pytest.raises(ValueError):
        launch.record(0, None, [], None, "none")
    with pytest.raises(ValueError):
        launch.record(1, None, [], {}, "proc")


def test_side_extra_env_is_in_the_plan():
    config = json.loads((REPO / "tests/fixtures/compare-local-mock.json").read_text())
    config["sides"][1]["env"] = {"GLIBC_TUNABLES": "glibc.malloc.hugetlb=0"}
    built = plans.build(config, "docker")
    assert built["sides"][1]["env"] == {"GLIBC_TUNABLES": "glibc.malloc.hugetlb=0"}
    assert plans.side_spec(built, built["sides"][1])["env"]["GLIBC_TUNABLES"] == "glibc.malloc.hugetlb=0"


# ---- the comparison's mode --------------------------------------------------------------------


def test_compare_mode_pairs_keeps_the_run_id():
    config = json.loads((REPO / "tests/fixtures/compare-local-mock.json").read_text())
    assert "mode" not in config
    assert plans.build(config, "docker")["run_id"] == plans.build({**config, "mode": "pairs"}, "docker")["run_id"]
    assert "mode" not in plans.build(config, "docker")


@pytest.mark.parametrize("mode, message", [
    ("scaleout", "mode scaleout (the scale-out benchmark) is designed but not built yet"),
    ("surge", "mode must be one of pairs, scaleout"),
])
def test_compare_mode_refused(mode, message):
    config = json.loads((REPO / "tests/fixtures/compare-local-mock.json").read_text())
    with pytest.raises(plans.PlanError, match=message.replace("(", r"\(").replace(")", r"\)")):
        plans.build({**config, "mode": mode}, "docker")
