"""Unit tests for the inference server's config handling, status helpers,
lock files and image definition. None of them need Ray, vLLM or a GPU."""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO / "inference-server"
sys.path.insert(0, str(SERVER_DIR))

import rhvp_status  # noqa: E402
import serve_app  # noqa: E402
from serve_app import ConfigError, build_llm_config_kwargs, load_config  # noqa: E402

# The versions the notebook's baseline label names.
BASELINE = {"Ray": "2.60.0", "vLLM": "0.29.0", "PyTorch": "2.13.0"}
LOCKS = {
    "x86_64": SERVER_DIR / "requirements-x86_64.lock",
    "aarch64": SERVER_DIR / "requirements-aarch64.lock",
}

# The llm.yaml the chart renders (CONTRACT), with a placeholder revision.
CONTRACT_YAML = """\
app_name: llm
model_id: qwen2.5-0.5b-instruct
model_source: Qwen/Qwen2.5-0.5B-Instruct
model_revision: "0123456789abcdef0123456789abcdef01234567"
engine_kwargs: {max_model_len: 8192, gpu_memory_utilization: 0.85, tensor_parallel_size: 1}
accelerator_type: null
min_replicas: 1
max_replicas: 1
object_store_memory_gb: 2
http_host: 0.0.0.0
http_port: 8000
"""
REV = "0123456789abcdef0123456789abcdef01234567"


def write(tmp_path: Path, text: str, name: str = "llm.yaml") -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def contract_with(**changes) -> str:
    data = yaml.safe_load(CONTRACT_YAML)
    for key, value in changes.items():
        if value is ...:
            data.pop(key, None)
        else:
            data[key] = value
    return yaml.safe_dump(data)


# --------------------------------------------------------------------------
# load_config
# --------------------------------------------------------------------------


def test_contract_file_loads_unchanged(tmp_path):
    cfg = load_config(write(tmp_path, CONTRACT_YAML), {})
    assert cfg == {
        "app_name": "llm",
        "model_id": "qwen2.5-0.5b-instruct",
        "model_source": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": REV,
        "engine_kwargs": {
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.85,
            "tensor_parallel_size": 1,
        },
        "accelerator_type": None,
        "min_replicas": 1,
        "max_replicas": 1,
        "object_store_memory_gb": 2,
        "http_host": "0.0.0.0",
        "http_port": 8000,
    }


def test_minimal_file_gets_contract_defaults(tmp_path):
    text = "model_id: m\nmodel_source: org/m\n"
    cfg = load_config(write(tmp_path, text), {})
    assert cfg["app_name"] == "llm"
    assert cfg["model_revision"] is None
    assert cfg["engine_kwargs"] == {
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.85,
        "tensor_parallel_size": 1,
    }
    assert (cfg["min_replicas"], cfg["max_replicas"]) == (1, 1)
    assert cfg["object_store_memory_gb"] == 2
    assert (cfg["http_host"], cfg["http_port"]) == ("0.0.0.0", 8000)
    assert cfg["accelerator_type"] is None


def test_partial_engine_kwargs_merge_with_defaults(tmp_path):
    text = "model_id: m\nmodel_source: org/m\nengine_kwargs: {max_model_len: 4096, enforce_eager: true}\n"
    cfg = load_config(write(tmp_path, text), {})
    assert cfg["engine_kwargs"] == {
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.85,
        "tensor_parallel_size": 1,
        "enforce_eager": True,
    }


def test_null_max_model_len_is_left_to_vllm(tmp_path):
    cfg = load_config(write(tmp_path, contract_with(engine_kwargs={"max_model_len": None})), {})
    assert "max_model_len" not in cfg["engine_kwargs"]


@pytest.mark.parametrize("revision", ["", "   ", None])
def test_empty_revision_means_latest(tmp_path, revision):
    cfg = load_config(write(tmp_path, contract_with(model_revision=revision)), {})
    assert cfg["model_revision"] is None
    assert "revision" not in build_llm_config_kwargs(cfg)["engine_kwargs"]


def test_unquoted_numeric_revision_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="quote it"):
        load_config(write(tmp_path, contract_with(model_revision=1234567)), {})


def test_unknown_key_is_ignored_with_a_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="rhvp.serve"):
        cfg = load_config(write(tmp_path, CONTRACT_YAML + "route_prefix: /x\n"), {})
    assert "route_prefix" not in cfg
    assert "route_prefix" in caplog.text


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(str(tmp_path / "nope.yaml"), {})


def test_bad_yaml_and_non_mapping(tmp_path):
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(write(tmp_path, "a: [1, 2\n"), {})
    with pytest.raises(ConfigError, match="mapping"):
        load_config(write(tmp_path, "- 1\n- 2\n"), {})


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"model_id": ...}, "model_id"),
        ({"model_source": ""}, "model_source"),
        ({"app_name": ""}, "app_name"),
        ({"min_replicas": 2, "max_replicas": 1}, "must not exceed"),
        ({"max_replicas": 0}, "max_replicas"),
        ({"min_replicas": -1}, "min_replicas"),
        ({"min_replicas": 0}, "min_replicas must be an integer >= 1"),
        ({"min_replicas": True}, "min_replicas"),
        ({"http_port": 70000}, "http_port"),
        ({"http_port": "8000"}, "http_port"),
        ({"object_store_memory_gb": 0}, "object_store_memory_gb"),
        ({"accelerator_type": ""}, "accelerator_type"),
        ({"engine_kwargs": {"gpu_memory_utilization": 0}}, "gpu_memory_utilization"),
        ({"engine_kwargs": {"gpu_memory_utilization": 1.5}}, "gpu_memory_utilization"),
        ({"engine_kwargs": {"max_model_len": 0}}, "max_model_len"),
        ({"engine_kwargs": {"max_model_len": 81.5}}, "max_model_len"),
        ({"engine_kwargs": {"tensor_parallel_size": True}}, "tensor_parallel_size"),
        ({"engine_kwargs": {"model": "x"}}, "engine_kwargs.model"),
        ({"engine_kwargs": {"served_model_name": "x"}}, "served_model_name"),
        ({"engine_kwargs": {"revision": "other"}}, "disagree"),
        ({"engine_kwargs": "not-a-map"}, "engine_kwargs"),
    ],
)
def test_invalid_values_are_rejected(tmp_path, changes, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, contract_with(**changes)), {})


def test_engine_kwargs_revision_alone_becomes_model_revision(tmp_path):
    text = contract_with(model_revision=..., engine_kwargs={"revision": "abc123"})
    cfg = load_config(write(tmp_path, text), {})
    assert cfg["model_revision"] == "abc123"
    assert "revision" not in cfg["engine_kwargs"]
    assert build_llm_config_kwargs(cfg)["engine_kwargs"]["revision"] == "abc123"


# --------------------------------------------------------------------------
# Environment overrides
# --------------------------------------------------------------------------


def test_each_env_override(tmp_path):
    env = {
        "RHVP_MODEL_ID": "other-id",
        "RHVP_MODEL_SOURCE": "org/other",
        "RHVP_MODEL_REVISION": "feedface",
        "RHVP_GPU_MEMORY_UTILIZATION": "0.5",
        "RHVP_MAX_MODEL_LEN": "2048",
    }
    cfg = load_config(write(tmp_path, CONTRACT_YAML), env)
    assert cfg["model_id"] == "other-id"
    assert cfg["model_source"] == "org/other"
    assert cfg["model_revision"] == "feedface"
    assert cfg["engine_kwargs"]["gpu_memory_utilization"] == 0.5
    assert cfg["engine_kwargs"]["max_model_len"] == 2048
    assert isinstance(cfg["engine_kwargs"]["max_model_len"], int)
    assert cfg["engine_kwargs"]["tensor_parallel_size"] == 1


def test_empty_env_overrides_change_nothing(tmp_path):
    env = {name: "" for name in serve_app.ENV_OVERRIDES}
    env["RHVP_MAX_MODEL_LEN"] = "   "
    assert load_config(write(tmp_path, CONTRACT_YAML), env) == load_config(
        write(tmp_path, CONTRACT_YAML), {}
    )


def test_env_beats_file_for_engine_kwargs(tmp_path):
    cfg = load_config(write(tmp_path, CONTRACT_YAML), {"RHVP_GPU_MEMORY_UTILIZATION": "0.3"})
    assert cfg["engine_kwargs"]["gpu_memory_utilization"] == 0.3


@pytest.mark.parametrize(
    "name, value",
    [
        ("RHVP_MAX_MODEL_LEN", "8192.0"),
        ("RHVP_MAX_MODEL_LEN", "lots"),
        ("RHVP_GPU_MEMORY_UTILIZATION", "most"),
    ],
)
def test_unparsable_env_override(tmp_path, name, value):
    with pytest.raises(ConfigError, match=name):
        load_config(write(tmp_path, CONTRACT_YAML), {name: value})


def test_env_override_is_still_validated(tmp_path):
    with pytest.raises(ConfigError, match="gpu_memory_utilization"):
        load_config(write(tmp_path, CONTRACT_YAML), {"RHVP_GPU_MEMORY_UTILIZATION": "2"})


def test_new_model_source_drops_the_files_revision(tmp_path):
    cfg = load_config(write(tmp_path, CONTRACT_YAML), {"RHVP_MODEL_SOURCE": "org/other"})
    assert cfg["model_revision"] is None


def test_new_model_source_with_its_own_revision(tmp_path):
    env = {"RHVP_MODEL_SOURCE": "org/other", "RHVP_MODEL_REVISION": "beef"}
    assert load_config(write(tmp_path, CONTRACT_YAML), env)["model_revision"] == "beef"


def test_same_model_source_keeps_the_files_revision(tmp_path):
    env = {"RHVP_MODEL_SOURCE": "Qwen/Qwen2.5-0.5B-Instruct"}
    assert load_config(write(tmp_path, CONTRACT_YAML), env)["model_revision"] == REV


# --------------------------------------------------------------------------
# build_llm_config_kwargs
# --------------------------------------------------------------------------


def test_llm_config_kwargs_for_the_contract_file(tmp_path):
    kwargs = build_llm_config_kwargs(load_config(write(tmp_path, CONTRACT_YAML), {}))
    assert kwargs == {
        "model_loading_config": {
            "model_id": "qwen2.5-0.5b-instruct",
            "model_source": "Qwen/Qwen2.5-0.5B-Instruct",
        },
        "engine_kwargs": {
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.85,
            "tensor_parallel_size": 1,
            "revision": REV,
        },
        "deployment_config": {"autoscaling_config": {"min_replicas": 1, "max_replicas": 1}},
    }
    # accelerator_type is omitted when null; LLMConfig's default is None.
    assert "accelerator_type" not in kwargs


def test_accelerator_type_is_passed_when_set(tmp_path):
    cfg = load_config(write(tmp_path, contract_with(accelerator_type="L4")), {})
    assert build_llm_config_kwargs(cfg)["accelerator_type"] == "L4"


def test_build_does_not_mutate_the_config(tmp_path):
    cfg = load_config(write(tmp_path, CONTRACT_YAML), {})
    build_llm_config_kwargs(cfg)
    assert "revision" not in cfg["engine_kwargs"]


def test_object_store_bytes(tmp_path):
    cfg = load_config(write(tmp_path, CONTRACT_YAML), {})
    assert serve_app.object_store_bytes(cfg) == 2 * 1024**3
    assert serve_app.object_store_bytes({"object_store_memory_gb": 0.5}) == 512 * 1024**2


# --------------------------------------------------------------------------
# Environment, ready file, status helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env, expected_env, removed",
    [
        ({"HF_TOKEN": "none"}, {}, ["HF_TOKEN"]),
        ({"HF_TOKEN": ""}, {}, ["HF_TOKEN"]),
        ({"HF_TOKEN": "hf_abc"}, {"HF_TOKEN": "hf_abc"}, []),
        ({"HF_TOKEN": " hf_abc\n"}, {"HF_TOKEN": "hf_abc"}, []),
        ({"HUGGING_FACE_HUB_TOKEN": "changeme", "X": "1"}, {"X": "1"}, ["HUGGING_FACE_HUB_TOKEN"]),
        ({}, {}, []),
    ],
)
def test_sanitize_env(env, expected_env, removed):
    env = dict(env)
    assert serve_app.sanitize_env(env) == removed
    assert env == expected_env


def test_ready_file_path():
    assert serve_app.ready_file_path({}) == "/tmp/r/rhvp-ready"
    assert serve_app.ready_file_path({"RAY_TMPDIR": "/tmp/r"}) == "/tmp/r/rhvp-ready"
    assert serve_app.ready_file_path({"RAY_TMPDIR": "/x"}) == "/x/rhvp-ready"


def test_write_and_remove_ready_file(tmp_path):
    path = str(tmp_path / "sub" / "rhvp-ready")
    serve_app.write_ready_file(path)
    assert Path(path).is_file()
    assert [p.name for p in Path(path).parent.iterdir()] == ["rhvp-ready"]  # no temp left
    serve_app.remove_ready_file(path)
    assert not Path(path).exists()
    serve_app.remove_ready_file(path)  # removing twice is fine


class _Enum:
    def __init__(self, value):
        self.value = value


def _status(proxies, apps):
    return SimpleNamespace(
        proxies=proxies,
        applications={name: SimpleNamespace(status=s) for name, s in apps.items()},
    )


@pytest.mark.parametrize(
    "status, expected",
    [
        (_status({}, {}), "NO_CONTROLLER"),
        (_status({}, {"llm": "RUNNING"}), "NO_CONTROLLER"),
        (_status({"n1": object()}, {}), "NOT_STARTED"),
        (_status({"n1": object()}, {"other": "RUNNING"}), "NOT_STARTED"),
        (_status({"n1": object()}, {"llm": "RUNNING"}), "RUNNING"),
        (_status({"n1": object()}, {"llm": _Enum("DEPLOYING")}), "DEPLOYING"),
        (_status({"n1": object()}, {"llm": _Enum("DEPLOY_FAILED")}), "DEPLOY_FAILED"),
        (_status({"n1": object()}, {"llm": None}), "UNKNOWN"),
    ],
)
def test_app_status_name(status, expected):
    assert serve_app.app_status_name(status, "llm") == expected


def test_watch_keeps_ready_file_in_step_with_status(tmp_path):
    ready = str(tmp_path / "rhvp-ready")
    serve_app.write_ready_file(ready)
    stop = threading.Event()
    seen = []
    script = ["RUNNING", "UNHEALTHY", RuntimeError("controller busy"), "RUNNING", "DEPLOYING"]

    def get_status():
        step = script[len(seen)]
        seen.append(Path(ready).exists())
        if len(seen) == len(script):
            stop.set()
        if isinstance(step, Exception):
            raise step
        return step

    serve_app.watch_ready_file(ready, stop, get_status, interval_s=0)
    # File state seen at the start of each check: present, present (RUNNING),
    # gone (after UNHEALTHY), gone (failed check leaves it), present (RUNNING).
    assert seen == [True, True, False, False, True]
    assert not Path(ready).exists()  # removed after the final DEPLOYING


def test_watch_returns_when_stopped(tmp_path):
    stop = threading.Event()
    stop.set()
    calls = []
    serve_app.watch_ready_file(str(tmp_path / "r"), stop, lambda: calls.append(1), interval_s=0)
    assert calls == []


# --------------------------------------------------------------------------
# main() without Ray
# --------------------------------------------------------------------------


def test_main_print_config_needs_no_ray(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RAY_TMPDIR", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "none")
    for name in serve_app.ENV_OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    stale = tmp_path / "rhvp-ready"
    stale.write_text("old\n")
    path = write(tmp_path, CONTRACT_YAML)
    assert serve_app.main(["--config", path, "--print-config"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["llm_config"]["engine_kwargs"]["revision"] == REV
    assert out["config"]["app_name"] == "llm"
    assert not stale.exists(), "a stale ready file must be removed at start"
    assert "HF_TOKEN" not in serve_app.os.environ
    assert "ray" not in sys.modules


def test_main_bad_config_exits_2(tmp_path, monkeypatch):
    monkeypatch.setenv("RAY_TMPDIR", str(tmp_path))
    assert serve_app.main(["--config", str(tmp_path / "missing.yaml")]) == 2


def test_rhvp_status_exit_codes_and_output(monkeypatch, capsys):
    assert rhvp_status.exit_code("RUNNING") == 0
    assert rhvp_status.exit_code("DEPLOY_FAILED") == 2
    for state in ("NO_CONTROLLER", "NOT_STARTED", "DEPLOYING", "UNHEALTHY", "DELETING"):
        assert rhvp_status.exit_code(state) == 1

    states = iter(["NO_CONTROLLER", "DEPLOYING", "RUNNING"])
    monkeypatch.setattr(rhvp_status, "check_once", lambda app, address: next(states))
    monkeypatch.setattr(rhvp_status, "POLL_INTERVAL_S", 0)
    assert rhvp_status.main(["--app", "llm", "--wait", "60"]) == 0
    assert capsys.readouterr().out.strip() == "RUNNING"

    monkeypatch.setattr(rhvp_status, "check_once", lambda app, address: "DEPLOYING")
    assert rhvp_status.main(["--app", "llm"]) == 1
    assert capsys.readouterr().out.strip() == "DEPLOYING"


def test_rhvp_status_reads_app_name_from_config(tmp_path, monkeypatch):
    monkeypatch.setenv("RHVP_CONFIG", write(tmp_path, contract_with(app_name="chat")))
    assert rhvp_status.default_app_name() == "chat"
    monkeypatch.setenv("RHVP_CONFIG", str(tmp_path / "missing.yaml"))
    assert rhvp_status.default_app_name() == "llm"


# --------------------------------------------------------------------------
# Lock files and requirements.in
# --------------------------------------------------------------------------


def _pins(path: Path) -> dict:
    pins = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\;]+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def test_requirements_in_pins():
    text = (SERVER_DIR / "requirements.in").read_text()
    reqs = [l.strip() for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    assert reqs == ["ray[llm]==2.60.0", "cuda-toolkit[nvcc,cccl]==13.0.3"]


@pytest.mark.parametrize("arch", sorted(LOCKS))
def test_lock_matches_requirements_and_baseline(arch):
    pins = _pins(LOCKS[arch])
    ray_in = re.search(r"^ray\[llm\]==(\S+)", (SERVER_DIR / "requirements.in").read_text(), re.M)
    assert ray_in and pins["ray"] == ray_in.group(1)
    label = {"Ray": pins["ray"], "vLLM": pins["vllm"], "PyTorch": pins["torch"]}
    assert label == BASELINE
    assert {f"{k} {v}" for k, v in label.items()} == {"vLLM 0.29.0", "Ray 2.60.0", "PyTorch 2.13.0"}
    # The pip nvcc and CCCL stay on CUDA 13.0, like torch's CUDA libraries.
    assert pins["cuda-toolkit"].startswith("13.0.3")
    assert pins["nvidia-cuda-nvcc"].startswith("13.0.")
    assert pins["nvidia-cuda-cccl"].startswith("13.0.")
    assert pins["nvidia-cuda-runtime"].startswith("13.0.")


def test_image_labels_name_the_locked_versions():
    pins = _pins(LOCKS["x86_64"])
    labels = dict(re.findall(r'^\s+(?:LABEL\s+)?([a-z0-9.\-]+)="([^"]*)"', CONTAINERFILE, re.M))
    assert labels["version"] == f"ray{pins['ray']}-vllm{pins['vllm']}"
    stack = f"Ray {pins['ray']}, vLLM {pins['vllm']} and PyTorch {pins['torch']}"
    assert labels["description"].startswith(stack), labels["description"]
    assert labels["io.k8s.description"].startswith(stack), labels["io.k8s.description"]
    # The base image's own build labels are replaced, not inherited.
    for name in ("build-date", "vcs-ref", "release", "org.opencontainers.image.revision", "cpe"):
        assert labels[name] == "", name
    assert labels["com.redhat.component"] == "rhvp-vllm-ray-inference-server"


def test_both_locks_pin_the_same_versions():
    assert _pins(LOCKS["x86_64"]) == _pins(LOCKS["aarch64"])


@pytest.mark.parametrize("arch", sorted(LOCKS))
def test_lock_is_fully_hashed_and_for_glibc_2_34(arch):
    text = LOCKS[arch].read_text()
    assert f"--python-platform {arch}-manylinux_2_34" in text.splitlines()[1]
    assert "--only-binary :all:" in text and "--generate-hashes" in text
    blocks = re.split(r"\n(?=[A-Za-z0-9])", text.split("\n", 2)[2])
    for block in blocks:
        name = block.split("==", 1)[0]
        assert "--hash=sha256:" in block, f"{name} has no hash"
    assert ";" not in text, "a platform lock must not carry environment markers"


# --------------------------------------------------------------------------
# Containerfile and entrypoint.sh (static checks)
# --------------------------------------------------------------------------

CONTAINERFILE = (SERVER_DIR / "Containerfile").read_text()
ENTRYPOINT = (SERVER_DIR / "entrypoint.sh").read_text()


def test_containerfile_base_is_pinned_by_digest():
    froms = re.findall(r"^FROM\s+(\S+)", CONTAINERFILE, re.M)
    assert froms == [
        "nvcr.io/nvidia/cuda:13.0.3-devel-ubi9@sha256:"
        "c99ab53f9d1e0ba4b5817014ac398aafec516042b73e09d74fac4d2598f8f5fb"
    ]


@pytest.mark.parametrize(
    "name, value",
    [
        ("RHVP_CONFIG", "/config/llm.yaml"),
        ("HF_HOME", "/models/hf"),
        ("TRITON_CACHE_DIR", "/cache/triton"),
        ("VLLM_CACHE_ROOT", "/cache/vllm"),
        ("FLASHINFER_WORKSPACE_BASE", "/cache/flashinfer"),
        ("XDG_CACHE_HOME", "/cache/xdg"),
        ("RAY_TMPDIR", "/tmp/r"),
        ("CUDA_HOME", "/usr/local/cuda"),
        ("HOME", "/opt/app-root/home"),
        ("USER", "rhvp"),
        ("LOGNAME", "rhvp"),
        ("RAY_USAGE_STATS_ENABLED", "0"),
        ("VLLM_NO_USAGE_STATS", "1"),
        ("DO_NOT_TRACK", "1"),
        ("RAYLLM_ENABLE_REQUEST_PROMPT_LOGS", "0"),
    ],
)
def test_containerfile_env(name, value):
    assert re.search(rf"^\s*(ENV\s+)?{name}={re.escape(value)}(\s|\\|$)", CONTAINERFILE, re.M), name


def test_containerfile_install_and_runtime_settings():
    assert "--only-binary :all: --require-hashes" in CONTAINERFILE
    assert "requirements-$(uname -m).lock" in CONTAINERFILE
    assert "python3.12 -m venv /opt/app-root/venv" in CONTAINERFILE
    assert "dnf clean all" in CONTAINERFILE
    assert not re.search(r"dnf\s+(-y\s+)?(update|upgrade)\s+-y|dnf\s+-y\s+(update|upgrade)", CONTAINERFILE)
    assert re.search(r"^USER 1001$", CONTAINERFILE, re.M)
    assert re.search(r"^EXPOSE 8000$", CONTAINERFILE, re.M)
    assert 'ENTRYPOINT ["/opt/app-root/src/entrypoint.sh"]' in CONTAINERFILE
    # No BuildKit-only syntax: OpenShift builds run buildah.
    assert "RUN <<" not in CONTAINERFILE and "--chmod" not in CONTAINERFILE
    # Each COPY source exists in the build context.
    for line in re.findall(r"^COPY\s+(.+)$", CONTAINERFILE, re.M):
        for src in line.split()[:-1]:
            assert (SERVER_DIR / src).is_file(), src


def test_pip_is_hash_pinned_and_installed_before_the_lock():
    text = (SERVER_DIR / "requirements-pip.lock").read_text()
    pins = re.findall(r"^pip==(\S+) \\\n\s+--hash=sha256:[0-9a-f]{64}$", text, re.M)
    assert len(pins) == 1, "requirements-pip.lock must pin exactly one pip, with its hash"
    major = int(pins[0].split(".")[0])
    # Checked against the full lock with pip 26.2.2; the venv's own pip 23.2.1 rejects a hash-locked dependency
    # that is asked for with an extra (nvidia-cutlass-dsl[cu13]) as unpinned.
    assert major >= 26, "use a pip that has been checked against the full lock"
    first = CONTAINERFILE.index("-r /opt/app-root/src/requirements-pip.lock")
    assert first < CONTAINERFILE.index('-r "$lock"')


def test_entrypoint_contract():
    assert ENTRYPOINT.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in ENTRYPOINT
    assert 'rm -f "$RAY_TMPDIR/rhvp-ready"' in ENTRYPOINT
    assert "hf_*) ;;" in ENTRYPOINT and "unset HF_TOKEN" in ENTRYPOINT
    assert ENTRYPOINT.rstrip().endswith('exec python /opt/app-root/src/serve_app.py "$@"')
    assert (SERVER_DIR / "entrypoint.sh").stat().st_mode & 0o111, "entrypoint.sh must be executable"


@pytest.mark.parametrize("name", ["serve_app.py", "rhvp_status.py"])
def test_ported_files_carry_the_license_header(name):
    first = (SERVER_DIR / name).read_text().splitlines()[0]
    assert first == (
        "# Copyright (c) 2026, Eugo. Licensed under the BSD 3-Clause License "
        "(see LICENSES/BSD-3-Clause.txt)."
    )
    assert (REPO / "LICENSES" / "BSD-3-Clause.txt").read_text().count("Copyright (c) 2026, Eugo") == 1
