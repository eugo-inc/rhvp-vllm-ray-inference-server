# Copyright (c) 2026, Eugo. Licensed under the BSD 3-Clause License (see LICENSES/BSD-3-Clause.txt).
"""OpenAI-compatible LLM server: Ray Serve LLM with the vLLM engine, in one process.

What ``main()`` does, in order:

1. Drops ``HF_TOKEN`` (and the older ``HUGGING_FACE_HUB_TOKEN``) from the
   environment unless it looks like a Hugging Face token (starts with
   ``hf_``), so a placeholder such as ``none`` is never sent to the Hub.
2. Removes a stale ready file left by an earlier run in the same pod.
3. Reads the config file (``$RHVP_CONFIG``, default ``/config/llm.yaml``)
   and the ``RHVP_*`` environment overrides.
4. Starts a single-node Ray cluster in this container, starts Ray Serve on
   ``http_host:http_port``, and deploys one app built by
   ``ray.serve.llm.build_openai_app``.
5. ``serve.run(..., blocking=False)`` returns only when the app is RUNNING,
   which in Ray 2.60.0 means the model replica has finished starting (weights
   downloaded and loaded into vLLM). Only then is the ready file
   ``$RAY_TMPDIR/rhvp-ready`` written. The Kubernetes startup and readiness
   probes test for that file. In Ray 2.60.0's source, ``/-/healthz`` and
   ``/v1/models`` do not wait for the model; ``scripts/local-smoke.sh`` logs
   when each first answers.
6. Keeps the ready file in step with the app status (removes it if the app
   stops being RUNNING) until SIGTERM or SIGINT, then shuts Serve and Ray down.

``load_config`` and ``build_llm_config_kwargs`` import only the standard
library and PyYAML, so they can be tested without Ray installed.

Config file keys (all but ``model_id`` and ``model_source`` are optional)::

    app_name: llm
    model_id: qwen2.5-0.5b-instruct          # name clients send as "model"
    model_source: Qwen/Qwen2.5-0.5B-Instruct  # Hugging Face repo or local path
    model_revision: "<commit sha>"            # empty or omitted: latest
    engine_kwargs: {max_model_len: 8192, gpu_memory_utilization: 0.85,
                    tensor_parallel_size: 1}  # vLLM engine arguments
    accelerator_type: null
    min_replicas: 1
    max_replicas: 1
    object_store_memory_gb: 2
    http_host: 0.0.0.0
    http_port: 8000

Environment overrides (a variable that is unset or empty changes nothing):
``RHVP_MODEL_ID``, ``RHVP_MODEL_SOURCE``, ``RHVP_MODEL_REVISION``,
``RHVP_GPU_MEMORY_UTILIZATION``, ``RHVP_MAX_MODEL_LEN``. If
``RHVP_MODEL_SOURCE`` names a different model than the file and
``RHVP_MODEL_REVISION`` is not set, the file's revision is dropped, because a
commit of one model repository means nothing in another.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import signal
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional

log = logging.getLogger("rhvp.serve")

DEFAULT_CONFIG_PATH = "/config/llm.yaml"
DEFAULT_RAY_TMPDIR = "/tmp/r"
READY_FILE_NAME = "rhvp-ready"
STATUS_POLL_INTERVAL_S = 10.0

REQUIRED_KEYS = ("model_id", "model_source")
DEFAULTS: Dict[str, Any] = {
    "app_name": "llm",
    "model_revision": None,
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
KNOWN_KEYS = frozenset(DEFAULTS) | frozenset(REQUIRED_KEYS)

# Environment variable -> (path in the config, parser).
ENV_OVERRIDES = {
    "RHVP_MODEL_ID": (("model_id",), str),
    "RHVP_MODEL_SOURCE": (("model_source",), str),
    "RHVP_MODEL_REVISION": (("model_revision",), str),
    "RHVP_GPU_MEMORY_UTILIZATION": (("engine_kwargs", "gpu_memory_utilization"), float),
    "RHVP_MAX_MODEL_LEN": (("engine_kwargs", "max_model_len"), int),
}

# Keys Ray Serve LLM sets itself from model_loading_config; it rejects them
# in engine_kwargs.
FORBIDDEN_ENGINE_KWARGS = ("model", "served_model_name")

HF_TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


class ConfigError(ValueError):
    """The config file or an override has a missing or invalid value."""


# --------------------------------------------------------------------------
# Config: pure functions, no Ray import.
# --------------------------------------------------------------------------


def load_config(path: str, env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Read the YAML config at ``path``, apply ``RHVP_*`` overrides from
    ``env``, fill defaults, validate, and return a normalised dict.

    Raises ``ConfigError`` with a readable message on any problem.
    """
    import yaml  # PyYAML; in the image through Ray's dependencies.

    env = {} if env is None else env
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file {path} is not valid YAML: {exc}") from None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {path} must hold a mapping, got {type(raw).__name__}")

    cfg = copy.deepcopy(DEFAULTS)
    for key, value in raw.items():
        if key not in KNOWN_KEYS:
            log.warning("config: ignoring unknown key %r", key)
            continue
        if key == "engine_kwargs":
            if value is None:
                continue
            if not isinstance(value, dict):
                raise ConfigError("engine_kwargs must be a mapping")
            cfg["engine_kwargs"].update(value)
        else:
            cfg[key] = value

    file_source = cfg.get("model_source")
    overridden = _apply_env_overrides(cfg, env)
    if (
        "RHVP_MODEL_SOURCE" in overridden
        and "RHVP_MODEL_REVISION" not in overridden
        and cfg.get("model_source") != file_source
        and cfg.get("model_revision")
    ):
        log.warning(
            "config: RHVP_MODEL_SOURCE names a different model than the file; "
            "dropping the file's model_revision (set RHVP_MODEL_REVISION to pin one)"
        )
        cfg["model_revision"] = None

    return _validate(cfg)


def _apply_env_overrides(cfg: Dict[str, Any], env: Mapping[str, str]) -> List[str]:
    applied = []
    for name, (path, parse) in ENV_OVERRIDES.items():
        raw = env.get(name)
        if raw is None or raw.strip() == "":
            continue
        try:
            value = parse(raw.strip())
        except ValueError:
            raise ConfigError(f"{name}={raw!r} is not a valid {parse.__name__}") from None
        target = cfg
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        applied.append(name)
    return applied


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _nonempty_str(cfg: Dict[str, Any], key: str) -> str:
    value = cfg.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string, got {value!r}")
    return value.strip()


def _validate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["app_name"] = _nonempty_str(cfg, "app_name")
    out["model_id"] = _nonempty_str(cfg, "model_id")
    out["model_source"] = _nonempty_str(cfg, "model_source")

    revision = cfg.get("model_revision")
    if revision is not None and not isinstance(revision, str):
        # An unquoted all-digit or digits-and-one-"e" commit id is read by
        # YAML as a number and cannot be recovered exactly.
        raise ConfigError(
            f"model_revision must be a string (quote it in YAML), got {revision!r}"
        )
    revision = (revision or "").strip() or None

    engine_kwargs = dict(cfg.get("engine_kwargs") or {})
    for key in engine_kwargs:
        if not isinstance(key, str):
            raise ConfigError(f"engine_kwargs keys must be strings, got {key!r}")
    for key in FORBIDDEN_ENGINE_KWARGS:
        if key in engine_kwargs:
            raise ConfigError(
                f"engine_kwargs.{key} is not allowed; set model_id / model_source instead"
            )
    if "revision" in engine_kwargs:
        kw_revision = engine_kwargs.pop("revision")
        kw_revision = str(kw_revision).strip() if kw_revision is not None else None
        if revision is None:
            revision = kw_revision or None
        elif kw_revision and kw_revision != revision:
            raise ConfigError(
                f"model_revision {revision!r} and engine_kwargs.revision "
                f"{kw_revision!r} disagree; set only model_revision"
            )
    out["model_revision"] = revision

    if engine_kwargs.get("max_model_len") is None:
        engine_kwargs.pop("max_model_len", None)  # null: let vLLM use the model's limit
    else:
        if not _is_int(engine_kwargs["max_model_len"]) or engine_kwargs["max_model_len"] <= 0:
            raise ConfigError(
                f"engine_kwargs.max_model_len must be a positive integer, "
                f"got {engine_kwargs['max_model_len']!r}"
            )
    if "gpu_memory_utilization" in engine_kwargs:
        gmu = engine_kwargs["gpu_memory_utilization"]
        if not _is_number(gmu) or not 0.0 < float(gmu) <= 1.0:
            raise ConfigError(
                f"engine_kwargs.gpu_memory_utilization must be a number in (0, 1], got {gmu!r}"
            )
        engine_kwargs["gpu_memory_utilization"] = float(gmu)
    if "tensor_parallel_size" in engine_kwargs:
        tps = engine_kwargs["tensor_parallel_size"]
        if not _is_int(tps) or tps < 1:
            raise ConfigError(
                f"engine_kwargs.tensor_parallel_size must be an integer >= 1, got {tps!r}"
            )
    out["engine_kwargs"] = engine_kwargs

    accel = cfg.get("accelerator_type")
    if accel is not None and (not isinstance(accel, str) or not accel.strip()):
        raise ConfigError(f"accelerator_type must be null or a non-empty string, got {accel!r}")
    out["accelerator_type"] = accel.strip() if isinstance(accel, str) else None

    min_r, max_r = cfg.get("min_replicas"), cfg.get("max_replicas")
    # At least 1: with 0 replicas the app is RUNNING before any model is loaded, so the ready file (and the pod's
    # Ready) would come before the model; the pod holds its GPU either way.
    if not _is_int(min_r) or min_r < 1:
        raise ConfigError(f"min_replicas must be an integer >= 1, got {min_r!r}")
    if not _is_int(max_r) or max_r < 1:
        raise ConfigError(f"max_replicas must be an integer >= 1, got {max_r!r}")
    if min_r > max_r:
        raise ConfigError(f"min_replicas ({min_r}) must not exceed max_replicas ({max_r})")
    out["min_replicas"], out["max_replicas"] = min_r, max_r

    osm = cfg.get("object_store_memory_gb")
    if not _is_number(osm) or osm <= 0:
        raise ConfigError(f"object_store_memory_gb must be a positive number, got {osm!r}")
    out["object_store_memory_gb"] = osm

    out["http_host"] = _nonempty_str(cfg, "http_host")
    port = cfg.get("http_port")
    if not _is_int(port) or not 1 <= port <= 65535:
        raise ConfigError(f"http_port must be an integer in 1..65535, got {port!r}")
    out["http_port"] = port
    return out


def build_llm_config_kwargs(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Keyword arguments for ``ray.serve.llm.LLMConfig`` from a config made
    by ``load_config``.

    The model revision goes into ``engine_kwargs["revision"]``: Ray Serve LLM
    2.60.0 passes every key that is a vLLM engine argument to vLLM, and vLLM
    0.29.0 uses the same revision for the tokenizer unless told otherwise.
    ``model_loading_config`` has no revision field (and rejects unknown keys).
    """
    engine_kwargs = dict(cfg["engine_kwargs"])
    if cfg.get("model_revision"):
        engine_kwargs["revision"] = cfg["model_revision"]
    kwargs: Dict[str, Any] = {
        "model_loading_config": {
            "model_id": cfg["model_id"],
            "model_source": cfg["model_source"],
        },
        "engine_kwargs": engine_kwargs,
        "deployment_config": {
            "autoscaling_config": {
                "min_replicas": cfg["min_replicas"],
                "max_replicas": cfg["max_replicas"],
            }
        },
    }
    if cfg.get("accelerator_type"):
        kwargs["accelerator_type"] = cfg["accelerator_type"]
    return kwargs


def object_store_bytes(cfg: Mapping[str, Any]) -> int:
    return int(cfg["object_store_memory_gb"] * 1024**3)


# --------------------------------------------------------------------------
# Process helpers: environment, ready file, status.
# --------------------------------------------------------------------------


def sanitize_env(env: MutableMapping[str, str]) -> List[str]:
    """Remove Hugging Face token variables that do not hold a real token.

    Returns the names removed. A real token starts with ``hf_``; anything
    else (the pattern's placeholder secret is ``none``) would make every Hub
    request fail with 401 instead of downloading a public model anonymously.
    """
    removed = []
    for name in HF_TOKEN_VARS:
        value = env.get(name)
        if value is None:
            continue
        stripped = value.strip()
        if stripped.startswith("hf_"):
            if stripped != value:
                env[name] = stripped
        else:
            del env[name]
            removed.append(name)
    return removed


def ready_file_path(env: Mapping[str, str]) -> str:
    return os.path.join(env.get("RAY_TMPDIR") or DEFAULT_RAY_TMPDIR, READY_FILE_NAME)


def write_ready_file(path: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".rhvp-ready.", dir=directory)
    with os.fdopen(fd, "w") as fh:
        fh.write(f"{time.time():.3f}\n")
    os.replace(tmp, path)


def remove_ready_file(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def app_status_name(serve_status: Any, app_name: str) -> str:
    """One word for the app's state, from a ``serve.status()`` result.

    ``NO_CONTROLLER`` when Serve has no proxies (Serve not running at all),
    ``NOT_STARTED`` when Serve runs but the app is not deployed, otherwise
    the app's Ray Serve status (``RUNNING``, ``DEPLOYING``, ``DEPLOY_FAILED``,
    ``UNHEALTHY``, ``DELETING``).
    """
    proxies = getattr(serve_status, "proxies", None) or {}
    if len(proxies) == 0:
        return "NO_CONTROLLER"
    apps = getattr(serve_status, "applications", None) or {}
    app = apps.get(app_name)
    if app is None:
        return "NOT_STARTED"
    status = getattr(app, "status", None)
    status = getattr(status, "value", status)  # ApplicationStatus is a str Enum
    return str(status) if status else "UNKNOWN"


def watch_ready_file(
    ready_path: str,
    stop: threading.Event,
    get_status: Callable[[], str],
    interval_s: float = STATUS_POLL_INTERVAL_S,
) -> None:
    """Until ``stop`` is set, keep the ready file present exactly while
    ``get_status()`` returns ``RUNNING``. A failed status call leaves the
    file as it is (the liveness probe covers a dead Serve proxy)."""
    last = "RUNNING"
    failing = False
    while not stop.wait(interval_s):
        try:
            state = get_status()
        except Exception as exc:  # noqa: BLE001 - keep serving; just report
            if not failing:
                log.warning("status check failed (ready file unchanged): %s", exc)
            failing = True
            continue
        failing = False
        if state == "RUNNING":
            if not os.path.exists(ready_path):
                write_ready_file(ready_path)
        else:
            remove_ready_file(ready_path)
        if state != last:
            log.warning("app status changed: %s -> %s", last, state)
            last = state


# --------------------------------------------------------------------------
# Entry point.
# --------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--config",
        default=os.environ.get("RHVP_CONFIG") or DEFAULT_CONFIG_PATH,
        help="path to llm.yaml (default: $RHVP_CONFIG or %(default)s)",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the resolved config and LLMConfig arguments as JSON and exit (no Ray)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s rhvp: %(message)s"
    )
    args = _parse_args(argv)

    # Before Ray starts: Ray's processes inherit this environment, and Ray
    # Serve LLM forwards HF_TOKEN to the engine from it.
    for name in sanitize_env(os.environ):
        log.info("%s does not start with 'hf_'; unset it (public models need no token)", name)
    ready_path = ready_file_path(os.environ)
    remove_ready_file(ready_path)

    try:
        cfg = load_config(args.config, os.environ)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2
    llm_kwargs = build_llm_config_kwargs(cfg)
    if args.print_config:
        print(json.dumps({"config": cfg, "llm_config": llm_kwargs}, indent=2, sort_keys=True))
        return 0

    log.info(
        "model_id=%s model_source=%s revision=%s engine_kwargs=%s",
        cfg["model_id"],
        cfg["model_source"],
        cfg["model_revision"] or "(latest)",
        json.dumps(cfg["engine_kwargs"], sort_keys=True),
    )

    import ray
    from ray import serve
    from ray.serve.llm import LLMConfig, build_openai_app

    # address="local" always starts a new single-node cluster in this
    # container, even if RAY_ADDRESS is set. ray.init installs a SIGTERM
    # handler that raises SystemExit; the finally block below cleans up.
    ray.init(address="local", object_store_memory=object_store_bytes(cfg))
    stop = threading.Event()
    try:
        serve.start(http_options={"host": cfg["http_host"], "port": cfg["http_port"]})
        app = build_openai_app({"llm_configs": [LLMConfig(**llm_kwargs)]})
        started = time.monotonic()
        # Returns once the app is RUNNING; raises if it is DEPLOY_FAILED.
        serve.run(app, name=cfg["app_name"], blocking=False)
        log.info("app %r is RUNNING after %.1f s", cfg["app_name"], time.monotonic() - started)

        def _on_signal(signum: int, _frame: Any) -> None:
            log.info("received signal %d; shutting down", signum)
            stop.set()

        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)
        write_ready_file(ready_path)
        log.info("ready file written: %s", ready_path)
        watch_ready_file(
            ready_path,
            stop,
            lambda: app_status_name(serve.status(), cfg["app_name"]),
        )
        return 0
    except Exception:
        log.exception("the server failed")
        return 1
    finally:
        remove_ready_file(ready_path)
        try:
            serve.shutdown()
        except Exception as exc:  # noqa: BLE001
            log.warning("serve.shutdown failed: %s", exc)
        try:
            ray.shutdown()
        except Exception as exc:  # noqa: BLE001
            log.warning("ray.shutdown failed: %s", exc)


if __name__ == "__main__":
    sys.exit(main())
