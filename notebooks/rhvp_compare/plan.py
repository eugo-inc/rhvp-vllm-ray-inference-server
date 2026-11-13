# SPDX-License-Identifier: Apache-2.0
"""The comparison's plan: the config file read, checked and turned into everything that is fixed before the first
round. The plan is written into the results file before any measurement, so the metric family, the tie band, alpha,
the test, the correction and the round order are on record as declared, not chosen after the fact.

The config file is JSON with the keys of `global.compare` in values-global.yaml (camelCase), plus `model` (from
`global.model`) and a few platform settings the chart adds. Standard library only.

Round order. With `pairs: n` the scored rounds are n pairs in ABBA order: A B, B A, A B, B A, ... (pair 1 is A then
B, pair 2 is B then A, and so on). An optional warm-up runs each side once first, unscored: it pulls the image and
fills the model cache, so no scored round pays for a download. Optional A/A pairs come last: side A twice, to show
how far two rounds of one image land apart (the noise floor).

Primary and secondary metrics. A few metrics, declared before the run, are primary: only they share the
multiple-comparison correction, so ten pairs can still give a FASTER or SLOWER verdict. By default they are the time
to first answer and, at the highest concurrency level, the median time to first token, the output tokens per second
and the output tokens per joule. Every other metric is secondary: it gets the same test, interval and tie band, but
its p-value is not corrected, so its verdict is descriptive, not a finding. `analysis.primary` lists other ids.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping

from . import stats

SCHEMA = "rhvp-image-compare/1"
A, B = "A", "B"
WARMUP, SCORED, AA = "warmup", "scored", "aa"
PRESETS = ("vllm-openai", "ray-serve-llm", "guidellm-mock", "custom")
ENERGY_SOURCES = ("auto", "dcgm", "nvidia-smi", "none")
ENERGY_LABEL = "GPU-only power"
DEFAULT_PORT = 8000
#: GuideLLM v0.8.1, the multi-arch index digest (linux/amd64 and linux/arm64).
GUIDELLM_IMAGE = ("ghcr.io/vllm-project/guidellm:v0.8.1"
                  "@sha256:015b22441020e6014b4c06d89882921eb29e9b9a6f2e274b253afef4029aa081")
NAME = re.compile(r"^[a-z0-9]([-a-z0-9.]{0,38}[a-z0-9])?$")

DEFAULTS: dict = {
    "runName": "",
    "sides": [],
    "rounds": {"pairs": 10, "warmup": True, "aaPairs": 0},
    "server": {"port": DEFAULT_PORT, "readyTimeoutSeconds": 1800, "warmupTimeoutSeconds": 3600,
               "gpuMemoryUtilization": 0.85, "maxModelLen": 8192, "seed": 0, "firstAnswerTokens": 16},
    "load": {"image": GUIDELLM_IMAGE, "concurrencies": [1, 8, 32], "promptTokens": 256, "outputTokens": 128,
             "secondsPerLevel": 60, "seed": 42, "requestFormat": "/v1/completions", "timeoutSeconds": None},
    "analysis": {"tieBand": 0.05, "alpha": 0.05, "test": "sign", "correction": "holm", "minPairs": 3,
                 "ciLevel": 0.95, "primary": None},
    "energy": {"source": "auto", "settleSeconds": 45,
               "thanosUrl": "https://thanos-querier.openshift-monitoring.svc:9091", "minSamples": 2},
}


class PlanError(ValueError):
    """The config cannot be run; the message says which key and why."""


def _merge(base: dict, extra: Mapping) -> dict:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(out.get(key), dict) and isinstance(value, Mapping):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _number(value, key: str, low: float, high: float, integer: bool = False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanError(f"{key} must be a number, got {value!r}")
    if integer and int(value) != value:
        raise PlanError(f"{key} must be a whole number, got {value!r}")
    if not low <= value <= high:
        raise PlanError(f"{key} must be between {low} and {high}, got {value!r}")
    return int(value) if integer else float(value)


def _string_map(value, key: str) -> dict[str, str]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise PlanError(f"{key} must be a map of names to values")
    out = {}
    for name, item in value.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise PlanError(f"{key}: {name!r} is not an environment variable name")
        out[name] = "" if item is None else str(item)
    return out


def _string_list(value, key: str) -> list[str]:
    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise PlanError(f"{key} must be a list")
    return [str(item) for item in value]


def _side(raw: Mapping, key: str, where: str) -> dict:
    if not isinstance(raw, Mapping):
        raise PlanError(f"{where} must be a map with name and image")
    name = raw.get("name")
    if not isinstance(name, str) or not NAME.match(name):
        raise PlanError(f"{where}.name must be lower-case letters, digits, '-' or '.', at most 40 characters; "
                        f"got {name!r}")
    image = raw.get("image")
    if not isinstance(image, str) or not image.strip() or any(c.isspace() for c in image.strip()):
        raise PlanError(f"{where}.image must be an image reference, got {image!r}")
    preset = raw.get("preset", "vllm-openai")
    if preset not in PRESETS:
        raise PlanError(f"{where}.preset must be one of {', '.join(PRESETS)}; got {preset!r}")
    command = raw.get("command")
    if command is not None:
        command = _string_list(command, f"{where}.command") or None
    if preset == "custom" and not command and not raw.get("args"):
        raise PlanError(f"{where}: preset custom needs command or args")
    gpu = raw.get("gpu", preset != "guidellm-mock")
    if not isinstance(gpu, bool):
        raise PlanError(f"{where}.gpu must be true or false")
    return {"key": key, "name": name, "image": image.strip(), "preset": preset, "command": command,
            "args": _string_list(raw.get("args"), f"{where}.args"), "env": _string_map(raw.get("env"), f"{where}.env"),
            "gpu": gpu}


def round_order(pairs: int, warmup: bool, aa_pairs: int) -> list[dict]:
    """Every round in run order: index (from 1), side, kind and pair (None for warm-up rounds)."""
    out: list[dict] = []

    def add(side: str, kind: str, pair: int | None) -> None:
        out.append({"index": len(out) + 1, "side": side, "kind": kind, "pair": pair})

    if warmup:
        add(A, WARMUP, None)
        add(B, WARMUP, None)
    for pair in range(1, pairs + 1):
        first, second = (A, B) if pair % 2 else (B, A)
        add(first, SCORED, pair)
        add(second, SCORED, pair)
    for pair in range(1, aa_pairs + 1):
        add(A, AA, pair)
        add(A, AA, pair)
    return out


PRIMARY, SECONDARY = "primary", "secondary"


def default_primary(concurrencies: list[int], energy: bool) -> list[str]:
    """The default primary metrics: time to first answer, and at the highest concurrency level the median time to
    first token, the output tokens per second and (when energy is measured) the output tokens per joule."""
    top = max(concurrencies)
    out = ["startup.first_answer_s", f"c{top}.ttft_ms_median", f"c{top}.output_tokens_per_s"]
    if energy:
        out.append(f"c{top}.tokens_per_joule")
    return out


def metric_family(concurrencies: list[int], energy: bool) -> list[dict]:
    """The declared metrics, in report order. Each has an id (the key in every round's `metrics` map), a label, a
    unit, the direction that is better, and its kind (speed or efficiency, which picks the verdict words). `role`
    is set by `build` (primary or secondary)."""
    out = [
        {"id": "startup.ready_s", "label": "time to ready", "unit": "s", "better": stats.LOWER, "kind": "speed",
         "concurrency": None},
        {"id": "startup.first_answer_s", "label": "time to first answer", "unit": "s", "better": stats.LOWER,
         "kind": "speed", "concurrency": None},
    ]
    for c in concurrencies:
        streams = f"{c} concurrent"
        out += [
            {"id": f"c{c}.ttft_ms_median", "label": f"time to first token, median, {streams}", "unit": "ms",
             "better": stats.LOWER, "kind": "speed", "concurrency": c},
            {"id": f"c{c}.itl_ms_median", "label": f"inter-token latency, median, {streams}", "unit": "ms",
             "better": stats.LOWER, "kind": "speed", "concurrency": c},
            {"id": f"c{c}.output_tokens_per_s", "label": f"output tokens per second, {streams}", "unit": "tokens/s",
             "better": stats.HIGHER, "kind": "speed", "concurrency": c},
        ]
        if energy:
            out.append({"id": f"c{c}.tokens_per_joule", "label": f"output tokens per joule ({ENERGY_LABEL}), "
                        f"{streams}", "unit": "tokens/J", "better": stats.HIGHER, "kind": "efficiency",
                        "concurrency": c})
    return out


def digest(config: Mapping) -> str:
    """A short, stable hash of the config: part of the run id, so a changed config is a new run."""
    text = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def build(config: Mapping, platform: str = "openshift") -> dict:
    """The plan for `config`, or PlanError. `platform` is openshift or docker: it decides what energy source
    `auto` means (DCGM through the cluster's monitoring on OpenShift, nvidia-smi sampling under docker)."""
    if not isinstance(config, Mapping):
        raise PlanError("the config must be a JSON object")
    cfg = _merge(DEFAULTS, config)
    run_name = cfg.get("runName")
    if not isinstance(run_name, str) or not NAME.match(run_name):
        raise PlanError(f"runName must be lower-case letters, digits, '-' or '.', at most 40 characters; "
                        f"got {run_name!r}")
    model = cfg.get("model") or {}
    for field in ("id", "source"):
        if not isinstance(model.get(field), str) or not model[field].strip():
            raise PlanError(f"model.{field} must be set (global.model.{field})")
    revision = model.get("revision") or ""
    if not isinstance(revision, str):
        raise PlanError("model.revision must be a string (quote it in YAML)")
    sides = cfg.get("sides")
    if not isinstance(sides, list) or len(sides) != 2:
        raise PlanError("sides must list exactly two sides: the baseline (A) first, then the candidate (B)")
    side_a, side_b = _side(sides[0], A, "sides[0]"), _side(sides[1], B, "sides[1]")
    if side_a["name"] == side_b["name"]:
        raise PlanError("the two sides need different names")

    rounds = cfg["rounds"]
    pairs = _number(rounds.get("pairs"), "rounds.pairs", 1, 50, integer=True)
    aa_pairs = _number(rounds.get("aaPairs", 0), "rounds.aaPairs", 0, 20, integer=True)
    warmup = rounds.get("warmup", True)
    if not isinstance(warmup, bool):
        raise PlanError("rounds.warmup must be true or false")

    server = cfg["server"]
    load = cfg["load"]
    concurrencies = load.get("concurrencies")
    if not isinstance(concurrencies, list) or not concurrencies:
        raise PlanError("load.concurrencies must be a non-empty list")
    levels = [_number(c, "load.concurrencies[]", 1, 4096, integer=True) for c in concurrencies]
    if len(set(levels)) != len(levels):
        raise PlanError("load.concurrencies must not repeat a value")
    seconds_per_level = _number(load.get("secondsPerLevel"), "load.secondsPerLevel", 1, 86400)
    if load.get("timeoutSeconds") is None:  # time for every level, the tokenizer download and GuideLLM's start
        timeout_s = float(round(len(levels) * (seconds_per_level * 1.5 + 60) + 300))
    else:
        timeout_s = _number(load.get("timeoutSeconds"), "load.timeoutSeconds", 30, 86400)
    request_format = load.get("requestFormat")
    if request_format not in ("/v1/completions", "/v1/chat/completions"):
        raise PlanError("load.requestFormat must be /v1/completions or /v1/chat/completions")

    analysis = cfg["analysis"]
    test = analysis.get("test")
    if test not in stats.TESTS:
        raise PlanError(f"analysis.test must be one of {', '.join(stats.TESTS)}")
    correction = analysis.get("correction")
    if correction not in stats.CORRECTIONS:
        raise PlanError(f"analysis.correction must be one of {', '.join(stats.CORRECTIONS)}")
    tie_band = _number(analysis.get("tieBand"), "analysis.tieBand", 0.0, 0.5)
    alpha = _number(analysis.get("alpha"), "analysis.alpha", 0.0001, 0.5)
    min_pairs = _number(analysis.get("minPairs"), "analysis.minPairs", 1, 50, integer=True)
    ci_level = _number(analysis.get("ciLevel"), "analysis.ciLevel", 0.5, 0.999)

    energy = cfg["energy"]
    source = energy.get("source")
    if source not in ENERGY_SOURCES:
        raise PlanError(f"energy.source must be one of {', '.join(ENERGY_SOURCES)}")
    if source == "auto":
        source = "dcgm" if platform == "openshift" else "nvidia-smi"
    if not (side_a["gpu"] and side_b["gpu"]):
        source = "none"  # a side without a GPU has no GPU power to read
    metrics = metric_family(levels, source != "none")
    ids = [m["id"] for m in metrics]
    primary = analysis.get("primary")
    if primary is None:
        primary = default_primary(levels, source != "none")
    elif (not isinstance(primary, list) or not primary or not all(isinstance(p, str) for p in primary)
          or len(set(primary)) != len(primary)):
        raise PlanError("analysis.primary must be a non-empty list of metric ids without repeats, or null for the "
                        "default")
    unknown = [p for p in primary if p not in ids]
    if unknown:
        raise PlanError(f"analysis.primary names metrics this run does not have: {', '.join(unknown)}; the metrics "
                        f"are {', '.join(ids)}")
    for metric in metrics:
        metric["role"] = PRIMARY if metric["id"] in primary else SECONDARY
    family = len(primary)
    floor = stats.floor_p(pairs, family, correction)

    plan = {
        "schema": SCHEMA,
        "run_name": run_name,
        "run_id": f"{run_name}-{digest(config)}",
        "platform": platform,
        "model": {"id": model["id"].strip(), "source": model["source"].strip(), "revision": revision.strip()},
        "sides": [side_a, side_b],
        "rounds": round_order(pairs, warmup, aa_pairs),
        "pairs": pairs,
        "aa_pairs": aa_pairs,
        "warmup": warmup,
        "server": {
            "port": _number(server.get("port"), "server.port", 1, 65535, integer=True),
            "ready_timeout_s": _number(server.get("readyTimeoutSeconds"), "server.readyTimeoutSeconds", 10, 86400),
            "warmup_timeout_s": _number(server.get("warmupTimeoutSeconds"), "server.warmupTimeoutSeconds", 10,
                                        86400),
            "gpu_memory_utilization": _number(server.get("gpuMemoryUtilization"), "server.gpuMemoryUtilization",
                                              0.05, 1.0),
            "max_model_len": _number(server.get("maxModelLen"), "server.maxModelLen", 16, 10_000_000, integer=True),
            "seed": _number(server.get("seed"), "server.seed", 0, 2**31 - 1, integer=True),
            "first_answer_tokens": _number(server.get("firstAnswerTokens"), "server.firstAnswerTokens", 1, 4096,
                                           integer=True),
        },
        "load": {
            "tool": "guidellm",
            "image": str(load.get("image") or GUIDELLM_IMAGE),
            "concurrencies": levels,
            "prompt_tokens": _number(load.get("promptTokens"), "load.promptTokens", 1, 1_000_000, integer=True),
            "output_tokens": _number(load.get("outputTokens"), "load.outputTokens", 1, 1_000_000, integer=True),
            "seconds_per_level": seconds_per_level,
            "seed": _number(load.get("seed"), "load.seed", 0, 2**31 - 1, integer=True),
            "request_format": request_format,
            "tokenizer": model["source"].strip(),
            "timeout_s": timeout_s,
        },
        "energy": {
            "source": source,
            "label": ENERGY_LABEL,
            "settle_s": _number(energy.get("settleSeconds"), "energy.settleSeconds", 0, 3600),
            "thanos_url": str(energy.get("thanosUrl") or DEFAULTS["energy"]["thanosUrl"]),
            "min_samples": _number(energy.get("minSamples"), "energy.minSamples", 2, 10000, integer=True),
        },
        "analysis": {
            "baseline": A, "candidate": B, "ratio": "B/A", "pairing": "ABBA, each pair one A round and the B round "
            "next to it", "unit": "one summary per round and metric; requests are never pooled across rounds",
            "test": test, "correction": correction, "tie_band": tie_band, "alpha": alpha, "min_pairs": min_pairs,
            "ci_level": ci_level, "zero_rule": "ratios exactly 1 carry no sign and are left out of both tests",
            "tie_rule": "TIE when the median ratio and both ends of its confidence interval lie within 1/(1+tie_band) "
            "and 1+tie_band; a median inside with an interval outside, or with no interval, is INCONCLUSIVE",
            "direction_rule": "FASTER or SLOWER only when the test leans the same way as the median ratio",
            "failure_rule": "a metric that lost a pair because one side failed on its own account (SIDE_FAILED, "
            "SIDE_NEVER_READY, LOAD_FAILED, failed requests at its level, or a value of 0) is INCONCLUSIVE, and so "
            "is every metric when the scored rounds ran on more than one node",
            "primary": [m["id"] for m in metrics if m["role"] == PRIMARY],
            "family_rule": "the correction runs over the primary metrics only; secondary metrics get the same test, "
            "interval and tie band with an uncorrected p-value, and their verdicts are descriptive",
        },
        "metrics": metrics,
        "floor": {"family_size": family, "min_adjusted_p": floor, "reachable": floor <= alpha,
                  "pairs_needed": stats.pairs_needed(family, correction, alpha)},
    }
    return plan


def load(path: str, platform: str = "openshift") -> dict:
    """Read the config file at `path` and build its plan."""
    try:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    except FileNotFoundError:
        raise PlanError(f"no config file at {path}") from None
    except ValueError as error:
        raise PlanError(f"the config file is not valid JSON: {error}") from None
    return build(config, platform)


# ---- what each side runs ----

#: Writable places for any image run as an arbitrary UID in group 0 (OpenShift's restricted SCC): every cache and
#: HOME on the pod's own emptyDir at /cache, the model weights on the volume at /models. USER and LOGNAME let
#: Python's getpass find a name for a UID that /etc/passwd does not list.
COMMON_ENV = {
    "HOME": "/cache/home",
    "USER": "server",
    "LOGNAME": "server",
    "HF_HOME": "/models/hf",
    "XDG_CACHE_HOME": "/cache/xdg",
    "XDG_CONFIG_HOME": "/cache/config",
    "VLLM_CACHE_ROOT": "/cache/vllm",
    "VLLM_CONFIG_ROOT": "/cache/vllm-config",
    "TRITON_CACHE_DIR": "/cache/triton",
    "TORCHINDUCTOR_CACHE_DIR": "/cache/inductor",
    "FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer",
    "NUMBA_CACHE_DIR": "/cache/numba",
    "OUTLINES_CACHE_DIR": "/cache/outlines",
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    "RAY_USAGE_STATS_ENABLED": "0",
}


def _fill(text: str, values: Mapping[str, object]) -> str:
    out = text
    for key, value in values.items():
        out = out.replace("{" + key + "}", str(value))
    return out


def side_spec(plan: Mapping, side: Mapping) -> dict:
    """What to run for one side: command (None keeps the image's entrypoint), args, env, whether it needs a GPU,
    and whether it reads llm.yaml (the pattern's own Ray Serve LLM image). Placeholders {model_id},
    {model_source}, {model_revision} and {port} in the side's args and env are filled in."""
    model, server = plan["model"], plan["server"]
    values = {"model_id": model["id"], "model_source": model["source"], "model_revision": model["revision"],
              "port": server["port"]}
    env = dict(COMMON_ENV)
    command = side.get("command")
    args: list[str] = []
    llm_config = False
    if side["preset"] == "vllm-openai":
        args = ["--model", model["source"]]
        if model["revision"]:
            args += ["--revision", model["revision"]]
        args += ["--served-model-name", model["id"], "--host", "0.0.0.0", "--port", str(server["port"]),
                 "--max-model-len", str(server["max_model_len"]),
                 "--gpu-memory-utilization", str(server["gpu_memory_utilization"]), "--seed", str(server["seed"])]
    elif side["preset"] == "ray-serve-llm":
        command = command or ["/opt/app-root/src/entrypoint.sh"]
        llm_config = True
        env.update({"RHVP_CONFIG": "/config/llm.yaml", "RAY_TMPDIR": "/tmp/r", "RHVP_MODEL_ID": model["id"],
                    "RHVP_MODEL_SOURCE": model["source"], "RHVP_MODEL_REVISION": model["revision"],
                    "RHVP_GPU_MEMORY_UTILIZATION": str(server["gpu_memory_utilization"]),
                    "RHVP_MAX_MODEL_LEN": str(server["max_model_len"])})
    elif side["preset"] == "guidellm-mock":
        command = command or ["guidellm", "mock-server"]
        args = ["--host", "0.0.0.0", "--port", str(server["port"]), "--model", model["id"]]
    args += [_fill(a, values) for a in side.get("args") or []]
    env.update({k: _fill(v, values) for k, v in (side.get("env") or {}).items()})
    return {"command": [_fill(c, values) for c in command] if command else None, "args": args, "env": env,
            "gpu": bool(side.get("gpu", True)), "llm_config": llm_config}


def llm_yaml(plan: Mapping) -> str:
    """llm.yaml for the pattern's own Ray Serve LLM image, from the plan's model and server settings (JSON is
    valid YAML, so no YAML library is needed)."""
    model, server = plan["model"], plan["server"]
    config = {"app_name": "llm", "model_id": model["id"], "model_source": model["source"],
              "model_revision": model["revision"] or None,
              "engine_kwargs": {"max_model_len": server["max_model_len"],
                                "gpu_memory_utilization": server["gpu_memory_utilization"],
                                "tensor_parallel_size": 1},
              "accelerator_type": None, "min_replicas": 1, "max_replicas": 1, "object_store_memory_gb": 2,
              "http_host": "0.0.0.0", "http_port": server["port"]}
    return json.dumps(config, indent=2) + "\n"
