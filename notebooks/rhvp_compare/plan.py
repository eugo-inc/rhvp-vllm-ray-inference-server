# SPDX-License-Identifier: Apache-2.0
"""The comparison's plan: the config file read, checked and turned into everything that is fixed before the first
launch. The plan is written into the results file before any measurement, so the cases, the primary cases, alpha
and the order of every launch are on record as declared, not chosen after the fact.

The config file is JSON with the keys of `global.compare` in values-global.yaml (camelCase), plus `model` (from
`global.model`) and a few platform settings the chart adds. Standard library only.

Words. A launch is one start of one server: start it, time its start, load it, take it down. A round runs every
entry of the lineup once. The lineup is the two sides, each followed by its twin, an identical copy of the side (the
same image, arguments and settings) that measures how far two launches of one image land apart:
A, A~same, B, B~same. Side A is the baseline and side B the subject; every ratio is B/A. `rounds.twins` says which
sides get a twin (both by default); without both twins no case can get a verdict.

Run order ("rotate-reverse"). For round r (from 1) over a lineup g of n entries: off = (r-1) % n,
cyc = ((r-1) // n) % 2, order[k] = g[(k+off) % n], reversed when cyc is 1. With four entries, 8 rounds give every
entry every position equally often and each compared pair (B with A, each side with its twin) each order equally
often. An optional warm-up launches each side once first, unmeasured: it pulls the image and fills the model cache,
so no measured launch pays for a download.

Primary and secondary cases. A few cases, declared before the run, are primary: only they share the correction for
testing several cases, so a run of 8 rounds can still give them a verdict. By default they are the time to first
answer and, at the highest concurrency level, the median time to first token, the output tokens per second and the
output tokens per joule. Every other case is secondary: it gets the same rule on an uncorrected p-value, so its
verdict describes the run and is not a finding. `analysis.primary` lists other ids.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping

from . import stats

SCHEMA = "rhvp-image-compare/2"
A, B = "A", "B"
TWIN = "~same"
WARMUP, SCORED = "warmup", "scored"
SCHEDULE = "rotate-reverse"
TWIN_CHOICES = ("both", A, B, "none")
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
    "rounds": {"count": 8, "warmup": True, "twins": "both"},
    "server": {"port": DEFAULT_PORT, "readyTimeoutSeconds": 1800, "warmupTimeoutSeconds": 3600,
               "gpuMemoryUtilization": 0.85, "maxModelLen": 8192, "seed": 0, "firstAnswerTokens": 16},
    "load": {"image": GUIDELLM_IMAGE, "concurrencies": [1, 8, 32], "promptTokens": 256, "outputTokens": 128,
             "secondsPerLevel": 60, "seed": 42, "requestFormat": "/v1/completions", "timeoutSeconds": None},
    "analysis": {"alpha": 0.05, "primary": None},
    "energy": {"source": "auto", "settleSeconds": 45,
               "thanosUrl": "https://thanos-querier.openshift-monitoring.svc:9091", "minSamples": 10},
}
#: Keys of the earlier rule, with what took their place.
REMOVED = {
    ("rounds", "pairs"): "rounds.count (rounds of A, A~same, B, B~same)",
    ("rounds", "aaPairs"): "rounds.twins (each side runs against its twin in every round)",
    ("analysis", "tieBand"): "the bar measured from the twins",
    ("analysis", "test"): "the exact sign test, the only test",
    ("analysis", "correction"): "Holm's correction over the primary cases, the only correction",
    ("analysis", "minPairs"): "the fewest rounds that can give a verdict, worked out from the primary cases",
    ("analysis", "ciLevel"): "the bar measured from the twins",
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


def twin_key(key: str) -> str:
    return f"{key}{TWIN}"


def token(key: str) -> str:
    """A lineup key as platforms take it in names and label values (no "~"): A, A-same, B, B-same."""
    return key.replace("~", "-")


def lineup(twins: str) -> list[dict]:
    """The lineup in order: A, A~same, B, B~same, with the twins `twins` asks for (both, A, B or none). Each entry:
    key, the side it launches (`of`), and the side it is the twin of (`twin_of`, None for a side)."""
    out = []
    for key in (A, B):
        out.append({"key": key, "of": key, "twin_of": None})
        if twins in ("both", key):
            out.append({"key": twin_key(key), "of": key, "twin_of": key})
    return out


def round_order(keys: list[str], r: int) -> list[str]:
    """The order of round r (from 1) over the lineup `keys` (rotate-reverse)."""
    n = len(keys)
    off = (r - 1) % n
    cyc = ((r - 1) // n) % 2
    order = [keys[(k + off) % n] for k in range(n)]
    return order[::-1] if cyc == 1 else order


def compared_pairs(entries: list[dict]) -> list[tuple[str, str]]:
    """(first, second) for every compared pair: B with A, then each side with its twin."""
    keys = [e["key"] for e in entries]
    out = [(B, A)]
    out += [(e["twin_of"], e["key"]) for e in entries if e["twin_of"]]
    return [p for p in out if p[0] in keys and p[1] in keys]


def balance(entries: list[dict], rounds: int) -> dict:
    """Whether `rounds` rounds are balanced, counted over the planned orders: every entry in every position equally
    often; in every compared pair each runs first equally often, and their positions differ by zero in sum. Also the
    smallest balanced round count (`step`)."""
    keys = [e["key"] for e in entries]

    def check(count: int) -> tuple[bool, dict]:
        orders = [round_order(keys, r) for r in range(1, count + 1)]
        positions = {k: [0] * len(keys) for k in keys}
        for order in orders:
            for at, key in enumerate(order):
                positions[key][at] += 1
        even = all(len(set(row)) == 1 for row in positions.values())
        pairs = {}
        for first, second in compared_pairs(entries):
            ahead = sum(1 for o in orders if o.index(first) < o.index(second))
            shift = sum(o.index(first) - o.index(second) for o in orders)
            pairs[f"{first} vs {second}"] = {"first_ran_first": ahead, "second_ran_first": count - ahead,
                                             "position_difference_sum": shift}
            even = even and 2 * ahead == count and shift == 0
        return even, pairs

    balanced, pairs = check(rounds)
    step = next(c for c in range(1, 4 * len(keys) + 1) if check(c)[0])
    return {"balanced": balanced, "step": step, "pairs": pairs}


def launch_order(entries: list[dict], rounds: int, warmup: bool) -> list[dict]:
    """Every launch in run order: index (from 1), round (None for a warm-up), position in its round (from 1), side
    (the lineup key), of (the side it launches), kind (warmup or scored)."""
    out: list[dict] = []

    def add(key: str, of: str, kind: str, r: int | None, pos: int | None) -> None:
        out.append({"index": len(out) + 1, "round": r, "pos": pos, "side": key, "of": of, "kind": kind})

    if warmup:
        for key in (A, B):
            add(key, key, WARMUP, None, None)
    keys = [e["key"] for e in entries]
    of = {e["key"]: e["of"] for e in entries}
    for r in range(1, rounds + 1):
        for pos, key in enumerate(round_order(keys, r), start=1):
            add(key, of[key], SCORED, r, pos)
    return out


PRIMARY, SECONDARY = "primary", "secondary"
STARTUP = "startup"


def level_id(concurrency: int) -> str:
    return f"c{concurrency}"


def case_id(level: str, metric: str) -> str:
    return f"{level}/{metric}"


def default_primary(concurrencies: list[int], energy: bool) -> list[str]:
    """The default primary cases: time to first answer, and at the highest concurrency level the median time to
    first token, the output tokens per second and (when energy is measured) the output tokens per joule."""
    top = level_id(max(concurrencies))
    out = [case_id(STARTUP, "first_answer_s"), case_id(top, "ttft_ms_median"), case_id(top, "output_tokens_per_s")]
    if energy:
        out.append(case_id(top, "tokens_per_joule[gpu]"))
    return out


def case_family(concurrencies: list[int], energy: bool) -> list[dict]:
    """The declared cases, in report order. Each has an id (the key in every launch's `values` map), the metric it
    measures (shared by the same figure at every level), a label, a unit, the direction that is better, its kind
    (time, rate or efficiency, which picks the verdict words) and its load level's concurrency. `role` and `group`
    are set by `build`."""
    out = [
        {"id": case_id(STARTUP, "ready_s"), "metric": "ready_s", "label": "time to ready", "unit": "s",
         "better": stats.LOWER, "kind": stats.TIME, "concurrency": None},
        {"id": case_id(STARTUP, "first_answer_s"), "metric": "first_answer_s", "label": "time to first answer",
         "unit": "s", "better": stats.LOWER, "kind": stats.TIME, "concurrency": None},
    ]
    for c in concurrencies:
        level, streams = level_id(c), f"{c} concurrent"
        out += [
            {"id": case_id(level, "ttft_ms_median"), "metric": "ttft_ms_median",
             "label": f"time to first token, median, {streams}", "unit": "ms", "better": stats.LOWER,
             "kind": stats.TIME, "concurrency": c},
            {"id": case_id(level, "itl_ms_median"), "metric": "itl_ms_median",
             "label": f"inter-token latency, median, {streams}", "unit": "ms", "better": stats.LOWER,
             "kind": stats.TIME, "concurrency": c},
            {"id": case_id(level, "output_tokens_per_s"), "metric": "output_tokens_per_s",
             "label": f"output tokens per second, {streams}", "unit": "tokens/s", "better": stats.HIGHER,
             "kind": stats.RATE, "concurrency": c},
        ]
        if energy:
            out.append({"id": case_id(level, "tokens_per_joule[gpu]"), "metric": "tokens_per_joule[gpu]",
                        "label": f"output tokens per joule ({ENERGY_LABEL}), {streams}", "unit": "tokens/J",
                        "better": stats.HIGHER, "kind": stats.EFFICIENCY, "concurrency": c})
    return out


def digest(config: Mapping) -> str:
    """A short, stable hash of the config: part of the run id, so a changed config is a new run."""
    text = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def power(rounds: int, family: int, alpha: float, twins: list[str]) -> dict:
    """What the planned rounds allow before any measurement: the fewest rounds a verdict on a primary case needs
    (min_rounds), whether the plan has them, how many rounds must then lean the same way, and the most primary cases
    this many rounds can give a verdict. Without a twin for both sides no case can get a verdict."""
    needed = stats.min_rounds(family, alpha)
    both = A in twins and B in twins
    return {"family_size": family, "min_rounds": needed, "reachable": rounds >= needed and both,
            "both_twins": both, "must_agree": stats.must_agree(rounds, family, alpha),
            "max_primary_cases": stats.max_cases(rounds, alpha), "plain": stats.plain_family(family)}


def _removed(config: Mapping) -> None:
    for (section, key), instead in REMOVED.items():
        if isinstance(config.get(section), Mapping) and key in config[section]:
            raise PlanError(f"{section}.{key} is no longer read; it was replaced by {instead}. Remove it from the "
                            f"config")


def build(config: Mapping, platform: str = "openshift") -> dict:
    """The plan for `config`, or PlanError. `platform` is openshift or docker: it decides what energy source
    `auto` means (DCGM through the cluster's monitoring on OpenShift, nvidia-smi sampling under docker)."""
    if not isinstance(config, Mapping):
        raise PlanError("the config must be a JSON object")
    _removed(config)
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

    rounds_cfg = cfg["rounds"]
    rounds = _number(rounds_cfg.get("count"), "rounds.count", 1, 50, integer=True)
    warmup = rounds_cfg.get("warmup", True)
    if not isinstance(warmup, bool):
        raise PlanError("rounds.warmup must be true or false")
    twins = rounds_cfg.get("twins", "both")
    if twins not in TWIN_CHOICES:
        raise PlanError(f"rounds.twins must be one of {', '.join(TWIN_CHOICES)}; got {twins!r}")
    entries = lineup(twins)
    for entry in entries:
        entry["name"] = (side_a if entry["of"] == A else side_b)["name"] + (TWIN if entry["twin_of"] else "")

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
    alpha = _number(analysis.get("alpha"), "analysis.alpha", 0.0001, 0.5)

    energy = cfg["energy"]
    source = energy.get("source")
    if source not in ENERGY_SOURCES:
        raise PlanError(f"energy.source must be one of {', '.join(ENERGY_SOURCES)}")
    if source == "auto":
        source = "dcgm" if platform == "openshift" else "nvidia-smi"
    if not (side_a["gpu"] and side_b["gpu"]):
        source = "none"  # a side without a GPU has no GPU power to read
    cases = case_family(levels, source != "none")
    ids = [c["id"] for c in cases]
    primary = analysis.get("primary")
    if primary is None:
        primary = default_primary(levels, source != "none")
    elif (not isinstance(primary, list) or not primary or not all(isinstance(p, str) for p in primary)
          or len(set(primary)) != len(primary)):
        raise PlanError("analysis.primary must be a non-empty list of case ids without repeats, or null for the "
                        "default")
    unknown = [p for p in primary if p not in ids]
    if unknown:
        raise PlanError(f"analysis.primary names cases this run does not have: {', '.join(unknown)}; the cases "
                        f"are {', '.join(ids)}")
    for case in cases:
        case["role"] = PRIMARY if case["id"] in primary else SECONDARY
        case["group"] = f"{case['role']}:{case['metric']}"  # the floor is the largest over this group
    family = len(primary)
    twinned = [e["twin_of"] for e in entries if e["twin_of"]]
    rule = stats.strict(alpha)

    plan = {
        "schema": SCHEMA,
        "run_name": run_name,
        "run_id": f"{run_name}-{digest(config)}",
        "platform": platform,
        "model": {"id": model["id"].strip(), "source": model["source"].strip(), "revision": revision.strip()},
        "sides": [side_a, side_b],
        "lineup": entries,
        "twins": twinned,
        "schedule": SCHEDULE,
        "rounds": rounds,
        "warmup": warmup,
        "launches": launch_order(entries, rounds, warmup),
        "balance": balance(entries, rounds),
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
            "scope": "gpu",
            "settle_s": _number(energy.get("settleSeconds"), "energy.settleSeconds", 0, 3600),
            "thanos_url": str(energy.get("thanosUrl") or DEFAULTS["energy"]["thanosUrl"]),
            "min_samples": _number(energy.get("minSamples"), "energy.minSamples", 2, 10000, integer=True),
        },
        "analysis": {
            "rule": rule.name,
            "rule_fields": rule.fields(),
            "baseline": A, "subject": B, "ratio": "B/A",
            "alpha": alpha,
            "pairing": "a pair is side B's and side A's figure from the same round",
            "unit": "one figure per launch and case; requests are never pooled across launches",
            "test": "exact two-sided sign test over the rounds where the two figures differ; a round with equal "
            "figures is not a trial, and one paired round gives no p-value",
            "family_rule": "Holm's correction over the primary cases only; the twin comparisons and the secondary "
            "cases are outside that family, and a secondary case is graded on its uncorrected p-value, as a "
            "description",
            "floor_rule": "a side's noise floor on a group of cases is the largest abs(median(side / twin) - 1) "
            "over the group (by default the cases of one metric and role); the bar is side B's floor plus side "
            "A's",
            "verdict_rule": "outside the bar and p below alpha: FASTER or SLOWER (time), BETTER or WORSE (rates, "
            "tokens per joule); inside the bar and p not below alpha: TIE; inside but significant, or outside but "
            "not significant: UNRESOLVED; NO VERDICT when the comparison cannot be graded",
            "primary": [c["id"] for c in cases if c["role"] == PRIMARY],
        },
        "cases": cases,
        "power": power(rounds, family, alpha, twinned),
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
