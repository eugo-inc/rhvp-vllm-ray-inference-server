# SPDX-License-Identifier: Apache-2.0
"""A synthetic results file, for the comparison notebook's REPLAY mode and the tests: made up by this script from a
fixed seed, never measured, and flagged as such (`run.synthetic: true`, the made-up images registry.example/...).

make_notebook.py writes it to compare-recordings/synthetic.json and checks it with --check, like the timing
notebook's synthetic recording. Standard library only.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import analyze, plan as plans

PATH = Path(__file__).resolve().parents[1] / "compare-recordings" / "synthetic.json"
SEED = 20260101
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
CONFIG = {
    "runName": "synthetic-example",
    "model": {"id": "example-model", "source": "example-org/example-model", "revision": ""},
    "sides": [
        {"name": "baseline", "image": "registry.example/inference-server:baseline", "preset": "vllm-openai"},
        {"name": "candidate", "image": "registry.example/inference-server:candidate", "preset": "vllm-openai"},
    ],
    "rounds": {"pairs": 10, "warmup": True, "aaPairs": 2},
}
#: Side A's typical figures, and side B's against them (made up).
A_BASE = {"ready_s": 92.0, "first_answer_s": 95.0, "ttft_ms": {1: 22.0, 8: 31.0, 32: 58.0},
          "itl_ms": {1: 6.1, 8: 7.4, 32: 11.8}, "tps": {1: 160.0, 8: 1050.0, 32: 2600.0},
          "power_w": {1: 95.0, 8: 160.0, 32: 230.0}}
B_FACTOR = {"ready_s": 0.99, "first_answer_s": 0.80, "ttft_ms": 1.01, "itl_ms": 0.90, "tps": 1.10, "power_w": 1.02}
PHASES = (("weights_load", "loading the weights", 0.20), ("torch_compile", "torch.compile in total", 0.30),
          ("cuda_graphs", "capturing CUDA graphs", 0.10), ("engine_init", "engine start (profile, KV cache, "
                                                                          "warm-up)", 0.45))


def _iso(at: datetime) -> str:
    return at.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _level(rng: random.Random, side: str, c: int, at: float, seconds: float) -> dict:
    b = side == "B"

    def jitter(value: float, key: str) -> float:
        return value * (B_FACTOR[key] if b else 1.0) * rng.uniform(0.98, 1.02)

    ttft, itl, tps = jitter(A_BASE["ttft_ms"][c], "ttft_ms"), jitter(A_BASE["itl_ms"][c], "itl_ms"), jitter(
        A_BASE["tps"][c], "tps")
    power = jitter(A_BASE["power_w"][c], "power_w")
    return {
        "concurrency": c, "started_at": _iso(datetime.fromtimestamp(at, timezone.utc)),
        "finished_at": _iso(datetime.fromtimestamp(at + seconds, timezone.utc)), "start_epoch": at,
        "end_epoch": at + seconds, "duration_s": seconds,
        "requests": {"successful": int(tps * seconds / 128), "errored": 0, "incomplete": c},
        "ttft_ms": {"median": ttft, "p90": ttft * 1.3, "mean": ttft * 1.05, "count": int(tps * seconds / 128)},
        "itl_ms": {"median": itl, "p90": itl * 1.2, "mean": itl * 1.02, "count": int(tps * seconds)},
        "output_tokens": int(tps * seconds), "output_tokens_per_s": tps, "requests_per_s": tps / 128,
        "usable": True, "note": None,
        "energy": {"source": "dcgm", "label": plans.ENERGY_LABEL, "method": "energy_counter",
                   "samples": int(seconds / 15), "mean_power_w": power, "energy_j": power * seconds,
                   "tokens_per_joule": tps / power, "joules_per_token": power / tps, "note": None},
    }


def synthesize() -> dict:
    """The whole synthetic results file."""
    rng = random.Random(SEED)
    plan = plans.build(CONFIG, "openshift")
    clock = START
    rounds = []
    for planned in plan["rounds"]:
        side = planned["side"]
        b = side == "B" and planned["kind"] != plans.AA
        started = clock
        ready = A_BASE["ready_s"] * (B_FACTOR["ready_s"] if b else 1.0) * rng.uniform(0.97, 1.03)
        first = A_BASE["first_answer_s"] * (B_FACTOR["first_answer_s"] if b else 1.0) * rng.uniform(0.97, 1.03)
        if planned["kind"] == plans.WARMUP:
            ready, first = ready + 240.0, first + 240.0  # the image pull and the model download
        breakdown = [{"phase": "first_log", "label": "the server writes its first log line", "seconds": None,
                      "at_s": round(ready * 0.05, 3), "source": "log"}]
        for key, label, share in PHASES:
            breakdown.append({"phase": key, "label": label, "seconds": round(ready * share, 2),
                              "at_s": None, "source": "log", "count": 1})
        record = {
            "index": planned["index"], "side": side, "kind": planned["kind"], "pair": planned["pair"],
            "status": "ok", "error": None, "error_code": None, "started_at": _iso(started), "finished_at": None,
            "node": "gpu-node-1",
            "image_digest": f"registry.example/inference-server@sha256:{('a' if side == 'A' else 'b') * 64}",
            "server_version": {"vllm": "0.0.0-example", "source": "endpoint"},
            "startup": {"ready_s": round(ready, 3), "first_answer_s": round(first, 3),
                        "clock": "runner, from just before the server was asked for", "breakdown": breakdown},
            "levels": [], "metrics": {"startup.ready_s": ready, "startup.first_answer_s": first},
            "artifacts": {"server_log": f"rounds/{planned['index']:02d}-{side}-{planned['kind']}/server.log"},
            "start_retries": 0,
        }
        at = (started + timedelta(seconds=first + 5)).timestamp()
        if planned["kind"] != plans.WARMUP:
            seconds = plan["load"]["seconds_per_level"]
            for c in plan["load"]["concurrencies"]:
                level = _level(rng, "B" if b else "A", c, at, seconds)
                record["levels"].append(level)
                at += seconds + 2
                record["metrics"].update({f"c{c}.ttft_ms_median": level["ttft_ms"]["median"],
                                          f"c{c}.itl_ms_median": level["itl_ms"]["median"],
                                          f"c{c}.output_tokens_per_s": level["output_tokens_per_s"],
                                          f"c{c}.tokens_per_joule": level["energy"]["tokens_per_joule"]})
            record["artifacts"]["load_report"] = (f"rounds/{planned['index']:02d}-{side}-{planned['kind']}/"
                                                  "guidellm.json")
        clock = datetime.fromtimestamp(at + 30, timezone.utc)
        record["finished_at"] = _iso(clock)
        rounds.append(record)
    results = {
        "schema": plans.SCHEMA,
        "run": {"id": plan["run_id"], "name": plan["run_name"], "status": "complete", "error_code": None,
                "started_at": _iso(START),
                "finished_at": _iso(clock), "platform": "openshift", "server_restored": True, "synthetic": True,
                "notes": ["Synthetic: written by notebooks/rhvp_compare/example.py, never measured."], "errors": []},
        "plan": plan,
        "provenance": {
            "harness": {"name": "rhvp_compare", "version": "1"},
            "load_tool": {"name": "guidellm", "image": plan["load"]["image"], "version": "0.8.1"},
            "seeds": {"load": plan["load"]["seed"], "server": plan["server"]["seed"]},
            "sides": {s["key"]: {"image": s["image"], "image_digest": rounds[i]["image_digest"],
                                 "server_version": {"vllm": "0.0.0-example", "source": "endpoint"}}
                      for i, s in enumerate(plan["sides"])},
            "gpu": {"product": "Example-GPU", "driver": "0.0", "cuda_driver": "0.0", "memory_mib": 24000, "count": 1,
                    "source": "made up"},
            "node": {"name": "gpu-node-1", "arch": "amd64", "instance_type": "example.large", "os_image": "example",
                     "kernel": "0.0", "container_runtime": "example"},
            "nodes": [{"name": "gpu-node-1", "gpu": "Example-GPU"}],
            "cluster": {"openshift_version": "4.20.0"},
        },
        "rounds": rounds,
        "analysis": None,
    }
    results["analysis"] = analyze.analyze(results)
    return results


def text(results: dict) -> str:
    return json.dumps(results, indent=1, sort_keys=False, ensure_ascii=False) + "\n"
