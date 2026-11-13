# SPDX-License-Identifier: Apache-2.0
"""Short, machine-readable views of a run, for readers that cannot open the results volume:

- `status(results)`: progress and outcome: the phase, pairs and rounds done, an estimate of the time left, the
  failure codes, the load levels left out because requests failed, whether the live server is back, and the primary
  verdicts. On OpenShift the runner keeps it in ConfigMap `compare-<run id>` (key status.json), with the whole
  results file (key result.json) as long as it fits, and updates both after every round.
- `termination(results)`: at most 4096 bytes for the runner container's termination message, which `oc get pod`
  shows after the run: status, failure code, whether the live server is back, usable pairs and the verdict counts.
- `log_lines(results)`: the whole results file as one compact JSON line between two marker lines, printed at the end
  of a run, for a reader that has only the Job's log.

Standard library only.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from collections.abc import Mapping

STATUS_SCHEMA = "rhvp-image-compare-status/1"
BEGIN = "===== rhvp-compare results begin ====="
END = "===== rhvp-compare results end ====="
TERMINATION_LIMIT = 4096
#: result.json is left out of the ConfigMap above this many bytes (a ConfigMap holds at most 1 MiB in all).
CONFIGMAP_RESULT_LIMIT = 900_000


def compact(data: Mapping) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def _epoch(text: str | None) -> float | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def pairs(results: Mapping) -> tuple[int, int]:
    """(scored pairs with both rounds run, scored pairs planned)."""
    if not results.get("plan"):
        return 0, 0
    planned = {r["pair"] for r in results["plan"]["rounds"] if r["kind"] == "scored"}
    seen: dict[int, int] = {}
    for r in results.get("rounds") or []:
        if r.get("kind") == "scored" and isinstance(r.get("pair"), int):
            seen[r["pair"]] = seen.get(r["pair"], 0) + 1
    return sum(1 for pair in planned if seen.get(pair, 0) >= 2), len(planned)


def eta_seconds(results: Mapping) -> int | None:
    """Seconds left while the run goes: the mean length of the rounds so far times the rounds still to run. None
    once the run has ended, and before the first round has finished."""
    if (results.get("run") or {}).get("status") != "running":
        return None
    done = results.get("rounds") or []
    lengths = [end - start for r in done
               if (start := _epoch(r.get("started_at"))) is not None and (end := _epoch(r.get("finished_at")))
               is not None]
    if not lengths:
        return None
    left = len(results["plan"]["rounds"]) - len(done)
    return max(0, round(sum(lengths) / len(lengths) * left))


def levels_unusable(results: Mapping) -> list[dict]:
    """The load levels of finished rounds (scored and A/A) where requests failed or none finished. A concurrency
    level is then left out of the comparison; a scale-out surge (concurrency None) only loses its latency medians,
    and its failed requests count as requests that missed the goals (guidellm.surge_metrics)."""
    out = []
    for r in results.get("rounds") or []:
        if r.get("status") != "ok" or r.get("kind") == "warmup":
            continue
        for level in r.get("levels") or []:
            if level.get("usable") is False:
                requests = level.get("requests") or {}
                out.append({"index": r["index"], "side": r["side"], "kind": r["kind"],
                            "concurrency": level.get("concurrency"), "errored": requests.get("errored"),
                            "successful": requests.get("successful")})
    return out


def surge_rate_fit(results: Mapping) -> str | None:
    """Mode scaleout, when the surge rate did not separate the sides: "all_met" when every scored surge of both
    sides met the latency goals for every request it judged and none failed, "all_failed" when requests failed in
    every scored surge of both sides; else None (also when a side has no finished scored surge). README, "Picking
    the surge rate", says what to change."""
    if (results.get("plan") or {}).get("mode") != "scaleout":
        return None
    surges: dict[str, list] = {"A": [], "B": []}
    for r in results.get("rounds") or []:
        if r.get("kind") == "scored" and r.get("status") == "ok" and r.get("side") in surges:
            surges[r["side"]] += [lv for lv in r.get("levels") or [] if lv.get("concurrency") is None]
    if not surges["A"] or not surges["B"]:
        return None
    every = surges["A"] + surges["B"]
    failed = [bool((lv.get("requests") or {}).get("errored")) for lv in every]
    if not any(failed) and all(lv.get("slo_attainment") == 1 for lv in every):
        return "all_met"
    if all(failed):
        return "all_failed"
    return None


def _primary(results: Mapping) -> list[dict]:
    rows = ((results.get("analysis") or {}).get("metrics")) or []
    return [{"id": row["id"], "verdict": row["verdict"], "median_ratio": row.get("median_ratio"),
             "ci": [(row.get("ci") or {}).get("low"), (row.get("ci") or {}).get("high")],
             "p_adjusted": row.get("p_adjusted"), "n_pairs": row.get("n_pairs")}
            for row in rows if row.get("role", "primary") == "primary"]


def status(results: Mapping, now: float) -> dict:
    run = results.get("run") or {}
    done, total = pairs(results)
    rounds = results.get("rounds") or []
    return {
        "schema": STATUS_SCHEMA,
        "run_id": run.get("id"),
        "phase": run.get("status"),
        "error_code": run.get("error_code"),
        "completed_pairs": done,
        "total_pairs": total,
        "rounds_done": len(rounds),
        "rounds_total": len(results["plan"]["rounds"]),
        "rounds_failed": [{"index": r["index"], "side": r["side"], "kind": r["kind"],
                           "error_code": r.get("error_code")} for r in rounds if r.get("status") != "ok"],
        "levels_unusable": levels_unusable(results),
        "server_restored": run.get("server_restored"),
        "eta_seconds": eta_seconds(results),
        "updated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "sides": {s["key"]: {"name": s["name"], "image": s["image"]} for s in results["plan"]["sides"]},
        "primary": _primary(results),
        "outcomes": (results.get("analysis") or {}).get("outcomes"),
        "results_file": f"{run.get('id')}/results.json",
    }


def result_text(results: Mapping) -> str | None:
    """The results file for the ConfigMap, or None when it is too big to keep there."""
    text = compact(results)
    return text if len(text.encode("utf-8")) <= CONFIGMAP_RESULT_LIMIT else None


def termination(results: Mapping) -> str:
    """The termination message: never more than TERMINATION_LIMIT bytes."""
    run = results.get("run") or {}
    analysis = results.get("analysis") or {}
    done, total = pairs(results)
    data = {"run_id": run.get("id"), "status": run.get("status"), "error_code": run.get("error_code"),
            "server_restored": run.get("server_restored"), "completed_pairs": done, "total_pairs": total,
            "pairs_usable_min": (analysis.get("method") or {}).get("pairs_usable_min"),
            "outcomes": analysis.get("outcomes"),
            "primary": {row["id"]: row["verdict"] for row in _primary(results)},
            "results_file": f"{run.get('id')}/results.json"}
    for drop in ("primary", "outcomes", "results_file"):
        text = compact(data)
        if len(text.encode("utf-8")) <= TERMINATION_LIMIT:
            return text
        data.pop(drop)
    return compact({"run_id": str(run.get("id"))[:200], "status": run.get("status"),
                    "error_code": run.get("error_code"), "server_restored": run.get("server_restored")})


def replica_gpus(results: Mapping) -> dict[int, list[str]]:
    """Mode scaleout: per replica number, the GPU requests its starts got in the scored rounds that finished, from
    the launch records (`gpus`), in first-seen order; empty when no record has one."""
    out: dict[int, list[str]] = {}
    for r in results.get("rounds") or []:
        if r.get("kind") != "scored" or r.get("status") != "ok":
            continue
        for item in r.get("launches") or []:
            if isinstance(item.get("gpus"), str) and item["gpus"] not in out.setdefault(item["replica"], []):
                out[item["replica"]].append(item["gpus"])
    return {n: given for n, given in sorted(out.items()) if given}


def not_started(code: str, message: str, run_id: str | None = None) -> dict:
    """What the log and the termination message say when no results file was written: the config cannot be run
    (PLAN_INVALID), another comparison is running (RUN_ACTIVE), or the run failed before it began (RUNNER_FAILED)."""
    return {"run": {"id": run_id, "status": "failed", "error_code": code, "errors": [message[:2000]]}}


def log_lines(results: Mapping) -> list[str]:
    return [BEGIN, compact(results), END]
