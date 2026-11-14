# SPDX-License-Identifier: Apache-2.0
"""Short, machine-readable views of a run, for readers that cannot open the results volume:

- `status(results)`: progress and outcome: the phase, rounds and launches done, an estimate of the time left, the
  failure codes, the load levels left out because requests failed, whether the live server is back, and the primary
  verdicts. On OpenShift the runner keeps it in ConfigMap `compare-<run id>` (key status.json), with the whole
  results file (key result.json) as long as it fits, and updates both after every launch.
- `termination(results)`: at most 4096 bytes for the runner container's termination message, which `oc get pod`
  shows after the run: status, failure code, whether the live server is back, rounds done and the verdict counts.
- `log_lines(results)`: the whole results file as one compact JSON line between two marker lines, printed at the end
  of a run, for a reader that has only the Job's log.

Standard library only.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from collections.abc import Mapping

STATUS_SCHEMA = "rhvp-image-compare-status/2"
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


def rounds(results: Mapping) -> tuple[int, int]:
    """(rounds whose every launch ran, rounds planned)."""
    if not results.get("plan"):
        return 0, 0
    planned: dict[int, int] = {}
    for launch in results["plan"]["launches"]:
        if launch["kind"] == "scored":
            planned[launch["round"]] = planned.get(launch["round"], 0) + 1
    seen: dict[int, int] = {}
    for launch in results.get("launches") or []:
        if launch.get("kind") == "scored" and isinstance(launch.get("round"), int):
            seen[launch["round"]] = seen.get(launch["round"], 0) + 1
    return sum(1 for r, n in planned.items() if seen.get(r, 0) >= n), len(planned)


def eta_seconds(results: Mapping) -> int | None:
    """Seconds left while the run goes: the mean length of the launches so far times the launches still to run.
    None once the run has ended, and before the first launch has finished."""
    if (results.get("run") or {}).get("status") != "running":
        return None
    done = results.get("launches") or []
    lengths = [end - start for r in done
               if (start := _epoch(r.get("started_at"))) is not None and (end := _epoch(r.get("finished_at")))
               is not None]
    if not lengths:
        return None
    left = len(results["plan"]["launches"]) - len(done)
    return max(0, round(sum(lengths) / len(lengths) * left))


def levels_unusable(results: Mapping) -> list[dict]:
    """The load levels of finished measured launches where requests failed or none finished. A concurrency level is
    then left out of the comparison; a scale-out surge (concurrency None) only loses its latency medians, and its
    failed requests count as requests that missed the goals (guidellm.surge_values)."""
    out = []
    for r in results.get("launches") or []:
        if r.get("status") != "ok" or r.get("kind") == "warmup":
            continue
        for level in r.get("levels") or []:
            if level.get("usable") is False:
                requests = level.get("requests") or {}
                out.append({"index": r["index"], "round": r.get("round"), "side": r["side"], "kind": r["kind"],
                            "concurrency": level.get("concurrency"), "errored": requests.get("errored"),
                            "successful": requests.get("successful")})
    return out


#: summary.surge_rate_fit: "all_failed" needs at least this share of a surge's judged requests (those that finished
#: or failed) to have failed, in every scored surge of both sides; "all_missed" needs at most this share within the
#: latency goals in every one. words.SCALE_RATE_ALL_FAILED and SCALE_RATE_ALL_MISSED state both.
SURGE_FAILED_SHARE = 0.5
SURGE_MET_SHARE_LOW = 0.05


def surge_rate_fit(results: Mapping) -> str | None:
    """Mode scaleout, when the surge rate did not separate the sides, from what every scored surge of both sides
    (their twins' too) measured: "all_met" when each met the latency goals for every request it judged and none
    failed; "all_failed" when in each at least SURGE_FAILED_SHARE of the judged requests (finished or failed) failed;
    "all_missed" when in each at most SURGE_MET_SHARE_LOW of them met the goals (a surge in which nothing finished or
    failed counts as 0). Else None (also when a side has no finished scored surge). README, "Picking the surge
    rate", says what to change."""
    if (results.get("plan") or {}).get("mode") != "scaleout":
        return None
    surges: dict[str, list] = {"A": [], "B": []}
    for r in results.get("launches") or []:
        side = r.get("of", r.get("side"))
        if r.get("kind") == "scored" and r.get("status") == "ok" and side in surges:
            surges[side] += [lv for lv in r.get("levels") or [] if lv.get("concurrency") is None]
    if not surges["A"] or not surges["B"]:
        return None
    every = surges["A"] + surges["B"]

    def counts(level: Mapping) -> tuple[int, int]:
        requests = level.get("requests") or {}
        failed, finished = requests.get("errored") or 0, requests.get("successful") or 0
        return failed, failed + finished

    if all(counts(lv)[0] == 0 and lv.get("slo_attainment") == 1 for lv in every):
        return "all_met"
    if all(judged and failed >= SURGE_FAILED_SHARE * judged for failed, judged in map(counts, every)):
        return "all_failed"

    def met(level: Mapping) -> float | None:
        value = level.get("slo_attainment")
        return 0.0 if value is None and not counts(level)[1] else value

    if all(isinstance(share := met(lv), (int, float)) and share <= SURGE_MET_SHARE_LOW for lv in every):
        return "all_missed"
    return None


def energy_methods_mixed(results: Mapping) -> list[dict]:
    """The load levels whose tokens per joule came from the energy counter in some finished measured launches and from
    the mean of the power samples in others (energy.window_power picks per level), with the launches of each."""
    by_level: dict[int, dict[str, list[int]]] = {}
    for r in results.get("launches") or []:
        if r.get("status") != "ok" or r.get("kind") == "warmup":
            continue
        for level in r.get("levels") or []:
            method = (level.get("energy") or {}).get("method")
            if method in ("energy_counter", "power_mean") and level.get("concurrency") is not None:
                by_level.setdefault(level["concurrency"], {}).setdefault(method, []).append(r["index"])
    return [{"concurrency": c, "energy_counter": methods["energy_counter"], "power_mean": methods["power_mean"]}
            for c, methods in sorted(by_level.items()) if len(methods) == 2]


def _primary(results: Mapping) -> list[dict]:
    rows = ((results.get("analysis") or {}).get("comparisons")) or []
    return [{"case": row["case"], "verdict": row["verdict"], "outcome": row["outcome"], "reason": row.get("reason"),
             "ratio": row.get("ratio"), "bar": row.get("bar"), "p_corrected": row.get("p_corrected"),
             "n_rounds": row.get("n_rounds")}
            for row in rows if row.get("role", "primary") == "primary"]


def status(results: Mapping, now: float) -> dict:
    run = results.get("run") or {}
    done, total = rounds(results)
    launches = results.get("launches") or []
    return {
        "schema": STATUS_SCHEMA,
        "run_id": run.get("id"),
        "phase": run.get("status"),
        "error_code": run.get("error_code"),
        "completed_rounds": done,
        "total_rounds": total,
        "launches_done": len(launches),
        "launches_total": len(results["plan"]["launches"]),
        "launches_failed": [{"index": r["index"], "round": r.get("round"), "side": r["side"], "kind": r["kind"],
                             "error_code": r.get("error_code")} for r in launches if r.get("status") != "ok"],
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
    done, total = rounds(results)
    data = {"run_id": run.get("id"), "status": run.get("status"), "error_code": run.get("error_code"),
            "server_restored": run.get("server_restored"), "completed_rounds": done, "total_rounds": total,
            "outcomes": analysis.get("outcomes"),
            "primary": {row["case"]: row["verdict"] for row in _primary(results)},
            "results_file": f"{run.get('id')}/results.json"}
    for drop in ("primary", "outcomes", "results_file"):
        text = compact(data)
        if len(text.encode("utf-8")) <= TERMINATION_LIMIT:
            return text
        data.pop(drop)
    return compact({"run_id": str(run.get("id"))[:200], "status": run.get("status"),
                    "error_code": run.get("error_code"), "server_restored": run.get("server_restored")})


def replica_gpus(results: Mapping) -> dict[int, list[dict]]:
    """Mode scaleout: per replica number, the GPUs its servers got in the scored launches that finished, from the
    server records, in first-seen order: each the request (`gpus`) with the UUIDs nvidia-smi read through it (`uuids`,
    empty when not read; for a MIG device, its parent GPU's UUID); empty when no record has a request."""
    out: dict[int, list[dict]] = {}
    for r in results.get("launches") or []:
        if r.get("kind") != "scored" or r.get("status") != "ok":
            continue
        for item in r.get("servers") or []:
            if not isinstance(item.get("gpus"), str):
                continue
            uuids = [u for u in item.get("gpu_uuids") or [] if isinstance(u, str)]
            entry = {"gpus": item["gpus"], "uuids": uuids}
            if entry not in out.setdefault(item["replica"], []):
                out[item["replica"]].append(entry)
    return {n: given for n, given in sorted(out.items()) if given}


def not_started(code: str, message: str, run_id: str | None = None) -> dict:
    """What the log and the termination message say when no results file was written: the config cannot be run
    (PLAN_INVALID), another comparison is running (RUN_ACTIVE), or the run failed before it began (RUNNER_FAILED)."""
    return {"run": {"id": run_id, "status": "failed", "error_code": code, "errors": [message[:2000]]}}


def log_lines(results: Mapping) -> list[str]:
    return [BEGIN, compact(results), END]
