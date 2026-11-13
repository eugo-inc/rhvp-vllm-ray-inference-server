# SPDX-License-Identifier: Apache-2.0
"""The adapter between the results file and the statistics: reads `plan` and `rounds`, writes `analysis`.

This is the boundary a replacement tool plugs into. It reads only:

- `plan.analysis` (test, correction, alpha, tie band, minimum pairs, interval level),
- `plan.metrics` (the declared metrics: id, label, unit, better, kind, role, concurrency),
- `rounds[]` with `kind`, `pair`, `side`, `status`, `error_code`, `node`, `levels` (each level's `concurrency` and
  `usable`) and the flat `metrics` map of each round,

and returns the `analysis` object that results.schema.json describes. The correction runs over the primary metrics
only (role "primary"); a secondary metric's p-value is left uncorrected and its verdict is descriptive. A results file
written before roles existed counts every metric as primary. Anything that reads the same fields and writes
an `analysis` of the same shape can replace this module and stats.py together. Standard library only.

Two checks run over the verdict stats.verdict gives, and make it INCONCLUSIVE:

- one side failing. A pair that lost a metric because one side failed in a way that is that side's doing (its round
  failed with SIDE_FAILED, SIDE_NEVER_READY or LOAD_FAILED, or requests failed at that load level) is evidence
  against that side, and the pairs that are left would overstate it. So any such pair makes the metric
  INCONCLUSIVE, whichever side failed. Rounds that failed for other reasons (stopped, the GPU was not free, the
  platform) and rounds not run leave no such mark. Every row says how many pairs it lost and why (`pairs_lost`).
- more than one node. Scored rounds that ran on different nodes compare the nodes as much as the images.
"""

from __future__ import annotations

from collections.abc import Mapping

from . import stats

TOOL = {"name": "rhvp_compare.analyze", "version": "2"}
#: Round failure codes that belong to the side itself (runner.ROUND_CODES; not imported, to keep this module alone).
SIDE_CODES = ("SIDE_FAILED", "SIDE_NEVER_READY", "LOAD_FAILED")


def _value(round_: Mapping, metric: str) -> float | None:
    value = (round_.get("metrics") or {}).get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        return None
    return float(value)


def paired(rounds: list[Mapping], kind: str, metric: str) -> list[tuple[float, float]]:
    """(first, second) values per pair: (A, B) for scored pairs; for A/A pairs, the two A rounds in run order."""
    by_pair: dict[int, list[Mapping]] = {}
    for r in rounds:
        if r.get("kind") == kind and r.get("status") == "ok" and isinstance(r.get("pair"), int):
            by_pair.setdefault(r["pair"], []).append(r)
    out = []
    for pair in sorted(by_pair):
        members = sorted(by_pair[pair], key=lambda r: r.get("index", 0))
        if kind == "scored":
            a = next((r for r in members if r.get("side") == "A"), None)
            b = next((r for r in members if r.get("side") == "B"), None)
        else:
            a, b = (members[0], members[1]) if len(members) >= 2 else (None, None)
        if a is None or b is None:
            continue
        va, vb = _value(a, metric), _value(b, metric)
        if va is not None and vb is not None:
            out.append((va, vb))
    return out


def _side_fault(round_: Mapping | None, metric: Mapping) -> bool | None:
    """Whether this round's value of the metric is missing through the side's own doing: True (its round failed
    with a side code, or requests failed at the metric's load level), False (missing for another reason), None (the
    value is there)."""
    if round_ is None:
        return False
    if round_.get("status") != "ok":
        return round_.get("error_code") in SIDE_CODES
    if _value(round_, metric["id"]) is not None:
        return None
    level = next((lv for lv in round_.get("levels") or [] if lv.get("concurrency") == metric.get("concurrency")),
                 None)
    return bool(metric.get("concurrency") is not None and level is not None and level.get("usable") is False)


def pairs_lost(rounds: list[Mapping], metric: Mapping) -> dict[str, int]:
    """Scored pairs that ran at least one round but give no ratio for the metric, by cause: "A" or "B" when only
    that side failed through its own doing, "both" when both did, "other" for any other cause (a round stopped, not
    run yet, the GPU not free, a missing power reading)."""
    by_pair: dict[int, dict[str, Mapping]] = {}
    for r in rounds:
        if r.get("kind") == "scored" and isinstance(r.get("pair"), int) and r.get("side") in ("A", "B"):
            by_pair.setdefault(r["pair"], {})[r["side"]] = r
    out = {"A": 0, "B": 0, "both": 0, "other": 0}
    for members in by_pair.values():
        fault = {side: _side_fault(members.get(side), metric) for side in ("A", "B")}
        if fault["A"] is None and fault["B"] is None:
            continue  # this pair gives a ratio
        if fault["A"] and fault["B"]:
            out["both"] += 1
        elif fault["A"]:
            out["A"] += 1
        elif fault["B"]:
            out["B"] += 1
        else:
            out["other"] += 1
    return out


def nodes(rounds: list[Mapping]) -> list[str]:
    """The nodes the scored rounds that finished ran on (none under docker)."""
    return sorted({r["node"] for r in rounds if r.get("kind") == "scored" and r.get("status") == "ok"
                   and isinstance(r.get("node"), str) and r["node"]})


def _side_median(rounds: list[Mapping], side: str, metric: str) -> float | None:
    values = [v for r in rounds if r.get("kind") == "scored" and r.get("status") == "ok" and r.get("side") == side
              and (v := _value(r, metric)) is not None]
    return stats.median(values) if values else None


def _role(metric: Mapping) -> str:
    return metric.get("role") or "primary"


def analyze(results: Mapping) -> dict:
    """The `analysis` section for a results file (complete or partial)."""
    plan = results["plan"]
    method = plan["analysis"]
    metrics = plan["metrics"]
    rounds = list(results.get("rounds") or [])
    family = sum(1 for m in metrics if _role(m) == "primary")

    rows, extra = [], {}
    split = len(nodes(rounds)) > 1
    for metric in metrics:
        pairs = paired(rounds, "scored", metric["id"])
        ratio_list = stats.ratios(pairs)
        sign = stats.sign_test(ratio_list)
        wilcoxon = stats.wilcoxon_test(ratio_list)
        interval = stats.median_interval(ratio_list, method["ci_level"])
        test = sign if method["test"] == "sign" else wilcoxon
        rows.append({
            "id": metric["id"], "label": metric["label"], "unit": metric["unit"], "better": metric["better"],
            "kind": metric["kind"], "role": _role(metric), "n_pairs": len(ratio_list),
            "pairs_lost": pairs_lost(rounds, metric),
            "a_median": _side_median(rounds, "A", metric["id"]), "b_median": _side_median(rounds, "B", metric["id"]),
            "ratios": ratio_list,
            "median_ratio": stats.median_ratio(ratio_list) if ratio_list else None,
            "ci": {"low": interval.low, "high": interval.high, "level": interval.level},
            "p_sign": sign.p, "p_wilcoxon": wilcoxon.p,
            "p_value": test.p,
        })
        extra[metric["id"]] = (interval, test.direction)

    primary = [row for row in rows if row["role"] == "primary"]
    for row, p_adj in zip(primary, stats.adjust([row["p_value"] for row in primary], method["correction"])):
        row["p_adjusted"] = p_adj
    primary_floor = dict(zip((row["id"] for row in primary),
                             stats.floors([row["n_pairs"] for row in primary], method["correction"])))
    for row in rows:
        if row["role"] == "primary":
            p_adj, reachable = row["p_adjusted"], primary_floor[row["id"]] <= method["alpha"]
        else:  # descriptive: the same rule on the uncorrected p-value
            p_adj, reachable = row["p_value"], stats.floor_p(row["n_pairs"], 1, "none") <= method["alpha"]
        interval, direction = extra[row["id"]]
        decided = stats.verdict(row["median_ratio"], p_adj, row["better"], method["tie_band"], method["alpha"],
                                row["n_pairs"], method["min_pairs"], reachable, interval, direction)
        if decided.outcome != stats.INCONCLUSIVE:
            if split:
                decided = stats.Verdict(stats.INCONCLUSIVE, stats.REASON_NODES)
            elif row["pairs_lost"]["A"] or row["pairs_lost"]["B"]:
                decided = stats.Verdict(stats.INCONCLUSIVE, stats.REASON_ONE_SIDE)
        words = stats.EFFICIENCY_WORDS if row["kind"] == "efficiency" else stats.SPEED_WORDS
        row.update({"p_adjusted": p_adj if row["role"] == "primary" else None, "reachable": reachable,
                    "outcome": decided.outcome, "verdict": words[decided.outcome], "reason": decided.reason})

    noise_rows = []
    if any(r.get("kind") == "aa" for r in rounds):
        for metric in metrics:
            aa = stats.ratios(paired(rounds, "aa", metric["id"]))
            floor = stats.noise(aa)
            noise_rows.append({"id": metric["id"], "pairs": floor.pairs, "median_abs_change": floor.median_abs_change,
                               "max_abs_change": floor.max_abs_change})

    usable = [row["n_pairs"] for row in primary]
    outcomes: dict[str, dict[str, int]] = {"primary": {}, "secondary": {}}
    for row in rows:
        outcomes[row["role"]][row["verdict"]] = outcomes[row["role"]].get(row["verdict"], 0) + 1
    return {
        "tool": dict(TOOL),
        "method": {**method, "family_size": family,
                   "pairs_planned": plan.get("pairs"),
                   "pairs_usable_min": min(usable) if usable else 0,
                   "min_adjusted_p": min(primary_floor.values()) if primary_floor else 1.0},
        "metrics": rows,
        "outcomes": outcomes,
        "noise_floor": noise_rows or None,
    }
