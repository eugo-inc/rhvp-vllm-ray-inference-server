# SPDX-License-Identifier: Apache-2.0
"""The adapter between the results file and the statistics: reads `plan` and `rounds`, writes `analysis`.

This is the boundary a replacement tool plugs into. It reads only:

- `plan.analysis` (test, correction, alpha, tie band, minimum pairs, interval level),
- `plan.metrics` (the declared family: id, label, unit, better, kind),
- `rounds[]` with `kind`, `pair`, `side`, `status` and the flat `metrics` map of each round,

and returns the `analysis` object that results.schema.json describes. Anything that reads the same fields and writes
an `analysis` of the same shape can replace this module and stats.py together. Standard library only.
"""

from __future__ import annotations

from collections.abc import Mapping

from . import stats

TOOL = {"name": "rhvp_compare.analyze", "version": "1"}


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


def _side_median(rounds: list[Mapping], side: str, metric: str) -> float | None:
    values = [v for r in rounds if r.get("kind") == "scored" and r.get("status") == "ok" and r.get("side") == side
              and (v := _value(r, metric)) is not None]
    return stats.median(values) if values else None


def analyze(results: Mapping) -> dict:
    """The `analysis` section for a results file (complete or partial)."""
    plan = results["plan"]
    method = plan["analysis"]
    metrics = plan["metrics"]
    rounds = list(results.get("rounds") or [])
    family = len(metrics)

    rows = []
    for metric in metrics:
        pairs = paired(rounds, "scored", metric["id"])
        ratio_list = stats.ratios(pairs)
        sign = stats.sign_test(ratio_list)
        wilcoxon = stats.wilcoxon_test(ratio_list)
        interval = stats.median_interval(ratio_list, method["ci_level"])
        rows.append({
            "id": metric["id"], "label": metric["label"], "unit": metric["unit"], "better": metric["better"],
            "kind": metric["kind"], "n_pairs": len(ratio_list),
            "a_median": _side_median(rounds, "A", metric["id"]), "b_median": _side_median(rounds, "B", metric["id"]),
            "ratios": ratio_list,
            "median_ratio": stats.median(ratio_list) if ratio_list else None,
            "ci": {"low": interval.low, "high": interval.high, "level": interval.level},
            "p_sign": sign.p, "p_wilcoxon": wilcoxon.p,
            "p_value": sign.p if method["test"] == "sign" else wilcoxon.p,
        })

    adjusted = stats.adjust([row["p_value"] for row in rows], method["correction"])
    for row, p_adj in zip(rows, adjusted):
        reachable = stats.floor_p(row["n_pairs"], family, method["correction"]) <= method["alpha"]
        decided = stats.verdict(row["median_ratio"], p_adj, row["better"], method["tie_band"], method["alpha"],
                                row["n_pairs"], method["min_pairs"], reachable)
        words = stats.EFFICIENCY_WORDS if row["kind"] == "efficiency" else stats.SPEED_WORDS
        row.update({"p_adjusted": p_adj, "reachable": reachable, "outcome": decided.outcome,
                    "verdict": words[decided.outcome], "reason": decided.reason})

    noise_rows = []
    if any(r.get("kind") == "aa" for r in rounds):
        for metric in metrics:
            aa = stats.ratios(paired(rounds, "aa", metric["id"]))
            floor = stats.noise(aa)
            noise_rows.append({"id": metric["id"], "pairs": floor.pairs, "median_abs_change": floor.median_abs_change,
                               "max_abs_change": floor.max_abs_change})

    usable = [row["n_pairs"] for row in rows]
    return {
        "tool": dict(TOOL),
        "method": {**method, "family_size": family,
                   "pairs_planned": plan.get("pairs"),
                   "pairs_usable_min": min(usable) if usable else 0,
                   "min_adjusted_p": stats.floor_p(min(usable) if usable else 0, family, method["correction"])},
        "metrics": rows,
        "noise_floor": noise_rows or None,
    }
