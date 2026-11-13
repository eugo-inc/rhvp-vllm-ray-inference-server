# SPDX-License-Identifier: Apache-2.0
"""The adapter between the results file and the statistics: reads `plan` and `launches`, writes `analysis`.

This is the boundary a replacement tool plugs into, in two steps:

1. `inputs(results)` turns the results file into plain rows: one (round, position, side, case, value) per measured
   launch and case, which side is whose twin, the cases, the launches that failed through the side's own doing, and
   the nodes the launches ran on.
2. `analyse(...)` takes those rows and returns the analysis: one comparison per case (side B against side A), the
   twin comparisons, the rule and its family. It reads nothing else.

Anything that takes the same rows and returns an `analysis` of the same shape can replace this module and stats.py
together. Standard library only.

Two checks this pattern adds to the rule, because a failed launch does not stop a run here:

- a launch of side A or B that failed through its own doing (its launch failed with SIDE_FAILED, SIDE_NEVER_READY
  or LOAD_FAILED, or requests failed at the case's load level) leaves the pairs that are left overstating that side,
  so the case gets NO VERDICT (side_failed), whichever side failed. A launch that failed for another reason (stopped,
  the GPU not free, the platform) only means one pair fewer;
- launches that ran on more than one node compare the nodes as much as the images: every case gets NO VERDICT
  (more_than_one_node).

A figure of 0 or less gives no ratio. On side A or B the case gets NO VERDICT (nonpositive_value), in a pair of B and
A or not: a round the other side has no figure for is not left out unseen. In a side's comparison with its twin, that
side has no noise floor on the case, and every case whose floor uses it (the case's floor group) gets NO VERDICT
(nonpositive_value) too. The floor is never taken from the rounds that are left: which rounds those are would be
chosen after seeing a figure. `nonpositive_twins` names the side, the rounds and the cases, and a note names them in
`notes`.

A launch whose GuideLLM timed out runs again from a new start of the side (runner.LOAD_RETRIES), so it can finish
with figures like any other. Every time-out still counts against its side: `load_timeouts` per side over the whole
run (twins included) and per comparison over its paired rounds, so a load that stalls on one side only is never
hidden behind launches that finished on their second try.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from . import stats

TOOL = {"name": "rhvp_compare.analyze", "version": "5"}
#: Launch failure codes that belong to the side itself (runner.LAUNCH_CODES; not imported, to keep this module alone).
SIDE_CODES = ("SIDE_FAILED", "SIDE_NEVER_READY", "LOAD_FAILED")
SUBJECT, BASELINE = "B", "A"


def _number(launch: Mapping, case: str) -> float | None:
    """The launch's figure for the case when it is a finite number, else None (not measured)."""
    value = (launch.get("values") or {}).get(case)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _measured(results: Mapping) -> list[Mapping]:
    return [r for r in results.get("launches") or [] if r.get("kind") == "scored" and isinstance(r.get("round"), int)]


def _side_fault(launch: Mapping, case: Mapping) -> bool:
    """Whether this launch gives no figure for the case through the side's own doing."""
    if launch.get("status") != "ok":
        return launch.get("error_code") in SIDE_CODES
    if _number(launch, case["id"]) is not None:
        return False
    level = next((lv for lv in launch.get("levels") or [] if lv.get("concurrency") == case.get("concurrency")),
                 None)
    return case.get("concurrency") is not None and level is not None and level.get("usable") is False


def inputs(results: Mapping) -> dict:
    """The rows analyse() takes, from a results file (complete or partial)."""
    plan = results["plan"]
    cases = plan["cases"]
    rows, faults = [], []
    for launch in _measured(results):
        for case in cases:
            value = _number(launch, case["id"]) if launch.get("status") == "ok" else None
            if value is not None:
                rows.append({"round": launch["round"], "pos": launch["pos"], "side": launch["side"],
                             "case": case["id"], "value": value})
            elif _side_fault(launch, case):
                faults.append({"round": launch["round"], "side": launch["side"], "case": case["id"]})
    positions = {(launch["round"], launch["side"]): launch["pos"] for launch in _measured(results)}
    nodes = sorted({r["node"] for r in _measured(results) if r.get("status") == "ok"
                    and isinstance(r.get("node"), str) and r["node"]})
    timeouts = [{"round": r["round"], "side": r["side"], "n": int(r.get("load_timeouts") or 0)}
                for r in _measured(results) if r.get("load_timeouts")]
    return {"rows": rows, "positions": [{"round": r, "side": s, "pos": p} for (r, s), p in sorted(positions.items())],
            "twins": {e["key"]: e["twin_of"] for e in plan["lineup"] if e["twin_of"]},
            "cases": cases, "faults": faults, "nodes": nodes, "alpha": plan["analysis"]["alpha"],
            "timeouts": timeouts}


def _by_round(rows: Iterable[Mapping], side: str, case: str) -> dict[int, float]:
    return {r["round"]: r["value"] for r in rows if r["side"] == side and r["case"] == case}


def _paired(first: Mapping[int, float], second: Mapping[int, float]) -> list[int]:
    return sorted(set(first) & set(second))


def _rounds(numbers: Iterable[int]) -> str:
    """"round 2" or "rounds 2, 5"."""
    numbers = list(numbers)
    return f"round{'s' if len(numbers) > 1 else ''} {', '.join(map(str, numbers))}"


def _floor_entry(floor: stats.Floor | None, case: str | None) -> dict | None:
    if floor is None:
        return None
    return {"value": floor.value, "n": floor.n, "provisional": floor.provisional, "spread": floor.spread,
            "outlier_rounds": list(floor.outlier_rounds), "case": case}


def analyse(rows: list[Mapping], positions: list[Mapping], twins: Mapping[str, str], cases: list[Mapping],
            alpha: float = 0.05, faults: Iterable[Mapping] = (), nodes: Iterable[str] = (),
            subject: str = SUBJECT, baseline: str = BASELINE, timeouts: Iterable[Mapping] = ()) -> dict:
    """The analysis from plain rows. `rows`: round, pos, side, case, value. `positions`: round, side, pos of every
    measured launch. `twins`: twin key -> the side it copies. `cases`: id, better, kind, role, group. `timeouts`:
    round, side, n for every measured launch whose GuideLLM timed out n times."""
    rule = stats.strict(alpha)
    pos = {(p["round"], p["side"]): p["pos"] for p in positions}
    twin_of = {side: twin for twin, side in twins.items()}
    fault_set = {(f["side"], f["case"], f["round"]) for f in faults}
    nodes = list(nodes)
    timed_out = {(t["round"], t["side"]): int(t["n"]) for t in timeouts}
    primary_ids = [c["id"] for c in cases if c.get("role", "primary") == "primary"]
    needed = {"primary": stats.min_rounds(len(primary_ids), alpha), "secondary": stats.min_rounds(1, alpha)}

    groups: dict[str, list[str]] = {}
    for case in cases:
        groups.setdefault(case.get("group") or case["id"], []).append(case["id"])

    def group_of(case: Mapping) -> list[str]:
        return groups[case.get("group") or case["id"]]

    # Floors: per side and case, then the largest over each group. A figure of 0 or less on either side of a
    # comparison with the twin leaves that side without a floor on the case (nonpositive), never with a smaller one.
    case_floor: dict[tuple[str, str], stats.Floor | None] = {}
    nonpositive: dict[tuple[str, str], dict] = {}
    same_vs_same, notes, differs = [], [], []
    for side, twin in sorted(twin_of.items()):
        for case in cases:
            own, copy_ = _by_round(rows, side, case["id"]), _by_round(rows, twin, case["id"])
            both = _paired(own, copy_)
            at_zero = {"side_rounds": [r for r in both if own[r] <= 0],
                       "twin_rounds": [r for r in both if copy_[r] <= 0]}
            if at_zero["side_rounds"] or at_zero["twin_rounds"]:
                nonpositive[(side, case["id"])] = {"side": side, "twin": twin, "case": case["id"], **at_zero,
                                                   "cases": list(group_of(case))}
                case_floor[(side, case["id"])] = None
                continue
            ratios = {r: own[r] / copy_[r] for r in both}
            floor = stats.floor(ratios)
            case_floor[(side, case["id"])] = floor
            if floor is None:
                continue
            lower, higher = stats.signs(list(ratios.values()))
            test = stats.sign_test(lower, higher) if floor.n > 1 else None
            first = sum(1 for r in both if pos.get((r, side), 0) < pos.get((r, twin), 0))
            same_vs_same.append({"side": side, "twin": twin, "case": case["id"], "n": floor.n, "ratio": floor.ratio,
                                 "floor": floor.value, "spread": floor.spread,
                                 "outlier_rounds": list(floor.outlier_rounds), "provisional": floor.provisional,
                                 "p_raw": test.p if test else None, "order_split": [first, len(both) - first]})
            if test and test.p < 0.05:
                differs.append((side, case["id"], max(lower, higher), lower + higher, test.p))
    for entry in nonpositive.values():  # the round and the side, then what it costs
        where = [f"side {entry['side']} in {_rounds(entry['side_rounds'])}"] if entry["side_rounds"] else []
        where += [f"its twin in {_rounds(entry['twin_rounds'])}"] if entry["twin_rounds"] else []
        notes.append(f"side {entry['side']}'s comparison with its twin on {entry['case']} has a figure of 0 "
                     f"({'; '.join(where)}): side {entry['side']} has no noise floor for {', '.join(entry['cases'])}, "
                     f"and those cases get NO VERDICT (nonpositive_value)")
    tested = sum(1 for row in same_vs_same if row["p_raw"] is not None)
    for side, case_id, most, untied, p in differs:  # what was measured; never a cause
        notes.append(f"side {side} and its twin differed on {case_id} the same way in {most} of {untied} rounds (p = "
                     f"{p:.3g}, not corrected for the {tested} comparisons of a side with its twin)")

    def group_floor(side: str, case: Mapping) -> tuple[stats.Floor | None, str | None]:
        if any((side, cid) in nonpositive for cid in group_of(case)):
            return None, None
        found = [(case_floor.get((side, cid)), cid) for cid in group_of(case)]
        found = [(f, cid) for f, cid in found if f is not None]
        if not found:
            return None, None
        return max(found, key=lambda pair: pair[0].value)

    comparisons = []
    for case in cases:
        cid = case["id"]
        s, b = _by_round(rows, subject, cid), _by_round(rows, baseline, cid)
        paired = _paired(s, b)
        positive = [r for r in paired if s[r] > 0 and b[r] > 0]
        ratios = [s[r] / b[r] for r in positive]
        lower, higher = stats.signs(ratios)
        test = stats.sign_test(lower, higher) if len(positive) > 1 else None
        ratio = stats.median(ratios) if ratios else None
        # A 0 of side A or B gives the case no verdict wherever it is, also in a round the other side and the twin
        # have no figure for: leaving it out would grade the case on the rounds that are left. One note names the
        # rounds the twin note above does not.
        own_zero = {side: [r for r in sorted(values) if values[r] <= 0]
                    for side, values in ((subject, s), (baseline, b))}
        for side, zero in own_zero.items():
            beside_twin = (nonpositive.get((side, cid)) or {}).get("side_rounds") or []
            zero = [r for r in zero if r not in beside_twin]
            if zero:
                notes.append(f"side {side} had a figure of 0 on {cid} in {_rounds(zero)}: {cid} gets NO VERDICT "
                             f"(nonpositive_value)")
        first = sum(1 for r in positive if pos.get((r, subject), 0) < pos.get((r, baseline), 0))
        fs, fs_case = group_floor(subject, case)
        fb, fb_case = group_floor(baseline, case)
        bar = fs.value + fb.value if fs is not None and fb is not None else None
        role = case.get("role", "primary")
        # Rounds one side lost through its own doing while the other did not: they would flatter the side that failed.
        lost = {side: sum(1 for (fs_side, fc, r) in fault_set if fs_side == side and fc == cid
                          and (other, cid, r) not in fault_set)
                for side, other in ((subject, baseline), (baseline, subject))}
        comparisons.append({
            "case": cid, "subject": subject, "baseline": baseline, "role": role, "kind": case["kind"],
            "better": case["better"], "group": case.get("group"),
            "n_rounds": len(paired), "n_unpaired": len(set(s) ^ set(b)), "n_untied": lower + higher,
            "n_lower": lower, "n_higher": higher,
            "subject_median": stats.median(list(s.values())) if s else None,
            "baseline_median": stats.median(list(b.values())) if b else None,
            "ratio": ratio, "gap": abs(ratio - 1.0) if ratio is not None else None,
            "ratios": ratios,
            "floor": {"source": "twins", "subject": _floor_entry(fs, fs_case), "baseline": _floor_entry(fb, fb_case),
                      "mode": "TWO-SIDED"},
            "bar": bar, "p_raw": test.p if test else None, "p_raw_fraction": list(test.fraction) if test else None,
            "p_corrected": None, "order_split": [first, len(positive) - first],
            "lost": lost,
            "load_timeouts": {side: sum(1 for r in paired if timed_out.get((r, side))) for side in (subject, baseline)},
            "_nonpositive": len(positive) != len(paired) or any(own_zero.values()) or any(
                (side, other) in nonpositive for side in (subject, baseline) for other in group_of(case)),
        })

    family = {c["case"]: c["p_raw"] for c in comparisons if c["role"] == "primary" and c["p_raw"] is not None}
    corrected = stats.holm(family)
    for c in comparisons:
        c["p_corrected"] = corrected.get(c["case"])
        c["outcome"], c["reason"] = _decide(c, rule, needed[c["role"]], twins, subject, baseline, len(nodes) > 1)
        c["verdict"] = stats.word(c["outcome"], c["kind"])
        c.pop("_nonpositive")

    outcomes: dict[str, dict[str, int]] = {"primary": {}, "secondary": {}}
    for c in comparisons:
        outcomes[c["role"]][c["verdict"]] = outcomes[c["role"]].get(c["verdict"], 0) + 1
    refusals = []
    if len(nodes) > 1:
        refusals.append({"code": "MORE_THAN_ONE_NODE", "scope": "experiment",
                         "message": f"the measured launches ran on {len(nodes)} nodes"})
    totals = {side: 0 for side in (subject, baseline)}
    for (_, side), n in timed_out.items():
        base = twins.get(side, side)
        if base in totals:
            totals[base] += n
    return {
        "rule": {"name": rule.name, "fields": rule.fields(), "family_size": len(family),
                 "plain": stats.plain_family(len(family)), "min_rounds": needed},
        "comparisons": comparisons,
        "same_vs_same": same_vs_same,
        "nonpositive_twins": list(nonpositive.values()),
        "refusals": refusals,
        "notes": notes,
        "outcomes": outcomes,
        "load_timeouts": totals,
    }


def _decide(c: Mapping, rule: stats.Rule, needed: int, twins: Mapping[str, str], subject: str, baseline: str,
            split: bool) -> tuple[str, str | None]:
    """(outcome, reason) of one comparison: the reasons it cannot be graded first, then the verdict table, then the
    order check."""
    if split:
        return stats.NO_VERDICT, stats.MORE_THAN_ONE_NODE
    if any(c["lost"].values()):
        return stats.NO_VERDICT, stats.SIDE_FAILED
    if c["_nonpositive"]:
        return stats.NO_VERDICT, stats.NONPOSITIVE_VALUE
    if c["n_rounds"] == 1:
        return stats.NO_VERDICT, stats.SINGLE_MEASUREMENT
    if c["n_rounds"] < needed:
        return stats.NO_VERDICT, stats.TOO_FEW_ROUNDS
    fs, fb = c["floor"]["subject"], c["floor"]["baseline"]
    if fs is None and fb is None:
        return stats.NO_VERDICT, stats.NO_FLOOR
    if fs is None or fb is None:
        return stats.NO_VERDICT, stats.ONE_SIDED_FLOOR
    if fs["value"] == 0 or fb["value"] == 0:
        return stats.NO_VERDICT, stats.ZERO_FLOOR
    p = c["p_corrected"] if c["role"] == "primary" else c["p_raw"]
    outcome, reason = stats.grade(c["ratio"], p, c["bar"], c["better"], rule.alpha)
    if outcome in (stats.BETTER, stats.WORSE, stats.TIE) and stats.order_unbalanced(c["order_split"]):
        return stats.UNRESOLVED, stats.ORDER_UNBALANCED
    return outcome, reason


def analyze(results: Mapping) -> dict:
    """The `analysis` section for a results file (complete or partial)."""
    given = inputs(results)
    out = analyse(given["rows"], given["positions"], given["twins"], given["cases"], given["alpha"],
                  given["faults"], given["nodes"], timeouts=given["timeouts"])
    return {"tool": dict(TOOL), **out, "nodes": given["nodes"]}
