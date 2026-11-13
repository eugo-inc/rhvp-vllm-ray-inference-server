# SPDX-License-Identifier: Apache-2.0
"""rhvp_compare.stats and analyze.analyse: the sign test, Holm's correction, the fewest rounds, the run order, the
noise floor from the twins and the verdict table, each against answers written here as constants.

Run with: python -m pytest -q tests/test_compare_stats.py
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

from rhvp_compare import analyze, plan as plans, stats  # noqa: E402


# ---- the sign test ----

@pytest.mark.parametrize("lower, higher, p", [
    (12, 0, 2 / 4096), (10, 2, 158 / 4096), (6, 0, 0.03125), (5, 1, 0.21875), (0, 2, 0.5), (0, 0, 1.0),
    (5, 0, 0.0625),
])
def test_sign_test_known_answers(lower, higher, p):
    assert stats.sign_test(lower, higher).p == p


def test_sign_test_fraction_and_symmetry():
    assert stats.sign_test(12, 0).fraction == (2, 4096)
    assert stats.sign_test(10, 2).fraction == (158, 4096)
    assert stats.sign_test(1, 1).fraction == (4, 4)  # capped at 1
    assert round(stats.sign_test(18, 6).p, 4) == 0.0227
    assert stats.sign_test(3, 9).p == stats.sign_test(9, 3).p
    # No p below 0.05 with fewer than 6 untied pairs.
    assert min(stats.sign_test(n, 0).p for n in range(1, 6)) >= 0.05


def test_ties_are_not_trials():
    # Three lower and three exact ties: 3 against 0, not 6 against 0.
    lower, higher = stats.signs([0.9, 0.8, 0.95, 1.0, 1.0, 1.0])
    assert (lower, higher) == (3, 0) and stats.sign_test(lower, higher).p == 0.25
    assert stats.sign_test(*stats.signs([1.0] * 9)).p == 1.0


# ---- Holm ----

@pytest.mark.parametrize("raw, adjusted", [
    ([0.01, 0.04, 0.5], [0.03, 0.08, 0.5]),
    ([0.01, 0.012, 0.5], [0.03, 0.03, 0.5]),
    ([0.01, 0.04, 0.03], [0.03, 0.06, 0.06]),
    ([0.03125, 0.03125, 0.21875], [0.09375, 0.09375, 0.21875]),
])
def test_holm_known_answers(raw, adjusted):
    got = stats.holm({f"c{i}": p for i, p in enumerate(raw)})
    assert [got[f"c{i}"] for i in range(len(raw))] == pytest.approx(adjusted)


def test_holm_wording():
    assert stats.plain_family(1) == "one case tested"
    assert stats.plain_family(4) == "corrected for testing 4 cases at once"


# ---- the fewest rounds ----

@pytest.mark.parametrize("m, n", [(1, 6), (2, 7), (4, 8), (10, 9), (35, 11), (100, 12)])
def test_min_rounds_known_answers(m, n):
    assert stats.min_rounds(m) == n


def test_eight_rounds():
    assert stats.max_cases(8) == 6  # at most 6 primary cases can get a verdict at 8 rounds
    assert stats.must_agree(8, 4) == 8  # and with 4 primaries all 8 rounds must agree
    assert stats.must_agree(16, 4) == 14  # 16 rounds: a verdict survives two disagreeing rounds
    assert stats.must_agree(7, 4) is None


# ---- the run order ----

def test_rotate_reverse_known_answers():
    two = [plans.round_order(["A", "B"], r) for r in range(1, 5)]
    assert ["".join(o) for o in two] == ["AB", "BA", "BA", "AB"]
    three = [plans.round_order(["A", "B", "C"], r) for r in range(1, 7)]
    assert ["".join(o) for o in three] == ["ABC", "BCA", "CAB", "CBA", "ACB", "BAC"]


def test_balance_with_both_twins():
    entries = plans.lineup("both")
    assert [e["key"] for e in entries] == ["A", "A~same", "B", "B~same"]
    for rounds in (8, 16, 24):
        assert plans.balance(entries, rounds)["balanced"]
    for rounds, behind in ((12, (7, 5)), (20, (11, 9))):
        got = plans.balance(entries, rounds)
        assert not got["balanced"]
        assert got["pairs"]["B vs A"]["first_ran_first"] * 2 == rounds  # subject and baseline even
        twin = got["pairs"]["A vs A~same"]
        assert (twin["first_ran_first"], twin["second_ran_first"]) == behind
    assert plans.balance(entries, 8)["step"] == 8


def test_launches_of_the_default_plan():
    entries = plans.lineup("both")
    launches = plans.launch_order(entries, 8, True)
    assert len(launches) == 34  # 2 warm-ups and 8 rounds of 4: 34 server starts
    assert [(x["side"], x["kind"]) for x in launches[:2]] == [("A", "warmup"), ("B", "warmup")]
    assert [x["side"] for x in launches[2:6]] == ["A", "A~same", "B", "B~same"]
    assert [x["side"] for x in launches[6:10]] == ["A~same", "B", "B~same", "A"]
    assert [x["side"] for x in launches[18:22]] == ["B~same", "B", "A~same", "A"]  # round 5, reversed
    assert {x["of"] for x in launches if x["side"] == "B~same"} == {"B"}
    assert [x["key"] for x in plans.lineup("A")] == ["A", "A~same", "B"]
    assert [x["key"] for x in plans.lineup("none")] == ["A", "B"]


# ---- the verdict table ----

@pytest.mark.parametrize("ratio, p, better, outcome, reason", [
    (0.90, 0.01, "lower", "better", None),       # M < 1 - bar, significant, lower is better: FASTER
    (0.90, 0.01, "higher", "worse", None),       # the same for a rate: WORSE
    (1.10, 0.01, "lower", "worse", None),        # SLOWER
    (1.10, 0.01, "higher", "better", None),      # BETTER
    (1.01, 0.30, "lower", "tie", None),          # inside the bar and not significant
    (1.01, 0.01, "lower", "unresolved", "inside_noise"),
    (1.10, 0.30, "lower", "unresolved", "not_significant"),
    (1.10, 0.05, "lower", "unresolved", "not_significant"),  # p equal to alpha is not significant
])
def test_grade(ratio, p, better, outcome, reason):
    assert stats.grade(ratio, p, 0.05, better, 0.05) == (outcome, reason)


def test_exactly_on_the_bar_is_inside():
    assert stats.grade(1.25, 0.30, 0.25, "lower", 0.05) == ("tie", None)
    assert stats.grade(0.75, 0.01, 0.25, "lower", 0.05) == ("unresolved", "inside_noise")


def test_words():
    assert stats.word("better", "time") == "FASTER" and stats.word("worse", "time") == "SLOWER"
    assert stats.word("better", "rate") == "BETTER" and stats.word("worse", "efficiency") == "WORSE"
    assert [stats.word(o, "time") for o in ("tie", "unresolved", "no_verdict")] == ["TIE", "UNRESOLVED", "NO VERDICT"]


def test_median_and_floor():
    assert stats.median([3.0, 1.0, 2.0]) == 2.0 and stats.median([1.0, 2.0, 4.0, 3.0]) == 2.5
    floor = stats.floor({1: 1.01, 2: 0.99, 3: 1.02, 4: 1.03, 5: 0.98, 6: 1.01, 7: 1.0, 8: 1.5})
    assert floor.value == pytest.approx(0.01) and floor.n == 8 and not floor.provisional
    assert floor.outlier_rounds == (8,)  # named, never dropped
    assert stats.floor({1: 1.0, 2: 1.0}).provisional
    assert stats.floor({}) is None


# ---- analyse: rows in, verdicts out ----

CASES = [{"id": "t", "better": "lower", "kind": "time", "role": "primary", "group": "primary:t"}]


def _rows(per_round: dict[str, list[float]], case: str = "t") -> tuple[list[dict], list[dict]]:
    """Rows and positions for rounds 1..n of the given sides, in the default rotate-reverse order."""
    keys = [k for k in ("A", "A~same", "B", "B~same") if k in per_round]
    rows, positions = [], []
    for i in range(len(next(iter(per_round.values())))):
        order = plans.round_order(keys, i + 1)
        for key in keys:
            positions.append({"round": i + 1, "side": key, "pos": order.index(key) + 1})
            rows.append({"round": i + 1, "pos": order.index(key) + 1, "side": key, "case": case,
                         "value": per_round[key][i]})
    return rows, positions


TWINS = {"A~same": "A", "B~same": "B"}
SECONDARY = [{**CASES[0], "role": "secondary"}]


def _noisy(a: float, b: float, n: int = 8, sd: float = 0.02, seed: int = 7, keys=("A", "A~same", "B", "B~same")):
    """Every launch its own noise: A and A~same around a, B and B~same around b."""
    rng = random.Random(seed)
    return {k: [(a if k.startswith("A") else b) * rng.gauss(1.0, sd) for _ in range(n)] for k in keys}


def _one(per_round, twins=TWINS, cases=CASES, **kw) -> dict:
    rows, positions = _rows(per_round)
    return analyze.analyse(rows, positions, twins, cases, **kw)["comparisons"][0]


def test_a_known_difference_reads_its_ratio_and_word():
    got = _one(_noisy(10.0, 5.0), cases=SECONDARY)
    assert got["ratio"] == pytest.approx(0.5, rel=0.03) and got["verdict"] == "FASTER" and got["outcome"] == "better"
    assert got["n_lower"] == 8 and got["p_raw"] == 2 / 256 and got["bar"] > 0


def test_identical_sides_tie_and_noise_free_sides_have_no_floor():
    got = _one(_noisy(10.0, 10.0), cases=SECONDARY)
    assert (got["outcome"], got["verdict"]) == ("tie", "TIE")
    flat = _one({"A": [10.0] * 8, "A~same": [10.0] * 8, "B": [10.0] * 8, "B~same": [10.0] * 8}, cases=SECONDARY)
    assert (flat["outcome"], flat["reason"]) == ("no_verdict", "zero_floor")


def test_swapping_the_sides_flips_the_word():
    base = _noisy(10.0, 12.0)
    swapped = {"A": base["B"], "A~same": base["B~same"], "B": base["A"], "B~same": base["A~same"]}
    one, two = _one(base, cases=SECONDARY), _one(swapped, cases=SECONDARY)
    assert one["verdict"] == "SLOWER" and two["verdict"] == "FASTER"
    assert one["ratio"] == pytest.approx(1 / two["ratio"], rel=0.02)


def test_a_missing_twin_gives_no_verdict():
    one = _one(_noisy(10.0, 5.0, keys=("A", "A~same", "B")), twins={"A~same": "A"})
    assert (one["outcome"], one["reason"]) == ("no_verdict", "one_sided_floor")
    none = _one(_noisy(10.0, 5.0, keys=("A", "B")), twins={})
    assert (none["outcome"], none["reason"]) == ("no_verdict", "no_floor")


def test_eighteen_of_twentyfour_alone_and_among_four():
    # 18 rounds lower and 6 higher: p = 0.0227 alone (a verdict), above alpha once corrected for four cases.
    signs = [0.8] * 18 + [1.2] * 6
    noise = [1.0 + 0.002 * (i % 4) for i in range(24)]
    base = {"A": [10.0] * 24, "A~same": [10.0 * x for x in noise], "B": [10.0 * s for s in signs],
            "B~same": [10.0 * s * x for s, x in zip(signs, noise)]}
    alone = _one(base)
    assert alone["verdict"] == "FASTER" and alone["p_corrected"] == pytest.approx(0.0227, abs=1e-4)
    rows, positions = _rows(base)
    cases = [{"id": c, "better": "lower", "kind": "time", "role": "primary", "group": f"primary:{c}"}
             for c in ("t", "u", "v", "w")]
    for c in ("u", "v", "w"):  # three more cases with no difference at all
        rows += [{**r, "case": c} for r in _rows(_noisy(10.0, 10.0, n=24, seed=ord(c)), c)[0]]
    got = {c["case"]: c for c in analyze.analyse(rows, positions, TWINS, cases)["comparisons"]}
    assert (got["t"]["outcome"], got["t"]["reason"]) == ("unresolved", "not_significant")
    assert got["t"]["p_corrected"] > 0.05


def test_significant_inside_the_bar_is_unresolved_never_tie():
    noisy = _noisy(10.0, 10.0, sd=0.03)
    per_round = {"A": [10.0] * 8, "A~same": noisy["A~same"], "B": [10.0 * 0.995] * 8, "B~same": noisy["B~same"]}
    got = _one(per_round, cases=SECONDARY)  # all eight lower, by 0.5 %, against a bar of several percent
    assert got["bar"] > 0.005 and (got["outcome"], got["reason"]) == ("unresolved", "inside_noise")


def test_too_few_rounds_and_one_measurement():
    got = _one(_noisy(10.0, 5.0, n=5))
    assert (got["outcome"], got["reason"]) == ("no_verdict", "too_few_rounds")  # 1 primary case needs 6 rounds
    one = _one(_noisy(10.0, 5.0, n=1))
    assert (one["outcome"], one["reason"]) == ("no_verdict", "single_measurement") and one["p_raw"] is None


def test_a_zero_figure_and_a_failed_side():
    base = _noisy(10.0, 5.0)
    zero = _one({**base, "B": [0.0] + base["B"][1:]})
    assert (zero["outcome"], zero["reason"]) == ("no_verdict", "nonpositive_value")
    rows, positions = _rows(base)
    rows = [r for r in rows if not (r["side"] == "B" and r["round"] == 3)]
    got = analyze.analyse(rows, positions, TWINS, CASES, faults=[{"round": 3, "side": "B", "case": "t"}])
    row = got["comparisons"][0]
    assert (row["outcome"], row["reason"]) == ("no_verdict", "side_failed") and row["lost"] == {"B": 1, "A": 0}
    split = analyze.analyse(*_rows(base), TWINS, CASES, nodes=["n1", "n2"])
    assert split["comparisons"][0]["reason"] == "more_than_one_node" and split["refusals"]


def test_an_order_that_never_changes_is_unresolved():
    rows, _ = _rows(_noisy(10.0, 5.0))
    fixed = {"A": 1, "A~same": 2, "B": 3, "B~same": 4}
    positions = [{"round": r["round"], "side": r["side"], "pos": fixed[r["side"]]} for r in rows]
    got = analyze.analyse(rows, positions, TWINS, SECONDARY)["comparisons"][0]
    assert (got["outcome"], got["reason"]) == ("unresolved", "order_unbalanced") and got["order_split"] == [0, 8]


def test_the_floor_is_the_largest_over_its_group_and_a_twin_that_differs_is_loud():
    cases = [{"id": i, "better": "lower", "kind": "time", "role": "secondary", "group": "secondary:t"}
             for i in ("c1/t", "c8/t")]
    rows, positions = [], []
    for case, wide in (("c1/t", 1.01), ("c8/t", 1.04)):
        r, p = _rows({"A": [10.0] * 8, "A~same": [10.0 / wide] * 8, "B": [10.0] * 8, "B~same": [10.0] * 8}, case)
        rows += r
        positions = p
    got = analyze.analyse(rows, positions, TWINS, cases)
    floors = {c["case"]: c["floor"]["baseline"]["value"] for c in got["comparisons"]}
    assert floors["c1/t"] == pytest.approx(0.04) and floors["c8/t"] == pytest.approx(0.04)
    assert any("differs from its twin" in note for note in got["notes"])
