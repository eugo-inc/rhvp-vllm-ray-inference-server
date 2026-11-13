# SPDX-License-Identifier: Apache-2.0
"""rhvp_compare.stats: the exact tests, the corrections, the interval, the floor and the verdict rule.

Reference p-values were computed with SciPy 1.16.3 (scipy.stats.wilcoxon on the log ratios, scipy.stats.binomtest,
scipy.stats.false_discovery_control), in a container; SciPy itself is not needed to run these tests.

Run with: python -m pytest -q tests/test_compare_stats.py
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

from rhvp_compare import stats  # noqa: E402

W1 = [1.10, 0.95, 1.20, 1.05, 1.30, 0.98, 1.15, 1.08]
W2 = [1.02, 1.02, 0.97, 1.05, 1.05, 1.05, 0.99, 1.10, 1.01, 1.03]  # tied magnitudes
W3 = [0.9, 0.8, 0.85, 0.95, 0.7, 0.92]
W4 = [1.1, 0.9, 1.2, 0.8, 1.05, 0.95]


@pytest.mark.parametrize("values, sign_p, wilcoxon_p", [
    (W1, 0.2890625, 0.0546875),
    (W2, 0.109375, 0.046875),
    (W3, 0.03125, 0.03125),
    (W4, 1.0, 0.84375),
])
def test_tests_match_scipy(values, sign_p, wilcoxon_p):
    assert stats.sign_test(values).p == pytest.approx(sign_p, abs=1e-12)
    assert stats.wilcoxon_test(values).p == pytest.approx(wilcoxon_p, abs=1e-12)


def test_sign_test_counts_and_zero_rule():
    assert stats.sign_test([1.1] * 6).p == pytest.approx(0.03125)
    assert stats.sign_test([1.1] * 5 + [0.9]).p == pytest.approx(0.21875)
    assert stats.sign_test([1.1] * 8 + [0.9] * 2).p == pytest.approx(0.109375)
    result = stats.sign_test([1.0, 1.0, 1.2, 0.8])
    assert (result.n, result.above, result.below) == (2, 1, 1)
    assert stats.sign_test([1.0, 1.0]).p == 1.0
    assert stats.sign_test([]).p == 1.0


def test_wilcoxon_drops_zeros_and_is_symmetric():
    assert stats.wilcoxon_test([1.0, 1.0]).p == 1.0
    up = stats.wilcoxon_test(W3)
    down = stats.wilcoxon_test([1 / r for r in W3])
    assert up.p == pytest.approx(down.p)
    assert up.w_plus == 0 and down.w_plus == pytest.approx(21)
    with pytest.raises(ValueError):
        stats.wilcoxon_test([1.1, -1.0])


def test_corrections_match_references():
    p = [0.01, 0.04, 0.03, 0.005, 0.2]
    assert stats.adjust(p, "bh") == pytest.approx([0.025, 0.05, 0.05, 0.025, 0.2])
    assert stats.adjust(p, "holm") == pytest.approx([0.04, 0.09, 0.09, 0.025, 0.2])
    assert stats.adjust(p, "bonferroni") == pytest.approx([0.05, 0.2, 0.15, 0.025, 1.0])
    assert stats.adjust(p, "none") == p
    assert stats.adjust([], "holm") == []
    with pytest.raises(ValueError):
        stats.adjust(p, "sidak")


def test_holm_is_monotone_and_capped():
    p = [0.9, 0.5, 0.6, 0.01]
    out = stats.adjust(p, "holm")
    assert all(0 <= x <= 1 for x in out)
    ordered = [out[i] for i in sorted(range(len(p)), key=lambda i: p[i])]
    assert ordered == sorted(ordered)


def test_median_interval_order_statistics():
    interval = stats.median_interval(list(range(1, 11)), 0.95)
    assert (interval.low, interval.high) == (2, 9)
    assert interval.level == pytest.approx(0.978515625)
    assert stats.median_interval([1.0, 2.0, 3.0], 0.95).low is None  # 3 values cannot reach 95 %
    assert stats.median_interval([5.0] * 6, 0.95).low == 5.0


def test_floor_and_pairs_needed():
    assert stats.floor_p(10, 14, "holm") == pytest.approx(0.02734375)
    assert stats.floor_p(10, 14, "bh") == pytest.approx(2 ** -9)
    assert stats.floor_p(0, 14, "holm") == 1.0
    assert stats.pairs_needed(14, "holm", 0.05) == 10
    assert stats.pairs_needed(14, "bh", 0.05) == 6
    assert stats.pairs_needed(1, "none", 0.05) == 6
    # the floor is what the tests really give with every pair one way
    assert stats.sign_test([1.2] * 6).p == pytest.approx(stats.floor_p(6, 1, "none"))
    assert stats.wilcoxon_test([1.2, 1.3, 1.1, 1.4, 1.5, 1.6]).p == pytest.approx(stats.floor_p(6, 1, "none"))


def test_tie_band_is_symmetric_in_log_space():
    assert stats.inside_band(1.05, 0.05)
    assert stats.inside_band(1 / 1.05, 0.05)
    assert not stats.inside_band(1.0501, 0.05)
    assert not stats.inside_band(0.95, 0.05)  # 0.95 < 1/1.05 = 0.952...
    assert stats.inside_band(1.0, 0.0)


TIGHT = stats.Interval(1.01, 1.04, 0.98)   # inside the 5% band
WIDE = stats.Interval(0.80, 1.30, 0.98)    # reaches outside it
NONE = stats.Interval(None, None, None)    # too few pairs for an interval


@pytest.mark.parametrize("median_ratio, p_adj, better, n, reachable, interval, direction, outcome, reason", [
    (0.80, 0.01, stats.LOWER, 10, True, WIDE, -1, stats.BETTER, stats.REASON_SIGNIFICANT),
    (1.25, 0.01, stats.LOWER, 10, True, WIDE, 1, stats.WORSE, stats.REASON_SIGNIFICANT),
    (1.25, 0.01, stats.HIGHER, 10, True, WIDE, 1, stats.BETTER, stats.REASON_SIGNIFICANT),
    (0.80, 0.01, stats.HIGHER, 10, True, WIDE, -1, stats.WORSE, stats.REASON_SIGNIFICANT),
    (1.03, 0.001, stats.LOWER, 10, True, TIGHT, 1, stats.TIE, stats.REASON_IN_BAND),
    (1.02, 1.0, stats.HIGHER, 10, True, WIDE, 0, stats.INCONCLUSIVE, stats.REASON_WIDE),
    (1.02, 1.0, stats.HIGHER, 3, True, NONE, 0, stats.INCONCLUSIVE, stats.REASON_NO_INTERVAL),
    (0.80, 0.01, stats.LOWER, 50, True, WIDE, 1, stats.INCONCLUSIVE, stats.REASON_DISAGREE),
    (0.80, 0.20, stats.LOWER, 10, True, WIDE, -1, stats.INCONCLUSIVE, stats.REASON_NOT_SIGNIFICANT),
    (0.80, 1.00, stats.LOWER, 4, False, NONE, -1, stats.INCONCLUSIVE, stats.REASON_FLOOR),
    (0.80, 0.01, stats.LOWER, 2, True, NONE, -1, stats.INCONCLUSIVE, stats.REASON_TOO_FEW),
    (None, None, stats.LOWER, 0, False, NONE, 0, stats.INCONCLUSIVE, stats.REASON_TOO_FEW),
])
def test_verdict_rule(median_ratio, p_adj, better, n, reachable, interval, direction, outcome, reason):
    decided = stats.verdict(median_ratio, p_adj, better, 0.05, 0.05, n, 3, reachable, interval, direction)
    assert (decided.outcome, decided.reason) == (outcome, reason)


def test_a_tie_needs_the_interval_inside_the_band():
    """A median inside the band is not enough: three pairs (no interval), or ten whose interval runs from 0.75 to
    1.35, do not show the two images are within 5%."""
    for values in ([0.8, 1.02, 1.3], [0.70, 0.75, 0.8, 0.9, 1.0, 1.04, 1.1, 1.3, 1.35, 1.40]):
        ratio, interval = stats.median_ratio(values), stats.median_interval(values)
        assert stats.inside_band(ratio, 0.05)
        decided = stats.verdict(ratio, 1.0, stats.HIGHER, 0.05, 0.05, len(values), 3, True, interval, 0)
        assert decided.outcome == stats.INCONCLUSIVE
    close = [1.01, 1.02, 0.99, 1.03, 1.0, 1.02, 0.98, 1.01, 1.02, 1.0]
    decided = stats.verdict(stats.median_ratio(close), 1.0, stats.HIGHER, 0.05, 0.05, 10, 3, True,
                            stats.median_interval(close), stats.sign_test(close).direction)
    assert (decided.outcome, decided.reason) == (stats.TIE, stats.REASON_IN_BAND)


def test_the_direction_comes_from_the_test_not_only_the_median():
    """27 pairs where B is 5% faster and 23 where it is 50% slower: the median says faster, the signed-rank test is
    significant because B is mostly slower. That is not FASTER."""
    values = [0.95] * 27 + [1.5] * 23
    wilcoxon = stats.wilcoxon_test(values)
    assert wilcoxon.p < 0.01 and wilcoxon.direction == 1
    assert stats.sign_test(values).direction == -1
    decided = stats.verdict(stats.median_ratio(values), wilcoxon.p * 4, stats.LOWER, 0.05, 0.05, 50, 8, True,
                            stats.median_interval(values), wilcoxon.direction)
    assert (decided.outcome, decided.reason) == (stats.INCONCLUSIVE, stats.REASON_DISAGREE)
    assert stats.wilcoxon_test([1.0, 1.0]).direction == 0


def test_the_median_ratio_does_not_depend_on_which_image_is_a():
    values = [0.9, 0.95, 0.97, 0.99, 1.0, 1.101, 1.11, 1.12, 1.15, 1.2]
    assert stats.median_ratio(values) == pytest.approx(1 / stats.median_ratio([1 / r for r in values]))
    assert stats.median_ratio(values) == pytest.approx(math.sqrt(1.0 * 1.101))
    assert stats.median_ratio([1.3, 1.02, 0.8]) == 1.02
    with pytest.raises(ValueError):
        stats.median_ratio([])


@pytest.mark.parametrize("pairs, correction, expected", [
    ([10, 10, 10, 10], "holm", [4 * 2 ** -9] * 4),
    ([10, 6, 6, 6], "holm", [2 ** -9 * 4, 3 * 2 ** -5, 3 * 2 ** -5, 3 * 2 ** -5]),
    ([10, 10, 10, 7], "holm", [4 * 2 ** -9, 4 * 2 ** -9, 4 * 2 ** -9, 2 ** -6]),
    ([10, 7], "bonferroni", [2 * 2 ** -9, 2 * 2 ** -6]),
    ([10, 7], "none", [2 ** -9, 2 ** -6]),
    ([0, 3], "holm", [1.0, 0.5]),
])
def test_each_metric_has_its_own_floor(pairs, correction, expected):
    assert stats.floors(pairs, correction) == pytest.approx(expected)


def test_holm_floors_are_what_holm_gives_at_the_floors():
    """With every metric at its own smallest p-value, Holm's adjusted p-value of the metric with the most pairs, and
    the floor of the one with the fewest, are what `floors` says."""
    pairs = [10, 6, 6, 6]
    adjusted = stats.adjust([2.0 ** (1 - n) for n in pairs], "holm")
    floors = stats.floors(pairs, "holm")
    assert adjusted[0] == pytest.approx(floors[0])
    assert min(adjusted[1:]) == pytest.approx(floors[1])
    assert floors[1] > 0.05  # three 6-pair metrics cannot reach alpha next to a 10-pair one


def test_verdict_words():
    assert stats.SPEED_WORDS[stats.BETTER] == "FASTER" and stats.SPEED_WORDS[stats.WORSE] == "SLOWER"
    assert stats.EFFICIENCY_WORDS[stats.BETTER] == "MORE_EFFICIENT"
    assert set(stats.SPEED_WORDS) == set(stats.EFFICIENCY_WORDS) == {stats.BETTER, stats.TIE, stats.WORSE,
                                                                      stats.INCONCLUSIVE}


def test_ratios_and_median():
    assert stats.ratios([(2.0, 1.0), (1.0, 3.0)]) == [0.5, 3.0]
    with pytest.raises(ValueError):
        stats.ratios([(0.0, 1.0)])
    with pytest.raises(ValueError):
        stats.ratios([(1.0, math.inf)])
    assert stats.median([3, 1, 2]) == 2
    assert stats.median([4, 1, 2, 3]) == 2.5
    with pytest.raises(ValueError):
        stats.median([])


def test_noise_floor():
    floor = stats.noise([1.02, 0.98, 1.10])
    assert floor.pairs == 3
    assert floor.median_abs_change == pytest.approx(math.exp(abs(math.log(0.98))) - 1)
    assert floor.max_abs_change == pytest.approx(0.10)
    assert stats.noise([]).median_abs_change is None
