# SPDX-License-Identifier: Apache-2.0
"""The statistics of a two-image comparison: pure functions on numbers, standard library only.

Nothing here knows about the results file, a server or a cluster; analyze.py turns the results file into the paired
values these functions take, and their answers back into the file's `analysis` section. That keeps this module a
drop-in: any tool that reads and writes the results file can replace analyze.py and this module together.

The unit of analysis is one round's summary per metric (for example the median time to first token of the round).
Requests inside a round are never pooled across rounds. Rounds run in ABBA order, and each pair is one A round and
the B round next to it, so a slow drift in the machine lands on both sides of every second pair the other way round.

For each metric:

- ratio r_i = B_i / A_i for every pair i with both values (A is the baseline, B the candidate);
- the median ratio, on the log scale (for an even count the geometric mean of the two middle ratios, so swapping A
  and B gives exactly the reciprocal), and a distribution-free confidence interval for it from the order statistics
  of the ratios (the interval that goes with the sign test);
- an exact two-sided sign test on the ratios: under "no difference", r_i > 1 and r_i < 1 are equally likely. Ratios
  exactly equal to 1 carry no sign and are left out (the usual zero rule), so they lower n;
- an exact two-sided Wilcoxon signed-rank test on log r_i, ties given average ranks, the null distribution counted
  exactly over all 2^n sign patterns;
- a multiple-comparison correction across the declared metric family: Holm (default), Bonferroni,
  Benjamini-Hochberg, or none;
- a verdict from a tie band declared before the run (see `verdict`).

The floor. With n pairs, no exact two-sided sign or signed-rank p-value can be smaller than 2^(1-n): all n ratios
on one side. Holm and Bonferroni multiply the smallest p by the family size m, so a FASTER or SLOWER verdict needs
m * 2^(1-n) <= alpha. `floor_p` computes it for a plan (every metric with the same pairs); plan.py warns before the
run when the plan cannot reach alpha. `floors` gives each metric its own floor once failed rounds have left the
metrics with different numbers of pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence

LOWER, HIGHER = "lower", "higher"
TESTS = ("sign", "wilcoxon")
CORRECTIONS = ("holm", "bonferroni", "bh", "none")

#: Machine outcomes, and the words a reader sees for each kind of metric.
BETTER, TIE, WORSE, INCONCLUSIVE = "better", "tie", "worse", "inconclusive"
SPEED_WORDS = {BETTER: "FASTER", TIE: "TIE", WORSE: "SLOWER", INCONCLUSIVE: "INCONCLUSIVE"}
EFFICIENCY_WORDS = {BETTER: "MORE_EFFICIENT", TIE: "TIE", WORSE: "LESS_EFFICIENT", INCONCLUSIVE: "INCONCLUSIVE"}

#: Why a verdict is what it is.
REASON_TOO_FEW = "too_few_pairs"
REASON_IN_BAND = "interval_inside_tie_band"
REASON_NO_INTERVAL = "median_inside_tie_band_but_too_few_pairs_for_an_interval"
REASON_WIDE = "median_inside_tie_band_but_interval_is_not"
REASON_SIGNIFICANT = "significant_outside_tie_band"
REASON_DISAGREE = "test_and_median_disagree"
REASON_NOT_SIGNIFICANT = "not_significant_outside_tie_band"
REASON_FLOOR = "outside_tie_band_but_too_few_pairs_to_reach_alpha"
#: Set by analyze.py over a verdict the rule above gave (never by `verdict`).
REASON_ONE_SIDE = "pairs_lost_to_one_side_failing"
REASON_NODES = "rounds_ran_on_more_than_one_node"
REASONS = (REASON_TOO_FEW, REASON_IN_BAND, REASON_NO_INTERVAL, REASON_WIDE, REASON_SIGNIFICANT, REASON_DISAGREE,
           REASON_NOT_SIGNIFICANT, REASON_FLOOR, REASON_ONE_SIDE, REASON_NODES)


# ---- basic summaries ----

def median(values: Sequence[float]) -> float:
    """The median; the mean of the two middle values for an even count."""
    data = sorted(values)
    if not data:
        raise ValueError("median of no values")
    mid = len(data) // 2
    return data[mid] if len(data) % 2 else (data[mid - 1] + data[mid]) / 2.0


def median_ratio(values: Sequence[float]) -> float:
    """The median of ratios, on the log scale: the middle ratio, or for an even count the geometric mean of the two
    middle ratios. So the median of the A/B ratios is exactly the reciprocal of the median of the B/A ratios, and
    which image is called A cannot move a ratio across the edge of the tie band."""
    data = sorted(values)
    if not data:
        raise ValueError("median of no values")
    mid = len(data) // 2
    return data[mid] if len(data) % 2 else math.sqrt(data[mid - 1] * data[mid])


def ratios(pairs: Sequence[tuple[float, float]]) -> list[float]:
    """B / A for each (A, B) pair. Both values must be positive."""
    out = []
    for a, b in pairs:
        if not (a > 0 and b > 0) or math.isinf(a) or math.isinf(b):
            raise ValueError(f"a ratio needs two positive finite values, got A={a!r} B={b!r}")
        out.append(b / a)
    return out


# ---- the binomial pieces of the sign test ----

def _binomial_cdf(k: int, n: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 1/2), exactly (integers until the last division)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n


@dataclass(frozen=True)
class SignTest:
    n: int        # ratios with a sign (r != 1)
    above: int    # r > 1: B larger than A
    below: int    # r < 1: B smaller than A
    p: float      # exact two-sided p-value; 1.0 when n == 0

    @property
    def direction(self) -> int:
        """+1 when the test leans to B larger than A, -1 to B smaller, 0 for neither."""
        return (self.above > self.below) - (self.above < self.below)


def sign_test(values: Sequence[float]) -> SignTest:
    """Exact two-sided sign test of the ratios against 1. Ratios equal to 1 are left out."""
    above = sum(1 for r in values if r > 1.0)
    below = sum(1 for r in values if r < 1.0)
    n = above + below
    if n == 0:
        return SignTest(0, 0, 0, 1.0)
    p = min(1.0, 2.0 * _binomial_cdf(min(above, below), n))
    return SignTest(n, above, below, p)


# ---- Wilcoxon signed-rank, exact ----

def _average_ranks(magnitudes: Sequence[float]) -> list[float]:
    """Ranks 1..n of the values, ties given the mean of the ranks they share."""
    order = sorted(range(len(magnitudes)), key=lambda i: magnitudes[i])
    ranks = [0.0] * len(magnitudes)
    at = 0
    while at < len(order):
        end = at
        while end + 1 < len(order) and magnitudes[order[end + 1]] == magnitudes[order[at]]:
            end += 1
        shared = (at + 1 + end + 1) / 2.0
        for j in range(at, end + 1):
            ranks[order[j]] = shared
        at = end + 1
    return ranks


@dataclass(frozen=True)
class WilcoxonTest:
    n: int          # log ratios that are not 0
    w_plus: float   # sum of the ranks of the positive log ratios
    p: float        # exact two-sided p-value; 1.0 when n == 0
    direction: int = 0  # +1: W+ above its mean under "no difference" (B mostly larger), -1 below, 0 at it


def wilcoxon_test(values: Sequence[float]) -> WilcoxonTest:
    """Exact two-sided Wilcoxon signed-rank test of log(ratio) against 0.

    Zero differences are dropped first. Tied magnitudes get average ranks; the exact null distribution is counted
    over all 2^n sign patterns with those ranks (doubled, so every rank is an integer), so ties are handled exactly.
    """
    diffs = [math.log(r) for r in values if r > 0 and r != 1.0]
    if any(r <= 0 for r in values):
        raise ValueError("ratios must be positive")
    n = len(diffs)
    if n == 0:
        return WilcoxonTest(0, 0.0, 1.0)
    ranks = _average_ranks([abs(d) for d in diffs])
    doubled = [round(2 * r) for r in ranks]
    w_plus2 = sum(r2 for r2, d in zip(doubled, diffs) if d > 0)
    total = sum(doubled)
    counts = [0] * (total + 1)  # counts[s]: sign patterns whose positive ranks (doubled) sum to s
    counts[0] = 1
    for r2 in doubled:
        for s in range(total, r2 - 1, -1):
            counts[s] += counts[s - r2]
    patterns = 2 ** n
    low = sum(counts[: w_plus2 + 1]) / patterns
    high = sum(counts[w_plus2:]) / patterns
    direction = (2 * w_plus2 > total) - (2 * w_plus2 < total)  # the mean of W+ (doubled) is total / 2
    return WilcoxonTest(n, w_plus2 / 2.0, min(1.0, 2.0 * min(low, high)), direction)


# ---- confidence interval for the median ratio ----

@dataclass(frozen=True)
class Interval:
    low: float | None
    high: float | None
    level: float | None  # the exact coverage of this interval (at least the level asked for), None if none exists


def median_interval(values: Sequence[float], level: float = 0.95) -> Interval:
    """The distribution-free interval for the median from order statistics: (x_(k), x_(n+1-k)) with the largest k
    whose exact binomial coverage 1 - 2 P(X <= k-1) is still at least `level`. None when n is too small."""
    data = sorted(values)
    n = len(data)
    best = None
    for k in range(1, n // 2 + 1):
        coverage = 1.0 - 2.0 * _binomial_cdf(k - 1, n)
        if coverage >= level:
            best = (k, coverage)
        else:
            break
    if best is None:
        return Interval(None, None, None)
    k, coverage = best
    return Interval(data[k - 1], data[n - k], coverage)


# ---- multiple comparisons ----

def adjust(pvalues: Sequence[float], method: str) -> list[float]:
    """Adjusted p-values, in the input order. Holm and Bonferroni control the family-wise error rate,
    Benjamini-Hochberg ("bh") the false discovery rate; "none" returns them unchanged."""
    m = len(pvalues)
    if method not in CORRECTIONS:
        raise ValueError(f"unknown correction {method!r}; one of {', '.join(CORRECTIONS)}")
    if m == 0 or method == "none":
        return list(pvalues)
    if method == "bonferroni":
        return [min(1.0, p * m) for p in pvalues]
    order = sorted(range(m), key=lambda i: pvalues[i])
    out = [0.0] * m
    if method == "holm":
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (m - rank) * pvalues[i]))
            out[i] = running
        return out
    running = 1.0  # bh: step up from the largest p
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, min(1.0, pvalues[i] * m / (rank + 1)))
        out[i] = running
    return out


def _base_floor(pairs: int) -> float:
    """The smallest exact two-sided p-value n pairs allow: 2^(1-n); 1.0 for fewer than one pair."""
    return 1.0 if pairs < 1 else min(1.0, 2.0 ** (1 - pairs))


def floor_p(pairs: int, family: int, correction: str) -> float:
    """The smallest adjusted p-value any metric can reach with `pairs` pairs and `family` metrics: 2^(1-n) for one
    metric, times m for Holm and Bonferroni (the smallest p is multiplied by m), unchanged for "bh" (when every
    metric sits at the floor) and for "none". 1.0 for fewer than one pair."""
    base = _base_floor(pairs)
    if correction in ("holm", "bonferroni"):
        return min(1.0, base * max(family, 1))
    return base


def floors(pairs: Sequence[int], correction: str) -> list[float]:
    """The smallest adjusted p-value each metric of a family can reach, from each metric's own number of pairs.

    - none: the metric's own 2^(1-n).
    - Bonferroni: m times that.
    - Holm, exactly: a metric ranked after k others gets (m-k) times its own p, and never less than the Holm values
      of the metrics ranked before it. So its floor is the smallest, over k, of the larger of (m-k) times its own
      floor and the Holm values of the k metrics with the smallest floors at those floors (only metrics whose floor
      is not above its own can rank before it). With equal pairs this is m * 2^(1-n), as `floor_p` says.
    - Benjamini-Hochberg: a lower bound, the metric's own 2^(1-n) (no adjusted p-value is below its raw p-value).
    """
    if correction not in CORRECTIONS:
        raise ValueError(f"unknown correction {correction!r}; one of {', '.join(CORRECTIONS)}")
    base = [_base_floor(n) for n in pairs]
    m = len(base)
    if correction in ("none", "bh"):
        return base
    if correction == "bonferroni":
        return [min(1.0, f * m) for f in base]
    out = []
    for i, own in enumerate(base):
        ahead = sorted(f for j, f in enumerate(base) if j != i and f <= own)
        best, before = m * own, 0.0
        for k, f in enumerate(ahead, start=1):
            before = max(before, (m - k + 1) * f)  # the Holm value of the metric ranked k-th
            best = min(best, max(before, (m - k) * own))
        out.append(min(1.0, best))
    return out


def pairs_needed(family: int, correction: str, alpha: float) -> int:
    """The fewest pairs with which `floor_p` reaches `alpha`."""
    n = 1
    while floor_p(n, family, correction) > alpha:
        n += 1
        if n > 64:
            raise ValueError("alpha is unreachable")
    return n


# ---- the verdict ----

def inside_band(ratio: float, tie_band: float) -> bool:
    """True when the ratio is within the tie band, symmetric in log space: 1/(1+band) <= ratio <= 1+band."""
    return 1.0 / (1.0 + tie_band) <= ratio <= 1.0 + tie_band


@dataclass(frozen=True)
class Verdict:
    outcome: str   # better / tie / worse / inconclusive (B against A)
    reason: str


def verdict(median_ratio: float | None, p_adjusted: float | None, better: str, tie_band: float, alpha: float,
            n_pairs: int, min_pairs: int, reachable: bool = True, interval: Interval | None = None,
            direction: int | None = None) -> Verdict:
    """The pre-declared rule, in this order:

    1. fewer than `min_pairs` usable pairs: inconclusive (too few pairs);
    2. the median ratio inside the tie band: tie only when the whole confidence interval for the median also lies
       inside the band (whatever the p-value; a difference smaller than the band was declared not to matter before
       the run). Otherwise inconclusive: a median inside the band with an interval that reaches outside it, or with
       too few pairs for an interval at all, does not show the two images are within the band;
    3. outside the band and the adjusted p-value <= alpha: better or worse, by the metric's direction, when the
       test leans the same way as the median (`direction`: +1 for B larger, -1 for B smaller; the sign test and the
       signed-rank test can come out significant on the other side of 1 from the median); otherwise inconclusive;
    4. outside the band and not significant: inconclusive, with the floor named when the plan could never have
       reached alpha (`reachable` False).
    """
    if better not in (LOWER, HIGHER):
        raise ValueError(f"better must be {LOWER!r} or {HIGHER!r}")
    if n_pairs < max(min_pairs, 1) or median_ratio is None:
        return Verdict(INCONCLUSIVE, REASON_TOO_FEW)
    if inside_band(median_ratio, tie_band):
        if interval is None or interval.low is None or interval.high is None:
            return Verdict(INCONCLUSIVE, REASON_NO_INTERVAL)
        if inside_band(interval.low, tie_band) and inside_band(interval.high, tie_band):
            return Verdict(TIE, REASON_IN_BAND)
        return Verdict(INCONCLUSIVE, REASON_WIDE)
    if p_adjusted is not None and p_adjusted <= alpha:
        b_larger = median_ratio > 1.0
        if direction is not None and direction != (1 if b_larger else -1):
            return Verdict(INCONCLUSIVE, REASON_DISAGREE)
        b_is_better = not b_larger if better == LOWER else b_larger
        return Verdict(BETTER if b_is_better else WORSE, REASON_SIGNIFICANT)
    return Verdict(INCONCLUSIVE, REASON_NOT_SIGNIFICANT if reachable else REASON_FLOOR)


# ---- the noise floor from A/A pairs ----

@dataclass(frozen=True)
class Noise:
    pairs: int
    median_abs_change: float | None  # exp(median |log r|) - 1: the typical change between two runs of one image
    max_abs_change: float | None


def noise(values: Sequence[float]) -> Noise:
    """How far apart two rounds of the same image land: the ratios of A/A pairs as a relative change."""
    logs = [abs(math.log(r)) for r in values if r > 0]
    if not logs:
        return Noise(0, None, None)
    return Noise(len(logs), math.exp(median(logs)) - 1.0, math.exp(max(logs)) - 1.0)
