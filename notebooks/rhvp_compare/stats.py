# SPDX-License-Identifier: Apache-2.0
"""The statistics of an image comparison: pure functions on numbers, standard library only.

Nothing here knows about the results file, a server or a cluster. analyze.py turns the results file into the paired
values these functions take, and their answers back into the file's `analysis` section, so this module and
analyze.py can be replaced together by any tool that reads and writes the same fields.

The unit of analysis is one launch's figure per case (for example the median time to first token of one launch at
one load level). Requests are never pooled across launches. A round runs every side once, and a pair is the
subject's and the baseline's figure from the same round.

The rule ("strict"), with M the median of the per-round ratios subject / baseline:

- the test is the exact two-sided sign test over the rounds where the two figures differ; a round where both are
  exactly equal is not a trial, and one paired round gives no p-value at all;
- the p-values of the primary cases are corrected with Holm's method, over the primary cases only (the comparisons
  of a side with its twin, and secondary cases, are outside that family);
- the noise is measured, never assumed: each side also runs against an identical copy of itself (its twin) in the
  same rounds. A side's noise floor on a group of cases is the largest abs(median(side / twin) - 1) over the
  group, and the bar is the subject's floor plus the baseline's;
- outside the bar and significant: the subject is FASTER or SLOWER (time-like cases), BETTER or WORSE (rates and
  tokens per joule). Inside the bar and not significant: TIE. Inside the bar but significant, or outside but not
  significant: UNRESOLVED. A comparison that cannot be graded (too few rounds, a missing or zero floor, a value
  that is not positive) has NO VERDICT. A win, a loss and a tie need the same evidence.

Programs read `outcome` and `reason`; the display word is for people.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

LOWER, HIGHER = "lower", "higher"
#: Kinds of case: time-like cases use FASTER / SLOWER, the others BETTER / WORSE.
TIME, RATE, EFFICIENCY = "time", "rate", "efficiency"
KINDS = (TIME, RATE, EFFICIENCY)

#: What programs read.
BETTER, WORSE, TIE, UNRESOLVED, NO_VERDICT = "better", "worse", "tie", "unresolved", "no_verdict"
OUTCOMES = (BETTER, WORSE, TIE, UNRESOLVED, NO_VERDICT)

#: Why a comparison is UNRESOLVED.
INSIDE_NOISE = "inside_noise"          # inside the bar, yet the sign test is significant
NOT_SIGNIFICANT = "not_significant"    # outside the bar, and the sign test is not significant
ORDER_UNBALANCED = "order_unbalanced"  # one side ran first more than three times as often as the other
#: Why a comparison has NO VERDICT.
SINGLE_MEASUREMENT = "single_measurement"  # one paired round: one measurement, no p-value
TOO_FEW_ROUNDS = "too_few_rounds"          # fewer paired rounds than any verdict needs (min_rounds)
NO_FLOOR = "no_floor"                      # neither side has a twin (or a twin that gave figures)
ONE_SIDED_FLOOR = "one_sided_floor"        # only one of the two sides has a floor
ZERO_FLOOR = "zero_floor"                  # a floor of exactly zero on either side
NONPOSITIVE_VALUE = "nonpositive_value"    # a paired figure of 0 or less (for example no output tokens)
ORDER_NOT_RECORDED = "order_not_recorded"  # the order the sides ran in is unknown (never in this pattern)
#: Two reasons only this pattern gives. A failed launch does not stop a run here, and a run whose launches landed on
#: more than one node is still analysed, so these say why such a comparison is not graded.
SIDE_FAILED = "side_failed"                # a launch of the subject or the baseline failed through its own doing
MORE_THAN_ONE_NODE = "more_than_one_node"  # the measured launches ran on more than one node

UNRESOLVED_REASONS = (INSIDE_NOISE, NOT_SIGNIFICANT, ORDER_UNBALANCED)
NO_VERDICT_REASONS = (SINGLE_MEASUREMENT, TOO_FEW_ROUNDS, NO_FLOOR, ONE_SIDED_FLOOR, ZERO_FLOOR, NONPOSITIVE_VALUE,
                      ORDER_NOT_RECORDED, SIDE_FAILED, MORE_THAN_ONE_NODE)
REASONS = UNRESOLVED_REASONS + NO_VERDICT_REASONS

#: A floor from fewer rounds than this is marked provisional.
PROVISIONAL_BELOW = 8


def word(outcome: str, kind: str) -> str:
    """The display word for an outcome: FASTER / SLOWER for time-like cases, BETTER / WORSE for the others."""
    if outcome == BETTER:
        return "FASTER" if kind == TIME else "BETTER"
    if outcome == WORSE:
        return "SLOWER" if kind == TIME else "WORSE"
    return {TIE: "TIE", UNRESOLVED: "UNRESOLVED", NO_VERDICT: "NO VERDICT"}[outcome]


# ---- the rule ----

@dataclass(frozen=True)
class Rule:
    """The rule's every setting, recorded in the results. Only the settings of "strict" are implemented; alpha may
    be changed."""
    name: str = "strict"
    scale: str = "ratio"
    floor_scope: str = "group"
    bar: str = "sum_of_both"
    bar_round_digits: int | None = None
    require_both_floors: bool = True
    p_source: str = "corrected"
    alpha: float = 0.05
    p_inclusive: bool = False
    gap_inclusive: bool = False
    tie: str = "inside_and_not_significant"
    min_rounds: int | None = None

    def fields(self) -> dict:
        return asdict(self)


def strict(alpha: float = 0.05) -> Rule:
    return Rule(alpha=alpha)


# ---- basic summaries ----

def median(values: Sequence[float]) -> float:
    """The median; with an even count, the arithmetic mean of the two middle values."""
    data = sorted(values)
    if not data:
        raise ValueError("median of no values")
    mid = len(data) // 2
    return data[mid] if len(data) % 2 else (data[mid - 1] + data[mid]) / 2.0


def quantile(values: Sequence[float], q: float) -> float:
    """The q-quantile by linear interpolation between closest ranks."""
    data = sorted(values)
    if not data:
        raise ValueError("quantile of no values")
    at = (len(data) - 1) * q
    low = math.floor(at)
    high = min(low + 1, len(data) - 1)
    return data[low] + (data[high] - data[low]) * (at - low)


# ---- the exact sign test ----

@dataclass(frozen=True)
class SignTest:
    n_lower: int              # pairs with subject / baseline below 1
    n_higher: int             # pairs with subject / baseline above 1
    p: float                  # exact two-sided p-value; 1.0 with no untied pair
    fraction: tuple[int, int]  # p as an exact fraction

    @property
    def n(self) -> int:
        return self.n_lower + self.n_higher


def sign_test(n_lower: int, n_higher: int) -> SignTest:
    """k = min(n_lower, n_higher); p = min(1, 2 * sum(C(n, i) for i <= k) / 2^n). Integers until the one division."""
    n = n_lower + n_higher
    if n == 0:
        return SignTest(0, 0, 1.0, (1, 1))
    tail = sum(math.comb(n, i) for i in range(min(n_lower, n_higher) + 1))
    whole = 1 << n
    top = min(2 * tail, whole)
    return SignTest(n_lower, n_higher, top / whole, (top, whole))


def signs(ratios: Sequence[float]) -> tuple[int, int]:
    """(ratios below 1, ratios above 1); a ratio of exactly 1 is a tie, not a trial."""
    return sum(1 for r in ratios if r < 1.0), sum(1 for r in ratios if r > 1.0)


# ---- correction for testing several cases ----

def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm's adjusted p-values by key: sorted ascending (ties by key), adj[i] = max(adj[i-1], min(1, (m-i) p[i]))."""
    order = sorted(pvalues, key=lambda key: (pvalues[key], key))
    m = len(order)
    out, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        out[key] = running
    return out


def plain_family(m: int) -> str:
    if m == 0:
        return "no case tested yet"
    return "one case tested" if m == 1 else f"corrected for testing {m} cases at once"


def min_rounds(m: int, alpha: float = 0.05) -> int:
    """The fewest rounds with which any of m cases can get a verdict: the smallest n with m * 2^(1-n) < alpha."""
    m = max(int(m), 1)
    n = 1
    while m * 2.0 ** (1 - n) >= alpha:
        n += 1
    return n


def max_cases(n: int, alpha: float = 0.05) -> int:
    """The most primary cases that n rounds can still give a verdict: the largest m with m * 2^(1-n) < alpha."""
    if n < 1:
        return 0
    m = math.ceil(alpha / 2.0 ** (1 - n)) - 1
    return max(m, 0)


def must_agree(n: int, m: int, alpha: float = 0.05) -> int | None:
    """How many of n rounds must lean the same way for a verdict on one of m cases (all n untied), or None when
    none can: the smallest a with m * p(sign test, n - a against a) < alpha."""
    for agree in range((n + 1) // 2, n + 1):
        if max(m, 1) * sign_test(n - agree, agree).p < alpha:
            return agree
    return None


# ---- noise floor ----

@dataclass(frozen=True)
class Floor:
    value: float          # abs(median(side / twin) - 1)
    n: int                # rounds with both figures
    ratio: float          # median(side / twin)
    spread: float | None  # interquartile range of side / twin over its median
    outlier_rounds: tuple[int, ...]  # rounds outside the Tukey fences; named, never dropped
    provisional: bool     # from fewer than PROVISIONAL_BELOW rounds


def floor(by_round: Mapping[int, float]) -> Floor | None:
    """The noise floor from the per-round ratios side / twin (round number: ratio); None without a round."""
    if not by_round:
        return None
    values = list(by_round.values())
    mid = median(values)
    spread = outliers = None
    if len(values) >= 2:
        q1, q3 = quantile(values, 0.25), quantile(values, 0.75)
        iqr = q3 - q1
        spread = iqr / mid if mid else None
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = tuple(sorted(r for r, v in by_round.items() if v < low or v > high))
    return Floor(abs(mid - 1.0), len(values), mid, spread, outliers or (), len(values) < PROVISIONAL_BELOW)


# ---- the verdict table ----

def grade(ratio: float, p: float, bar: float, better: str, alpha: float) -> tuple[str, str | None]:
    """(outcome, reason) from the median ratio, the p-value the rule uses and the bar, for a comparison that can be
    graded. `better` is the direction that is better for the case (lower or higher)."""
    if better not in (LOWER, HIGHER):
        raise ValueError(f"better must be {LOWER!r} or {HIGHER!r}")
    significant = p < alpha
    inside = abs(ratio - 1.0) <= bar
    if inside:
        return (UNRESOLVED, INSIDE_NOISE) if significant else (TIE, None)
    if not significant:
        return UNRESOLVED, NOT_SIGNIFICANT
    subject_lower = ratio < 1.0
    return (BETTER if subject_lower == (better == LOWER) else WORSE), None


def order_unbalanced(split: Sequence[int]) -> bool:
    """True when, over a comparison's pairs, one side ran first more than three times as often as the other."""
    first, second = split
    return max(first, second) > 3 * min(first, second)
