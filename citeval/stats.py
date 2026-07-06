"""Statistical rigor for the faithfulness study.

The whole point of the controlled study is to report differences between
configurations *with uncertainty*, not as bare point estimates. This module
provides the three tools that turn a table of per-question scores into
defensible claims:

* ``bootstrap_ci`` — a percentile bootstrap 95% confidence interval for any
  statistic (default: the mean) of a metric across questions.
* ``paired_diff`` — a paired bootstrap for the mean difference between two
  configurations evaluated on the *same* questions, with a two-sided p-value.
  Pairing removes per-question difficulty as a nuisance factor, which is why
  it is far more powerful than comparing two independent CIs.
* ``mcnemar_exact`` — an exact (binomial) McNemar test for paired *binary*
  outcomes, e.g. "was this question fully citation-supported: yes/no" under
  two configs.

Deliberately dependency-free (stdlib ``random``/``math`` only) so it installs
and unit-tests without numpy/torch, and every result is reproducible via an
explicit ``seed``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

Number = float


def mean(xs: Sequence[Number]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


@dataclass(frozen=True)
class CI:
    """A point estimate with a (1 - alpha) confidence interval."""

    point: float
    lo: float
    hi: float
    n: int
    alpha: float = 0.05

    def __str__(self) -> str:
        pct = round((1 - self.alpha) * 100)
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}] ({pct}% CI, n={self.n})"


@dataclass(frozen=True)
class DiffResult:
    """Paired mean difference (a - b) with a bootstrap CI and p-value."""

    diff: float
    lo: float
    hi: float
    p_value: float
    n: int
    alpha: float = 0.05

    @property
    def significant(self) -> bool:
        """True when the CI excludes 0 (equivalently p < alpha)."""
        return self.lo > 0 or self.hi < 0

    def __str__(self) -> str:
        star = " *" if self.significant else ""
        return f"{self.diff:+.3f} [{self.lo:+.3f}, {self.hi:+.3f}], p={self.p_value:.4f}{star}"


@dataclass(frozen=True)
class McNemarResult:
    n01: int  # A=0, B=1 (B wins)
    n10: int  # A=1, B=0 (A wins)
    p_value: float

    @property
    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def _percentile(sorted_xs: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list, q in [0, 1]."""
    if not sorted_xs:
        return 0.0
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    idx = q * (len(sorted_xs) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_xs[lo]
    frac = idx - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def bootstrap_ci(
    xs: Sequence[Number],
    *,
    statistic: Callable[[Sequence[Number]], float] = mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """Percentile-bootstrap CI for ``statistic`` over ``xs``.

    Resamples ``xs`` with replacement ``n_boot`` times, recomputes the
    statistic each time, and takes the alpha/2 and 1-alpha/2 percentiles.
    """
    n = len(xs)
    point = statistic(xs)
    if n <= 1:
        return CI(point=point, lo=point, hi=point, n=n, alpha=alpha)
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(n_boot):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        boot.append(statistic(sample))
    boot.sort()
    return CI(
        point=point,
        lo=_percentile(boot, alpha / 2),
        hi=_percentile(boot, 1 - alpha / 2),
        n=n,
        alpha=alpha,
    )


def paired_diff(
    a: Sequence[Number],
    b: Sequence[Number],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> DiffResult:
    """Paired bootstrap for mean(a - b) where a[i], b[i] are the same question.

    Resamples the *question indices* (keeping each a/b pair together) so the
    pairing is preserved. The two-sided p-value is twice the smaller tail mass
    of the bootstrap distribution of the mean difference around 0.
    """
    if len(a) != len(b):
        raise ValueError(f"paired_diff needs equal-length inputs, got {len(a)} vs {len(b)}")
    n = len(a)
    diffs = [a[i] - b[i] for i in range(n)]
    observed = mean(diffs)
    if n <= 1:
        return DiffResult(observed, observed, observed, 1.0, n, alpha)

    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(n_boot):
        boot.append(mean([diffs[rng.randrange(n)] for _ in range(n)]))
    boot.sort()

    lo = _percentile(boot, alpha / 2)
    hi = _percentile(boot, 1 - alpha / 2)
    # Two-sided bootstrap p-value with the standard +1 smoothing so it is
    # never exactly 0 (you can't prove p=0 from a finite resample).
    ge = sum(1 for x in boot if x >= 0)
    le = sum(1 for x in boot if x <= 0)
    p = min(1.0, 2 * (min(ge, le) + 1) / (n_boot + 1))
    return DiffResult(diff=observed, lo=lo, hi=hi, p_value=p, n=n, alpha=alpha)


def _binom_two_sided_p(n01: int, n10: int) -> float:
    """Exact two-sided binomial p-value for McNemar's test (p=0.5)."""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    # P(X <= k) under Binomial(n, 0.5), doubled for two-sided.
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2 * tail)


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> McNemarResult:
    """Exact McNemar test for paired binary outcomes.

    ``a[i]``, ``b[i]`` are 0/1 outcomes for the same question under two configs.
    Only the discordant pairs (where the two configs disagree) carry signal.
    """
    if len(a) != len(b):
        raise ValueError(f"mcnemar_exact needs equal-length inputs, got {len(a)} vs {len(b)}")
    n01 = sum(1 for i in range(len(a)) if a[i] == 0 and b[i] == 1)
    n10 = sum(1 for i in range(len(a)) if a[i] == 1 and b[i] == 0)
    return McNemarResult(n01=n01, n10=n10, p_value=_binom_two_sided_p(n01, n10))
