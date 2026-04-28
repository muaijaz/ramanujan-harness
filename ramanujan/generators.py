"""Parameterized generators that produce candidate constants from integer parameters.

Each generator is a callable that takes parameters and a precision (dps) and
returns an mpmath value. Generators are deliberately simple so the parameter
sweep stays interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import mpmath as mp


@dataclass(frozen=True)
class Generator:
    """One named generator family parameterized by an integer tuple."""

    name: str
    arity: int
    evaluate: Callable[..., mp.mpf]

    def at(self, params: tuple[int, ...], dps: int, depth: int) -> mp.mpf:
        if len(params) != self.arity:
            raise ValueError(f"{self.name} expects {self.arity} params, got {len(params)}")
        old = mp.mp.dps
        mp.mp.dps = dps + 10
        try:
            return mp.mpf(self.evaluate(*params, depth=depth))
        finally:
            mp.mp.dps = old


def polynomial_cf_value(
    a0: int,
    a1: int,
    a2: int,
    a3: int,
    b0: int,
    b1: int,
    b2: int,
    b3: int,
    head: int,
    *,
    depth: int,
) -> mp.mpf:
    """Continued fraction with cubic-in-n numerator/denominator coefficients.

    Tail value of:
        head + a(1) / ( b(1) + a(2) / ( b(2) + ... ) )
    where a(n) = a0 + a1*n + a2*n^2 + a3*n^3
    and   b(n) = b0 + b1*n + b2*n^2 + b3*n^3.

    The separate ``head`` parameter lets us model CFs whose 0-th term
    differs from the polynomial pattern (e.g. Brouncker: head=1, b(n>=1)=2).
    Backward recurrence for numerical stability.
    """

    def a(n: int) -> mp.mpf:
        n2 = n * n
        return mp.mpf(a0 + a1 * n + a2 * n2 + a3 * n2 * n)

    def b(n: int) -> mp.mpf:
        n2 = n * n
        return mp.mpf(b0 + b1 * n + b2 * n2 + b3 * n2 * n)

    x = b(depth)
    for n in range(depth, 0, -1):
        if x == 0:
            return mp.inf
        x = b(n - 1) + a(n) / x if n - 1 >= 1 else mp.mpf(head) + a(n) / x
    return x


def apery_like_value(
    p_a: tuple[int, ...],
    p_b: tuple[int, ...],
    *,
    depth: int,
) -> mp.mpf:
    """Generic Apery-like ratio.

    Recurrence:
        u_{n+1} = ( P(n) * u_n + Q(n) * u_{n-1} ) / R(n)
    with P, Q, R polynomial in n. Run two seeded recurrences (a-series and
    b-series) and return a_n / b_n at the truncation depth.

    p_a, p_b each encode (P, Q, R) as a flat tuple of nine integers
    (three per polynomial, constant + linear + quadratic).
    """

    if len(p_a) != 9 or len(p_b) != 9:
        raise ValueError("apery-like generator needs 9 ints per series")

    def poly3(coefs: tuple[int, ...], n: int) -> mp.mpf:
        c0, c1, c2 = coefs
        return mp.mpf(c0 + c1 * n + c2 * n * n)

    Pa, Qa, Ra = (p_a[0:3], p_a[3:6], p_a[6:9])
    Pb, Qb, Rb = (p_b[0:3], p_b[3:6], p_b[6:9])

    a_prev, a_curr = mp.mpf(0), mp.mpf(1)
    b_prev, b_curr = mp.mpf(1), mp.mpf(1)

    for n in range(depth):
        Ra_n = poly3(Ra, n)
        Rb_n = poly3(Rb, n)
        if Ra_n == 0 or Rb_n == 0:
            return mp.nan
        a_next = (poly3(Pa, n) * a_curr + poly3(Qa, n) * a_prev) / Ra_n
        b_next = (poly3(Pb, n) * b_curr + poly3(Qb, n) * b_prev) / Rb_n
        a_prev, a_curr = a_curr, a_next
        b_prev, b_curr = b_curr, b_next

    if b_curr == 0:
        return mp.nan
    return a_curr / b_curr


def central_binomial_value(
    sign: int,
    p: int,
    q: int,
    shift: int,
    *,
    depth: int,
) -> mp.mpf:
    """Apery-shaped central-binomial series.

    S = sum_{k=1..depth} (-1)^(sign*k) / ( (k+shift)^p * C(2k, k)^q )

    Known fast-converging cases:
        zeta(2) = 3 *  S(0, 2, 1, 0)
        zeta(3) = (5/2) * S(1, 3, 1, 0)            [Apery 1978]
        zeta(4) = (36/17) * S(0, 4, 1, 0)
    The harness sweeps (sign, p, q, shift) over small integers and lets PSLQ
    discover the rational multiplier against a target.
    """

    if p < 1 or q < 0 or q > 4 or shift < 0 or shift > 4:
        raise ValueError("central-binomial: p>=1, q in [0,4], shift in [0,4]")
    if sign not in (0, 1):
        raise ValueError("sign must be 0 (no alternation) or 1 (alternating)")

    total = mp.mpf(0)
    binom = mp.mpf(1)  # C(0,0)
    for k in range(1, depth + 1):
        # update C(2k, k) from C(2(k-1), k-1) using the standard recurrence
        # C(2k, k) = C(2(k-1), k-1) * (2k-1) * 2 / k
        binom = binom * (2 * k - 1) * 2 / k
        denom = mp.mpf(k + shift) ** p
        if q > 0:
            denom *= binom**q
        if denom == 0:
            return mp.nan
        term = mp.mpf(1) / denom
        if sign == 1:
            term *= mp.mpf((-1) ** (k - 1))
        total += term
    return total


def central_binomial_power_value(
    sign: int,
    p: int,
    q: int,
    r: int,
    s: int,
    *,
    depth: int,
) -> mp.mpf:
    """Generalized central-binomial series.

    S = sum_{k=1..depth} (-1)^(sign*k) / ( k^p * C(2k, k)^q * (2k+1)^r * 4^(s*k) )

    Includes Comtet's zeta(4) variant, BBP-style 1/16^k weights (s=2),
    and Lehmer's families. Wide enough that the Apery and folklore zeta(2),
    zeta(3), zeta(4) identities are special cases.
    """

    if p < 0 or p > 8 or q < 0 or q > 4 or r < 0 or r > 4 or s < 0 or s > 3:
        raise ValueError("central-binomial-power: p in [0,8], q in [0,4], r in [0,4], s in [0,3]")
    if sign not in (0, 1):
        raise ValueError("sign must be 0 or 1")

    total = mp.mpf(0)
    binom = mp.mpf(1)  # C(0, 0)
    for k in range(1, depth + 1):
        binom = binom * (2 * k - 1) * 2 / k  # C(2k, k)
        denom = mp.mpf(1)
        if p > 0:
            denom *= mp.mpf(k) ** p
        if q > 0:
            denom *= binom**q
        if r > 0:
            denom *= mp.mpf(2 * k + 1) ** r
        if s > 0:
            denom *= mp.mpf(4) ** (s * k)
        if denom == 0:
            return mp.nan
        term = mp.mpf(1) / denom
        if sign == 1:
            term *= mp.mpf((-1) ** (k - 1))
        total += term
    return total


def modulated_binomial_value(
    sign: int,
    p: int,
    q: int,
    alpha: int,
    beta: int,
    *,
    depth: int,
) -> mp.mpf:
    """Apery-Schmidt-style modulated binomial series.

    S = sum_{k=1..depth} (-1)^(sign*k) * (k + alpha) / ( (k + beta) * k^p * C(2k,k)^q )

    Captures shapes like Σ (k+1)/(k^4 * C(2k,k)) etc., which are not in the
    pure central-binomial family but appear in Apery-Schmidt's accelerated
    representations of zeta values.
    """

    if p < 0 or p > 6 or q < 0 or q > 4 or alpha < -3 or alpha > 6 or beta < 0 or beta > 6:
        raise ValueError("modulated-binomial: out of range")
    if sign not in (0, 1):
        raise ValueError("sign must be 0 or 1")

    total = mp.mpf(0)
    binom = mp.mpf(1)
    for k in range(1, depth + 1):
        binom = binom * (2 * k - 1) * 2 / k
        denom = mp.mpf(k + beta)
        if p > 0:
            denom *= mp.mpf(k) ** p
        if q > 0:
            denom *= binom**q
        if denom == 0:
            return mp.nan
        term = mp.mpf(k + alpha) / denom
        if sign == 1:
            term *= mp.mpf((-1) ** (k - 1))
        total += term
    return total


def power_weighted_value(
    sign: int,
    p: int,
    q: int,
    base_p2: int,
    pow_p2: int,
    *,
    depth: int,
) -> mp.mpf:
    """Power-weighted central-binomial series.

    S = sum_{k=1..depth} (-1)^(sign*k) / ( k^p * C(2k,k)^q * base_p2^(pow_p2 * k) )

    Captures BBP/Borwein-style geometric weights:
        log 2 = sum_{k>=1} 1/(k * 2^k)         params: (0, 1, 0, 2, 1)
        pi = sum_{k>=0} (1/16^k) * BBP_term     not directly captured here
        log((1+x)/(1-x))/2 = sum x^(2k+1)/(2k+1)
    """

    if base_p2 < 2 or base_p2 > 16:
        raise ValueError("base_p2 in [2,16]")
    if pow_p2 < 0 or pow_p2 > 4:
        raise ValueError("pow_p2 in [0,4]")
    if p < 0 or p > 6 or q < 0 or q > 3:
        raise ValueError("p in [0,6], q in [0,3]")
    if sign not in (0, 1):
        raise ValueError("sign in {0,1}")

    total = mp.mpf(0)
    binom = mp.mpf(1)
    for k in range(1, depth + 1):
        binom = binom * (2 * k - 1) * 2 / k
        denom = mp.mpf(1)
        if p > 0:
            denom *= mp.mpf(k) ** p
        if q > 0:
            denom *= binom**q
        if pow_p2 > 0:
            denom *= mp.mpf(base_p2) ** (pow_p2 * k)
        if denom == 0:
            return mp.nan
        term = mp.mpf(1) / denom
        if sign == 1:
            term *= mp.mpf((-1) ** (k - 1))
        total += term
    return total


def hypergeometric_sum_value(
    alpha: int,
    beta: int,
    gamma: int,
    *,
    depth: int,
) -> mp.mpf:
    """Generic hypergeometric-like sum.

    Sum_{k=0..depth} (alpha + k)! / ((beta + k)!^2 (gamma + k)!) * (-1)^k
    Provides a flexible family that includes Apery's series for zeta(3) when
    (alpha, beta, gamma) = (0, 0, 0) up to a normalization.
    """

    if min(alpha, beta, gamma) < 0 or max(alpha, beta, gamma) > 6:
        raise ValueError("hypergeometric params bounded to [0, 6] for safety")

    total = mp.mpf(0)
    for k in range(depth + 1):
        num = mp.factorial(alpha + k)
        den = mp.factorial(beta + k) ** 2 * mp.factorial(gamma + k)
        total += ((-1) ** k) * num / den
    return total


GENERATORS: dict[str, Generator] = {
    "polynomial-cf": Generator(
        name="polynomial-cf",
        arity=9,
        evaluate=lambda a0, a1, a2, a3, b0, b1, b2, b3, head, depth: polynomial_cf_value(
            a0, a1, a2, a3, b0, b1, b2, b3, head, depth=depth
        ),
    ),
    "apery-like": Generator(
        name="apery-like",
        arity=18,
        evaluate=lambda *args, depth: apery_like_value(
            tuple(args[0:9]), tuple(args[9:18]), depth=depth
        ),
    ),
    "hypergeometric": Generator(
        name="hypergeometric",
        arity=3,
        evaluate=lambda a, b, c, depth: hypergeometric_sum_value(a, b, c, depth=depth),
    ),
    "central-binomial": Generator(
        name="central-binomial",
        arity=4,
        evaluate=lambda sign, p, q, shift, depth: central_binomial_value(
            sign, p, q, shift, depth=depth
        ),
    ),
    "central-binomial-power": Generator(
        name="central-binomial-power",
        arity=5,
        evaluate=lambda sign, p, q, r, s, depth: central_binomial_power_value(
            sign, p, q, r, s, depth=depth
        ),
    ),
    "modulated-binomial": Generator(
        name="modulated-binomial",
        arity=5,
        evaluate=lambda sign, p, q, alpha, beta, depth: modulated_binomial_value(
            sign, p, q, alpha, beta, depth=depth
        ),
    ),
    "power-weighted": Generator(
        name="power-weighted",
        arity=5,
        evaluate=lambda sign, p, q, base_p2, pow_p2, depth: power_weighted_value(
            sign, p, q, base_p2, pow_p2, depth=depth
        ),
    ),
}


def grid(ranges: Iterable[tuple[int, int]]) -> Iterable[tuple[int, ...]]:
    """Cartesian product of integer ranges (inclusive)."""

    ranges = list(ranges)
    if not ranges:
        yield ()
        return
    head, *rest = ranges
    for value in range(head[0], head[1] + 1):
        for tail in grid(rest):
            yield (value, *tail)
