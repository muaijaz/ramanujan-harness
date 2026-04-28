"""PSLQ wrapper for finding short integer relations between candidate values."""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp


@dataclass(frozen=True)
class Relation:
    """An integer-relation hit: sum_i coeffs[i] * values[i] ~ 0."""

    coeffs: tuple[int, ...]
    labels: tuple[str, ...]
    residual: mp.mpf
    norm: int

    def pretty(self) -> str:
        terms = [
            f"{c:+d}*{lbl}"
            for c, lbl in zip(self.coeffs, self.labels)
            if c != 0
        ]
        return " ".join(terms) + f"  ~ 0  (residual={mp.nstr(self.residual, 5)})"


def find_relation(
    values: list[mp.mpf],
    labels: list[str],
    *,
    dps: int,
    max_coeff: int = 10**8,
) -> Relation | None:
    """Run PSLQ on the given high-precision values.

    Returns a non-trivial relation if PSLQ finds one with bounded coefficients,
    otherwise None. Filters out the trivial all-zero result and any relation
    whose coefficients exceed ``max_coeff`` (likely numerical noise).
    """

    if len(values) != len(labels):
        raise ValueError("values and labels must have equal length")
    if len(values) < 2:
        return None

    old = mp.mp.dps
    mp.mp.dps = dps
    try:
        try:
            coeffs = mp.pslq(values, tol=mp.mpf(10) ** (-(dps - 5)), maxcoeff=max_coeff)
        except (ValueError, ZeroDivisionError):
            return None
    finally:
        mp.mp.dps = old

    if coeffs is None:
        return None
    coeffs = tuple(int(c) for c in coeffs)
    if all(c == 0 for c in coeffs):
        return None
    if any(abs(c) > max_coeff for c in coeffs):
        return None

    residual = mp.mpf(0)
    for c, v in zip(coeffs, values):
        residual += c * v
    norm = max(abs(c) for c in coeffs)

    return Relation(
        coeffs=coeffs,
        labels=tuple(labels),
        residual=mp.fabs(residual),
        norm=norm,
    )
