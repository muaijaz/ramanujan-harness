"""Symbolic verification of PSLQ hits.

PSLQ returns a numerical relation; we want to either (a) re-derive it in
closed form via sympy, or (b) at minimum re-evaluate it at much higher
precision to rule out aliasing. Discovery is heuristic; verification has
to be defensible.
"""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp

from .constants import basis as basis_values
from .generators import GENERATORS


@dataclass(frozen=True)
class VerificationResult:
    high_precision_residual_log10: float
    survived_dps: int
    note: str


def pretty_relation(
    coeffs: tuple[int, ...],
    labels: tuple[str, ...],
) -> str:
    """Render a PSLQ relation as ``candidate = (a/b) * something + ...``.

    Solves the relation for the candidate term and reports the rational
    coefficient on each basis element, dropping zeros.
    """

    cand_idx = labels.index("candidate")
    a = coeffs[cand_idx]
    if a == 0:
        return "(no candidate coupling)"
    parts = []
    for c, lbl in zip(coeffs, labels):
        if c == 0 or lbl == "candidate":
            continue
        # candidate = -(c/a) * lbl
        from fractions import Fraction

        rational = Fraction(-c, a)
        if rational == 0:
            continue
        parts.append((rational, lbl))
    body = " + ".join(f"({r})*{l}" for r, l in parts) if parts else "0"
    return f"candidate = {body}"


def reverify_at_higher_precision(
    family: str,
    params: tuple[int, ...],
    coeffs: tuple[int, ...],
    labels: tuple[str, ...],
    *,
    depth: int,
    survived_dps: int = 200,
) -> VerificationResult:
    """Re-evaluate the candidate and basis at much higher precision.

    Aliasing produces relations that vanish at one precision and re-emerge
    elsewhere. A real identity stays close to zero as dps grows. We use
    survived_dps to test that the residual scales like 10^(-dps).
    """

    gen = GENERATORS[family]
    value = gen.at(params, survived_dps + 20, depth=depth)
    basis_vals = basis_values(list(labels[1:]), survived_dps + 20)
    full = [value, *basis_vals]

    old = mp.mp.dps
    mp.mp.dps = survived_dps + 20
    try:
        residual = mp.mpf(0)
        for c, v in zip(coeffs, full):
            residual += c * v
        residual = mp.fabs(residual)
        if residual == 0:
            log10 = -float(survived_dps + 20)
        else:
            log10 = float(mp.log10(residual))
    finally:
        mp.mp.dps = old

    if log10 < -(survived_dps - 5):
        note = "robust: residual scales with precision"
    elif log10 < -(survived_dps // 2):
        note = "weak: residual non-trivially small but not at full precision"
    else:
        note = "noise: relation likely a precision artifact"

    return VerificationResult(
        high_precision_residual_log10=log10,
        survived_dps=survived_dps,
        note=note,
    )
