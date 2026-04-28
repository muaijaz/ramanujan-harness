"""End-to-end pipeline test: prove sweep + PSLQ find a known identity."""

from __future__ import annotations

import tempfile
from pathlib import Path

import mpmath as mp

from ramanujan.generators import GENERATORS, Generator
from ramanujan.search import sweep


def _install_e_series_generator():
    """Register a one-parameter generator whose value is e * p (fast-converging)."""

    def evaluate(p: int, *, depth: int) -> mp.mpf:
        total = mp.mpf(0)
        for k in range(depth + 1):
            total += mp.mpf(1) / mp.factorial(k)
        return total * p

    GENERATORS["e-test"] = Generator(name="e-test", arity=1, evaluate=evaluate)


def test_sweep_recovers_e_series():
    _install_e_series_generator()
    with tempfile.TemporaryDirectory() as tmp:
        hits = sweep(
            family="e-test",
            param_ranges=[(1, 1)],
            target="e",
            depth=40,
            dps=40,
            max_coeff=10,
            out_dir=Path(tmp),
            progress_every=0,
        )
    assert len(hits) >= 1, "expected sweep to pair candidate with e"
    h = hits[0]
    cand_idx = 0
    e_idx = list(h.basis).index("e") + 1
    assert h.coeffs[cand_idx] != 0 and h.coeffs[e_idx] != 0
    ratio = -h.coeffs[e_idx] / h.coeffs[cand_idx]
    assert abs(abs(ratio) - 1) < 1e-9
