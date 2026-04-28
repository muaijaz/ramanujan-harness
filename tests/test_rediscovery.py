"""Lock in: PSLQ rediscovers Apery and folklore zeta identities from scratch."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ramanujan.search import sweep


def _scan_for(target: str, params: tuple[int, ...]):
    with tempfile.TemporaryDirectory() as tmp:
        hits = sweep(
            family="central-binomial",
            param_ranges=[(p, p) for p in params],
            target=target,
            extra_basis=["zeta2", "zeta3", "zeta4", "pi", "pi2", "pi4"],
            depth=200,
            dps=80,
            max_coeff=1000,
            out_dir=Path(tmp),
            progress_every=0,
        )
    return hits


def test_rediscover_zeta2_folklore():
    # zeta(2) = 3 * sum 1/(k^2 * C(2k,k)) -> S(0,2,1,0)
    hits = _scan_for("zeta2", (0, 2, 1, 0))
    assert len(hits) == 1
    h = hits[0]
    cand_idx = 0
    target_idx = list(h.basis).index("zeta2") + 1
    # Expect 3*candidate - 1*zeta2 = 0
    ratio = -h.coeffs[target_idx] / h.coeffs[cand_idx]
    assert abs(abs(ratio) - 1 / 3) < 1e-12


def test_rediscover_apery_zeta3():
    # Apery: zeta(3) = (5/2) * sum (-1)^(k-1)/(k^3 * C(2k,k)) -> S(1,3,1,0)
    hits = _scan_for("zeta3", (1, 3, 1, 0))
    assert len(hits) == 1
    h = hits[0]
    cand_idx = 0
    target_idx = list(h.basis).index("zeta3") + 1
    # Expect 5*candidate - 2*zeta3 = 0  (or its negation)
    ratio = -h.coeffs[target_idx] / h.coeffs[cand_idx]
    assert abs(abs(ratio) - 2 / 5) < 1e-12


def test_rediscover_catalan_classical():
    # Catalan: 8G = 3*sum_{k>=0} 1/(C(2k,k)*(2k+1)^2) + pi*log(2+sqrt(3))
    # Our candidate sums from k=1, so candidate = sum_{k>=0} - 1.
    # Family: central-binomial-power (sign=0, p=0, q=1, r=2, s=0)
    with tempfile.TemporaryDirectory() as tmp:
        hits = sweep(
            family="central-binomial-power",
            param_ranges=[(0, 0), (0, 0), (1, 1), (2, 2), (0, 0)],
            target="catalan",
            extra_basis=["pi", "pi_log2sqrt3", "log2sqrt3"],
            depth=200,
            dps=80,
            max_coeff=100,
            out_dir=Path(tmp),
            progress_every=0,
        )
    assert len(hits) == 1
    h = hits[0]
    nonzero = {l: c for c, l in zip(h.coeffs, ("candidate",) + h.basis) if c}
    # Expect candidate, catalan, pi_log2sqrt3 nonzero, with the right ratios.
    assert "candidate" in nonzero
    assert "catalan" in nonzero
    assert "pi_log2sqrt3" in nonzero
    one_c = nonzero.get("1", 0)
    a = nonzero["candidate"]
    g = nonzero["catalan"]
    L = nonzero["pi_log2sqrt3"]
    # Relation: a*candidate + one_c + g*catalan + L*pi_log2sqrt3 = 0
    # candidate = sum_{k>=1} = sum_{k>=0} - 1, so:
    #   8*G = 3*sum_{k>=0} + pi*log(2+sqrt(3))
    #   8*G = 3*candidate + 3 + pi*log(2+sqrt(3))
    #   3*candidate + 3 + pi*log(2+sqrt(3)) - 8*G = 0
    # Normalize so coefficient on candidate is +3:
    from fractions import Fraction
    scale = Fraction(3, a)
    one_norm = scale * one_c
    cat_norm = scale * g
    log_norm = scale * L
    assert one_norm == 3
    assert cat_norm == -8
    assert log_norm == 1


def test_rediscover_zeta4_folklore():
    # zeta(4) = (36/17) * sum 1/(k^4 * C(2k,k)) -> S(0,4,1,0)
    hits = _scan_for("zeta4", (0, 4, 1, 0))
    assert len(hits) == 1
    h = hits[0]
    cand_idx = 0
    target_idx = list(h.basis).index("zeta4") + 1
    ratio = -h.coeffs[target_idx] / h.coeffs[cand_idx]
    assert abs(abs(ratio) - 17 / 36) < 1e-12


def test_rediscover_lehmer_golden_ratio():
    # Σ (-1)^(k-1)/(k²·C(2k,k)) = 2·log²(φ)
    # via Maclaurin of arcsin²(x) at x = i/2  →  arcsin²(i/2) = -log²(φ)
    with tempfile.TemporaryDirectory() as tmp:
        hits = sweep(
            family="central-binomial",
            param_ranges=[(1, 1), (2, 2), (1, 1), (0, 0)],
            target="log_phi_sq",
            extra_basis=["phi", "log_phi", "sqrt5", "pi", "log2"],
            depth=200,
            dps=70,
            max_coeff=200,
            out_dir=Path(tmp),
            progress_every=0,
        )
    assert len(hits) == 1
    h = hits[0]
    cand_idx = 0
    target_idx = list(h.basis).index("log_phi_sq") + 1
    ratio = -h.coeffs[target_idx] / h.coeffs[cand_idx]
    assert abs(abs(ratio) - 2) < 1e-12


def test_rediscover_catalan_via_dirichlet_chi4():
    # G = L(2, chi_4) = sum_{k>=1} chi_4(k)/k^2
    with tempfile.TemporaryDirectory() as tmp:
        hits = sweep(
            family="dirichlet",
            param_ranges=[(3, 3), (2, 2), (0, 0)],
            target="catalan",
            extra_basis=["pi", "log2"],
            depth=100, dps=70, max_coeff=10,
            out_dir=Path(tmp), progress_every=0,
        )
    assert len(hits) == 1
    h = hits[0]
    cand_idx = 0
    target_idx = list(h.basis).index("catalan") + 1
    ratio = -h.coeffs[target_idx] / h.coeffs[cand_idx]
    assert abs(abs(ratio) - 1) < 1e-12


def test_rediscover_leibniz_pi_over_4():
    # pi = 4 * L(1, chi_4) = 4 * (1 - 1/3 + 1/5 - ...)
    with tempfile.TemporaryDirectory() as tmp:
        hits = sweep(
            family="dirichlet",
            param_ranges=[(3, 3), (1, 1), (0, 0)],
            target="pi",
            extra_basis=["log2", "catalan"],
            depth=100, dps=70, max_coeff=10,
            out_dir=Path(tmp), progress_every=0,
        )
    assert len(hits) == 1
    h = hits[0]
    cand_idx = 0
    target_idx = list(h.basis).index("pi") + 1
    ratio = -h.coeffs[target_idx] / h.coeffs[cand_idx]
    assert abs(abs(ratio) - 1 / 4) < 1e-12


def test_rediscover_euler_sum_zeta3():
    # Euler 1775: sum H_k / k^2 = 2 * zeta(3)
    with tempfile.TemporaryDirectory() as tmp:
        hits = sweep(
            family="harmonic-weighted",
            param_ranges=[(0, 0), (2, 2), (0, 0), (1, 1)],
            target="zeta3",
            extra_basis=["pi", "log2", "zeta2"],
            depth=100, dps=70, max_coeff=10,
            out_dir=Path(tmp), progress_every=0,
        )
    assert len(hits) == 1
    h = hits[0]
    cand_idx = 0
    target_idx = list(h.basis).index("zeta3") + 1
    ratio = -h.coeffs[target_idx] / h.coeffs[cand_idx]
    assert abs(abs(ratio) - 2) < 1e-12


def test_rediscover_euler_sum_zeta4():
    # Euler: sum H_k / k^3 = (5/4) * zeta(4)
    with tempfile.TemporaryDirectory() as tmp:
        hits = sweep(
            family="harmonic-weighted",
            param_ranges=[(0, 0), (3, 3), (0, 0), (1, 1)],
            target="zeta4",
            extra_basis=["pi", "log2", "zeta2", "zeta3"],
            depth=100, dps=70, max_coeff=10,
            out_dir=Path(tmp), progress_every=0,
        )
    assert len(hits) == 1
    h = hits[0]
    cand_idx = 0
    target_idx = list(h.basis).index("zeta4") + 1
    ratio = -h.coeffs[target_idx] / h.coeffs[cand_idx]
    assert abs(abs(ratio) - 5 / 4) < 1e-12


def test_rediscover_bbp_pi():
    # Bailey-Borwein-Plouffe (1995):
    # pi = sum_{k>=0} 1/16^k (4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6))
    with tempfile.TemporaryDirectory() as tmp:
        hits = sweep(
            family="bbp-multirational",
            param_ranges=[(16,16),(8,8),(4,4),(0,0),(0,0),(-2,-2),(-1,-1),(-1,-1),(0,0),(0,0)],
            target="pi",
            extra_basis=["log2", "catalan"],
            depth=80, dps=60, max_coeff=10,
            out_dir=Path(tmp), progress_every=0,
        )
    assert len(hits) == 1
    h = hits[0]
    cand_idx = 0
    target_idx = list(h.basis).index("pi") + 1
    ratio = -h.coeffs[target_idx] / h.coeffs[cand_idx]
    assert abs(abs(ratio) - 1) < 1e-12


def test_rediscover_ramanujan_sato_sqrt5():
    # Generating function: sum C(2k,k) x^k = 1/sqrt(1-4x). At x=1/5: sum C(2k,k)/5^k = sqrt(5).
    with tempfile.TemporaryDirectory() as tmp:
        hits = sweep(
            family="ramanujan-sato",
            param_ranges=[(2,2),(1,1),(1,1),(0,0),(5,5)],
            target="sqrt5",
            extra_basis=["pi", "log2"],
            depth=800, dps=60, max_coeff=10,
            out_dir=Path(tmp), progress_every=0,
        )
    assert len(hits) == 1
    h = hits[0]
    cand_idx = 0
    target_idx = list(h.basis).index("sqrt5") + 1
    ratio = -h.coeffs[target_idx] / h.coeffs[cand_idx]
    assert abs(abs(ratio) - 1) < 1e-12


def test_rediscover_harmonic_central_binomial_zeta4():
    # Σ H_k^(2) / (k^2 · C(2k,k)) = (14/27) · zeta(4)
    # Found via harness sweep; verified to 100 digits. Likely in
    # Borwein-Bradley 1997 catalog of Apery-like Euler sums but not
    # checked against the literature.
    with tempfile.TemporaryDirectory() as tmp:
        hits = sweep(
            family="harmonic-weighted",
            param_ranges=[(0, 0), (2, 2), (1, 1), (2, 2)],
            target="zeta4",
            extra_basis=["pi", "log2", "zeta2", "zeta3"],
            depth=200, dps=80, max_coeff=100,
            out_dir=Path(tmp), progress_every=0,
        )
    assert len(hits) == 1
    h = hits[0]
    cand_idx = 0
    target_idx = list(h.basis).index("zeta4") + 1
    ratio = -h.coeffs[target_idx] / h.coeffs[cand_idx]
    assert abs(abs(ratio) - 14 / 27) < 1e-12


def test_rediscover_log2_geometric():
    # log 2 = sum_{k=1} 1/(k * 2^k)
    # Family: power-weighted, params (sign=0, p=1, q=0, base_p2=2, pow_p2=1)
    with tempfile.TemporaryDirectory() as tmp:
        hits = sweep(
            family="power-weighted",
            param_ranges=[(0, 0), (1, 1), (0, 0), (2, 2), (1, 1)],
            target="log2",
            extra_basis=["pi", "log3"],
            depth=200,
            dps=60,
            max_coeff=100,
            out_dir=Path(tmp),
            progress_every=0,
        )
    assert len(hits) == 1
    h = hits[0]
    cand_idx = 0
    target_idx = list(h.basis).index("log2") + 1
    ratio = -h.coeffs[target_idx] / h.coeffs[cand_idx]
    assert abs(abs(ratio) - 1) < 1e-12
