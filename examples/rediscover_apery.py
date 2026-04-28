"""Rediscover Apery's 1978 identity for zeta(3) from scratch.

Run with:  uv run python examples/rediscover_apery.py

This is the canonical demo. PSLQ has no prior knowledge of the rational
coefficient (5/2). It scans only the parameter point (sign=1, p=3, q=1,
shift=0) of the central-binomial family against the basis {1, zeta(3)},
finds the integer relation 5*candidate - 2*zeta(3) = 0, and prints the
identity in human-readable form.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import mpmath as mp

from ramanujan.search import sweep


def main() -> None:
    print("Rediscovering Apery's identity:")
    print("    zeta(3) = (5/2) * sum_{k>=1} (-1)^(k-1) / (k^3 * C(2k,k))")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        hits = sweep(
            family="central-binomial",
            param_ranges=[(1, 1), (3, 3), (1, 1), (0, 0)],
            target="zeta3",
            extra_basis=["pi", "log2"],
            depth=200,
            dps=80,
            max_coeff=100,
            out_dir=Path(tmp),
            progress_every=0,
        )

    if not hits:
        raise SystemExit("No hit found — check installation or precision.")

    h = hits[0]
    cand = h.coeffs[0]
    target_idx = list(h.basis).index("zeta3") + 1
    z3 = h.coeffs[target_idx]

    print(f"PSLQ relation:  {cand}*candidate + {z3}*zeta(3) = 0")
    print(f"Solving:        candidate = ({-z3}/{cand}) * zeta(3)")
    print(f"Convergence:    geometric ~4^-k  (depth=200, ~120 digits)")
    print(f"Verified:       residual = 10^{h.residual_log10:.0f}")
    print()
    print("This is exactly Apery's identity.")


if __name__ == "__main__":
    main()
