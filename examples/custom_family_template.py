"""Template for adding your own generator family and sweeping it.

Run with:  uv run python examples/custom_family_template.py

Demonstrates:
 1. Defining a custom generator (here: a simple 1/k^p geometric blend)
 2. Registering it with the GENERATORS table
 3. Running a single-target sweep against zeta values

Copy this file as a starting point for new family explorations.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import mpmath as mp

from ramanujan.generators import GENERATORS, Generator
from ramanujan.search import sweep


def my_custom_value(a: int, b: int, c: int, *, depth: int) -> mp.mpf:
    """Toy family: S = sum_{k=1..depth} (k^a) / (k^b + c).

    This is intentionally simple. Real new families should target a
    structure that's missing from the existing set (see Gemini's gap
    analysis in the README).
    """

    if a < 0 or a > 4 or b < 1 or b > 6 or c < 0 or c > 100:
        raise ValueError("a in [0,4], b in [1,6], c in [0,100]")

    total = mp.mpf(0)
    for k in range(1, depth + 1):
        denom = mp.mpf(k) ** b + mp.mpf(c)
        if denom == 0:
            return mp.nan
        total += mp.mpf(k) ** a / denom
    return total


GENERATORS["my-toy"] = Generator(
    name="my-toy",
    arity=3,
    evaluate=lambda a, b, c, depth: my_custom_value(a, b, c, depth=depth),
)


def main() -> None:
    # Try (a=0, b=2, c=0): sum 1/k^2 = zeta(2). Sanity check.
    print("Custom-family demo: sum 1/k^2 should rediscover zeta(2)")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        hits = sweep(
            family="my-toy",
            param_ranges=[(0, 0), (2, 2), (0, 0)],
            target="zeta2",
            extra_basis=["pi", "pi2"],
            depth=10000,
            dps=40,
            max_coeff=10,
            out_dir=Path(tmp),
            progress_every=0,
        )

    if hits:
        h = hits[0]
        nz = [(c, l) for c, l in zip(h.coeffs, ("candidate",) + h.basis) if c]
        print(f"PSLQ relation: {nz}")
    else:
        print("No hit (depth too small for 1/k^2 convergence at the chosen dps).")
        print("Try depth=10^6 or use mpmath.nsum acceleration in the generator.")


if __name__ == "__main__":
    main()
