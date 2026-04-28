"""Find Dirichlet L-function values via the harness's Dirichlet family.

Run with:  uv run python examples/find_dirichlet_L_values.py

Sweeps the Dirichlet generator at small p across all available characters
(chi_2, chi_3, chi_4, chi_6) and identifies which targets each L-value
matches. Reproduces:
    L(1, chi_4) = pi/4              (Leibniz, 1674)
    L(2, chi_4) = G                 (Catalan)
    L(1, chi_3) = pi / (3*sqrt(3))
    L(2, chi_3) = (4/(3*sqrt(3))) * Cl_2(pi/3)
"""

from __future__ import annotations

from pathlib import Path
import tempfile

from ramanujan.search import sweep

CHARACTERS = {1: "chi_2 (mod 2)", 2: "chi_3", 3: "chi_4 (Catalan)", 4: "chi_6"}
TARGETS = ["pi", "catalan", "pi_sqrt3", "Cl2_pi3", "log2"]


def main() -> None:
    print("Scanning Dirichlet L-values L(p, chi) for p in {1, 2, 3}, all characters:")
    print()

    for char_id, char_name in CHARACTERS.items():
        for p in (1, 2, 3):
            params = [(char_id, char_id), (p, p), (0, 0)]
            for target in TARGETS:
                with tempfile.TemporaryDirectory() as tmp:
                    hits = sweep(
                        family="dirichlet",
                        param_ranges=params,
                        target=target,
                        extra_basis=["pi", "log2", "sqrt3", "catalan", "Cl2_pi3"],
                        depth=100,
                        dps=70,
                        max_coeff=30,
                        out_dir=Path(tmp),
                        progress_every=0,
                    )
                for h in hits:
                    nz = [(c, l) for c, l in zip(h.coeffs, ("candidate",) + h.basis) if c]
                    print(f"  L({p}, {char_name}) ~  {nz}")
                    break  # one hit per (char, p) is enough


if __name__ == "__main__":
    main()
