"""Multi-candidate PSLQ: search for relations involving two series at once.

Some target identities are only realizable as linear combinations of multiple
parameterized series. For example, several Catalan-G representations couple
a binomial sum with a logarithm:

    G = (3/8) * Σ 1/(C(2k,k)*(2k+1)^2) - (π/8) * log(2 + sqrt(3)).

A single-candidate sweep can't find this. Sweeping pairs of candidates with
PSLQ over {cand_a, cand_b, 1, target, basis...} can.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mpmath as mp

from .constants import basis as basis_values
from .generators import GENERATORS, grid
from .pslq import find_relation


@dataclass(frozen=True)
class PairHit:
    family_a: str
    params_a: tuple[int, ...]
    family_b: str
    params_b: tuple[int, ...]
    target: str
    basis: tuple[str, ...]
    coeffs: tuple[int, ...]
    norm: int
    residual_log10: float
    depth: int
    dps: int

    def to_json_dict(self) -> dict:
        d = asdict(self)
        for key in ("params_a", "params_b", "basis", "coeffs"):
            d[key] = list(getattr(self, key))
        return d


def pair_sweep(
    family_a: str,
    ranges_a: list[tuple[int, int]],
    family_b: str,
    ranges_b: list[tuple[int, int]],
    *,
    target: str,
    extra_basis: list[str] | None = None,
    depth: int = 150,
    dps: int = 70,
    max_coeff: int = 5000,
    out_dir: Path | None = None,
) -> list[PairHit]:
    """Sweep pairs of generator values; PSLQ over {a, b, 1, target, extras}."""

    gen_a = GENERATORS[family_a]
    gen_b = GENERATORS[family_b]
    if len(ranges_a) != gen_a.arity or len(ranges_b) != gen_b.arity:
        raise ValueError("range arity mismatch")

    extras = extra_basis or []
    basis_labels = ["1", target] + [c for c in extras if c not in {"1", target}]
    basis_vals = basis_values(basis_labels, dps + 10)

    out_dir = out_dir or Path("hits-pair")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cache evaluations for family_b so we don't redo them per family_a candidate.
    b_cache: dict[tuple[int, ...], mp.mpf] = {}
    for params_b in grid(ranges_b):
        try:
            v = gen_b.at(params_b, dps + 10, depth=depth)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if mp.isfinite(v):
            b_cache[params_b] = v

    hits: list[PairHit] = []
    for params_a in grid(ranges_a):
        try:
            v_a = gen_a.at(params_a, dps + 10, depth=depth)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if not mp.isfinite(v_a):
            continue

        for params_b, v_b in b_cache.items():
            # Skip identical-parameter self-pairs when families match.
            if family_a == family_b and params_a == params_b:
                continue
            values = [v_a, v_b, *basis_vals]
            labels = ("cand_a", "cand_b", *basis_labels)
            rel = find_relation(list(values), list(labels), dps=dps, max_coeff=max_coeff)
            if rel is None:
                continue
            # Require both candidates and the target involved.
            ca = rel.coeffs[labels.index("cand_a")]
            cb = rel.coeffs[labels.index("cand_b")]
            ct = rel.coeffs[labels.index(target)]
            if ca == 0 or cb == 0 or ct == 0:
                continue

            residual_log10 = (
                float(mp.log10(rel.residual)) if rel.residual > 0 else -float(dps)
            )
            hit = PairHit(
                family_a=family_a,
                params_a=tuple(params_a),
                family_b=family_b,
                params_b=tuple(params_b),
                target=target,
                basis=tuple(labels),
                coeffs=rel.coeffs,
                norm=rel.norm,
                residual_log10=residual_log10,
                depth=depth,
                dps=dps,
            )
            hits.append(hit)
            stem = f"{family_a}_{'_'.join(map(str, params_a))}__{family_b}_{'_'.join(map(str, params_b))}__{target}"
            (out_dir / f"{stem}.json").write_text(
                json.dumps(hit.to_json_dict(), indent=2, sort_keys=True)
            )

    return hits
