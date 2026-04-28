"""Parameter sweep harness: grid over generator params, PSLQ each candidate."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mpmath as mp

from .constants import basis as basis_values
from .generators import GENERATORS, grid
from .pslq import Relation, find_relation
from typing import Iterable


def _find_candidate_relation(
    values: list[mp.mpf],
    labels: list[str],
    *,
    target: str,
    dps: int,
    max_coeff: int,
) -> Relation | None:
    """Find a PSLQ relation that involves both candidate and target.

    PSLQ over a redundant basis can return the shortest relation, which is
    sometimes a known identity between basis elements (e.g. 90*zeta4 - pi^4
    = 0) that excludes the candidate. We work around this by trying multiple
    basis subsets, starting minimal and adding extras one at a time.
    """

    cand_idx = labels.index("candidate")
    one_idx = labels.index("1")
    target_idx = labels.index(target) if target in labels else None
    if target_idx is None:
        return None

    # Pre-filter: if candidate is itself a small rational, the target-coupled
    # relation PSLQ will find is just (rational identity) + (basis identity)
    # glued together. Reject early.
    rat_check = find_relation(
        [values[cand_idx], values[one_idx]],
        ["candidate", "1"],
        dps=dps,
        max_coeff=10**6,
    )
    if rat_check is not None and rat_check.coeffs[0] != 0:
        return None

    extras = [
        i
        for i, lbl in enumerate(labels)
        if i not in {cand_idx, one_idx, target_idx}
    ]

    # Try progressively richer basis subsets, capped at size 2 of extras.
    # Minimal {candidate, 1, target} catches all "candidate = (a/b) * target" identities.
    # Single extras catch identities like "a*candidate + b*target + c*extra = 0".
    # Pairs catch identities involving two extras (rare but real).
    from itertools import combinations
    subset_choices: list[list[int]] = [[]]
    if extras:
        for k in (1, 2):
            for combo in combinations(extras, k):
                subset_choices.append(list(combo))

    # Two-tier scan: cheap small-coefficient pass first across all subsets,
    # then a single full-coefficient pass on the minimal subset.
    cheap_cap = min(max_coeff, 500)

    seen: set[tuple[int, ...]] = set()
    for chosen in subset_choices:
        active = (cand_idx, one_idx, target_idx, *chosen)
        if active in seen:
            continue
        seen.add(active)

        sub_values = [values[i] for i in active]
        sub_labels = [labels[i] for i in active]
        rel = find_relation(sub_values, sub_labels, dps=dps, max_coeff=cheap_cap)
        if rel is None:
            continue
        c_local = sub_labels.index("candidate")
        t_local = sub_labels.index(target)
        if rel.coeffs[c_local] == 0 or rel.coeffs[t_local] == 0:
            continue

        # Spurious-composite detector: if removing the candidate AND the constant
        # '1' from the relation leaves a basis-only sub-relation that is itself
        # ~zero, then PSLQ glued a known basis identity (e.g. 90*zeta4 - pi^4 = 0)
        # to a rational candidate identity. The "discovery" is parasitic.
        one_local = sub_labels.index("1")
        sub_residual = mp.mpf(0)
        for j, c in enumerate(rel.coeffs):
            if j == c_local or j == one_local:
                continue
            sub_residual += c * sub_values[j]
        if mp.fabs(sub_residual) < mp.mpf(10) ** -(dps - 5):
            continue

        full_coeffs = [0] * len(values)
        for c, lbl in zip(rel.coeffs, sub_labels):
            full_coeffs[labels.index(lbl)] = c
        return Relation(
            coeffs=tuple(full_coeffs),
            labels=tuple(labels),
            residual=rel.residual,
            norm=rel.norm,
        )

    # Last-chance: full max_coeff on minimal basis only.
    if cheap_cap < max_coeff:
        active = (cand_idx, one_idx, target_idx)
        sub_values = [values[i] for i in active]
        sub_labels = [labels[i] for i in active]
        rel = find_relation(sub_values, sub_labels, dps=dps, max_coeff=max_coeff)
        if rel is not None:
            c_local = sub_labels.index("candidate")
            t_local = sub_labels.index(target)
            if rel.coeffs[c_local] != 0 and rel.coeffs[t_local] != 0:
                full_coeffs = [0] * len(values)
                for c, lbl in zip(rel.coeffs, sub_labels):
                    full_coeffs[labels.index(lbl)] = c
                return Relation(
                    coeffs=tuple(full_coeffs),
                    labels=tuple(labels),
                    residual=rel.residual,
                    norm=rel.norm,
                )

    return None


@dataclass(frozen=True)
class Hit:
    family: str
    params: tuple[int, ...]
    target: str
    basis: tuple[str, ...]
    coeffs: tuple[int, ...]
    norm: int
    residual_log10: float
    depth: int
    dps: int
    elapsed_seconds: float

    def to_json_dict(self) -> dict:
        d = asdict(self)
        d["params"] = list(self.params)
        d["basis"] = list(self.basis)
        d["coeffs"] = list(self.coeffs)
        return d


def _candidate_basis(target: str, extra: list[str]) -> list[str]:
    return ["1", target] + [c for c in extra if c not in {"1", target}]


def sweep(
    family: str,
    param_ranges: list[tuple[int, int]],
    *,
    target: str,
    extra_basis: list[str] | None = None,
    depth: int = 80,
    dps: int = 60,
    max_coeff: int = 10**6,
    out_dir: Path | None = None,
    progress_every: int = 100,
) -> list[Hit]:
    """Sweep generator parameters; PSLQ each candidate against the basis.

    Writes JSON files for each hit into ``out_dir`` (default ``./hits``).
    """

    if family not in GENERATORS:
        raise ValueError(f"unknown generator family: {family}")
    gen = GENERATORS[family]
    if len(param_ranges) != gen.arity:
        raise ValueError(
            f"{family} expects {gen.arity} param ranges, got {len(param_ranges)}"
        )

    extras = extra_basis or []
    labels = _candidate_basis(target, extras)
    basis_vals = basis_values(labels, dps + 10)

    out_dir = out_dir or Path("hits")
    out_dir.mkdir(parents=True, exist_ok=True)

    hits: list[Hit] = []
    started = time.time()
    explored = 0

    for params in grid(param_ranges):
        explored += 1
        try:
            value = gen.at(params, dps + 10, depth=depth)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if not mp.isfinite(value):
            continue

        values = [value, *basis_vals]
        full_labels = ["candidate", *labels]

        relation = _find_candidate_relation(
            values, full_labels, target=target, dps=dps, max_coeff=max_coeff
        )
        if relation is None:
            continue

        residual_log10 = (
            float(mp.log10(relation.residual)) if relation.residual > 0 else -float(dps)
        )
        hit = Hit(
            family=family,
            params=tuple(params),
            target=target,
            basis=tuple(labels),
            coeffs=relation.coeffs,
            norm=relation.norm,
            residual_log10=residual_log10,
            depth=depth,
            dps=dps,
            elapsed_seconds=time.time() - started,
        )
        hits.append(hit)
        path = out_dir / f"{family}_{target}_{'_'.join(map(str, params))}.json"
        path.write_text(json.dumps(hit.to_json_dict(), indent=2, sort_keys=True))

        if progress_every and len(hits) % max(1, progress_every // 10) == 0:
            print(f"[hit #{len(hits)}] {relation.pretty()}  params={params}")

        if progress_every and explored % progress_every == 0:
            print(f"  ... explored {explored} param tuples, hits so far: {len(hits)}")

    return hits


def relation_for_params(
    family: str,
    params: tuple[int, ...],
    *,
    target: str,
    extra_basis: list[str] | None = None,
    depth: int = 200,
    dps: int = 80,
    max_coeff: int = 10**8,
) -> Relation | None:
    """One-shot PSLQ on a specific generator parameter tuple."""

    gen = GENERATORS[family]
    extras = extra_basis or []
    labels = ["candidate", "1", target] + [c for c in extras if c not in {"1", target}]
    basis_labels = labels[1:]
    basis_vals = basis_values(basis_labels, dps + 10)
    value = gen.at(params, dps + 10, depth=depth)
    return find_relation([value, *basis_vals], labels, dps=dps, max_coeff=max_coeff)
