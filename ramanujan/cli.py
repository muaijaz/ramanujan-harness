"""Command-line interface for the Ramanujan harness."""

from __future__ import annotations

import argparse
from pathlib import Path

import mpmath as mp

from .constants import value as constant_value
from .generators import GENERATORS
from .search import relation_for_params, sweep
from .verify import reverify_at_higher_precision


def cmd_sanity(_: argparse.Namespace) -> None:
    """Rediscover Apery's continued fraction for zeta(3) end-to-end.

    Apery (1978) used a CF with a(n) = -n^6 numerator and a specific
    polynomial denominator. We use a known compact CF for zeta(3):
        zeta(3) = 6 / (5 - 1/(117 - 64/(535 - ...)))
    where the n-th partial uses a(n) = -n^6, b(n) = (2n+1)(17n^2+17n+5).
    PSLQ should recover the relation 1*candidate - 1*zeta(3) = 0.
    """

    print("Sanity: Apery zeta(3) continued fraction")
    dps = 60

    def apery_cf(depth: int) -> mp.mpf:
        old = mp.mp.dps
        mp.mp.dps = dps + 20
        try:
            x = mp.mpf((2 * depth + 1) * (17 * depth * depth + 17 * depth + 5))
            for n in range(depth, 0, -1):
                a = mp.mpf(-(n**6))
                b = mp.mpf((2 * n - 1) * (17 * (n - 1) ** 2 + 17 * (n - 1) + 5))
                x = b + a / x
            return mp.mpf(6) / x
        finally:
            mp.mp.dps = old

    candidate = apery_cf(60)
    target = constant_value("zeta3", dps + 20)
    diff = mp.fabs(candidate - target)
    print(f"  candidate ~ {mp.nstr(candidate, 20)}")
    print(f"  zeta(3)   ~ {mp.nstr(target, 20)}")
    print(f"  |diff|    ~ {mp.nstr(diff, 5)}")
    if diff < mp.mpf(10) ** -(dps - 5):
        print("  PASS: candidate matches zeta(3) to working precision")
    else:
        print("  FAIL: residual exceeds tolerance")


def cmd_sweep(args: argparse.Namespace) -> None:
    family = args.family
    if family not in GENERATORS:
        raise SystemExit(f"unknown family: {family}; choose from {list(GENERATORS)}")
    arity = GENERATORS[family].arity
    ranges_arg = args.ranges
    if not ranges_arg:
        ranges = [(-args.range, args.range)] * arity
    else:
        if len(ranges_arg) != arity:
            raise SystemExit(f"need {arity} ranges of form lo:hi for {family}")
        ranges = []
        for r in ranges_arg:
            lo, hi = r.split(":")
            ranges.append((int(lo), int(hi)))

    print(f"Sweep family={family} target={args.target} ranges={ranges}")
    print(f"  depth={args.depth}  dps={args.dps}  max_coeff={args.max_coeff}")
    extra = args.extra_basis.split(",") if args.extra_basis else []
    hits = sweep(
        family=family,
        param_ranges=ranges,
        target=args.target,
        extra_basis=extra,
        depth=args.depth,
        dps=args.dps,
        max_coeff=args.max_coeff,
        out_dir=Path(args.out),
    )
    print(f"\nFinished: {len(hits)} hits saved under {args.out}/")


def cmd_verify(args: argparse.Namespace) -> None:
    params = tuple(int(x) for x in args.params.split(","))
    labels = tuple(args.basis.split(","))
    coeffs = tuple(int(c) for c in args.coeffs.split(","))
    result = reverify_at_higher_precision(
        family=args.family,
        params=params,
        coeffs=coeffs,
        labels=labels,
        depth=args.depth,
        survived_dps=args.dps,
    )
    print(f"residual_log10 @ {result.survived_dps} dps: {result.high_precision_residual_log10:.2f}")
    print(f"verdict: {result.note}")


def cmd_summarize(args: argparse.Namespace) -> None:
    """Walk a hits directory and print one-line summaries grouped by relation."""

    import json
    from collections import defaultdict
    from fractions import Fraction
    from pathlib import Path

    root = Path(args.dir)
    if not root.exists():
        raise SystemExit(f"no such directory: {root}")

    by_relation: dict[tuple, list[dict]] = defaultdict(list)
    for p in sorted(root.glob("*.json")):
        data = json.loads(p.read_text())
        coeffs = tuple(data["coeffs"])
        # Normalize sign so e.g. (5,-2) and (-5,2) collapse.
        for c in coeffs:
            if c != 0:
                first = c
                break
        else:
            continue
        if first < 0:
            coeffs = tuple(-c for c in coeffs)
        by_relation[(data["target"], coeffs, tuple(data["basis"]))].append(data)

    if not by_relation:
        print(f"no hits under {root}")
        return

    print(f"hits in {root}:  {len(by_relation)} unique relations")
    for (target, coeffs, basis), entries in by_relation.items():
        # Re-compose pretty: solve for first nonzero candidate-like term.
        labels = ["candidate"] + list(basis) if "candidate" not in basis else list(basis)
        cand_idx = 0
        a = coeffs[cand_idx]
        if a == 0:
            print(f"  [no-candidate] target={target} {coeffs}")
            continue
        rhs_terms = []
        for c, lbl in zip(coeffs, labels):
            if lbl == "candidate" or c == 0:
                continue
            rhs_terms.append(f"({Fraction(-c, a)})*{lbl}")
        rhs = " + ".join(rhs_terms) if rhs_terms else "0"
        print(f"  target={target}  candidate = {rhs}   ({len(entries)} param tuple(s))")
        for entry in entries[:3]:
            print(f"    family={entry['family']}  params={entry['params']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s_sanity = sub.add_parser("sanity", help="rediscover Apery's CF for zeta(3)")
    s_sanity.set_defaults(func=cmd_sanity)

    s_sweep = sub.add_parser("sweep", help="grid sweep + PSLQ")
    s_sweep.add_argument("--family", required=True)
    s_sweep.add_argument("--target", default="zeta3")
    s_sweep.add_argument(
        "--range",
        type=int,
        default=3,
        help="default symmetric range [-r, r] for each integer parameter",
    )
    s_sweep.add_argument(
        "--ranges",
        nargs="*",
        help="explicit per-parameter ranges as lo:hi (length must match arity)",
    )
    s_sweep.add_argument("--depth", type=int, default=80)
    s_sweep.add_argument("--dps", type=int, default=60)
    s_sweep.add_argument("--max-coeff", type=int, default=10**6, dest="max_coeff")
    s_sweep.add_argument("--extra-basis", default="", dest="extra_basis")
    s_sweep.add_argument("--out", default="hits")
    s_sweep.set_defaults(func=cmd_sweep)

    s_verify = sub.add_parser("verify", help="re-verify a hit at higher precision")
    s_verify.add_argument("--family", required=True)
    s_verify.add_argument("--params", required=True, help="comma-separated ints")
    s_verify.add_argument("--basis", required=True, help="comma-separated label list (incl. candidate)")
    s_verify.add_argument("--coeffs", required=True, help="comma-separated ints")
    s_verify.add_argument("--depth", type=int, default=200)
    s_verify.add_argument("--dps", type=int, default=200)
    s_verify.set_defaults(func=cmd_verify)

    s_sum = sub.add_parser("summarize", help="summarize hits in a directory")
    s_sum.add_argument("--dir", default="hits")
    s_sum.set_defaults(func=cmd_summarize)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
