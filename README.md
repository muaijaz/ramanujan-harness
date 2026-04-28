# Ramanujan Harness

A reproducible identity-discovery framework in the spirit of the Ramanujan
Machine (Raayoni et al., *Nature* 2021). It evaluates parameterized series
families at high precision, runs PSLQ to find short integer relations
between candidate values and a basis of standard mathematical constants,
and verifies any hits at higher precision.

This is a **discovery framework**, not a proof system. It blind-rediscovers
known closed-form identities from scratch (proving the pipeline works), and
honestly returns nothing when nothing is found in the searched parameter
box (proving it doesn't hallucinate).

## About

I'm Mohammad Aijaz. I built this in an evening with Claude (Anthropic's
AI assistant) doing most of the typing while I steered, asked questions,
and pushed back when results looked sketchy. I was bored and wanted to
see if the "amateur + AI" vibe-math template — the one behind the
Scientific American story about an amateur cracking a 60-year problem
using ChatGPT — could be applied to identity discovery. We didn't crack
anything new. But the harness is honest, the rediscoveries are real, and
the negative results are reproducible. If any of this is useful to
someone working on a related problem, take it. If it's useful as a
teaching artifact for what AI-assisted experimental math actually looks
like — wins, dead ends, and corrections together — even better.

## What it does

```
parameter sweep  →  high-precision evaluation  →  PSLQ over basis  →  hit
       ↓                                                              ↓
generator family                                          symbolic verification
                                                                      ↓
                                                              hits/<...>.json
```

Given a generator family `G(p₁, …, pₙ)` and a target constant `T`, the
harness searches for short-coefficient integer relations of the form

```
a · G(params) + b · T + c₁ · k₁ + c₂ · k₂ + … = 0
```

where `k_i` are basis constants (π, ζ-values, log products, golden-ratio
logs, Clausen values, etc.). When such a relation is found, it implies a
closed-form identity for `G(params)` in terms of `T` and the basis.

## Validated rediscoveries

These thirteen identities are blind-rediscovered by `pytest tests/test_rediscovery.py`.
PSLQ has no prior knowledge of the rational coefficients — they fall out
of the integer-relation search.

| Target | Identity | Family | Notable |
|---|---|---|---|
| ζ(2) | 3 · Σ 1/(k²·C(2k,k)) = ζ(2) | central-binomial | folklore |
| ζ(3) | (5/2) · Σ (−1)^(k−1)/(k³·C(2k,k)) = ζ(3) | central-binomial | **Apéry, 1978** |
| ζ(4) | (36/17) · Σ 1/(k⁴·C(2k,k)) = ζ(4) | central-binomial | folklore |
| log²(φ) | (1/2) · Σ (−1)^(k−1)/(k²·C(2k,k)) = log²(φ) | central-binomial | arcsin² at x=i/2 |
| Catalan G | 8G = 3·Σ_{k≥0} 1/(C(2k,k)·(2k+1)²) + π·log(2+√3) | central-binomial-power | classical |
| log 2 | Σ 1/(k·2^k) = log 2 | power-weighted | Mercator |
| π | 4 · Σ_{k≥1} χ₄(k)/k = π | dirichlet | **Leibniz** |
| Catalan G | Σ_{k≥1} χ₄(k)/k² = G | dirichlet | L(2, χ₄) |
| π√3 | 9 · Σ_{k≥1} χ₃(k)/k = π√3 | dirichlet | L(1, χ₃) |
| ζ(3) | Σ H_k/k² = 2·ζ(3) | harmonic-weighted | **Euler, 1775** |
| ζ(4) | Σ H_k/k³ = (5/4)·ζ(4) | harmonic-weighted | Euler |
| π | Σ_{k≥0} 1/16^k · (4/(8k+1) − 2/(8k+4) − 1/(8k+5) − 1/(8k+6)) = π | bbp-multirational | **BBP, 1995** |
| √5 | Σ_{k≥0} C(2k,k)/5^k = √5 | ramanujan-sato | gen. function |
| ζ(4) | Σ Hₖ⁽²⁾/(k²·C(2k,k)) = (14/27)·ζ(4) | harmonic-weighted | arcsin⁴ Maclaurin at x=1/2 |

The ζ(3) result is **Apéry, 1978**. The π result is **Bailey-Borwein-Plouffe, 1995** — the famous "n-th hex digit of π without computing the previous ones" formula. The Euler sums are **1775**.

## Honest negative results

Across four generator families, five recognition algorithms (linear PSLQ,
multi-element subset PSLQ, polynomial root test up to degree 5,
mp.identify functional/multiplicative recognition, intra-family
combination PSLQ), and a 27-element basis including standard zeta values,
π powers, log products, golden-ratio logs, Clausen values, K(1/2) and
related elliptic constants, **no short-coefficient closed-form expression
was found** for:

- `Σ 1/(k³·C(2k,k))` — non-alternating cousin of Apéry
- `Σ (−1)^(k−1)/(k⁴·C(2k,k))` — alternating ζ(4)-shape
- `Σ 1/(k^p·C(2k,k)²)` for p ∈ {1, 2, 3, 4} — squared-binomial line
- `Σ (−1)^(k−1)/(k⁵·C(2k,k))` — natural ζ(5) candidate by sign-parity rule
- ζ(5), ζ(7) as combinations of central-binomial values for p ∈ {2,3,4,5}

The natural ζ(5) candidate's absence is consistent with 45+ years of
Apéry-style search by humans turning up nothing. The harness corroborates
that emptiness across multiple recognizer angles.

## Install

```bash
git clone https://github.com/<you>/ramanujan-harness
cd ramanujan-harness
uv venv && uv pip install -e ".[dev]"
uv run pytest -q
```

Requires Python ≥ 3.11. Dependencies: `mpmath`, `sympy`, `pytest`.

## Quick start

```bash
# Validate the pipeline on Apéry's ζ(3) continued fraction
uv run python -m ramanujan.cli sanity

# Re-run a known sweep (Apéry rediscovery)
uv run python -m ramanujan.cli sweep \
  --family central-binomial --target zeta3 \
  --ranges "1:1" "3:3" "1:1" "0:0" \
  --depth 200 --dps 80 --max-coeff 100

# Summarize hits in any directory
uv run python -m ramanujan.cli summarize --dir hits/
```

## Layout

```
ramanujan/
  constants.py      mpmath constants at configurable precision (27 included)
  generators.py     9 parameterized generator families
  pslq.py           PSLQ wrapper with spurious-relation filtering
  search.py         single-candidate grid sweep + retry logic
  multi.py          pair-candidate sweep (linear combinations)
  verify.py         high-precision re-verification
  reports.py        formatting helpers
  cli.py            sanity / sweep / verify / summarize subcommands
tests/
  test_constants.py
  test_generators.py
  test_pslq.py
  test_search.py
  test_rediscovery.py    locks in the 6 blind-rediscoveries
hits/                     example output (the rediscoveries)
```

## How it differs from the Ramanujan Machine

The Ramanujan Machine project (Raayoni et al. 2021) ran on distributed
infrastructure with millions of polynomial-CF candidates and reported a
catalog of ~100 conjectured identities. This harness:

- Runs on a single machine in seconds-to-minutes per sweep
- Includes a richer constant basis (golden ratio, Clausen, AGM, products)
- Adds a spurious-relation filter (PSLQ over a redundant basis can return
  basis-only identities glued to rational candidates; we detect and drop
  those)
- Adds a multi-element subset retry (PSLQ over too many basis elements
  often returns the wrong relation; we try minimal + small extras)
- Adds two-tier coefficient escalation (cheap pass at coeff ≤ 500 across
  subsets, then escalation to user-supplied coeff on minimal basis)
- Locks in known rediscoveries as `pytest` tests so regressions show up

It is not aiming to replace Ramanujan Machine — it's a single-machine
exploratory tool with stronger correctness guarantees per hit.

## What it is not

- Not a proof system. Hits are conjectural identities until proven by hand
  or by computer-algebra means.
- Not a replacement for human creativity. The genuine open problems
  (ζ(5), ζ(7) Apéry-like) require either much bigger compute or
  structurally new generator families that nobody has found.
- Not a discovery machine that promises to find new identities. It
  rediscovers known ones reliably and reports nothing when nothing exists
  in the searched space — that emptiness is itself useful information.

## Citation

If this is useful for your work, cite the underlying ideas:

- Ferguson, H. R. P., & Bailey, D. H. (1992). *A Polynomial Time,
  Numerically Stable Integer Relation Algorithm.* (PSLQ.)
- Raayoni, G. et al. (2021). *Generating conjectures on fundamental
  constants with the Ramanujan Machine.* Nature 590, 67–73.
- Apéry, R. (1979). *Irrationalité de ζ(2) et ζ(3).* Astérisque 61.

## License

MIT. See `LICENSE`.
