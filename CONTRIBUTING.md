# Contributing

Thanks for taking a look. This project is small and pragmatic — the
contribution surface is mostly *adding new generator families* or
*adding new basis constants* to widen the search territory.

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest -q
uv run python -m ramanujan.cli sanity
```

## Adding a new generator family

A "family" is a parameterized series whose values you want to feed into
PSLQ. Three steps:

**1. Write the evaluator.** Add a function to `ramanujan/generators.py`:

```python
def my_family_value(a: int, b: int, *, depth: int) -> mp.mpf:
    """One-line description.

    S = sum_{k=1..depth} <your formula in a, b, k>
    """
    if a < 0 or a > 5:                      # always validate ranges
        raise ValueError("a in [0, 5]")
    total = mp.mpf(0)
    for k in range(1, depth + 1):
        total += mp.mpf(<your term>) / mp.mpf(<your denom>)
    return total
```

For slowly-converging series (anything not geometric), use closed-form
evaluation via Hurwitz zeta / digamma when available — see
`dirichlet_value` for the pattern.

**2. Register it.** Add to the `GENERATORS` dict at the bottom of
`generators.py`:

```python
"my-family": Generator(
    name="my-family",
    arity=2,
    evaluate=lambda a, b, depth: my_family_value(a, b, depth=depth),
),
```

**3. Add a sanity test.** In `tests/test_generators.py`, prove your
family computes a known closed-form value somewhere (use a small
integer-coefficient case so PSLQ can verify):

```python
def test_my_family_at_known_point():
    val = GENERATORS["my-family"].at((0, 1), dps=50, depth=200)
    assert mp.fabs(val - <known constant>) < mp.mpf(10) ** -40
```

## Adding a new basis constant

Add to `ramanujan/constants.py`:

```python
"my_constant": Constant("Display Name", lambda: <mpmath expression>),
```

Use this when you have a target whose closed form involves a
transcendental that isn't yet in the basis (e.g., `Cl_3(pi/3)`,
`K(1/sqrt(2))`, AGM-related values).

## Adding a rediscovery test

Once a sweep produces a hit you want to lock in, add it to
`tests/test_rediscovery.py`. The pattern:

```python
def test_rediscover_my_identity():
    with tempfile.TemporaryDirectory() as tmp:
        hits = sweep(
            family="my-family",
            param_ranges=[(<exact param>, <exact param>), ...],
            target="<target name>",
            extra_basis=["<aux constants>"],
            depth=200, dps=70, max_coeff=100,
            out_dir=Path(tmp), progress_every=0,
        )
    assert len(hits) == 1
    h = hits[0]
    cand_idx = 0
    target_idx = list(h.basis).index("<target name>") + 1
    ratio = -h.coeffs[target_idx] / h.coeffs[cand_idx]
    assert abs(abs(ratio) - <expected rational>) < 1e-12
```

## Honest expectations

- Most sweeps return zero hits. That's the right behaviour — the
  parameter space for known small-coefficient identities is well-mined.
- A new family is most valuable when it covers a *shape* that none of
  the existing families do (e.g. characters mod p, factorial ratios,
  multi-rational numerators). Reading the README's "What we may have
  missed" section gives concrete suggestions.
- If you find an identity that *does* appear new, please open an issue
  with the JSON hit + a numerical verification at higher precision
  before claiming discovery.

## Style

- One-line module docstrings; multi-line where the math needs
  explanation.
- No emojis in code or docs.
- `uv run pytest -q` should always pass before opening a PR.
