import mpmath as mp

from ramanujan.pslq import find_relation


def test_recovers_pi_squared_over_six():
    mp.mp.dps = 60
    values = [mp.zeta(2), mp.pi**2]
    rel = find_relation(values, ["zeta2", "pi^2"], dps=50, max_coeff=100)
    assert rel is not None
    a, b = rel.coeffs
    assert (a, b) == (6, -1) or (a, b) == (-6, 1)


def test_no_spurious_relation_for_independent():
    mp.mp.dps = 60
    values = [mp.pi, mp.e]
    rel = find_relation(values, ["pi", "e"], dps=50, max_coeff=1000)
    assert rel is None
