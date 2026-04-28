import mpmath as mp

from ramanujan.generators import GENERATORS


def test_polynomial_cf_matches_golden_ratio():
    # phi = 1 + 1/(1 + 1/(1 + ...)) -> a(n) = 1, b(n) = 1
    gen = GENERATORS["polynomial-cf"]
    # a(n)=1, b(n)=1, head=1
    val = gen.at((1, 0, 0, 0, 1, 0, 0, 0, 1), dps=50, depth=100)
    mp.mp.dps = 60
    expected = (1 + mp.sqrt(5)) / 2
    assert mp.fabs(val - expected) < mp.mpf(10) ** -40


def test_polynomial_cf_brouncker_4_over_pi():
    # 4/pi = 1 + 1^2/(2 + 3^2/(2 + 5^2/(2 + ...)))
    # a(n) = (2n-1)^2 = 4n^2 - 4n + 1, b(n>=1) = 2, head = 1
    gen = GENERATORS["polynomial-cf"]
    val = gen.at((1, -4, 4, 0, 2, 0, 0, 0, 1), dps=50, depth=2000)
    mp.mp.dps = 60
    expected = 4 / mp.pi
    # Brouncker converges slowly; allow looser tolerance
    assert mp.fabs(val - expected) < mp.mpf("1e-2")


def test_hypergeometric_finite():
    gen = GENERATORS["hypergeometric"]
    val = gen.at((0, 0, 0), dps=40, depth=20)
    assert mp.isfinite(val)
