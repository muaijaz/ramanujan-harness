import mpmath as mp

from ramanujan.constants import basis, value


def test_zeta3_known_digits():
    z = value("zeta3", 30)
    assert mp.nstr(z, 10) == "1.202056903"


def test_basis_lengths_match():
    vals = basis(["1", "pi", "zeta3"], 40)
    assert len(vals) == 3
    assert vals[0] == 1
    assert mp.nstr(vals[1], 10) == "3.141592654"
