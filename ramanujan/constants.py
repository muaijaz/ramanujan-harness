"""High-precision target constants and basis vectors for PSLQ."""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp


@dataclass(frozen=True)
class Constant:
    """One target/basis constant computable at arbitrary precision."""

    name: str
    compute: callable

    def at(self, dps: int) -> mp.mpf:
        old = mp.mp.dps
        mp.mp.dps = dps
        try:
            return mp.mpf(self.compute())
        finally:
            mp.mp.dps = old


CONSTANTS: dict[str, Constant] = {
    "1": Constant("1", lambda: mp.mpf(1)),
    "pi": Constant("pi", lambda: mp.pi),
    "pi2": Constant("pi^2", lambda: mp.pi**2),
    "pi3": Constant("pi^3", lambda: mp.pi**3),
    "pi4": Constant("pi^4", lambda: mp.pi**4),
    "e": Constant("e", lambda: mp.e),
    "log2": Constant("log 2", lambda: mp.log(2)),
    "log3": Constant("log 3", lambda: mp.log(3)),
    "zeta2": Constant("zeta(2)", lambda: mp.zeta(2)),
    "zeta3": Constant("zeta(3)", lambda: mp.zeta(3)),
    "zeta4": Constant("zeta(4)", lambda: mp.zeta(4)),
    "zeta5": Constant("zeta(5)", lambda: mp.zeta(5)),
    "zeta6": Constant("zeta(6)", lambda: mp.zeta(6)),
    "zeta7": Constant("zeta(7)", lambda: mp.zeta(7)),
    "catalan": Constant("Catalan", lambda: mp.catalan),
    "apery": Constant("Apery (zeta(3))", lambda: mp.zeta(3)),
    # Compound constants that appear in real Catalan / Apery-Schmidt identities
    "log2sqrt3": Constant("log(2+sqrt(3))", lambda: mp.log(2 + mp.sqrt(3))),
    "log1sqrt2": Constant("log(1+sqrt(2))", lambda: mp.log(1 + mp.sqrt(2))),
    "sqrt2": Constant("sqrt(2)", lambda: mp.sqrt(2)),
    "sqrt3": Constant("sqrt(3)", lambda: mp.sqrt(3)),
    "sqrt5": Constant("sqrt(5)", lambda: mp.sqrt(5)),
    "pi_log2": Constant("pi*log(2)", lambda: mp.pi * mp.log(2)),
    "pi_log2sqrt3": Constant("pi*log(2+sqrt(3))", lambda: mp.pi * mp.log(2 + mp.sqrt(3))),
    "pi_log1sqrt2": Constant("pi*log(1+sqrt(2))", lambda: mp.pi * mp.log(1 + mp.sqrt(2))),
    "pi_sqrt3": Constant("pi*sqrt(3)", lambda: mp.pi * mp.sqrt(3)),
    "Cl2_pi3": Constant("Cl_2(pi/3)", lambda: mp.clsin(2, mp.pi / 3)),
    "Cl2_pi4": Constant("Cl_2(pi/4)", lambda: mp.clsin(2, mp.pi / 4)),
    # Apery's other constant: zeta(2) * log(2)
    "zeta2_log2": Constant("zeta(2)*log(2)", lambda: mp.zeta(2) * mp.log(2)),
    # Golden ratio family (Lehmer's central-binomial identities)
    "phi": Constant("phi (golden ratio)", lambda: (1 + mp.sqrt(5)) / 2),
    "log_phi": Constant("log(phi)", lambda: mp.log((1 + mp.sqrt(5)) / 2)),
    "log_phi_sq": Constant("log(phi)^2", lambda: mp.log((1 + mp.sqrt(5)) / 2) ** 2),
    "log_phi_cu": Constant("log(phi)^3", lambda: mp.log((1 + mp.sqrt(5)) / 2) ** 3),
    # Elliptic / AGM (squared-binomial identities)
    "K_half": Constant("K(1/2)", lambda: mp.ellipk(mp.mpf(1) / 2)),
    "K_third": Constant("K(1/3)", lambda: mp.ellipk(mp.mpf(1) / 3)),
    "K_quarter": Constant("K(1/4)", lambda: mp.ellipk(mp.mpf(1) / 4)),
    "agm_1_sqrt2": Constant("AGM(1, 1/sqrt(2))", lambda: mp.agm(1, 1 / mp.sqrt(2))),
    # Compound: log^2(phi) * sqrt(5), pi^2 * log(phi)
    "sqrt5_log_phi_sq": Constant(
        "sqrt(5) * log(phi)^2", lambda: mp.sqrt(5) * mp.log((1 + mp.sqrt(5)) / 2) ** 2
    ),
    # Compound targets nobody searches: products and ratios of known constants
    "pi_catalan": Constant("pi * G", lambda: mp.pi * mp.catalan),
    "G_log2": Constant("G * log(2)", lambda: mp.catalan * mp.log(2)),
    "G_sq": Constant("G^2", lambda: mp.catalan**2),
    "zeta3_log2": Constant("zeta(3) * log(2)", lambda: mp.zeta(3) * mp.log(2)),
    "zeta3_pi": Constant("zeta(3) * pi", lambda: mp.zeta(3) * mp.pi),
    "zeta3_over_pi3": Constant("zeta(3) / pi^3", lambda: mp.zeta(3) / mp.pi**3),
    "pi_log_phi": Constant("pi * log(phi)", lambda: mp.pi * mp.log((1 + mp.sqrt(5)) / 2)),
    "log2_log3": Constant("log(2) * log(3)", lambda: mp.log(2) * mp.log(3)),
    "pi_cu_log2": Constant("pi^3 * log(2)", lambda: mp.pi**3 * mp.log(2)),
    # Exotic constants rarely searched
    "khinchin": Constant("Khinchin", lambda: mp.khinchin),
    "glaisher": Constant("Glaisher-Kinkelin", lambda: mp.glaisher),
    "twin_prime": Constant("Twin Prime", lambda: mp.mpf("0.66016181584686957392781211001455577843262336028473341331")),
    "zeta_apery_sq": Constant("zeta(3)^2", lambda: mp.zeta(3) ** 2),
    "feigenbaum_d": Constant("Feigenbaum delta", lambda: mp.mpf("4.6692016091029906718532")),
}


def basis(names: list[str], dps: int) -> list[mp.mpf]:
    """Evaluate a list of basis constants at the given precision."""

    return [CONSTANTS[n].at(dps) for n in names]


def value(name: str, dps: int) -> mp.mpf:
    """Evaluate one named constant at the given precision."""

    return CONSTANTS[name].at(dps)
