from dataclasses import dataclass

import numpy as np
import xraylib

_DENSITIES: dict[str, float] = {
    "diamond": 3515.0,
    "beryllium": 1848.0,
    "aluminum": 2700.0,
    "silicon": 2329.0,
    "nickel": 8908.0,
    "ip-dip": 1200.0,
    "ip-s": 1200.0,
}
_FORMULAS = {
    "diamond": "C",
    "beryllium": "Be",
    "aluminum": "Al",
    "silicon": "Si",
    "nickel": "Ni",
    "ip-dip": "IP-Dip",
    "ip-s": "IP-S",
}
_ALIASES = {
    "c": "diamond",
    "be": "beryllium",
    "al": "aluminum",
    "si": "silicon",
    "ni": "nickel",
    "ipdip": "ip-dip",
    "ip dip": "ip-dip",
    "ip-d": "ip-dip",
    "ips": "ip-s",
    "ip s": "ip-s",
}


@dataclass(frozen=True)
class OpticalConstants:
    delta: float
    beta: float
    mu: float


@dataclass(frozen=True)
class Material:
    formula: str
    density: float
    name: str = ""

    def __post_init__(self):
        if not self.formula or not np.isfinite(self.density) or self.density <= 0:
            raise ValueError("material requires a formula and positive SI density")

    @classmethod
    def preset(cls, name: str) -> "Material":
        key = _ALIASES.get(name.lower(), name.lower())
        if key not in _DENSITIES:
            raise ValueError("unknown material preset")
        return cls(_FORMULAS[key], _DENSITIES[key], key)

    def optical_constants(self, energy_keV: float) -> OpticalConstants:
        if not np.isfinite(energy_keV) or energy_keV <= 0:
            raise ValueError("energy must be positive keV")
        key = _ALIASES.get(self.name.lower(), self.name.lower())
        if not key:
            formula_key = self.formula.lower()
            key = _ALIASES.get(formula_key, formula_key)
        if key in ("ip-dip", "ip-s"):
            delta_reference = 1.95e-6 if key == "ip-dip" else 1.87e-6
            mu_reference = 256.0 if key == "ip-dip" else 246.0
            ratio = 11.867 / energy_keV
            delta = delta_reference * ratio**2
            mu = mu_reference * ratio**3
            wavelength = 1.239841984e-9 / energy_keV
            beta = mu * wavelength / (4.0 * np.pi)
            return OpticalConstants(delta, float(beta), mu)
        density_g_cm3 = self.density / 1000.0
        real = xraylib.Refractive_Index_Re(self.formula, energy_keV, density_g_cm3)
        imag = xraylib.Refractive_Index_Im(self.formula, energy_keV, density_g_cm3)
        beta = float(abs(imag))
        mass_attenuation_cm2_g = xraylib.CS_Total_CP(self.formula, energy_keV)
        mu_total_m = mass_attenuation_cm2_g * density_g_cm3 * 100.0
        return OpticalConstants(float(1.0 - real), beta, float(mu_total_m))


DIAMOND = Material.preset("diamond")
BERYLLIUM = Material.preset("beryllium")
ALUMINUM = Material.preset("aluminum")
SILICON = Material.preset("silicon")
NICKEL = Material.preset("nickel")
IP_DIP = Material.preset("ip-dip")
IP_S = Material.preset("ip-s")
