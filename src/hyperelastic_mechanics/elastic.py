"""Elastic energy and pressure calculations for neo-Hookean and strain-stiffening materials.

All formulas assume spherical symmetry and incompressibility (J=1) unless noted.
The hoop stretch lambda_a = r(R0)/R0 is the independent variable for cavity problems.

Reference: Ronceray et al. SI B12 (NH energy density expansion),
           Kothari & Cohen 2020 (pressure integral formula).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def _I1(lam: float | np.ndarray) -> float | np.ndarray:
    """First invariant for incompressible spherical cavity: lam^{-4} + 2*lam^2."""
    return lam**-4 + 2.0 * lam**2


def _dI1_dlam(lam: float | np.ndarray) -> float | np.ndarray:
    return -4.0 * lam**-5 + 4.0 * lam


def nh_pressure(E: float, lam_a: float | np.ndarray) -> float | np.ndarray:
    """Cavity pressure for incompressible neo-Hookean material.

    Analytical result of the Kothari pressure integral for W = E/2*(I1-3):
      p(lam_a) = E * (5/2 - 2/lam_a - 1/(2*lam_a^4))

    Parameters
    ----------
    E : Young's modulus (or shear modulus G, same for incompressible NH).
    lam_a : hoop stretch >= 1.
    """
    lam_a = np.asarray(lam_a, dtype=float)
    return E * (2.5 - 2.0 / lam_a - 0.5 / lam_a**4)


def ss_pressure(
    E: float,
    epsilon_c: float,
    lam_a: float | np.ndarray,
) -> float | np.ndarray:
    """Cavity pressure for incompressible strain-stiffening material.

    Uses W = E/2 * (I1-3 + ((I1-3)/(6*epsilon_c))^3) and the Kothari
    pressure integral, evaluated numerically.

    Parameters
    ----------
    E : elastic modulus.
    epsilon_c : stiffening parameter (larger = later stiffening onset).
    lam_a : hoop stretch.
    """
    lam_a = np.atleast_1d(np.asarray(lam_a, dtype=float))
    result = np.empty_like(lam_a)

    def _integrand(t: float) -> float:
        i1 = _I1(t)
        di1 = _dI1_dlam(t)
        dW = E / 2.0 * di1 * (1.0 + 3.0 * ((i1 - 3.0) / (6.0 * epsilon_c)) ** 2 / (6.0 * epsilon_c))
        return dW / (1.0 - t**3)

    for k, la in enumerate(lam_a):
        if la <= 1.0:
            result[k] = 0.0
        else:
            val, _ = quad(_integrand, la, 1.0)
            result[k] = val

    return result.squeeze() if result.size == 1 else result


def surface_energy_density(gamma: float, lam_a: float | np.ndarray) -> float | np.ndarray:
    """Surface energy per unit droplet volume: 3*gamma / lam_a.

    Derived from E_surf = 4*pi*(lam_a*R0)^2*gamma and V = (4/3)*pi*(lam_a*R0)^3.
    """
    return 3.0 * gamma / np.asarray(lam_a, dtype=float)


def nh_elastic_energy_ronceray(
    G: float,
    beta: float,
    lam: float | np.ndarray,
) -> float | np.ndarray:
    """Neo-Hookean elastic energy per droplet volume (Ronceray SI B12).

    Expansion to order 1/beta^2 around the incompressible limit.

    Parameters
    ----------
    G : shear modulus.
    beta : compressibility parameter beta = 1/(1-2*nu).
    lam : hoop stretch (>= 1).
    """
    lam = np.asarray(lam, dtype=float)
    incompressible = G * (2.5 - 3.0 / lam - 1.0 / lam**3 + 1.5 / lam**4)
    order1 = (1.0 / beta) * G * (
        -3.0 / 40.0
        + 6.0 / 5.0 / lam**3
        - 9.0 / 4.0 / lam**4
        + 6.0 / 5.0 / lam**5
        - 3.0 / 40.0 / lam**8
    )
    order2 = (1.0 / beta**2) * (
        1.0 / 48.0
        - 2.0 / 15.0 / lam**3
        + 9.0 / 80.0 / lam**4
        + 9.0 / 80.0 / lam**8
        - 2.0 / 15.0 / lam**9
        + 1.0 / 48.0 / lam**12
    )
    return incompressible + order1 + order2


def mr_elastic_energy(
    G: float,
    n: float,
    lam: float | np.ndarray,
) -> float | np.ndarray:
    """Mooney-Rivlin elastic energy per droplet volume.

    f_out = n*G*(5/6 - 1/lam - 1/(3*lam^3) + 1/(2*lam^4))
           + (1-n)*G*(lam/2 - 1/3 - 1/lam^2 + 5/(6*lam^3))

    n=1 is the C10-only (neo-Hookean) limit; n=0 is the C01-only limit.

    Parameters
    ----------
    G : shear modulus.
    n : mixing parameter in [0, 1].
    lam : hoop stretch (>= 1).
    """
    lam = np.asarray(lam, dtype=float)
    f1 = n * G * (5.0 / 6.0 - 1.0 / lam - 1.0 / (3.0 * lam**3) + 0.5 / lam**4)
    f2 = (1.0 - n) * G * (lam / 2.0 - 1.0 / 3.0 - 1.0 / lam**2 + 5.0 / (6.0 * lam**3))
    return f1 + f2
