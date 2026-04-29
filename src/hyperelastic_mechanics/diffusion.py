"""Solvent exchange diffusion model (erfc semi-infinite slab).

Models the concentration at a fixed distance from the solvent inlet as a
function of time, and the spatial concentration profile at fixed time.

Used to estimate the interfacial tension (gamma_ow) as a function of time
during solvent exchange in the mesh-scale phase separation experiments.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc


def concentration_vs_time(
    D: float,
    Cs: float,
    C0: float,
    x_dist: float,
    t_arr: np.ndarray,
) -> np.ndarray:
    """Concentration at fixed distance x_dist as a function of time.

    Semi-infinite slab solution:
      C(x, t) = C0 + (Cs - C0) * erfc(x / (2*sqrt(D*t)))

    Parameters
    ----------
    D : diffusivity (m^2/s).
    Cs : concentration at the inlet boundary (x=0).
    C0 : initial concentration in the slab.
    x_dist : distance from inlet (m).
    t_arr : time array (s); t=0 is excluded (returns C0 for t→0).

    Returns
    -------
    np.ndarray
        Concentration at x_dist for each time in t_arr.
    """
    t_arr = np.asarray(t_arr, dtype=float)
    C = np.where(
        t_arr > 0,
        C0 + (Cs - C0) * erfc(x_dist / (2.0 * np.sqrt(D * t_arr))),
        C0,
    )
    return C


def concentration_profile(
    D: float,
    Cs: float,
    C0: float,
    x_arr: np.ndarray,
    t: float,
) -> np.ndarray:
    """Spatial concentration profile C(x) at a fixed time t.

    Parameters
    ----------
    D : diffusivity (m^2/s).
    Cs : concentration at inlet (x=0).
    C0 : initial concentration.
    x_arr : distance array (m).
    t : time (s); must be > 0.

    Returns
    -------
    np.ndarray
        Concentration profile across x_arr.
    """
    x_arr = np.asarray(x_arr, dtype=float)
    if t <= 0:
        return np.full_like(x_arr, C0)
    return C0 + (Cs - C0) * erfc(x_arr / (2.0 * np.sqrt(D * t)))


def gamma_vs_time(
    t_arr: np.ndarray,
    C_t: np.ndarray,
    w_H2O: np.ndarray,
    gamma_ow: np.ndarray,
    C_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Interfacial tension as a function of time via solvent-exchange diffusion.

    Combines concentration-vs-time from diffusion with a gamma_ow vs w_H2O
    calibration curve (from tensiometry) to give gamma_ow(t).

    Parameters
    ----------
    t_arr : time array (s).
    C_t : water concentration at the observation point vs time (from
          concentration_vs_time).
    w_H2O : water content values for the gamma calibration curve.
    gamma_ow : corresponding gamma_ow values (mN/m).
    C_threshold : only include times where C_t > C_threshold (above the
                  miscibility threshold where phase separation starts).

    Returns
    -------
    t_out : time values where C_t > C_threshold.
    gamma_out : interpolated gamma_ow at each t_out.
    """
    mask = C_t > C_threshold
    t_out = t_arr[mask]
    C_out = C_t[mask]
    gamma_out = np.interp(C_out, w_H2O[::-1], gamma_ow[::-1])
    return t_out, gamma_out
