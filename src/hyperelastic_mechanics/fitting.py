"""Peak density fitting with optical blur for neo-Hookean cavity profiles.

Computes the blurred peak density as a function of hoop stretch lambda,
used to fit the compressibility parameter beta to experimental data.

The optical blur accounts for finite microscope resolution: a Gaussian
kernel of width proportional to the mesh size is convolved with a density
profile that includes the zero-density cavity interior.
"""

from __future__ import annotations

import numpy as np
from scipy.signal.windows import gaussian

from .solver import nh_solver


def _gausswin(n: int) -> np.ndarray:
    """Gaussian window matching MATLAB's gausswin(n) with alpha=2.5."""
    n = max(int(round(n)), 1)
    std = (n - 1) / 5.0  # alpha=2.5 -> std = (N-1)/(2*alpha)
    g = gaussian(n, std=std)
    return g / g.sum()


def nh_blur_peak(
    beta: float,
    lambda_: float,
    blur_wid: float = 0.5 / 1.3,
    x_end: float = 6.0,
    n_eval: int = 50000,
) -> float:
    """Peak blurred density for a single (beta, lambda) pair.

    Computes the cavity density profile, prepends a zero-density hole region,
    applies a Gaussian blur of width blur_wid (in units of r(R)/R0), and
    returns the peak of the blurred profile.

    Parameters
    ----------
    beta : float
        Compressibility parameter.
    lambda_ : float
        Hoop stretch at the cavity surface.
    blur_wid : float
        Gaussian blur half-width in units of deformed radius. Default 0.5/1.3
        corresponds to half a mesh size in the reference geometry.
    x_end, n_eval : float, int
        Integration domain passed to nh_solver.

    Returns
    -------
    float
        Peak value of the blurred density profile.
    """
    rdim, den = nh_solver(beta, lambda_, x_end=x_end, n_eval=n_eval)

    drdim = (rdim[-1] - rdim[0]) / len(rdim)
    n_blur = blur_wid / drdim  # blur kernel width in pixels

    g = _gausswin(n_blur)

    # prepend zero-density hole (cavity interior)
    n_hole = int(round(rdim[0] / drdim))
    rdim_hole = np.concatenate([np.linspace(0, rdim[0], n_hole), rdim])
    den_hole = np.concatenate([np.zeros(n_hole), den])

    blurred = np.convolve(den_hole, g, mode="same")
    return float(blurred.max())


def nh_fit_fn(
    beta: float,
    lambda_list: np.ndarray,
    blur_wid: float = 0.5 / 1.3,
    x_end: float = 6.0,
    n_eval: int = 50000,
) -> np.ndarray:
    """Blurred peak density vs hoop stretch for a given beta.

    Parameters
    ----------
    beta : float
        Compressibility parameter to evaluate.
    lambda_list : array-like
        Hoop stretch values to evaluate.
    blur_wid : float
        Gaussian blur half-width in deformed radius units.
    x_end, n_eval : float, int
        Integration parameters passed to nh_solver.

    Returns
    -------
    np.ndarray
        Peak blurred density for each lambda in lambda_list.
    """
    lambda_list = np.asarray(lambda_list, dtype=float)
    return np.array(
        [
            nh_blur_peak(beta, lam, blur_wid=blur_wid, x_end=x_end, n_eval=n_eval)
            for lam in lambda_list
        ]
    )


def fit_beta(
    lambda_list: np.ndarray,
    pk_data: np.ndarray,
    beta0: float = 1.4,
    beta_bounds: tuple[float, float] = (1.001, 3.0),
    blur_wid: float = 0.5 / 1.3,
) -> tuple[float, float]:
    """Fit beta to experimental peak density data via least-squares.

    Parameters
    ----------
    lambda_list : array-like
        Observed hoop stretch values (e.g. droplet radius / mesh size).
    pk_data : array-like
        Observed peak density values (e.g. peak intensity / background).
    beta0 : float
        Initial guess for beta.
    beta_bounds : tuple
        (lower, upper) bounds on beta.
    blur_wid : float
        Blur width passed to nh_fit_fn.

    Returns
    -------
    beta_fit : float
        Best-fit compressibility parameter.
    nu_fit : float
        Corresponding Poisson ratio nu = (1 - 1/beta) / 2.
    """
    from scipy.optimize import least_squares

    lambda_list = np.asarray(lambda_list, dtype=float)
    pk_data = np.asarray(pk_data, dtype=float)

    def residuals(params):
        beta = params[0]
        return nh_fit_fn(beta, lambda_list, blur_wid=blur_wid) - pk_data

    result = least_squares(
        residuals,
        x0=[beta0],
        bounds=([beta_bounds[0]], [beta_bounds[1]]),
        diff_step=0.001,
        ftol=0.01,
    )

    beta_fit = float(result.x[0])
    nu_fit = (1.0 - 1.0 / beta_fit) / 2.0
    return beta_fit, nu_fit
