"""Neo-Hookean spherical cavity ODE solver.

Solves the second-order ODE for radial displacement r(R) around a spherical
cavity in a slightly compressible neo-Hookean material.

The dimensionless displacement r1 = r(R)/R0 - 1 satisfies Eq.(9) from
"Cavitation_instability_in_arbitrary_D.pdf":

    dv/dx = -2v/x - 2v/x * (1 + beta*(r+1)^3*(x*v+r+1)) / (1 + beta*(r+1)^4)
    dr/dx = v

where x = R/R0, r = r1, v = dr1/dx, and beta = 1/(1-2*nu) is the
compressibility parameter.

The boundary condition at the cavity surface (x=1) is computed via the
integral in Eq.(18):

    del1 = (sqrt(1+beta) - 2 * integral_1^lambda 1/sqrt(1+beta*tau^4) dtau)
            / sqrt(1+beta*lambda^4)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad, solve_ivp


def _radial_ic(beta: float, lambda_: float) -> tuple[float, float]:
    """Compute initial conditions at the cavity surface x=1.

    Returns (v0, r0) where r0 = lambda - 1 and v0 = del1 - lambda,
    with del1 the radial stretch at the cavity surface from Eq.(18).
    """
    dum, _ = quad(lambda tau: 1.0 / np.sqrt(1.0 + beta * tau**4), 1.0, lambda_)
    del1 = (np.sqrt(1.0 + beta) - 2.0 * dum) / np.sqrt(1.0 + beta * lambda_**4)
    return del1 - lambda_, lambda_ - 1.0


def _ode_rhs(x: float, y: np.ndarray, beta: float) -> list[float]:
    """RHS of the cavity ODE system [dr/dx, dv/dx]."""
    r, v = y
    hoop = r + 1.0  # hoop stretch t = r1 + 1
    radial = x * v + r + 1.0  # radial stretch s = x*v + r1 + 1
    dvdx = (
        -2.0 * v / x
        - 2.0 * v / x * (1.0 + beta * hoop**3 * radial) / (1.0 + beta * hoop**4)
    )
    return [v, dvdx]


def nh_solver(
    beta: float,
    lambda_: float,
    x_end: float = 6.0,
    n_eval: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the spherical cavity ODE for a slightly compressible neo-Hookean material.

    Parameters
    ----------
    beta : float
        Compressibility parameter beta = 1/(1-2*nu). Must satisfy beta > 1.
        Larger beta means less compressible (closer to nu = 0.5).
    lambda_ : float
        Hoop stretch at the cavity surface r(R0)/R0.
    x_end : float
        Maximum dimensionless reference radius x = R/R0 to integrate to.
    n_eval : int
        Number of evenly spaced output points on [1, x_end].

    Returns
    -------
    rdim : np.ndarray
        Dimensionless deformed radius r(R)/R0 = x*(r1+1).
    den : np.ndarray
        Relative density 1/J = 1/((hoop)^2 * radial).
    """
    v0, r0 = _radial_ic(beta, lambda_)
    x_eval = np.linspace(1.0 + (x_end - 1.0) / n_eval, x_end, n_eval)

    sol = solve_ivp(
        _ode_rhs,
        (1.0, x_end),
        [r0, v0],
        args=(beta,),
        method="RK45",
        t_eval=x_eval,
        rtol=1e-8,
        atol=1e-10,
    )

    xx = sol.t
    r = sol.y[0]
    v = sol.y[1]

    hoop = r + 1.0
    radial = xx * v + r + 1.0

    rdim = xx * hoop  # r(R)/R0 in deformed coordinates
    den = 1.0 / (hoop**2 * radial)  # relative density

    return rdim, den


def incompressible_rdim(
    lambda_: float, xx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Analytical incompressible (nu=0.5) solution for comparison.

    rth(x) = (1 + (lambda^3 - 1)/x^3)^(1/3) - 1
    """
    rth = (1.0 + (lambda_**3 - 1.0) / xx**3) ** (1.0 / 3.0) - 1.0
    return rth, xx * (rth + 1.0)
