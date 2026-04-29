"""Solve and plot neo-Hookean spherical cavity profiles.

Usage:
    uv run python scripts/solve.py
    uv run python scripts/solve.py output_dir=outputs/myrun
    uv run python scripts/solve.py lambda_sweep.beta=1.1

All parameters correspond to keys in configs/solve.yaml.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.signal.windows import gaussian

from hyperelastic_mechanics.fitting import nh_fit_fn
from hyperelastic_mechanics.figures import (
    plot_blurred_profiles,
    plot_density_profiles,
    plot_peak_density,
    plot_radial_displacements,
)
from hyperelastic_mechanics.solver import nh_solver


@hydra.main(version_base=None, config_path="../configs", config_name="solve")
def main(cfg: DictConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_end = cfg.x_end
    n_eval = cfg.n_eval
    blur_wid = cfg.blur_wid

    density_profiles = []
    disp_profiles = []
    labels = []

    xx = np.linspace(1.0 + (x_end - 1.0) / n_eval, x_end, n_eval)

    for case in cfg.cases:
        beta = case.beta
        lambda_ = case.lambda_
        label = case.label

        rdim, den = nh_solver(beta, lambda_, x_end=x_end, n_eval=n_eval)
        nu = (1.0 - 1.0 / beta) / 2.0

        density_profiles.append((rdim, den))
        disp_profiles.append((xx, rdim))
        labels.append(label)

        print(f"  {label}: nu={nu:.3f}, peak density={den.max():.4f}")

    plot_density_profiles(
        density_profiles,
        labels,
        str(out_dir / "density_profiles.png"),
        xlim=(1.0, x_end),
        title="Relative density in deformed coordinates",
    )

    plot_radial_displacements(
        disp_profiles,
        labels,
        str(out_dir / "radial_displacements.png"),
        xlim=(1.0, x_end),
        title=r"Deformed radius $r(R)/R_0$ vs reference radius $R/R_0$",
    )

    if cfg.lambda_sweep.enabled:
        beta = cfg.lambda_sweep.beta
        nu = (1.0 - 1.0 / beta) / 2.0
        lambda_list = np.linspace(
            cfg.lambda_sweep.lambda_min,
            cfg.lambda_sweep.lambda_max,
            cfg.lambda_sweep.n_lambda,
        )

        print(f"\nLambda sweep: beta={beta:.3f}, nu={nu:.3f}")

        rdim_hole_list = []
        den_blur_list = []
        rdim_list = []
        den_list = []
        pk_list = []
        pk_blur_list = []

        for lam in lambda_list:
            rdim, den = nh_solver(beta, lam, x_end=x_end, n_eval=n_eval)

            drdim = (rdim[-1] - rdim[0]) / len(rdim)
            n_blur = blur_wid / drdim

            n_win = max(int(round(n_blur)), 1)
            std = (n_win - 1) / 5.0
            g = gaussian(n_win, std=std)
            g = g / g.sum()

            n_hole = int(round(rdim[0] / drdim))
            rdim_hole = np.concatenate([np.linspace(0, rdim[0], n_hole), rdim])
            den_hole = np.concatenate([np.zeros(n_hole), den])
            blurred = np.convolve(den_hole, g, mode="same")

            rdim_hole_list.append(rdim_hole)
            den_blur_list.append(blurred)
            rdim_list.append(rdim)
            den_list.append(den)
            pk_list.append(float(den.max()))
            pk_blur_list.append(float(blurred.max()))

        plot_blurred_profiles(
            rdim_hole_list,
            den_blur_list,
            rdim_list,
            den_list,
            list(lambda_list),
            str(out_dir / "blurred_profiles.png"),
            xlim=(0.0, x_end),
            title=rf"Density profiles $\beta={beta:.3f}$, $\nu={nu:.3f}$",
        )

        plot_peak_density(
            lambda_list,
            np.array(pk_list),
            np.array(pk_blur_list),
            beta,
            str(out_dir / "peak_density.png"),
        )

    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
