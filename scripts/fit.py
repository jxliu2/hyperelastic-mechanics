"""Fit the neo-Hookean compressibility parameter beta to experimental data.

Usage:
    uv run python scripts/fit.py
    uv run python scripts/fit.py beta0=1.2 output_dir=outputs/myfit

Expects experimental data as two arrays in a .npz file with keys
'lambda_list' and 'pk_data' (peak density normalized by background).
Pass the path via the data_path config key.

If data_path is null (default), runs a synthetic demonstration using
NH model predictions at beta=1.5 as mock data.

All parameters correspond to keys in configs/fit.yaml.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from hyperelastic_mechanics.figures import plot_fit_result
from hyperelastic_mechanics.fitting import fit_beta, nh_fit_fn


@hydra.main(version_base=None, config_path="../configs", config_name="fit")
def main(cfg: DictConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = getattr(cfg, "data_path", None)

    if data_path is not None:
        data = np.load(data_path)
        lambda_list = np.asarray(data["lambda_list"], dtype=float)
        pk_data = np.asarray(data["pk_data"], dtype=float)
    else:
        print("No data_path provided — running synthetic demo (beta_true=1.5)")
        lambda_list = np.linspace(5.5, 20.0, 50)
        pk_data = nh_fit_fn(1.5, lambda_list, blur_wid=cfg.blur_wid)
        pk_data += np.random.default_rng(42).normal(0, 0.005, size=len(pk_data))

    # filter and downsample
    mask = lambda_list > cfg.lambda_min
    lambda_list = lambda_list[mask]
    pk_data = pk_data[mask]

    if len(lambda_list) > cfg.n_fit_points:
        idx = np.round(np.linspace(0, len(lambda_list) - 1, cfg.n_fit_points)).astype(int)
        lambda_list = lambda_list[idx]
        pk_data = pk_data[idx]

    print(f"Fitting beta to {len(lambda_list)} data points...")

    beta_fit, nu_fit = fit_beta(
        lambda_list,
        pk_data,
        beta0=cfg.beta0,
        beta_bounds=(cfg.beta_lb, cfg.beta_ub),
        blur_wid=cfg.blur_wid,
    )

    print(f"  beta_fit = {beta_fit:.4f}")
    print(f"  nu_fit   = {nu_fit:.4f}")

    # dense model curve for plotting
    lambda_dense = np.linspace(lambda_list.min(), lambda_list.max(), 200)
    pk_fit = nh_fit_fn(beta_fit, lambda_dense, blur_wid=cfg.blur_wid)

    plot_fit_result(
        lambda_list,
        pk_data,
        lambda_dense,
        pk_fit,
        beta_fit,
        nu_fit,
        str(out_dir / "fit_result.png"),
    )

    np.savez(
        out_dir / "fit_result.npz",
        lambda_list=lambda_list,
        pk_data=pk_data,
        lambda_dense=lambda_dense,
        pk_fit=pk_fit,
        beta_fit=beta_fit,
        nu_fit=nu_fit,
    )

    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
