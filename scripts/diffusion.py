"""Solvent exchange diffusion, ternary phase diagram, and cavitation phase diagram.

Usage:
    uv run python scripts/diffusion.py
    uv run python scripts/diffusion.py output_dir=outputs/mydiffusion
    uv run python scripts/diffusion.py diffusion.window_dist=2e-3

All parameters correspond to keys in configs/diffusion.yaml.

Ternary diagram: reads binodal data from data/water-etoh-decane.xlsx
(Skrzecz et al., J. Phys. Chem. Ref. Data 28(4), 1999, p. 1094).
Requires openpyxl: `uv add openpyxl`.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from hyperelastic_mechanics.diffusion import concentration_vs_time, gamma_vs_time
from hyperelastic_mechanics.figures import (
    plot_cavitation_phase_diagram,
    plot_concentration_vs_time,
    plot_gamma_vs_time,
    plot_ternary_diagram,
)


def _load_ternary_data(xlsx_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ethanol/decane/water weight fractions from the xlsx binodal table."""
    try:
        import openpyxl
    except ImportError as exc:
        raise ImportError(
            "openpyxl is required for the ternary diagram. "
            "Install with: uv add openpyxl"
        ) from exc

    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb.active

    etoh, decane, water = [], [], []
    for row in ws.iter_rows(min_row=4, max_row=15, min_col=6, max_col=8, values_only=True):
        if all(v is not None for v in row):
            etoh.append(float(row[0]))
            decane.append(float(row[1]))
            water.append(float(row[2]))

    return np.array(etoh), np.array(decane), np.array(water)


@hydra.main(version_base=None, config_path="../configs", config_name="diffusion")
def main(cfg: DictConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    D = cfg.diffusion.D
    Cs = cfg.diffusion.Cs
    C0 = cfg.diffusion.C0
    window_dist = cfg.diffusion.window_dist
    t_max = cfg.diffusion.t_max

    t_arr = np.linspace(1.0, t_max, cfg.diffusion.n_t)
    C_t = concentration_vs_time(D, Cs, C0, window_dist, t_arr)

    highlight = None
    if cfg.diffusion.get("highlight_t") is not None:
        th = float(cfg.diffusion.highlight_t)
        Ch = float(concentration_vs_time(D, Cs, C0, window_dist, np.array([th]))[0])
        highlight = [(th, Ch)]

    plot_concentration_vs_time(
        t_arr,
        C_t,
        str(out_dir / "concentration_vs_time.png"),
        x_dist=window_dist,
        highlight_points=highlight,
        title=f"Water content at {window_dist*1e3:.1f} mm vs time",
    )
    print(f"  C at t={t_arr[-1]:.0f}s: {C_t[-1]:.4f}")

    gamma_data_path = cfg.diffusion.get("gamma_data_path")
    if gamma_data_path is not None and Path(gamma_data_path).exists():
        gdata = np.load(gamma_data_path)
        w_H2O = gdata["w_H2O"]
        gamma_ow = gdata["gamma_ow"]
        t_out, gamma_out = gamma_vs_time(
            t_arr, C_t, w_H2O, gamma_ow,
            C_threshold=float(cfg.diffusion.get("C_threshold", 0.0)),
        )
        plot_gamma_vs_time(
            t_out,
            gamma_out,
            str(out_dir / "gamma_vs_time.png"),
            title=r"$\gamma_{ow}$ vs time (solvent exchange)",
        )
    else:
        print("  No gamma calibration data — skipping gamma_vs_time plot.")
        print("  Provide gamma_data_path (npz with w_H2O and gamma_ow keys).")

    xlsx_path = cfg.diffusion.get("ternary_xlsx")
    if xlsx_path is not None and Path(xlsx_path).exists():
        try:
            etoh, decane, water = _load_ternary_data(str(xlsx_path))
            plot_ternary_diagram(
                etoh,
                decane,
                water,
                str(out_dir / "ternary_diagram.png"),
            )
        except ImportError as e:
            print(f"  Skipping ternary diagram: {e}")
    else:
        print("  No ternary xlsx found — skipping ternary_diagram.")
        print("  Set diffusion.ternary_xlsx in config or pass on command line.")

    c_gel = np.array(cfg.phase_diagram.c_gel, dtype=float)
    c_surf = np.array(cfg.phase_diagram.c_surf, dtype=float)
    cavitated = np.array(cfg.phase_diagram.cavitated, dtype=bool)

    plot_cavitation_phase_diagram(
        c_surf,
        c_gel,
        cavitated,
        str(out_dir / "cavitation_phase_diagram.png"),
    )

    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
