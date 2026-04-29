"""Elastic energy and cavity pressure curves for NH and strain-stiffening models.

Usage:
    uv run python scripts/elastic.py
    uv run python scripts/elastic.py output_dir=outputs/myelastic
    uv run python scripts/elastic.py elastic.E=2.0 elastic.beta=50

All parameters correspond to keys in configs/elastic.yaml.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from hyperelastic_mechanics.elastic import (
    mr_elastic_energy,
    nh_elastic_energy_ronceray,
    nh_pressure,
    ss_pressure,
    surface_energy_density,
)
from hyperelastic_mechanics.figures import (
    plot_elastic_energies,
    plot_pressure_curves,
    plot_total_energy,
)


@hydra.main(version_base=None, config_path="../configs", config_name="elastic")
def main(cfg: DictConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    E = cfg.elastic.E
    beta = cfg.elastic.beta
    gamma = cfg.elastic.gamma
    epsilon_c = cfg.elastic.epsilon_c

    lam_a = np.linspace(cfg.elastic.lam_min, cfg.elastic.lam_max, cfg.elastic.n_lam)

    p_nh = nh_pressure(E, lam_a)
    curves = [p_nh]
    labels = ["Neo-Hookean"]
    for ec in cfg.elastic.epsilon_c_list:
        p_ss = ss_pressure(E, float(ec), lam_a)
        curves.append(p_ss)
        labels.append(rf"Strain-stiffening $\epsilon_c={ec}$")

    e_surf = surface_energy_density(gamma, lam_a)
    curves.append(e_surf)
    labels.append("Surface energy")

    plot_pressure_curves(
        lam_a,
        curves,
        labels,
        str(out_dir / "pressure_curves.png"),
        ylim=(0.0, cfg.elastic.p_ylim),
        title="Pressure / energy density vs hoop stretch",
    )
    print(f"  NH saturation pressure = {p_nh[-1]:.4f}")

    p_ss_default = ss_pressure(E, epsilon_c, lam_a)
    plot_total_energy(
        lam_a,
        p_nh,
        p_ss_default,
        e_surf,
        str(out_dir / "total_energy.png"),
        ylim=(0.0, cfg.elastic.p_ylim),
        title=rf"Total energy: $\epsilon_c={epsilon_c}$, $\gamma={gamma}$",
    )

    lam_e = np.linspace(1.0, cfg.elastic.lam_energy_max, cfg.elastic.n_lam)
    f_nh = nh_elastic_energy_ronceray(1.0, beta, lam_e)
    energy_curves = [f_nh]
    energy_labels = [rf"NH Ronceray ($\beta={beta}$)"]

    for n_val in cfg.elastic.n_list:
        f_mr = mr_elastic_energy(1.0, float(n_val), lam_e)
        energy_curves.append(f_mr)
        energy_labels.append(f"MR $n={n_val:.2f}$")

    plot_elastic_energies(
        lam_e,
        energy_curves,
        energy_labels,
        str(out_dir / "elastic_energies.png"),
        ylim=(0.0, 2.5),
        title="Elastic energy per droplet volume (G=1)",
    )

    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
