"""Experimental mesh-scale phase separation figures.

Usage:
    uv run python scripts/mesh_figures.py
    uv run python scripts/mesh_figures.py output_dir=outputs/myfigs

All parameters correspond to keys in configs/mesh_figures.yaml.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from hyperelastic_mechanics.figures import plot_mesh_figures


@hydra.main(version_base=None, config_path="../configs", config_name="mesh_figures")
def main(cfg: DictConfig) -> None:
    plot_mesh_figures(cfg.output_dir)


if __name__ == "__main__":
    main()
