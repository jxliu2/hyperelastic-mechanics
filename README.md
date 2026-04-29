# hyperelastic-mechanics

Neo-Hookean spherical cavity solver for slightly compressible hyperelastic materials. Code supporting:

> Liu, J. et al. "Mesh-scale phase separation in hydrogels driven by solvent exchange." *Nature Communications* 2023.
> https://www.nature.com/articles/s41467-023-41528-8

## Overview

Models radial deformation around a spherical cavity in a slightly compressible neo-Hookean material. The compressibility parameter beta = 1/(1-2*nu) controls how much material accumulates around an expanding cavity, producing density rings observable by optical microscopy.

Five pipelines are available:

| Pipeline | What it does |
|---|---|
| `solve` | Solves the cavity ODE and plots density and displacement profiles |
| `fit` | Fits the compressibility parameter beta to experimental peak density data |
| `mesh_figures` | Generates experimental data figures from the phase separation study |
| `elastic` | Plots elastic energy densities and cavity pressure curves for NH and strain-stiffening models |
| `diffusion` | Solvent exchange diffusion model, ternary phase diagram, and cavitation phase diagram |

## Installation

Requires Python 3.13+. Dependencies are managed with [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Running

Scripts live in `scripts/`. Parameters are configured via [Hydra](https://hydra.cc) — any key in the config files can be overridden on the command line.

### Solve

```bash
uv run python scripts/solve.py
```

Solves the cavity ODE for the default parameter cases (beta=1.5/lambda=3 and beta=1.3/lambda=5) and produces density profiles, radial displacement curves, and blurred density profiles for a lambda sweep. Output goes to `outputs/solve/`.

```bash
uv run python scripts/solve.py lambda_sweep.beta=1.1 lambda_sweep.lambda_max=5.0
uv run python scripts/solve.py 'cases=[{beta: 1.1, lambda_: 2.0, label: "nu~0.05"}, {beta: 2.4, lambda_: 2.0, label: "nu~0.29"}]'
```

### Fit

```bash
uv run python scripts/fit.py
```

Fits beta to experimental peak density data via nonlinear least squares. By default runs a synthetic demo using NH model predictions as mock data. To fit real data, pass a `.npz` file with `lambda_list` and `pk_data` keys:

```bash
uv run python scripts/fit.py data_path=my_data.npz beta0=1.2
```

Output goes to `outputs/fit/`.

### Mesh figures

```bash
uv run python scripts/mesh_figures.py
```

Generates the experimental data figures (cavitation timing, peak velocity, mesh size, shear modulus, cavitation threshold pressure, surface tension vs TX-100). Output goes to `outputs/mesh_figures/`.

### Elastic

```bash
uv run python scripts/elastic.py
```

Plots cavity pressure curves for the incompressible neo-Hookean and strain-stiffening models alongside the surface energy density, and plots elastic energy per droplet volume using the Ronceray SI B12 expansion and Mooney-Rivlin formulas. Output goes to `outputs/elastic/`.

```bash
uv run python scripts/elastic.py elastic.epsilon_c=10 elastic.gamma=5.0
```

### Diffusion

```bash
uv run python scripts/diffusion.py
```

Computes the water concentration at the observation window as a function of time during solvent exchange (erfc semi-infinite slab model), plots the cavitation phase diagram (cavitated vs mesh-scale droplets as a function of gel and surfactant concentration), and optionally plots the ternary ethanol-decane-water phase diagram and gamma vs time. Output goes to `outputs/diffusion/`.

To plot gamma_ow vs time, provide a `.npz` file with `w_H2O` and `gamma_ow` arrays from tensiometry:

```bash
uv run python scripts/diffusion.py diffusion.gamma_data_path=data/gamma_calibration.npz
```

## Configuration

Default parameters live in `configs/`. Key parameters:

**`configs/solve.yaml`**
- `cases` — list of `{beta, lambda_, label}` cases to compare
- `x_end`, `n_eval` — integration domain and resolution
- `lambda_sweep` — beta and lambda range for the sweep plots
- `blur_wid` — Gaussian blur width in deformed radius units (default 0.5/1.3)

**`configs/fit.yaml`**
- `beta0`, `beta_lb`, `beta_ub` — initial guess and bounds for beta
- `blur_wid` — must match the blur width used to compute experimental peak densities
- `lambda_min` — filter out low-lambda points before fitting
- `n_fit_points` — downsample data to this many points

**`configs/elastic.yaml`**
- `elastic.E` — elastic modulus (normalised to 1 by default)
- `elastic.beta` — compressibility for the Ronceray expansion
- `elastic.gamma` — interfacial energy (normalised)
- `elastic.epsilon_c` — strain-stiffening onset parameter
- `elastic.epsilon_c_list` — list of epsilon_c values to sweep for pressure curves
- `elastic.n_list` — Mooney-Rivlin n values to compare

**`configs/diffusion.yaml`**
- `diffusion.D` — diffusivity in m²/s (default 1e-9, ethanol in water)
- `diffusion.window_dist` — observation distance from inlet in metres
- `diffusion.t_max` — maximum time in seconds
- `diffusion.C_threshold` — water content above which phase separation begins
- `diffusion.ternary_xlsx` — path to the binodal data xlsx
- `phase_diagram.cavitated` — 2D grid of cavitation observations

## Data

`data/water-etoh-decane.xlsx` — binodal curve data for the ethanol-decane-water ternary system (Skrzecz et al., *J. Phys. Chem. Ref. Data* 28(4), 1999, p. 1094), used by the ternary diagram plot.

## Output

All outputs are saved as `.png` figures. Fit results are additionally saved as `.npz` files loadable with `numpy.load`.
