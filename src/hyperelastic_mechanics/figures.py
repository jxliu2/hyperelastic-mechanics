"""Figure generation for neo-Hookean cavity and experimental data results.

All figures are saved to disk (no interactive display).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_density_profiles(
    profiles: list[tuple[np.ndarray, np.ndarray]],
    labels: list[str],
    save_path: str,
    xlim: tuple[float, float] | None = (1.0, 20.0),
    title: str = "",
) -> None:
    """Relative density vs deformed radius for multiple (beta, lambda) cases.

    Parameters
    ----------
    profiles : list of (rdim, den) arrays
    labels : legend labels, one per profile
    save_path : output file path
    xlim : x-axis limits
    title : figure title
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for (rdim, den), label in zip(profiles, labels, strict=True):
        ax.plot(rdim, den, label=label)

    ax.set_xlabel(r"$r(R)/R_0$")
    ax.set_ylabel("Relative density")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_radial_displacements(
    profiles: list[tuple[np.ndarray, np.ndarray]],
    labels: list[str],
    save_path: str,
    xlim: tuple[float, float] | None = (1.0, 20.0),
    title: str = "",
) -> None:
    """Deformed radius r(R)/R0 vs reference radius R/R0.

    Parameters
    ----------
    profiles : list of (xx, rdim) arrays where xx = R/R0
    labels : legend labels
    save_path : output file path
    xlim : x-axis limits on R/R0
    title : figure title
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for (xx, rdim), label in zip(profiles, labels, strict=True):
        ax.plot(xx, rdim, label=label)

    ax.set_xlabel(r"$R/R_0$")
    ax.set_ylabel(r"$r(R)/R_0$")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_blurred_profiles(
    rdim_hole_list: list[np.ndarray],
    den_blur_list: list[np.ndarray],
    rdim_list: list[np.ndarray],
    den_list: list[np.ndarray],
    lambda_list: list[float],
    save_path: str,
    xlim: tuple[float, float] | None = (0.0, 10.0),
    title: str = "",
) -> None:
    """Raw and blurred density profiles for a sweep of lambda values.

    Parameters
    ----------
    rdim_hole_list : deformed radius arrays including the hole (zero-density) region
    den_blur_list : blurred density arrays
    rdim_list : deformed radius arrays (no hole)
    den_list : unblurred density arrays
    lambda_list : hoop stretch values (used for color cycling)
    save_path : output file path
    xlim : x-axis limits
    title : figure title
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambda_list)))

    for i, lam in enumerate(lambda_list):
        ax.plot(rdim_list[i], den_list[i], color=colors[i], alpha=0.5, linewidth=1)
        ax.plot(
            rdim_hole_list[i],
            den_blur_list[i],
            color=colors[i],
            linewidth=1.5,
            label=rf"$\lambda={lam:.2f}$",
        )

    ax.set_xlabel(r"$r(R)/R_0$")
    ax.set_ylabel("Relative density")
    if xlim is not None:
        ax.set_xlim(xlim)
    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=min(lambda_list), vmax=max(lambda_list)),
    )
    fig.colorbar(sm, ax=ax, label=r"$\lambda$")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_peak_density(
    lambda_list: np.ndarray,
    pk_list: np.ndarray,
    pk_blur_list: np.ndarray,
    beta: float,
    save_path: str,
) -> None:
    """Peak density (raw and blurred) vs hoop stretch lambda.

    Parameters
    ----------
    lambda_list : hoop stretch values
    pk_list : peak density without blur
    pk_blur_list : peak density with optical blur
    beta : compressibility parameter (for title)
    save_path : output file path
    """
    nu = (1.0 - 1.0 / beta) / 2.0
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(lambda_list, pk_list, "o-", label="unblurred")
    ax.plot(lambda_list, pk_blur_list, "o-", label="blurred")
    ax.set_xlabel(r"Final $\lambda$")
    ax.set_ylabel("Peak density")
    ax.set_title(rf"$\beta={beta:.3f}$, $\nu={nu:.3f}$")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_fit_result(
    lambda_list_data: np.ndarray,
    pk_data: np.ndarray,
    lambda_list_fit: np.ndarray,
    pk_fit: np.ndarray,
    beta_fit: float,
    nu_fit: float,
    save_path: str,
) -> None:
    """Overlay of experimental data and fitted neo-Hookean model.

    Parameters
    ----------
    lambda_list_data : hoop stretch from data
    pk_data : peak density from data (normalized)
    lambda_list_fit : hoop stretch for model curve
    pk_fit : peak blurred density from model
    beta_fit : fitted beta
    nu_fit : fitted Poisson ratio
    save_path : output file path
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(lambda_list_data, pk_data, "o", label="data", zorder=3)
    ax.plot(
        lambda_list_fit,
        pk_fit,
        "-",
        label=rf"NH fit $\beta={beta_fit:.3f}$, $\nu={nu_fit:.3f}$",
    )
    ax.set_xlabel(r"$\lambda = s/\xi$")
    ax.set_ylabel("Peak density / background")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_mesh_figures(save_dir: str) -> None:
    """Experimental data figures from mesh-scale phase separation study.

    Plots hardcoded experimental results:
      1. Time until cavitation vs gel concentration
      2. Peak cavitation velocity vs gel concentration
      3. Mesh size vs gel concentration
      4. Shear modulus vs gel concentration
      5. Cavitation threshold pressure vs gel concentration
      6. Surface tension vs TX-100 concentration

    Parameters
    ----------
    save_dir : directory to write figure files
    """
    from pathlib import Path

    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    style = {"linewidth": 2, "markersize": 8}

    # 1. Time until cavitation
    c_gel = np.array([3, 8, 13, 20])
    t_cav = np.array(
        [13.5 - 21.5, 26.5 - 19.5, 43.5 - 20.0, (34 - 19) + (49 - 19)]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(c_gel, t_cav, "o-", **style)
    ax.set_xlabel(r"$c_\mathrm{gel}$ (mg/mL)", fontsize=14)
    ax.set_ylabel(r"$t_\mathrm{cav}$ (min)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out / "t_until_cavitation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out / 't_until_cavitation.png'}")

    # 2. Peak cavitation velocity
    c_gel_pct = np.array([0.3, 0.8, 1.3, 2.0])
    v_max = np.array(
        [(4.3 - 3.6) / 2, (3.5 - 2.87) / 2, (3.3 - 2.4) / 2, (4.27 - 3.08) / 2]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(c_gel_pct, v_max, ".-", linewidth=3, markersize=20)
    ax.set_xlabel(r"$c_\mathrm{gel}$ (% w/w)", fontsize=14)
    ax.set_ylabel(r"$v_\mathrm{max}$ (μm/s)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out / "cavitation_velocity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out / 'cavitation_velocity.png'}")

    # 3. Mesh size
    xi = np.array([3.7, 1.3, 0.72, 0.6])
    err = np.array([2.0, 1.0, 0.5, 0.25])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(c_gel_pct, xi, err, fmt=".-", linewidth=3, markersize=20)
    ax.set_xlim([0, 2.1])
    ax.set_xlabel(r"$c_\mathrm{gel}$ (% w/w)", fontsize=14)
    ax.set_ylabel(r"$\xi$ (μm)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out / "mesh_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out / 'mesh_size.png'}")

    # 4. Shear modulus
    G = np.array([123, 594, 7834, 11326])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.semilogy(c_gel, G, "o-", **style)
    ax.set_xlabel(r"$c_\mathrm{gel}$ (mg/mL)", fontsize=14)
    ax.set_ylabel(r"$G$ (Pa)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out / "shear_modulus.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out / 'shear_modulus.png'}")

    # 5. Cavitation threshold pressure gamma*DeltaR/R^2
    c_gel_f = np.array([3, 8, 13, 20], dtype=float)
    gamma = np.array([1.0, 4.0, 10.0, 18.0])
    xi_f = np.array([3.7, 1.3, 0.72, 0.6])
    R = xi_f.copy()
    deltaR = np.array([2.0, 1.0, 0.5, 0.25])
    pressure = gamma * deltaR / R**2
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(c_gel_f, pressure, "o-", **style)
    ax.set_xlabel(r"$c_\mathrm{gel}$ (mg/mL)", fontsize=14)
    ax.set_ylabel(r"$\gamma \Delta R / R^2$ (pressure)", fontsize=14)
    ax.set_title("Pressure required to fracture", fontsize=12)
    plt.tight_layout()
    plt.savefig(out / "cavitation_threshold_pressure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out / 'cavitation_threshold_pressure.png'}")

    # 6. Surface tension vs TX-100 concentration
    c_tx100 = np.array(
        [
            1.2e-2,
            1.5e-3,
            1.9e-4,
            2.5e-2,
            3.1e-3,
            3.9e-4,
            4.8e-5,
            5e-2,
            6.2e-3,
            7.8e-4,
            9.7e-5,
        ]
    )
    gamma_tx = np.array([2.06, 3.55, 9.99, 1.7, 3.02, 5.82, 17.9, 1.38, 2.57, 4.31, 14.1])
    sort_idx = np.argsort(c_tx100)[::-1]
    c_tx100 = c_tx100[sort_idx]
    gamma_tx = gamma_tx[sort_idx]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.semilogx(c_tx100, gamma_tx, ".-", linewidth=3, markersize=20)
    ax.set_xlabel("[TX-100] (v/v)", fontsize=14)
    ax.set_ylabel(r"$\gamma_{ow}$ (mN/m)", fontsize=14)
    ax.set_xlim([1e-5, 1e-1])
    plt.tight_layout()
    plt.savefig(out / "surface_tension_tx100.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out / 'surface_tension_tx100.png'}")



def plot_pressure_curves(
    lam_a: np.ndarray,
    curves: list[np.ndarray],
    labels: list[str],
    save_path: str,
    ylim: tuple[float, float] | None = (0.0, 10.0),
    title: str = "",
) -> None:
    """Cavity pressure vs hoop stretch for multiple constitutive models.

    Parameters
    ----------
    lam_a : hoop stretch array (x-axis).
    curves : list of pressure arrays, one per model.
    labels : legend labels.
    save_path : output file path.
    ylim : y-axis limits.
    title : figure title.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for p, label in zip(curves, labels, strict=True):
        ax.plot(lam_a, p, label=label)
    ax.set_xlabel(r"$\lambda_\theta$", fontsize=14)
    ax.set_ylabel("$p$ (normalised)", fontsize=14)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_elastic_energies(
    lam: np.ndarray,
    curves: list[np.ndarray],
    labels: list[str],
    save_path: str,
    ylim: tuple[float, float] | None = None,
    title: str = "",
) -> None:
    """Elastic energy per droplet volume vs hoop stretch.

    Parameters
    ----------
    lam : hoop stretch array.
    curves : list of energy density arrays.
    labels : legend labels.
    save_path : output file path.
    ylim : optional y-axis limits.
    title : figure title.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for f, label in zip(curves, labels, strict=True):
        ax.plot(lam, f, label=label)
    ax.set_xlabel(r"$\lambda$", fontsize=14)
    ax.set_ylabel(r"$f_\mathrm{el}$ (normalised)", fontsize=14)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_total_energy(
    lam_a: np.ndarray,
    e_nh: np.ndarray,
    e_ss: np.ndarray,
    e_surf: np.ndarray,
    save_path: str,
    ylim: tuple[float, float] | None = (0.0, 10.0),
    title: str = "",
) -> None:
    """NH pressure, SS pressure, surface energy, and totals vs hoop stretch.

    Parameters
    ----------
    lam_a : hoop stretch array.
    e_nh : NH pressure / elastic energy array.
    e_ss : strain-stiffening pressure array.
    e_surf : surface energy density array.
    save_path : output file path.
    ylim : y-axis limits.
    title : figure title.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lam_a, e_nh, label="Neo-Hookean", linewidth=2)
    ax.plot(lam_a, e_ss, label="Strain-stiffening", linewidth=2)
    ax.plot(lam_a, e_surf, label="Surface energy", linewidth=2)
    ax.plot(lam_a, e_nh + e_surf, "--", label="Total (NH)", linewidth=1.5)
    ax.plot(lam_a, e_ss + e_surf, "--", label="Total (SS)", linewidth=1.5)
    ax.set_xlabel(r"$\lambda_\theta$", fontsize=14)
    ax.set_ylabel("Energy density (normalised)", fontsize=14)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(fontsize=9)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")



def plot_concentration_vs_time(
    t_arr: np.ndarray,
    C_arr: np.ndarray,
    save_path: str,
    x_dist: float | None = None,
    highlight_points: list[tuple[float, float]] | None = None,
    title: str = "",
) -> None:
    """Water concentration at fixed distance from inlet vs time.

    Parameters
    ----------
    t_arr : time array (s).
    C_arr : concentration array.
    save_path : output file path.
    x_dist : observation distance in mm (for axis label), or None.
    highlight_points : list of (t, C) to mark with 'o'.
    title : figure title.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_arr, C_arr, linewidth=3)
    if highlight_points:
        for t_pt, c_pt in highlight_points:
            ax.plot(t_pt, c_pt, "o", markersize=12, linewidth=4,
                    color="#7B2D8B")
            ax.plot([0, t_pt], [c_pt, c_pt], "-", linewidth=3,
                    color="#7B2D8B")
    xlabel = "Time (s)"
    ylabel = r"$w_{\mathrm{H_2O}}$"
    if x_dist is not None:
        ylabel = rf"$w_{{\mathrm{{H_2O}}}}$ at {x_dist*1e3:.1f} mm"
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_gamma_vs_time(
    t_arr: np.ndarray,
    gamma_arr: np.ndarray,
    save_path: str,
    highlight_points: list[tuple[float, float]] | None = None,
    title: str = "",
) -> None:
    """Interfacial tension vs time during solvent exchange.

    Parameters
    ----------
    t_arr : time array (s).
    gamma_arr : gamma_ow array (mN/m).
    save_path : output file path.
    highlight_points : list of (t, gamma) to mark.
    title : figure title.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_arr, gamma_arr, linewidth=3)
    if highlight_points:
        for t_pt, g_pt in highlight_points:
            ax.plot(t_pt, g_pt, "o", markersize=10, linewidth=4,
                    color="#7B2D8B")
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel(r"$\gamma_{ow}$ (mN/m)", fontsize=14)
    ax.set_ylim([0, 20])
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_ternary_diagram(
    etoh: np.ndarray,
    decane: np.ndarray,
    water: np.ndarray,
    save_path: str,
    dilution_start: tuple[float, float, float] = (0.9643, 0.0357, 0.0),
    dilution_end: tuple[float, float, float] = (0.0, 0.0357, 0.9643),
    intersection: tuple[float, float, float] | None = (0.03, 0.3, 0.67),
) -> None:
    """Ternary phase diagram (ethanol-decane-water) with binodal curve.

    Converts (phi_A=etoh, phi_B=decane, phi_C=water) weight fractions to
    equilateral-triangle Cartesian coordinates via:
      x = (phi_A + 2*phi_B) / (2*(phi_A+phi_B+phi_C))
      y = sqrt(3)/2 * phi_A / (phi_A+phi_B+phi_C)

    Parameters
    ----------
    etoh, decane, water : weight fraction arrays for the binodal curve.
    save_path : output file path.
    dilution_start : (etoh, decane, water) at the start of the dilution path.
    dilution_end : (etoh, decane, water) at the end of the dilution path.
    intersection : (etoh, decane, water) at the binodal crossing, or None.
    """

    def _to_xy(phi_A, phi_B, phi_C):
        s = phi_A + phi_B + phi_C
        x = 0.5 * (phi_A + 2.0 * phi_B) / s
        y = np.sqrt(3.0) / 2.0 * phi_A / s
        return x, y

    fig, ax = plt.subplots(figsize=(6, 6))

    # triangle outline
    xt = [0.0, 0.5, 1.0, 0.0]
    yt = [0.0, np.sqrt(3.0) / 2.0, 0.0, 0.0]
    ax.plot(xt, yt, "-", linewidth=3, color="k")

    # binodal curve (interpolated)
    phi_a_arr = np.asarray(etoh, dtype=float)
    phi_b_arr = np.asarray(decane, dtype=float)
    phi_c_arr = np.asarray(water, dtype=float)
    xb, yb = _to_xy(phi_a_arr, phi_b_arr, phi_c_arr)
    sort_idx = np.argsort(xb)
    xb, yb = xb[sort_idx], yb[sort_idx]
    xq = np.linspace(xb.min(), xb.max(), 2000)
    yq = np.interp(xq, xb, yb)
    ax.plot(xq, yq, color="#D55E00", linewidth=3, label="Binodal")

    # dilution trajectory
    phi_a0 = np.array([dilution_start[0], dilution_end[0]])
    phi_b0 = np.array([dilution_start[1], dilution_end[1]])
    phi_c0 = np.array([dilution_start[2], dilution_end[2]])
    x0, y0 = _to_xy(phi_a0, phi_b0, phi_c0)
    ax.plot(x0, y0, "o-", linewidth=3, color="#44AA00", label="Dilution path")

    # binodal crossing
    if intersection is not None:
        xi, yi = _to_xy(*intersection)
        ax.plot(xi, yi, "o", markersize=12, color="#7B2D8B",
                linewidth=2, label="Intersection")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_cavitation_phase_diagram(
    c_surf: np.ndarray,
    c_gel: np.ndarray,
    cavitated: np.ndarray,
    save_path: str,
) -> None:
    """2D phase diagram: cavitated (red) vs not (blue) as function of
    surfactant concentration and gel concentration.

    Parameters
    ----------
    c_surf : surfactant concentration array (v/v), 1-D.
    c_gel : gel concentration array (mg/mL), 1-D.
    cavitated : 2-D boolean array, shape (len(c_gel), len(c_surf)).
                cavitated[i, j] = True if cavitated at (c_gel[i], c_surf[j]).
    save_path : output file path.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    C_surf, C_gel = np.meshgrid(c_surf, c_gel)
    for i in range(C_surf.size):
        cs = C_surf.ravel()[i]
        cg = C_gel.ravel()[i]
        cav = cavitated.ravel()[i]
        color = "r" if cav else "b"
        ax.plot(cs, cg, "o", color=color, markersize=10, linewidth=5)
    ax.set_xscale("log")
    ax.set_xlabel("TX-100 concentration (v/v)", fontsize=14)
    ax.set_ylabel(r"$c_\mathrm{gel}$ (mg/mL)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
