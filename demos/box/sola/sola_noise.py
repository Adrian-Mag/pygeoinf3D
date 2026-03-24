"""
sola_noise.py — surrogate script for sola_noise.ipynb
======================================================
Runs the full SOLA demo and saves all figures to ./sola_noise_figures/.

Figure naming convention
------------------------
  sensitivity_kernels.html         – Plotly interactive (dropdown + slider)
  target_kernels.html              – Plotly interactive (dropdown + slider)
  true_model.html                  – Plotly interactive (z-slider)
  reconstruction.html              – Plotly interactive (side-by-side slider)
  property_estimate.html           – Plotly interactive (mean | 2σ side-by-side)
  unimodular_mean.html             – Plotly interactive (unconstrained vs unimodular mean)
  unimodular_uncertainty.html      – Plotly interactive (unconstrained vs unimodular 2σ)
  resolving_kernels.html           – Plotly interactive (dropdown + slider)

Usage
-----
    conda activate inferences3
    python sola_noise.py [--out-dir PATH] [--n-slices N] [--resolution R]
"""

import argparse
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[3]

# Allow running the script directly from the demo folder without requiring
# editable installs of the sibling packages on the current machine.
for repo_root in [WORKSPACE_ROOT / "pygeoinf", WORKSPACE_ROOT / "pygeoinf3D"]:
    repo_root_str = str(repo_root)
    if repo_root.exists() and repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — no GUI needed
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from pygeoinf import (
    LinearOperator,
)
from pygeoinf3d import (
    BoxDomain,
    Lebesgue3D,
    Function3D,
)
from pygeoinf3d.providers.function_providers import GaussianProvider3D
from pygeoinf3d.solvers import DenseVolumeSolaSolver
from pygeoinf3d.plot import (
    plot_function3d_axis_slider,
    compare_function3d_axis_sliders,
    plot_function3d_axis_dropdown_slider,
)

# ─────────────────────────── Utilities ──────────────────────────────────────

def _fmt_sec(sec: float) -> str:
    """Format seconds as a human-readable duration."""
    if sec < 1:
        return f"{sec*1000:.0f}ms"
    elif sec < 60:
        return f"{sec:.2f}s"
    else:
        return f"{sec//60:.0f}m {sec%60:.1f}s"


# ─────────────────────────── CLI ─────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Run the SOLA noise demo and save figures.")
    p.add_argument("--out-dir",    default="sola_noise_figures", help="Output directory for figures.")
    p.add_argument("--n-slices",   type=int,   default=10,  help="Number of slider slices (default 10).")
    p.add_argument("--resolution", type=int,   default=80,  help="Grid resolution per axis (default 80).")
    p.add_argument("--no-viz",     action="store_true",      help="Skip all visualisation (solver timing only).")
    p.add_argument("--memory-budget-gb", type=float, default=2.0, help="Memory budget for DenseVolumeSolaSolver K cache, in GB (default 2).")
    return p.parse_args()


def _save_html(fig, out_dir: Path, name: str) -> None:
    """Write a Plotly figure as a self-contained interactive HTML file."""
    path = out_dir / name
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    print(f"  [saved] {path}")


def _save_mpl(fig, out_dir: Path, name: str) -> None:
    """Write a matplotlib figure as a high-resolution PNG."""
    path = out_dir / name
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ─────────────────────────── main ────────────────────────────────────────────

def main():
    t_start = time.time()
    args = _parse_args()
    OUT_DIR   = Path(args.out_dir)
    N_SLICES  = args.n_slices
    RESOLUTION = args.resolution
    NO_VIZ    = args.no_viz

    if not NO_VIZ:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\nFigures will be saved to: {OUT_DIR.resolve()}\n")
    else:
        print("\n[--no-viz] Visualisation disabled — solver timing only.\n")

    np.random.seed(42)

    # ── Spaces & operators ────────────────────────────────────────────────────
    t_sec = time.time()
    print("=== Building spaces and operators ===")
    box   = BoxDomain(np.zeros(3), np.ones(3))
    space = Lebesgue3D(box)

    QUAD_RESOLUTION = 20  # quadrature grid for DenseVolumeSolaSolver
    N_d = 100
    target_grid = np.array([[x, y, z]
                             for x in np.linspace(0.1, 0.9, 10)
                             for y in np.linspace(0.1, 0.9, 10)
                             for z in np.linspace(0.1, 0.9, 10)])
    N_p = len(target_grid)   # 1000 (10×10×10 grid)

    centers_G = np.random.uniform(0.1, 0.9, size=(N_d, 3))
    sigmas_G  = np.random.uniform(0.05, 0.10, size=(N_d, 3))
    _memory_budget = int(args.memory_budget_gb * 1024**3)
    fp_G     = GaussianProvider3D(box, centers=centers_G, sigmas=sigmas_G, normalized=True)
    solver_G = DenseVolumeSolaSolver(box, fp_G, resolution=QUAD_RESOLUTION, memory_budget_bytes=_memory_budget)
    G        = solver_G.forward_operator(space)   # LinearOperator: M → R^N_d
    D        = G.codomain                         # EuclideanSpace(N_d)

    fp_T     = GaussianProvider3D(box, centers=target_grid, sigmas=0.08, normalized=True)
    solver_T = DenseVolumeSolaSolver(box, fp_T, resolution=QUAD_RESOLUTION, memory_budget_bytes=_memory_budget)
    T_op     = solver_T.forward_operator(space)   # LinearOperator: M → R^N_p
    P        = T_op.codomain                      # EuclideanSpace(N_p)

    print(f"  Forward operator G  : {space.__class__.__name__} → R^{N_d}")
    print(f"  Target operator T   : {space.__class__.__name__} → R^{N_p}")
    print(f"  Target locations    : {N_p} (10×10×10 grid)")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

    # ── Kernel visualisation ──────────────────────────────────────────────────
    if not NO_VIZ:
        t_sec = time.time()
        print("\n=== Kernel visualisation ===")

        N_PLOT = 10  # max kernels to show in each dropdown
        g_idx = np.round(np.linspace(0, N_d - 1, min(N_PLOT, N_d))).astype(int)
        g_kernels = [fp_G.get_function(i) for i in g_idx]
        g_labels  = [
            f"G kernel {i}  ({centers_G[i,0]:.2f},{centers_G[i,1]:.2f},{centers_G[i,2]:.2f})"
            for i in g_idx
        ]
        fig_g = plot_function3d_axis_dropdown_slider(
            g_kernels, axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
            labels=g_labels, title=f"Sensitivity kernels  ({len(g_idx)} of {N_d}, select from dropdown)",
            show_plot=False,
        )
        _save_html(fig_g, OUT_DIR, "sensitivity_kernels.html")

        t_idx = np.round(np.linspace(0, N_p - 1, min(N_PLOT, N_p))).astype(int)
        t_kernels = [fp_T.get_function(k) for k in t_idx]
        t_labels  = [
            f"T k={k} ({target_grid[k,0]:.2f},{target_grid[k,1]:.2f},{target_grid[k,2]:.2f})"
            for k in t_idx
        ]
        fig_t = plot_function3d_axis_dropdown_slider(
            t_kernels, axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
            labels=t_labels, title=f"Target kernels  ({len(t_idx)} of {N_p}, select from dropdown)",
            show_plot=False,
        )
        _save_html(fig_t, OUT_DIR, "target_kernels.html")
        print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")
    else:
        print("\n=== Kernel visualisation [SKIPPED — no-viz] ===")

    # ── Synthetic data ────────────────────────────────────────────────────────
    t_sec = time.time()
    print("\n=== Synthetic data ===")

    def gaussian_blob(x, center, sigma):
        diff = x - center[None, :]
        return np.exp(-0.5 * np.sum((diff / sigma) ** 2, axis=1))

    def true_model_fn(x):
        if np.ndim(x) == 1:
            x = x[None, :]
        vals = (
            + 1.5 * gaussian_blob(x, np.array([0.30, 0.30, 0.50]), 0.12)
            + 1.0 * gaussian_blob(x, np.array([0.70, 0.65, 0.55]), 0.10)
            - 0.8 * gaussian_blob(x, np.array([0.55, 0.40, 0.45]), 0.08)
        )
        return vals.squeeze()

    m_true    = Function3D(box, evaluate_callable=true_model_fn)
    d_bar     = G(m_true)
    noise_std = 0.05 * np.abs(d_bar).max()
    d_tilde   = d_bar + np.random.normal(0.0, noise_std, size=N_d)

    # Data noise covariance — replace with any SPD matrix for correlated noise.
    # (scalar noise_std passed as float to noise_cov uses the fast scalar path;
    #  a full C_D_matrix exercises the general dense path.)
    C_D_matrix = noise_std ** 2 * np.eye(N_d)

    print(f"  Noise std : {noise_std:.4f}")
    print(f"  SNR       : {np.linalg.norm(d_bar) / np.linalg.norm(d_tilde - d_bar):.1f}")

    if not NO_VIZ:
        fig_true = plot_function3d_axis_slider(
            m_true, axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
            title="True model  (z-axis slider)", show_plot=False,
        )
        _save_html(fig_true, OUT_DIR, "true_model.html")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

    # ── Shared helper: interpolate N_p values → Function3D ───────────────────
    # Grid axes derived from the actual target_grid (works for any n×n×n layout)
    _grid_ax = np.unique(target_grid[:, 0])
    _n_ax    = len(_grid_ax)

    def _make_prop_fn(values):
        grid_vals = values.reshape(_n_ax, _n_ax, _n_ax)
        _lo, _hi  = _grid_ax[0], _grid_ax[-1]
        _interp   = RegularGridInterpolator(
            (_grid_ax, _grid_ax, _grid_ax), grid_vals,
            method='linear', bounds_error=False, fill_value=0.0,
        )
        def _fn(x):
            pts_c = np.clip(np.atleast_2d(x), _lo, _hi)
            return _interp(pts_c).squeeze()
        return Function3D(box, evaluate_callable=_fn)

    # ── SOLA construction ─────────────────────────────────────────────────────
    t_sec = time.time()
    print("\n=== SOLA construction ===")
    X_mat = solver_G.compute_sola_matrix(solver_T, noise_cov=C_D_matrix)  # (N_p, N_d)
    print(f"  SOLA matrix : R^{N_d}  →  R^{N_p}")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

    # ── Model reconstruction ──────────────────────────────────────────────────────
    t_sec = time.time()
    print("\n=== Model reconstruction ===")
    m_hat    = solver_G.reconstruct(d_tilde, noise_cov=C_D_matrix)
    d_hat    = G(m_hat)
    residual = np.linalg.norm(d_hat - d_tilde) / np.linalg.norm(d_tilde)
    print(f"  Relative data residual: {residual:.4f}")
    if not NO_VIZ:
        fig_recon = compare_function3d_axis_sliders(
            [m_true, m_hat], axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
            titles=["True model", "Reconstruction"],
            show_plot=False, width=1000, height=500,
        )
        _save_html(fig_recon, OUT_DIR, "reconstruction.html")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

    # ── Property inference ────────────────────────────────────────────────────────
    t_sec = time.time()
    print("\n=== Property inference ===")
    p_mean    = X_mat @ d_tilde                            # (N_p,)
    P_cov_mat = X_mat @ C_D_matrix @ X_mat.T               # X C_D X^T  (general formula)
    p_std     = np.sqrt(np.diag(P_cov_mat))
    p_true    = T_op(m_true)                               # (N_p,)

    N_PRINT = min(20, N_p)
    print(f"  (showing first {N_PRINT} of {N_p} properties)")
    print("  Property  |  true    |  mean    |  std")
    print("  " + "-" * 46)
    for k in range(N_PRINT):
        cx, cy, cz = target_grid[k]
        print(f"  k={k}  ({cx:.2f},{cy:.2f},{cz:.2f})  |  {p_true[k]:.4f}  |  {p_mean[k]:.4f}  |  {p_std[k]:.4f}")

    p_mean_field = _make_prop_fn(p_mean)
    p_2std_field = _make_prop_fn(2 * p_std)
    if not NO_VIZ:
        fig_prop = compare_function3d_axis_sliders(
            [p_mean_field, p_2std_field], axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
            titles=["SOLA estimate (mean)", "Uncertainty (2\u03c3)"],
            show_plot=False, width=1000, height=500,
        )
        _save_html(fig_prop, OUT_DIR, "property_estimate.html")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

    # ── Unimodularity constraint ──────────────────────────────────────────────────
    t_sec = time.time()
    print("\n=== Unimodularity constraint ===")
    ones = Function3D(box, evaluate_callable=lambda x: np.ones(np.atleast_2d(x).shape[0]))
    v    = G(ones)                                         # G applied to constant field → R^N_d
    w    = np.ones(N_p)                                    # desired unimodular output
    u    = solver_G.solve(v, noise_cov=C_D_matrix)         # (C_D + GG*)^{-1} v → R^N_d
    beta = float(np.dot(v, u))                             # ⟨v, u⟩ scalar

    correction_matrix = np.outer((X_mat @ v - w) / beta, u)    # (N_p, N_d) rank-1
    X_mat_uni = X_mat - correction_matrix

    check = X_mat_uni @ v
    print(f"  X_uni(G 1) \u2014 should equal ones (first 5): {np.asarray(check).round(4)[:5]}")

    p_mean_uni = X_mat_uni @ d_tilde
    P_cov_uni  = X_mat_uni @ C_D_matrix @ X_mat_uni.T
    p_std_uni  = np.sqrt(np.diag(P_cov_uni))

    p_mean_uni_field = _make_prop_fn(p_mean_uni)
    p_2std_uni_field = _make_prop_fn(2 * p_std_uni)
    if not NO_VIZ:
        fig_uni_mean = compare_function3d_axis_sliders(
            [p_mean_field, p_mean_uni_field], axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
            titles=["Mean \u2014 unconstrained X", "Mean \u2014 unimodular X"],
            show_plot=False, width=1000, height=500,
        )
        _save_html(fig_uni_mean, OUT_DIR, "unimodular_mean.html")
        fig_uni_unc = compare_function3d_axis_sliders(
            [p_2std_field, p_2std_uni_field], axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
            titles=["2\u03c3 \u2014 unconstrained X", "2\u03c3 \u2014 unimodular X"],
            show_plot=False, width=1000, height=500,
        )
        _save_html(fig_uni_unc, OUT_DIR, "unimodular_uncertainty.html")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

    # ── Resolving kernels ─────────────────────────────────────────────────────────
    t_sec = time.time()
    print("\n=== Resolving kernels ===")

    # A_k(x) = (XG)*(e_k)(x) — resolving kernel for property k
    X_op = LinearOperator.from_matrix(D, P, X_mat)
    A_adjoint = (X_op @ G).adjoint   # P → M

    # ~10 evenly-spaced pairs to keep the dropdown manageable
    N_PLOT_RK  = 10
    rk_idx     = np.round(np.linspace(0, N_p - 1, min(N_PLOT_RK, N_p))).astype(int)
    rk_fns  = [A_adjoint(np.eye(N_p)[k]) for k in rk_idx]
    tgt_fns = [fp_T.get_function(k)      for k in rk_idx]

    combined_fns    = [fn for pair in zip(rk_fns, tgt_fns) for fn in pair]
    combined_labels = [
        label
        for k in rk_idx
        for label in [
            f"Resolving k={k} ({target_grid[k,0]:.2f},{target_grid[k,1]:.2f},{target_grid[k,2]:.2f})",
            f"Target    k={k} ({target_grid[k,0]:.2f},{target_grid[k,1]:.2f},{target_grid[k,2]:.2f})",
        ]
    ]
    if not NO_VIZ:
        fig_rk = plot_function3d_axis_dropdown_slider(
            combined_fns, axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
            labels=combined_labels,
            title=f"Resolving & target kernels  ({len(rk_idx)} of {N_p}, select from dropdown)",
            per_function_colorscale=True,
            show_plot=False,
        )
        _save_html(fig_rk, OUT_DIR, "resolving_kernels.html")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

    # ── Summary ───────────────────────────────────────────────────────────────
    t_total = time.time() - t_start
    if not NO_VIZ:
        print(f"\n✓ All figures saved to {OUT_DIR.resolve()}")
    print(f"✓ Total runtime: {_fmt_sec(t_total)}")
    print("  Open any .html file in a browser to interact with the Plotly figures.")


if __name__ == "__main__":
    main()
