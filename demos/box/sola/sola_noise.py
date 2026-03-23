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
import scipy.linalg

from pygeoinf import (
    EuclideanSpace,
    LinearOperator,
    CholeskySolver,
    GaussianMeasure,
)
from pygeoinf3d import (
    BoxDomain,
    Lebesgue3D,
    Function3D,
    FunctionToKernelAdapter,
    SOLAOperator3D,
)
from pygeoinf3d.providers.function_providers import GaussianProvider3D
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
    p.add_argument("--optimize",   action="store_true",      help="Use dense BLAS fast path for SOLA assembly.")
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
    OPTIMIZE  = args.optimize

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

    N_d = 35
    D   = EuclideanSpace(N_d)

    target_grid = np.array([[x, y, z]
                             for x in np.linspace(0.2, 0.8, 5)
                             for y in np.linspace(0.2, 0.8, 5)
                             for z in np.linspace(0.2, 0.8, 5)])
    N_p = len(target_grid)   # 125 (5×5×5 grid)
    P   = EuclideanSpace(N_p)

    centers_G = np.random.uniform(0.1, 0.9, size=(N_d, 3))
    sigmas_G  = np.random.uniform(0.12, 0.20, size=(N_d, 3))
    fp_G = GaussianProvider3D(box, centers=centers_G, sigmas=sigmas_G, normalized=True)
    kp_G = FunctionToKernelAdapter(fp_G, space)
    G    = SOLAOperator3D(space, D, kp_G)

    fp_T  = GaussianProvider3D(box, centers=target_grid, sigmas=0.08, normalized=True)
    kp_T  = FunctionToKernelAdapter(fp_T, space)
    T_op  = SOLAOperator3D(space, P, kp_T)

    print(f"  Forward operator G  : {type(G.domain).__name__} → R^{N_d}")
    print(f"  Target operator T   : {type(T_op.domain).__name__} → R^{N_p}")
    print(f"  Target locations    : {N_p} (2×2×2 grid)")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

    # ── Kernel visualisation ──────────────────────────────────────────────────
    if not NO_VIZ:
        t_sec = time.time()
        print("\n=== Kernel visualisation ===")

        g_kernels = [kp_G.get_kernel(i).kernel for i in range(N_d)]
        g_labels  = [
            f"G kernel {i}  ({centers_G[i,0]:.2f},{centers_G[i,1]:.2f},{centers_G[i,2]:.2f})"
            for i in range(N_d)
        ]
        fig_g = plot_function3d_axis_dropdown_slider(
            g_kernels, axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
            labels=g_labels, title="Sensitivity kernels  (select from dropdown)",
            show_plot=False,
        )
        _save_html(fig_g, OUT_DIR, "sensitivity_kernels.html")

        t_kernels = [kp_T.get_kernel(k).kernel for k in range(N_p)]
        t_labels  = [
            f"T k={k} ({target_grid[k,0]:.2f},{target_grid[k,1]:.2f},{target_grid[k,2]:.2f})"
            for k in range(N_p)
        ]
        fig_t = plot_function3d_axis_dropdown_slider(
            t_kernels, axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
            labels=t_labels, title="Target kernels  (select from dropdown)",
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

    print(f"  Noise std : {noise_std:.4f}")
    print(f"  SNR       : {np.linalg.norm(d_bar) / np.linalg.norm(d_tilde - d_bar):.1f}")

    if not NO_VIZ:
        fig_true = plot_function3d_axis_slider(
            m_true, axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
            title="True model  (z-axis slider)", show_plot=False,
        )
        _save_html(fig_true, OUT_DIR, "true_model.html")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

    # ── Data noise model ──────────────────────────────────────────────────────
    t_sec = time.time()
    print("\n=== Data noise model ===")
    C_D_matrix = noise_std**2 * np.eye(N_d)
    gaussian_D = GaussianMeasure.from_covariance_matrix(D, C_D_matrix, expectation=d_tilde)
    print(f"  Data measure expectation (first 5): {gaussian_D.expectation[:5].round(4)}")
    print(f"  Data covariance diagonal (var):     {C_D_matrix[0, 0]:.6f}")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

    # ── Shared helper: interpolate N_p values → Function3D ───────────────────
    _grid_ax = np.linspace(0.2, 0.8, 5)

    def _make_prop_fn(values):
        grid_vals = values.reshape(5, 5, 5)
        _interp = RegularGridInterpolator(
            (_grid_ax, _grid_ax, _grid_ax), grid_vals,
            method='linear', bounds_error=False, fill_value=0.0,
        )
        def _fn(x):
            pts_c = np.clip(np.atleast_2d(x), 0.25, 0.75)
            return _interp(pts_c).squeeze()
        return Function3D(box, evaluate_callable=_fn)

    if OPTIMIZE:
        print("\n[OPTIMIZE] Dense BLAS fast path enabled.\n")

        # ── [OPT] Kernel matrix pre-computation ──────────────────────────────
        t_sec = time.time()
        print("=== [OPT] Kernel matrix pre-computation ===")
        pts = box.structured_mesh(20, location="cell_centers")
        w = box.volume_weights(20, location="cell_centers")
        K_G = np.array([fp_G.get_function(i)(pts) for i in range(N_d)])
        K_T = np.array([fp_T.get_function(k)(pts) for k in range(N_p)])
        sqrt_w = np.sqrt(w)
        K_G_w = K_G * sqrt_w[None, :]
        K_T_w = K_T * sqrt_w[None, :]
        print(f"  K_G shape: {K_G.shape}, K_T shape: {K_T.shape}")
        print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

        # ── [OPT] SOLA construction (dense BLAS) ─────────────────────────────
        t_sec = time.time()
        print("\n=== [OPT] SOLA construction (dense BLAS) ===")
        Lambda_mat = K_G_w @ K_G_w.T
        W_mat = Lambda_mat + C_D_matrix
        L_chol = scipy.linalg.cholesky(W_mat, lower=True)
        M_mat = K_T_w @ K_G_w.T
        W_inv_mat = scipy.linalg.cho_solve((L_chol, True), np.eye(N_d))
        X_mat = M_mat @ W_inv_mat
        print(f"  X_mat shape: {X_mat.shape}")
        print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

        # ── [OPT] Model reconstruction ────────────────────────────────────────
        t_sec = time.time()
        print("\n=== [OPT] Model reconstruction ===")
        w_inv_d = scipy.linalg.cho_solve((L_chol, True), d_tilde)
        def _m_hat_fn(x):
            x = np.atleast_2d(x)
            vals = np.array([fp_G.get_function(i)(x) for i in range(N_d)])
            return (w_inv_d @ vals).squeeze()
        m_hat = Function3D(box, evaluate_callable=_m_hat_fn)
        d_hat = K_G @ (w * m_hat(pts))
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

        # ── [OPT] Property inference ──────────────────────────────────────────
        t_sec = time.time()
        print("\n=== [OPT] Property inference (dense) ===")
        p_mean = X_mat @ d_tilde
        P_cov_mat = X_mat @ C_D_matrix @ X_mat.T
        p_std = np.sqrt(np.diag(P_cov_mat))
        m_true_vals = m_true(pts)
        p_true = (K_T * w[None, :]) @ m_true_vals

        print("  Property  |  true    |  mean    |  std")
        print("  " + "-" * 46)
        for k in range(N_p):
            cx, cy, cz = target_grid[k]
            print(f"  k={k}  ({cx},{cy},{cz})  |  {p_true[k]:.4f}  |  {p_mean[k]:.4f}  |  {p_std[k]:.4f}")

        p_mean_field = _make_prop_fn(p_mean)
        p_2std_field = _make_prop_fn(2 * p_std)
        if not NO_VIZ:
            fig_prop = compare_function3d_axis_sliders(
                [p_mean_field, p_2std_field], axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
                titles=["SOLA estimate (mean)", "Uncertainty (2σ)"],
                show_plot=False, width=1000, height=500,
            )
            _save_html(fig_prop, OUT_DIR, "property_estimate.html")
        print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

        # ── [OPT] Unimodularity constraint ────────────────────────────────────
        t_sec = time.time()
        print("\n=== [OPT] Unimodularity constraint (dense) ===")
        v = K_G @ w
        u = scipy.linalg.cho_solve((L_chol, True), v)
        beta = float(np.dot(v, u))
        correction_mat = np.outer((X_mat @ v - np.ones(N_p)) / beta, u)
        X_uni_mat = X_mat - correction_mat

        check = X_uni_mat @ v
        print(f"  X_uni(G 1) — should equal ones: {check.round(4)}")

        p_mean_uni = X_uni_mat @ d_tilde
        C_P_uni_mat = X_uni_mat @ C_D_matrix @ X_uni_mat.T
        p_std_uni = np.sqrt(np.diag(C_P_uni_mat))

        p_mean_uni_field = _make_prop_fn(p_mean_uni)
        p_2std_uni_field = _make_prop_fn(2 * p_std_uni)
        if not NO_VIZ:
            fig_uni_mean = compare_function3d_axis_sliders(
                [p_mean_field, p_mean_uni_field], axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
                titles=["Mean — unconstrained X", "Mean — unimodular X"],
                show_plot=False, width=1000, height=500,
            )
            _save_html(fig_uni_mean, OUT_DIR, "unimodular_mean.html")
            fig_uni_unc = compare_function3d_axis_sliders(
                [p_2std_field, p_2std_uni_field], axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
                titles=["2σ — unconstrained X", "2σ — unimodular X"],
                show_plot=False, width=1000, height=500,
            )
            _save_html(fig_uni_unc, OUT_DIR, "unimodular_uncertainty.html")
        print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

        # ── [OPT] Resolving kernels ───────────────────────────────────────────
        t_sec = time.time()
        print("\n=== [OPT] Resolving kernels (dense) ===")

        def _make_rk_fn(coeffs_row):
            c = coeffs_row.copy()
            def _rk_fn(x):
                x = np.atleast_2d(x)
                vals = np.array([fp_G.get_function(i)(x) for i in range(N_d)])
                return (c @ vals).squeeze()
            return Function3D(box, evaluate_callable=_rk_fn)

        rk_fns = [_make_rk_fn(X_mat[k]) for k in range(N_p)]
        tgt_fns = [kp_T.get_kernel(k).kernel for k in range(N_p)]

        combined_fns = [fn for pair in zip(rk_fns, tgt_fns) for fn in pair]
        combined_labels = [
            label
            for k in range(N_p)
            for label in [
                f"Resolving k={k} ({target_grid[k,0]:.2f},{target_grid[k,1]:.2f},{target_grid[k,2]:.2f})",
                f"Target    k={k} ({target_grid[k,0]:.2f},{target_grid[k,1]:.2f},{target_grid[k,2]:.2f})",
            ]
        ]
        if not NO_VIZ:
            fig_rk = plot_function3d_axis_dropdown_slider(
                combined_fns, axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
                labels=combined_labels,
                title="Resolving & target kernels  (select from dropdown)",
                per_function_colorscale=True,
                show_plot=False,
            )
            _save_html(fig_rk, OUT_DIR, "resolving_kernels.html")
        print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

    else:
        # ── SOLA operator construction ────────────────────────────────────────────
        t_sec = time.time()
        print("\n=== SOLA operator construction ===")
        cholesky_solver = CholeskySolver(galerkin=True, parallel=True, n_jobs=-1)

        Lambda = G @ G.adjoint
        W_inv  = cholesky_solver(Lambda + gaussian_D.covariance)
        G_inv  = G.adjoint @ W_inv
        X      = T_op @ G_inv

        print("  SOLA operator X assembled.")
        print(f"    domain   : R^{N_d}  (data space)")
        print(f"    codomain : R^{N_p}  (property space)")
        print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

        # ── Model reconstruction ──────────────────────────────────────────────────
        t_sec = time.time()
        print("\n=== Model reconstruction ===")
        m_hat    = G_inv(d_tilde)
        residual = np.linalg.norm(G(m_hat) - d_tilde) / np.linalg.norm(d_tilde)
        print(f"  Relative data residual: {residual:.4f}")

        if not NO_VIZ:
            fig_recon = compare_function3d_axis_sliders(
                [m_true, m_hat], axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
                titles=["True model", "Reconstruction"],
                show_plot=False,
                width=1000, height=500,
            )
            _save_html(fig_recon, OUT_DIR, "reconstruction.html")
        print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

        # ── Property inference ────────────────────────────────────────────────────
        t_sec = time.time()
        print("\n=== Property inference ===")
        property_measure = gaussian_D.affine_mapping(operator=X)
        p_mean    = property_measure.expectation
        P_cov_mat = property_measure.covariance.matrix(
            dense=True, galerkin=False, parallel=True, n_jobs=-1
        )
        p_std  = np.sqrt(np.diag(P_cov_mat))
        p_true = T_op(m_true)

        print("  Property  |  true    |  mean    |  std")
        print("  " + "-" * 46)
        for k in range(N_p):
            cx, cy, cz = target_grid[k]
            print(f"  k={k}  ({cx},{cy},{cz})  |  {p_true[k]:.4f}  |  {p_mean[k]:.4f}  |  {p_std[k]:.4f}")

        p_mean_field = _make_prop_fn(p_mean)
        p_2std_field = _make_prop_fn(2 * p_std)

        if not NO_VIZ:
            fig_prop = compare_function3d_axis_sliders(
                [p_mean_field, p_2std_field], axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
                titles=["SOLA estimate (mean)", "Uncertainty (2σ)"],
                show_plot=False,
                width=1000, height=500,
            )
            _save_html(fig_prop, OUT_DIR, "property_estimate.html")
        print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

        # ── Unimodularity constraint ──────────────────────────────────────────────
        t_sec = time.time()
        print("\n=== Unimodularity constraint ===")
        ones = Function3D(box, evaluate_callable=lambda x: np.ones(np.atleast_2d(x).shape[0]))

        v    = G(ones)
        w    = np.ones(N_p)
        u    = W_inv(v)
        beta = float(np.dot(v, u))

        correction_matrix = np.outer((X(v) - w) / beta, u)
        correction        = LinearOperator.from_matrix(D, P, correction_matrix)
        X_unimodular      = X - correction

        check = X_unimodular(v)
        print(f"  X_uni(G 1) — should equal ones: {check.round(4)}")

        property_measure_uni = gaussian_D.affine_mapping(operator=X_unimodular)
        p_mean_uni    = property_measure_uni.expectation
        P_cov_uni_mat = property_measure_uni.covariance.matrix(
            dense=True, galerkin=False, parallel=True, n_jobs=-1
        )
        p_std_uni = np.sqrt(np.diag(P_cov_uni_mat))

        p_mean_uni_field = _make_prop_fn(p_mean_uni)
        p_2std_uni_field = _make_prop_fn(2 * p_std_uni)

        if not NO_VIZ:
            fig_uni_mean = compare_function3d_axis_sliders(
                [p_mean_field, p_mean_uni_field], axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
                titles=["Mean — unconstrained X", "Mean — unimodular X"],
                show_plot=False,
                width=1000, height=500,
            )
            _save_html(fig_uni_mean, OUT_DIR, "unimodular_mean.html")

            fig_uni_unc = compare_function3d_axis_sliders(
                [p_2std_field, p_2std_uni_field], axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
                titles=["2σ — unconstrained X", "2σ — unimodular X"],
                show_plot=False,
                width=1000, height=500,
            )
            _save_html(fig_uni_unc, OUT_DIR, "unimodular_uncertainty.html")
        print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

        # ── Resolving kernels ─────────────────────────────────────────────────────
        t_sec = time.time()
        print("\n=== Resolving kernels ===")
        A_adjoint = (X @ G).adjoint
        rk_fns  = [A_adjoint(np.eye(N_p)[k]) for k in range(N_p)]
        tgt_fns = [kp_T.get_kernel(k).kernel  for k in range(N_p)]

        combined_fns    = [fn for pair in zip(rk_fns, tgt_fns) for fn in pair]
        combined_labels = [
            label
            for k in range(N_p)
            for label in [
                f"Resolving k={k} ({target_grid[k,0]:.2f},{target_grid[k,1]:.2f},{target_grid[k,2]:.2f})",
                f"Target    k={k} ({target_grid[k,0]:.2f},{target_grid[k,1]:.2f},{target_grid[k,2]:.2f})",
            ]
        ]

        if not NO_VIZ:
            fig_rk = plot_function3d_axis_dropdown_slider(
                combined_fns, axis='z', n_slices=N_SLICES, resolution=RESOLUTION,
                labels=combined_labels,
                title="Resolving & target kernels  (select from dropdown)",
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
