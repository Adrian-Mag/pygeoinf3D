"""
sola_noise.py — script companion to sola_noise.ipynb
====================================================
Runs the full SOLA demo and saves all figures to ./sola_noise_figures/.

Figure naming convention
------------------------
    sensitivity_kernels.html    – dropdown + slider
    target_kernels.html         – dropdown + slider
    true_model.html             – z-slider
    reconstruction.html         – true vs reconstruction
    property_estimate.html      – mean | 2σ
    unimodular_mean.html        – unconstrained vs unimodular mean
    unimodular_uncertainty.html – unconstrained vs unimodular 2σ
    unimodular_estimate.html    – unimodular mean | 2σ
    resolving_kernels.html      – two panels + shared slider

Usage
-----
    conda activate inferences
    python sola_noise.py [--out-dir PATH] [--n-slices N] [--resolution R]
"""

import argparse
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

import numpy as np  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from pygeoinf import LinearOperator  # noqa: E402
from pygeoinf3d import BoxDomain, Lebesgue3D, Function3D  # noqa: E402
from pygeoinf3d.providers.function_providers import (  # noqa: E402
    GaussianProvider3D,
    NormalModesProvider3D,
)
from pygeoinf3d.solvers import DenseVolumeSolaSolver  # noqa: E402
from pygeoinf3d.plot import (  # noqa: E402
    plot_function3d_axis_slider,
    compare_function3d_axis_sliders,
    plot_function3d_axis_dropdown_slider,
    compare_panels_axis_dropdown_sliders,
)


def _fmt_sec(sec: float) -> str:
    """Format seconds as a human-readable duration."""
    if sec < 1:
        return f"{sec*1000:.0f}ms"
    if sec < 60:
        return f"{sec:.2f}s"
    return f"{sec//60:.0f}m {sec%60:.1f}s"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the SOLA noise demo and save figures."
    )
    parser.add_argument(
        "--out-dir",
        default="sola_noise_figures",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--n-slices",
        type=int,
        default=10,
        help="Number of slider slices (default 10).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=80,
        help="Grid resolution per axis (default 80).",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip all visualisation (solver timing only).",
    )
    return parser.parse_args()


def _save_html(fig, out_dir: Path, name: str) -> None:
    """Write a Plotly figure as a self-contained interactive HTML file."""
    path = out_dir / name
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    print(f"  [saved] {path}")


def main():
    t_start = time.time()
    args = _parse_args()
    out_dir = Path(args.out_dir)
    n_slices = args.n_slices
    resolution = args.resolution
    no_viz = args.no_viz

    if not no_viz:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nFigures will be saved to: {out_dir.resolve()}\n")
    else:
        print("\n[--no-viz] Visualisation disabled — solver timing only.\n")

    np.random.seed(42)

    t_sec = time.time()
    print("=== Building spaces and operators ===")
    box = BoxDomain(np.zeros(3), np.ones(3))
    space = Lebesgue3D(box)

    quad_resolution = 20

    N_d = 1000
    centers_G = np.random.uniform(0.1, 0.9, size=(N_d, 3))
    fp_G = NormalModesProvider3D(
        box,
        dim=N_d,
        freq_range=(1, 3),
        gaussian_width_percent_range=(5, 30),
    )
    solver_G = DenseVolumeSolaSolver(box, fp_G, resolution=quad_resolution)
    G = solver_G.forward_operator(space)

    target_grid = np.array([
        [x, y, z]
        for x in np.linspace(0.1, 0.9, 5)
        for y in np.linspace(0.1, 0.9, 5)
        for z in np.linspace(0.1, 0.9, 5)
    ])
    N_p = len(target_grid)
    fp_T = GaussianProvider3D(
        box,
        centers=target_grid,
        sigmas=0.08,
        normalized=True,
    )
    solver_T = DenseVolumeSolaSolver(box, fp_T, resolution=quad_resolution)
    T_op = solver_T.forward_operator(space)

    print(f"  Forward operator G  : {space.__class__.__name__} -> R^{N_d}")
    print(f"  Target operator T   : {space.__class__.__name__} -> R^{N_p}")
    print(f"  Target locations    : {N_p} (5x5x5 grid)")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

    if not no_viz:
        t_sec = time.time()
        print("\n=== Kernel visualisation ===")

        g_kernels = [fp_G.get_function(i) for i in range(N_d)]
        g_labels = [
            f"G kernel {i}  ("
            f"{centers_G[i,0]:.2f},{centers_G[i,1]:.2f},{centers_G[i,2]:.2f})"
            for i in range(N_d)
        ]
        fig_g = plot_function3d_axis_dropdown_slider(
            g_kernels[0:10],
            axis="z",
            n_slices=n_slices,
            resolution=resolution,
            labels=g_labels[0:10],
            title="Sensitivity kernels  (select from dropdown)",
            show_plot=False,
        )
        _save_html(fig_g, out_dir, "sensitivity_kernels.html")

        t_kernels = [fp_T.get_function(k) for k in range(N_p)]
        t_labels = [
            f"T k={k} ("
            f"{target_grid[k,0]:.2f},{target_grid[k,1]:.2f},"
            f"{target_grid[k,2]:.2f})"
            for k in range(N_p)
        ]
        fig_t = plot_function3d_axis_dropdown_slider(
            t_kernels[0:10],
            axis="z",
            n_slices=n_slices,
            resolution=resolution,
            labels=t_labels[0:10],
            title="Target kernels  (select from dropdown)",
            show_plot=False,
        )
        _save_html(fig_t, out_dir, "target_kernels.html")
        print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")
    else:
        print("\n=== Kernel visualisation [SKIPPED — no-viz] ===")

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

    m_true = Function3D(box, evaluate_callable=true_model_fn)

    d_bar = G(m_true)
    noise_std = 0.05 * np.abs(d_bar).max()
    d_tilde = d_bar + np.random.normal(0.0, noise_std, size=N_d)

    print(f"  Noise std : {noise_std:.4f}")
    snr = np.linalg.norm(d_bar) / np.linalg.norm(d_tilde - d_bar)
    print(f"  SNR       : {snr:.1f}")

    C_D_matrix = noise_std**2 * np.eye(N_d)
    print(f"  Data noise std      : {noise_std:.2e}")
    print(f"  Data noise variance : {C_D_matrix[0, 0]:.2e}")

    if not no_viz:
        fig_true = plot_function3d_axis_slider(
            m_true, axis="z", n_slices=n_slices, resolution=resolution,
            title="True model  (z-axis slider)", show_plot=False,
        )
        _save_html(fig_true, out_dir, "true_model.html")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

    _grid_ax = np.unique(target_grid[:, 0])
    _n_ax = len(_grid_ax)

    def _make_prop_fn(values):
        grid_vals = values.reshape(_n_ax, _n_ax, _n_ax)
        _lo, _hi = _grid_ax[0], _grid_ax[-1]
        _interp = RegularGridInterpolator(
            (_grid_ax, _grid_ax, _grid_ax), grid_vals,
            method="linear", bounds_error=False, fill_value=0.0,
        )

        def _fn(x):
            pts_c = np.clip(np.atleast_2d(x), _lo, _hi)
            return _interp(pts_c).squeeze()

        return Function3D(box, evaluate_callable=_fn)

    t_sec = time.time()
    print("\n=== SOLA construction ===")
    X_mat = solver_G.compute_sola_matrix(solver_T, noise_cov=C_D_matrix)
    print("  SOLA matrix X assembled.")
    print(f"    shape    : {X_mat.shape}  (N_p x N_d)")
    print(f"    domain   : R^{N_d}  (data space)")
    print(f"    codomain : R^{N_p}  (property space)")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

    t_sec = time.time()
    print("\n=== Model reconstruction ===")
    m_hat = solver_G.reconstruct(d_tilde, noise_cov=C_D_matrix)
    residual = np.linalg.norm(G(m_hat) - d_tilde) / np.linalg.norm(d_tilde)
    print(f"  Relative data residual: {residual:.4f}")
    if not no_viz:
        fig_recon = compare_function3d_axis_sliders(
            [m_true, m_hat],
            axis="z",
            n_slices=n_slices,
            resolution=resolution,
            titles=["True model", "Reconstruction"],
            show_plot=False, width=1000, height=500,
        )
        _save_html(fig_recon, out_dir, "LSN_reconstruction.html")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

    t_sec = time.time()
    print("\n=== Property inference ===")
    p_mean = X_mat @ d_tilde
    P_cov_mat = X_mat @ C_D_matrix @ X_mat.T
    p_std = np.sqrt(np.diag(P_cov_mat))
    p_true = T_op(m_true)

    n_print = min(20, N_p)
    print(f"  (showing first {n_print} of {N_p} properties)")
    print("  Property  |  true    |  mean    |  std")
    print("  " + "-" * 46)
    for k in range(n_print):
        cx, cy, cz = target_grid[k]
        print(
            f"  k={k}  ({cx:.2f},{cy:.2f},{cz:.2f})"
            f"  |  {p_true[k]:.4f}  |  {p_mean[k]:.4f}  |  {p_std[k]:.4f}"
        )

    p_mean_field = _make_prop_fn(p_mean)
    p_2std_field = _make_prop_fn(2 * p_std)
    if not no_viz:
        fig_prop = compare_function3d_axis_sliders(
            [p_mean_field, p_2std_field],
            axis="z",
            n_slices=n_slices,
            resolution=resolution,
            titles=["SOLA estimate (mean)", "Uncertainty (2σ)"],
            show_plot=False, width=1000, height=500,
        )
        _save_html(fig_prop, out_dir, "SOLA_unconstrained.html")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

    t_sec = time.time()
    print("\n=== Unimodularity constraint ===")
    ones = Function3D(
        box,
        evaluate_callable=lambda x: np.ones(np.atleast_2d(x).shape[0]),
    )

    v = G(ones)
    w = np.ones(N_p)
    u = solver_G.solve(v, noise_cov=C_D_matrix)
    beta = float(np.dot(v, u))

    correction_matrix = np.outer((X_mat @ v - w) / beta, u)
    X_mat_uni = X_mat - correction_matrix

    check = X_mat_uni @ v
    print(
        "  X_uni(G 1) — should equal ones (first 5): "
        f"{np.asarray(check).round(4)[:5]}"
    )

    p_mean_uni = X_mat_uni @ d_tilde
    P_cov_uni = X_mat_uni @ C_D_matrix @ X_mat_uni.T
    p_std_uni = np.sqrt(np.diag(P_cov_uni))

    p_mean_uni_field = _make_prop_fn(p_mean_uni)
    p_2std_uni_field = _make_prop_fn(2 * p_std_uni)
    if not no_viz:
        fig_uni_mean = compare_function3d_axis_sliders(
            [p_mean_field, p_mean_uni_field],
            axis="z",
            n_slices=n_slices,
            resolution=resolution,
            titles=["Mean — unconstrained X", "Mean — unimodular X"],
            show_plot=False, width=1000, height=500,
        )
        _save_html(fig_uni_mean, out_dir, "SOLA_mean_comparison.html")

        fig_uni_unc = compare_function3d_axis_sliders(
            [p_2std_field, p_2std_uni_field],
            axis="z",
            n_slices=n_slices,
            resolution=resolution,
            titles=["2σ — unconstrained X", "2σ — unimodular X"],
            show_plot=False, width=1000, height=500,
        )
        _save_html(fig_uni_unc, out_dir, "SOLA_uncertainties_comparison.html")

        fig_uni_est = compare_function3d_axis_sliders(
            [p_mean_uni_field, p_2std_uni_field],
            axis="z",
            n_slices=n_slices,
            resolution=resolution,
            titles=[
                "SOLA estimate (mean) — unimodular X",
                "Uncertainty (2σ) — unimodular X",
            ],
            show_plot=False, width=1000, height=500,
        )
        _save_html(fig_uni_est, out_dir, "SOLA_unimodular.html")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

    t_sec = time.time()
    print("\n=== Resolving kernels ===")
    X_op = LinearOperator.from_matrix(G.codomain, T_op.codomain, X_mat)
    A_adjoint = (X_op @ G).adjoint

    def _e(n, k):
        """Cheap basis vector - avoids allocating an nxn identity matrix."""
        vec = np.zeros(n)
        vec[k] = 1.0
        return vec

    PREVIEW_IDX = [0, 31, 62, 93, 124]
    rk_fns = [A_adjoint(_e(N_p, k)) for k in PREVIEW_IDX]
    tgt_fns = [fp_T.get_function(k) for k in PREVIEW_IDX]

    rk_labels = [
        f"Resolving k={k} "
        f"({target_grid[k,0]:.2f},"
        f"{target_grid[k,1]:.2f},{target_grid[k,2]:.2f})"
        for k in PREVIEW_IDX
    ]
    tgt_labels = [
        f"Target k={k} "
        f"({target_grid[k,0]:.2f},"
        f"{target_grid[k,1]:.2f},{target_grid[k,2]:.2f})"
        for k in PREVIEW_IDX
    ]

    if not no_viz:
        fig_rk = compare_panels_axis_dropdown_sliders(
            panel_functions=[rk_fns, tgt_fns],
            panel_labels=[rk_labels, tgt_labels],
            panel_titles=["Resolving kernels", "Target kernels"],
            axis="z",
            n_slices=n_slices,
            resolution=resolution,
            show_plot=False,
        )
        _save_html(fig_rk, out_dir, "SOLA_resolving_vs_target_kernels.html")
    print(
        "  Each resolving kernel should be concentrated near its matching "
        "target centre."
    )
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]")

    t_total = time.time() - t_start
    if not no_viz:
        print(f"\n✓ All figures saved to {out_dir.resolve()}")
    print(f"✓ Total runtime: {_fmt_sec(t_total)}")
    print(
        "  Open any .html file in a browser to interact with the Plotly "
        "figures."
    )


if __name__ == "__main__":
    main()
