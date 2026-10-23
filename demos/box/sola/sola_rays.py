"""
sola_rays.py — SOLA with Sobolev H² Model Space and Straight-Ray Data
======================================================================
Runs the full SOLA demo using straight-ray path-integral measurements
and a Sobolev H² model space.  Saves all figures to ./sola_rays_figures/.

Figure naming convention
------------------------
    ray_geometry.html           – Plotly 3D ray/source/station/target scatter
    true_model.html             – z-slider of the true model
    representers.html           – dropdown of 5 Riesz representers (z-slider)
    total_sensitivity.html      – z-slider of S(x) = Σ rᵢ(x)
    property_estimate.html      – SOLA mean | 2σ (unconstrained)
    unimodular_mean.html        – unconstrained vs unimodular mean
    unimodular_uncertainty.html – unconstrained vs unimodular 2σ
    unimodular_estimate.html    – unimodular mean | unimodular 2σ
    resolving_kernels.html      – resolving kernels vs target Gaussians

Usage
-----
    conda activate inferences
    python sola_rays.py [--out-dir PATH] [--n-sources N] [--resolution R]
    python sola_rays.py --no-viz          # timing only, skip all figures
"""

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[3]

# Allow running the script directly without requiring editable installs of the
# sibling packages on the current machine.
for repo_root in [WORKSPACE_ROOT / "pygeoinf", WORKSPACE_ROOT / "pygeoinf3D"]:
    repo_root_str = str(repo_root)
    if repo_root.exists() and repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

import numpy as np  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from pygeoinf3d import BoxDomain, Lebesgue3D, Function3D, SOLAOperator3D  # noqa: E402
from pygeoinf import EuclideanSpace  # noqa: E402
from pygeoinf3d.providers.tensor_product import (  # noqa: E402
    BoxLaplacianSpectrumProvider,
)
from pygeoinf3d.operators.bessel_sobolev_box import BesselSobolevBox  # noqa: E402
from pygeoinf3d.spaces.sobolev import Sobolev3D  # noqa: E402
from pygeoinf3d.spaces.forms import PathIntegralFunctional, VolumeKernel3D  # noqa: E402
from pygeoinf3d.core.paths import Path3D  # noqa: E402
from pygeoinf3d.providers.function_providers import GaussianProvider3D  # noqa: E402
from pygeoinf3d.plot import (  # noqa: E402
    plot_function3d_axis_slider,
    compare_function3d_axis_sliders,
    compare_panels_axis_dropdown_sliders,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_sec(sec: float) -> str:
    """Format seconds as a human-readable duration."""
    if sec < 1:
        return f"{sec * 1000:.0f}ms"
    if sec < 60:
        return f"{sec:.2f}s"
    return f"{sec // 60:.0f}m {sec % 60:.1f}s"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run SOLA with Sobolev H² model space and straight-ray data."
    )
    parser.add_argument(
        "--out-dir",
        default="sola_rays_figures",
        help="Output directory for figures (default: sola_rays_figures).",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip all visualisation (solver timing only).",
    )
    parser.add_argument(
        "--n-sources",
        type=int,
        default=30,
        help="Number of seismic sources (default: 30).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=80,
        help="Grid resolution per axis for visualisation (default: 80).",
    )
    return parser.parse_args()


def _save_html(fig, out_dir: Path, name: str) -> None:
    """Write a Plotly figure as a self-contained interactive HTML file."""
    path = out_dir / name
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    print(f"  [saved] {path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    args = _parse_args()
    out_dir = Path(args.out_dir)
    N_d = args.n_sources
    resolution = args.resolution
    no_viz = args.no_viz

    if not no_viz:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nFigures will be saved to: {out_dir.resolve()}\n", flush=True)
    else:
        print("\n[--no-viz] Visualisation disabled — solver timing only.\n", flush=True)

    # ===================================================================
    # 1. Sobolev H² model space
    # ===================================================================
    t_sec = time.time()
    print("=== 1. Building Sobolev H² model space ===", flush=True)

    np.random.seed(42)
    box = BoxDomain(np.zeros(3), np.ones(3))
    base_space = Lebesgue3D(box)
    spec_provider = BoxLaplacianSpectrumProvider(
        box, truncation=(8, 8, 8), bc_type="neumann"
    )
    W = BesselSobolevBox(base_space, spec_provider, shift=1.0, order=2.0)
    W_inv = BesselSobolevBox(base_space, spec_provider, shift=1.0, order=-2.0)
    sobolev = Sobolev3D(base_space, W, W_inv)

    n_modes = 8 ** 3
    print(f"  Box              : [0,1]^3", flush=True)
    print(f"  Spectral trunc.  : (8,8,8)  →  {n_modes} modes", flush=True)
    print(f"  Weight operator  : BesselSobolevBox  shift=1.0, order=2.0 (H²)", flush=True)
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]", flush=True)

    # ===================================================================
    # 2. Ray geometry
    # ===================================================================
    t_sec = time.time()
    print("\n=== 2. Building ray geometry ===", flush=True)

    N_stations = 10
    np.random.seed(42)
    station_xy = np.random.uniform(0.05, 0.95, size=(N_stations, 2))
    stations = np.column_stack([station_xy, np.ones(N_stations)])

    sources = np.random.uniform(0.05, 0.95, size=(N_d, 3))
    station_idx = np.random.randint(0, N_stations, size=N_d)

    paths = [
        Path3D.from_polyline(np.array([sources[i], stations[station_idx[i]]]))
        for i in range(N_d)
    ]

    print(f"  Stations : {N_stations} at z=1 surface (random x,y in [0.05,0.95])", flush=True)
    print(f"  Sources  : {N_d} random in [0.05,0.95]^3", flush=True)
    print(f"  Rays     : {len(paths)} (each source → its assigned station)", flush=True)
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]", flush=True)

    if not no_viz:
        t_sec = time.time()
        print("\n  Building ray geometry figure...", flush=True)

        # Build the 3×3×3 target grid for the geometry plot
        _tgx = np.linspace(0.1, 0.9, 3)
        target_grid_preview = np.array(
            [[x, y, z] for x in _tgx for y in _tgx for z in _tgx]
        )

        fig_geo = go.Figure()
        for i in range(N_d):
            src = sources[i]
            sta = stations[station_idx[i]]
            fig_geo.add_trace(
                go.Scatter3d(
                    x=[src[0], sta[0]],
                    y=[src[1], sta[1]],
                    z=[src[2], sta[2]],
                    mode="lines",
                    line=dict(color="steelblue", width=2),
                    showlegend=(i == 0),
                    name="Rays" if i == 0 else None,
                    legendgroup="rays",
                )
            )
        fig_geo.add_trace(
            go.Scatter3d(
                x=stations[:, 0],
                y=stations[:, 1],
                z=stations[:, 2],
                mode="markers",
                marker=dict(color="red", size=7, symbol="square"),
                name="Stations",
            )
        )
        fig_geo.add_trace(
            go.Scatter3d(
                x=sources[:, 0],
                y=sources[:, 1],
                z=sources[:, 2],
                mode="markers",
                marker=dict(color="limegreen", size=4, symbol="circle"),
                name="Sources",
            )
        )
        fig_geo.add_trace(
            go.Scatter3d(
                x=target_grid_preview[:, 0],
                y=target_grid_preview[:, 1],
                z=target_grid_preview[:, 2],
                mode="markers",
                marker=dict(color="orange", size=4, symbol="diamond"),
                name="Target locations",
            )
        )
        fig_geo.update_layout(
            title="Ray geometry: sources → stations, and SOLA target locations",
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                zaxis=dict(range=[0, 1]),
            ),
        )
        _save_html(fig_geo, out_dir, "ray_geometry.html")
        print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]", flush=True)

    # ===================================================================
    # 3. Forward forms, true model, and synthetic data
    # ===================================================================
    t_sec = time.time()
    print("\n=== 3. Forward forms and true model ===", flush=True)

    forms_G = [PathIntegralFunctional(sobolev, p, n_quad=200) for p in paths]
    print(f"  Forward forms G: {len(forms_G)} PathIntegralFunctional on Sobolev3D", flush=True)

    def _true_model_callable(x):
        if np.ndim(x) == 1:
            x = x[None, :]
        diff = lambda c: x - c[None, :]  # noqa: E731
        vals = (
            + 1.5 * np.exp(-0.5 * np.sum((diff(np.array([0.30, 0.30, 0.50])) / 0.12) ** 2, axis=1))
            + 1.0 * np.exp(-0.5 * np.sum((diff(np.array([0.70, 0.65, 0.55])) / 0.10) ** 2, axis=1))
            - 0.8 * np.exp(-0.5 * np.sum((diff(np.array([0.55, 0.40, 0.45])) / 0.08) ** 2, axis=1))
        )
        return vals.squeeze()

    m_true = Function3D(box, evaluate_callable=_true_model_callable, name="true model")

    d_bar = np.array([f(m_true) for f in forms_G])
    noise_std = 0.05 * np.max(np.abs(d_bar))
    C_D = noise_std ** 2 * np.eye(N_d)
    rng = np.random.default_rng(123)
    d_tilde = d_bar + rng.normal(0, noise_std, size=N_d)

    snr = np.linalg.norm(d_bar) / np.linalg.norm(d_tilde - d_bar)
    print(f"  Noise std  : {noise_std:.4e}", flush=True)
    print(f"  SNR        : {snr:.1f}", flush=True)
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]", flush=True)

    if not no_viz:
        fig_true = plot_function3d_axis_slider(
            m_true,
            axis="z",
            n_slices=10,
            resolution=resolution,
            title="True model  (z-axis slider)",
            show_plot=False,
        )
        _save_html(fig_true, out_dir, "true_model.html")

    # ===================================================================
    # 4. SOLA matrix construction
    # ===================================================================
    t_sec = time.time()
    print("\n=== 4. SOLA matrix construction ===", flush=True)

    # --- Target forms: 3×3×3 Gaussian volume kernels ---
    target_grid = np.array(
        [
            [x, y, z]
            for x in np.linspace(0.1, 0.9, 3)
            for y in np.linspace(0.1, 0.9, 3)
            for z in np.linspace(0.1, 0.9, 3)
        ]
    )
    N_p = len(target_grid)  # 27
    fp_T = GaussianProvider3D(box, centers=target_grid, sigmas=0.10, normalized=True)
    forms_T = [VolumeKernel3D(base_space, fp_T.get_function(k)) for k in range(N_p)]
    print(f"  Target forms T : {N_p} VolumeKernel3D on 3×3×3 grid", flush=True)

    # — Wrap forms as LinearOperators (H² → ℝ^N) —
    D_space = EuclideanSpace(N_d)
    P_space = EuclideanSpace(N_p)
    G = SOLAOperator3D(sobolev, D_space, forms_G)   # forward operator
    T = SOLAOperator3D(sobolev, P_space, forms_T)   # target operator

    # --- Step 1: Sobolev representers r_i = W^{-1} ℓ_i^* ---
    print(f"  Computing {N_d} Sobolev representers (this is the bottleneck)...", flush=True)
    representers = []
    t_rep = time.time()
    for i, f in enumerate(forms_G):
        representers.append(sobolev.from_dual(f))
        if (i + 1) % 10 == 0:
            print(
                f"    representers {i+1}/{N_d}  [{_fmt_sec(time.time() - t_rep)}]",
                flush=True,
            )
    print(f"  All {N_d} representers done.  [{_fmt_sec(time.time() - t_rep)}]", flush=True)

    # — Gram matrices via operator application (fast: forms on cached representers) —
    # G(rⱼ) = [ℓ₁(rⱼ), …, ℓ_Nd(rⱼ)]  :  forward data vector for representer j
    # T(rⱼ) = [τ₁(rⱼ), …, τ_Np(rⱼ)]  :  target integral vector for representer j
    print(f"  Computing Gram matrices via G and T operators...", flush=True)
    C_GG = np.column_stack([G(r) for r in representers])   # Nd × Nd
    C_TG = np.column_stack([T(r) for r in representers])   # Np × Nd

    # — SOLA weights:  X = C_TG @ (C_GG + C_D)⁻¹ —
    A    = C_GG + C_D
    X    = C_TG @ np.linalg.solve(A, np.eye(N_d))
    cond = np.linalg.cond(A)
    print(f"  SOLA matrix X assembled.  shape: {X.shape}", flush=True)
    print(f"  Condition number of C_GG + C_D : {cond:.2e}", flush=True)
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]", flush=True)

    # ===================================================================
    # 4b. Riesz representers (5 selected rays, dropdown)
    # ===================================================================
    t_sec = time.time()
    print("\n=== 4b. Riesz representers ===", flush=True)

    REP_IDX = [0, 6, 14, 21, 28]
    rep_fns = [representers[i] for i in REP_IDX]

    def _ray_label(i):
        src = sources[i]
        sta = stations[station_idx[i]]
        return (
            f"Ray {i}  src=({src[0]:.2f},{src[1]:.2f},{src[2]:.2f})"
            f" → sta=({sta[0]:.2f},{sta[1]:.2f},{sta[2]:.2f})"
        )

    rep_labels = [_ray_label(i) for i in REP_IDX]
    print(f"  Representer indices : {REP_IDX}", flush=True)

    if not no_viz:
        fig_rep = compare_panels_axis_dropdown_sliders(
            panel_functions=[rep_fns],
            panel_labels=[rep_labels],
            panel_titles=["Riesz representer  r_i = W⁻¹ℓ_i*"],
            axis="z",
            n_slices=12,
            resolution=resolution,
            show_plot=False,
        )
        _save_html(fig_rep, out_dir, "representers.html")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]", flush=True)

    # ===================================================================
    # 4c. Total sensitivity  S(x) = Σ rᵢ(x)
    # ===================================================================
    t_sec = time.time()
    print("\n=== 4c. Total sensitivity ===", flush=True)

    def _total_sensitivity_eval(x):
        pts = np.atleast_2d(x)
        result = np.zeros(len(pts))
        for r in representers:
            result += r(pts)
        if np.ndim(x) == 1:
            return float(result[0])
        return result

    total_sensitivity = Function3D(
        box,
        evaluate_callable=_total_sensitivity_eval,
        name="Total sensitivity  S(x) = Σ rᵢ(x)",
    )

    if not no_viz:
        fig_sens = plot_function3d_axis_slider(
            total_sensitivity,
            axis="z",
            n_slices=15,
            resolution=resolution,
            title="Total sensitivity  S(x) = Σ rᵢ(x)  (z-slider)",
            show_plot=False,
        )
        _save_html(fig_sens, out_dir, "total_sensitivity.html")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]", flush=True)

    # ===================================================================
    # 5. Property inference
    # ===================================================================
    t_sec = time.time()
    print("\n=== 5. Property inference ===", flush=True)

    p_mean = X @ d_tilde
    p_cov = X @ C_D @ X.T
    p_std = np.sqrt(np.diag(p_cov))
    p_true = np.array([forms_T[k](m_true) for k in range(N_p)])

    n_print = min(20, N_p)
    print(f"  (showing first {n_print} of {N_p} properties)", flush=True)
    print(
        "  k  |  (cx,  cy,  cz)      |   true    |   mean    |   std    ",
        flush=True,
    )
    print("  " + "-" * 62, flush=True)
    for k in range(n_print):
        cx, cy, cz = target_grid[k]
        print(
            f"  {k:2d} | ({cx:.2f}, {cy:.2f}, {cz:.2f})"
            f" |  {p_true[k]:8.4f} |  {p_mean[k]:8.4f} |  {p_std[k]:8.4f}",
            flush=True,
        )

    # Build Function3D fields from the 3×3×3 grid values via interpolation
    _grid_ax = np.unique(target_grid[:, 0])  # [0.1, 0.5, 0.9]
    _n_ax = len(_grid_ax)

    def _make_prop_fn(values, name=""):
        grid_vals = values.reshape(_n_ax, _n_ax, _n_ax)
        _lo, _hi = _grid_ax[0], _grid_ax[-1]
        _interp = RegularGridInterpolator(
            (_grid_ax, _grid_ax, _grid_ax),
            grid_vals,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

        def _fn(x):
            pts_c = np.clip(np.atleast_2d(x), _lo, _hi)
            return _interp(pts_c).squeeze()

        return Function3D(box, evaluate_callable=_fn, name=name)

    p_mean_field = _make_prop_fn(p_mean, "SOLA mean")
    p_2std_field = _make_prop_fn(2 * p_std, "Uncertainty 2σ")

    if not no_viz:
        fig_prop = compare_function3d_axis_sliders(
            [p_mean_field, p_2std_field],
            axis="z",
            n_slices=10,
            resolution=resolution,
            titles=["SOLA estimate (mean)", "Uncertainty (2σ)"],
            show_plot=False,
            width=1000,
            height=500,
        )
        _save_html(fig_prop, out_dir, "property_estimate.html")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]", flush=True)

    # ===================================================================
    # 6. Unimodularity constraint  (rank-1 correction)
    # ===================================================================
    t_sec = time.time()
    print("\n=== 6. Unimodularity constraint ===", flush=True)

    ones_fn = Function3D(
        box,
        evaluate_callable=lambda x: np.ones(np.atleast_2d(x).shape[0]),
        name="ones",
    )
    v = np.array([f(ones_fn) for f in forms_G])   # G(1) — forward forms applied to ones
    w = np.ones(N_p)                               # desired property values of ones
    # Rank-1 correction using the data-space metric (C_GG + C_D).
    # Minimises the change to X while enforcing X_uni @ v = w.
    u = np.linalg.solve(C_GG + C_D, v)            # (C_GG + C_D)^{-1} v
    beta = float(np.dot(v, u))
    X_uni = X - np.outer((X @ v - w) / beta, u)

    check = X_uni @ v
    print(
        "  X_uni(G 1) — should equal ones (first 5): "
        f"{np.asarray(check).round(4)[:5]}",
        flush=True,
    )

    p_mean_uni = X_uni @ d_tilde
    P_cov_uni = X_uni @ C_D @ X_uni.T
    p_std_uni = np.sqrt(np.diag(P_cov_uni))

    p_mean_uni_field = _make_prop_fn(p_mean_uni, "Unimodular mean")
    p_2std_uni_field = _make_prop_fn(2 * p_std_uni, "Unimodular 2σ")

    if not no_viz:
        fig_uni_mean = compare_function3d_axis_sliders(
            [p_mean_field, p_mean_uni_field],
            axis="z",
            n_slices=10,
            resolution=resolution,
            titles=["Mean — unconstrained X", "Mean — unimodular X"],
            show_plot=False,
            width=1000,
            height=500,
        )
        _save_html(fig_uni_mean, out_dir, "unimodular_mean.html")

        fig_uni_unc = compare_function3d_axis_sliders(
            [p_2std_field, p_2std_uni_field],
            axis="z",
            n_slices=10,
            resolution=resolution,
            titles=["2σ — unconstrained X", "2σ — unimodular X"],
            show_plot=False,
            width=1000,
            height=500,
        )
        _save_html(fig_uni_unc, out_dir, "unimodular_uncertainty.html")

        fig_uni_est = compare_function3d_axis_sliders(
            [p_mean_uni_field, p_2std_uni_field],
            axis="z",
            n_slices=10,
            resolution=resolution,
            titles=["SOLA estimate (unimodular mean)", "Uncertainty (2σ) — unimodular"],
            show_plot=False,
            width=1000,
            height=500,
        )
        _save_html(fig_uni_est, out_dir, "unimodular_estimate.html")
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]", flush=True)

    # ===================================================================
    # 7. Resolving kernels
    # ===================================================================
    t_sec = time.time()
    print("\n=== 7. Resolving kernels ===", flush=True)

    PREVIEW_IDX = [0, 6, 13, 20, 26]

    def make_resolving_kernel(k, X_mat, reps, bx):
        """Build the resolving kernel for target k as a Function3D."""
        weights = X_mat[k, :]

        def _eval(x):
            x_arr = np.asarray(x, dtype=float)
            single = x_arr.ndim == 1
            pts = x_arr[None, :] if single else x_arr
            result = np.zeros(len(pts))
            for idx, w in enumerate(weights):
                result += w * reps[idx](pts)
            return float(result[0]) if single else result

        return Function3D(bx, evaluate_callable=_eval, name=f"resolving kernel k={k}")

    rk_fns = [make_resolving_kernel(k, X_uni, representers, box) for k in PREVIEW_IDX]
    tgt_fns = [fp_T.get_function(k) for k in PREVIEW_IDX]

    rk_labels = [
        f"Resolving k={k} "
        f"({target_grid[k, 0]:.2f},{target_grid[k, 1]:.2f},{target_grid[k, 2]:.2f})"
        for k in PREVIEW_IDX
    ]
    tgt_labels = [
        f"Target k={k} "
        f"({target_grid[k, 0]:.2f},{target_grid[k, 1]:.2f},{target_grid[k, 2]:.2f})"
        for k in PREVIEW_IDX
    ]

    if not no_viz:
        fig_rk = compare_panels_axis_dropdown_sliders(
            panel_functions=[rk_fns, tgt_fns],
            panel_labels=[rk_labels, tgt_labels],
            panel_titles=["Resolving kernels", "Target Gaussians"],
            axis="z",
            n_slices=10,
            resolution=resolution,
            show_plot=False,
        )
        _save_html(fig_rk, out_dir, "resolving_kernels.html")

    print(
        "  Each resolving kernel should be concentrated near its "
        "matching target centre.",
        flush=True,
    )
    print(f"  [elapsed: {_fmt_sec(time.time() - t_sec)}]", flush=True)

    # ===================================================================
    # Summary
    # ===================================================================
    t_total = time.time() - t_start
    if not no_viz:
        print(f"\n✓ All figures saved to {out_dir.resolve()}", flush=True)
    print(f"✓ Total runtime: {_fmt_sec(t_total)}", flush=True)
    print(
        "  Open any .html file in a browser to interact with the Plotly figures.",
        flush=True,
    )


if __name__ == "__main__":
    main()
