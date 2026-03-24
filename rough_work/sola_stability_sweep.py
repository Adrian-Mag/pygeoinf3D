#!/usr/bin/env python
"""
sola_stability_sweep.py — Sweep over data-space dimension N_d and random seeds
to diagnose numerical instability in the SOLA inversion pipeline.

For each (N_d, seed) pair, this script computes:
  1. Gram matrix GG* eigenvalue spectrum & condition number
  2. Regularised matrix (GG* + C_D) eigenvalue spectrum & condition number
  3. Quadrature convergence of the Gram at multiple resolutions
  4. Reconstruction quality: relative L2 error on a grid
  5. Property inference quality: RMS error of means vs true values
  6. Solution operator norm ||X||

Outputs (saved to rough_work/sola_stability_results/):
  - sweep_results.npz           — all numerical data
  - eigenvalue_spectra.png      — eigenvalue decay per N_d
  - condition_numbers.png       — cond(GG*) and cond(GG*+C_D) vs N_d
  - quad_convergence.png        — Gram Frobenius-norm change vs resolution
  - reconstruction_quality.png  — relative error vs N_d
  - property_quality.png        — RMS property error vs N_d

Usage:
    conda activate inferences2   # on europa
    python sola_stability_sweep.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup (same as sola_noise.py) ────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]  # pygeoinf3D/rough_work -> pygeoinf3D -> Inferences
# Actually we need the Inferences root which is one more level up
INFERENCES_ROOT = SCRIPT_DIR.parents[1]
# pygeoinf3D is a sibling of pygeoinf under Inferences
for repo_root in [INFERENCES_ROOT / "pygeoinf", INFERENCES_ROOT / "pygeoinf3D"]:
    repo_str = str(repo_root)
    if repo_root.exists() and repo_str not in sys.path:
        sys.path.insert(0, repo_str)

from pygeoinf import LinearOperator, CholeskySolver, GaussianMeasure
from pygeoinf3d import BoxDomain, Lebesgue3D, Function3D
from pygeoinf3d.providers.function_providers import GaussianProvider3D
from pygeoinf3d.solvers import DenseVolumeSolaSolver

# ── Configuration ─────────────────────────────────────────────────────────
N_D_VALUES = [10, 20, 35, 50, 75, 100, 150, 200]
SEEDS = [42, 123, 7]
QUAD_RESOLUTIONS = [10, 15, 20, 30, 40]
BASE_QUAD_RES = 20           # resolution used for the main SOLA pipeline
TARGET_GRID_N = 5            # 5×5×5 = 125 property targets
EVAL_GRID_RES = 30           # grid for evaluating reconstruction error
NOISE_FRAC = 0.05            # noise as fraction of max |d_bar|

OUT_DIR = SCRIPT_DIR / "sola_stability_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _gaussian_blob(x, center, sigma):
    diff = x - center[None, :]
    return np.exp(-0.5 * np.sum((diff / sigma) ** 2, axis=1))


def _true_model_fn(x):
    if np.ndim(x) == 1:
        x = x[np.newaxis, :]
    vals = (
        + 1.5 * _gaussian_blob(x, np.array([0.30, 0.30, 0.50]), 0.12)
        + 1.0 * _gaussian_blob(x, np.array([0.70, 0.65, 0.55]), 0.10)
        - 0.8 * _gaussian_blob(x, np.array([0.55, 0.40, 0.45]), 0.08)
    )
    return vals.squeeze()


def run_single(N_d, seed, box, space, target_grid, N_p, eval_pts, m_true_vals):
    """Run one (N_d, seed) experiment. Returns a dict of metrics."""
    rng = np.random.RandomState(seed)

    # ── Build forward operator G ──────────────────────────────────────────
    centers_G = rng.uniform(0.1, 0.9, size=(N_d, 3))
    sigmas_G  = rng.uniform(0.12, 0.20, size=(N_d, 3))
    fp_G     = GaussianProvider3D(box, centers=centers_G, sigmas=sigmas_G, normalized=True)
    solver_G = DenseVolumeSolaSolver(box, fp_G, resolution=BASE_QUAD_RES)
    G        = solver_G.forward_operator(space)
    D        = G.codomain

    # ── Build target operator T ───────────────────────────────────────────
    fp_T     = GaussianProvider3D(box, centers=target_grid, sigmas=0.08, normalized=True)
    solver_T = DenseVolumeSolaSolver(box, fp_T, resolution=BASE_QUAD_RES)
    T_op     = solver_T.forward_operator(space)
    P        = T_op.codomain

    # ── Synthetic data ────────────────────────────────────────────────────
    m_true   = Function3D(box, evaluate_callable=_true_model_fn)
    d_bar    = G(m_true)
    noise_std = NOISE_FRAC * np.abs(d_bar).max()
    d_tilde  = d_bar + rng.normal(0.0, noise_std, size=N_d)

    # ── 1. Gram matrix GG* eigenvalues ────────────────────────────────────
    # Use the pre-computed K matrix from solver_G for exact GG*
    K_G = solver_G.backend.interaction_matrix()     # (N_d, n_pts)
    w_G = solver_G.backend._weights                 # (n_pts,)
    # Gram in kernel-function space: G_ij = <k_i, k_j>_L2
    Gram_kernel = K_G @ (w_G[:, None] * K_G.T)     # (N_d, N_d)

    # GG* in data space is the same matrix (data space IS R^N_d)
    # The forward operator G maps f -> (integral k_i f dV)
    # GG* maps d -> G(G*(d)) which is Gram_kernel @ d
    # So GG* IS the kernel Gram matrix
    eigvals_GGT = np.linalg.eigvalsh(Gram_kernel)
    eigvals_GGT = np.sort(eigvals_GGT)[::-1]       # descending

    cond_GGT = eigvals_GGT[0] / max(eigvals_GGT[-1], 1e-300)

    # ── 2. Regularised matrix (GG* + C_D) eigenvalues ────────────────────
    C_D_diag = noise_std**2
    reg_matrix = Gram_kernel + C_D_diag * np.eye(N_d)
    eigvals_reg = np.linalg.eigvalsh(reg_matrix)
    eigvals_reg = np.sort(eigvals_reg)[::-1]

    cond_reg = eigvals_reg[0] / max(eigvals_reg[-1], 1e-300)

    # ── 3. Quadrature convergence ─────────────────────────────────────────
    gram_by_res = {}
    for qres in QUAD_RESOLUTIONS:
        solver_q = DenseVolumeSolaSolver(box, fp_G, resolution=qres)
        K_q = solver_q.backend.interaction_matrix()
        w_q = solver_q.backend._weights
        gram_by_res[qres] = K_q @ (w_q[:, None] * K_q.T)

    # Use highest resolution as reference
    ref_res = max(QUAD_RESOLUTIONS)
    G_ref = gram_by_res[ref_res]
    quad_errors = {}
    for qres in QUAD_RESOLUTIONS:
        if qres == ref_res:
            quad_errors[qres] = 0.0
        else:
            diff = gram_by_res[qres] - G_ref
            quad_errors[qres] = np.linalg.norm(diff, 'fro') / max(np.linalg.norm(G_ref, 'fro'), 1e-300)

    # ── 4. SOLA construction ──────────────────────────────────────────────
    C_D_matrix = noise_std**2 * np.eye(N_d)
    gaussian_D = GaussianMeasure.from_covariance_matrix(D, C_D_matrix, expectation=d_tilde)

    # Direct dense construction (avoids lazy operator materialisation overhead)
    W = Gram_kernel + C_D_matrix
    try:
        from scipy.linalg import cho_factor, cho_solve
        cho = cho_factor(W)
        W_inv_dense = cho_solve(cho, np.eye(N_d))
    except np.linalg.LinAlgError:
        # Cholesky failed — matrix is not SPD (severe instability)
        return {
            'cholesky_failed': True,
            'cond_GGT': cond_GGT,
            'cond_reg': cond_reg,
            'eigvals_GGT': eigvals_GGT,
            'eigvals_reg': eigvals_reg,
            'quad_errors': quad_errors,
        }

    # G_inv = G* W^{-1}  (as a dense matrix: K_G^T @ diag(w) applied then W_inv)
    # Actually: G maps via K @ diag(w) @ f_vals, and G* maps via sum_i d_i k_i(x)
    # In the discrete setting: G* maps d -> function whose values at quad pts are K^T d
    # The composition G_inv = G* @ W^{-1} acts on data d as:
    #   G_inv(d)(x) = sum_i [W^{-1} d]_i k_i(x)

    # X = T @ G_inv maps d -> properties
    # X_dense[p, j] = sum_i T_dense[p, i] * [G* W^{-1}]_ij
    # where G* W^{-1} in the K-basis is just W^{-1} (coefficients)
    # and T maps function -> integrals, so T(sum c_i k_i) = sum c_i T(k_i)
    # T(k_i) = [K_T @ diag(w_T) @ K_G_at_T_pts^T]  ... this is cross-gram

    # Simpler: materialise via the pygeoinf operators
    cholesky_solver = CholeskySolver(galerkin=True, parallel=False)
    Lambda  = G @ G.adjoint
    W_inv_op = cholesky_solver(Lambda + gaussian_D.covariance)
    G_inv   = G.adjoint @ W_inv_op
    X       = T_op @ G_inv

    # ── 5. Reconstruction quality ─────────────────────────────────────────
    m_hat = G_inv(d_tilde)
    m_hat_vals = np.array([m_hat(pt) for pt in eval_pts])  # slow but safe
    rel_recon_err = np.linalg.norm(m_hat_vals - m_true_vals) / max(np.linalg.norm(m_true_vals), 1e-300)

    # ── 6. Property inference quality ─────────────────────────────────────
    property_measure = gaussian_D.affine_mapping(operator=X)
    p_mean = property_measure.expectation
    P_cov_mat = property_measure.covariance.matrix(dense=True, galerkin=True)
    p_std  = np.sqrt(np.maximum(np.diag(P_cov_mat), 0.0))
    p_true = T_op(m_true)

    rms_prop_err = np.sqrt(np.mean((p_mean - p_true)**2))
    mean_std     = np.mean(p_std)

    # ── 7. Solution operator norm ─────────────────────────────────────────
    X_dense = X.matrix(dense=True, galerkin=False)
    X_norm  = np.linalg.norm(X_dense, 2)  # spectral norm

    # ── Data residual ─────────────────────────────────────────────────────
    d_hat = G(m_hat)
    rel_data_residual = np.linalg.norm(d_hat - d_tilde) / max(np.linalg.norm(d_tilde), 1e-300)

    return {
        'cholesky_failed': False,
        'cond_GGT': cond_GGT,
        'cond_reg': cond_reg,
        'eigvals_GGT': eigvals_GGT,
        'eigvals_reg': eigvals_reg,
        'quad_errors': quad_errors,
        'rel_recon_err': rel_recon_err,
        'rms_prop_err': rms_prop_err,
        'mean_prop_std': mean_std,
        'X_norm': X_norm,
        'rel_data_residual': rel_data_residual,
        'noise_std': noise_std,
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("SOLA Stability Sweep")
    print(f"  N_d values: {N_D_VALUES}")
    print(f"  Seeds:      {SEEDS}")
    print(f"  Quad resolutions (convergence test): {QUAD_RESOLUTIONS}")
    print(f"  Target grid: {TARGET_GRID_N}^3 = {TARGET_GRID_N**3}")
    print("=" * 70)

    # ── Fixed setup ───────────────────────────────────────────────────────
    box   = BoxDomain(np.zeros(3), np.ones(3))
    space = Lebesgue3D(box)

    target_grid = np.array([[x, y, z]
                             for x in np.linspace(0.15, 0.85, TARGET_GRID_N)
                             for y in np.linspace(0.15, 0.85, TARGET_GRID_N)
                             for z in np.linspace(0.15, 0.85, TARGET_GRID_N)])
    N_p = len(target_grid)

    # Evaluation grid for reconstruction error
    ax = np.linspace(0.05, 0.95, EVAL_GRID_RES)
    eval_pts = np.array([[x, y, z] for x in ax for y in ax for z in ax])
    m_true_vals = _true_model_fn(eval_pts)

    # ── Storage ───────────────────────────────────────────────────────────
    # Scalars indexed by (N_d_idx, seed_idx)
    n_nd = len(N_D_VALUES)
    n_seeds = len(SEEDS)

    cond_GGT_arr    = np.full((n_nd, n_seeds), np.nan)
    cond_reg_arr    = np.full((n_nd, n_seeds), np.nan)
    recon_err_arr   = np.full((n_nd, n_seeds), np.nan)
    prop_err_arr    = np.full((n_nd, n_seeds), np.nan)
    prop_std_arr    = np.full((n_nd, n_seeds), np.nan)
    X_norm_arr      = np.full((n_nd, n_seeds), np.nan)
    data_resid_arr  = np.full((n_nd, n_seeds), np.nan)
    noise_std_arr   = np.full((n_nd, n_seeds), np.nan)
    chol_failed_arr = np.full((n_nd, n_seeds), False)

    # Eigenvalue spectra — ragged, store as dict then pack
    eigval_spectra_GGT = {}
    eigval_spectra_reg = {}

    # Quad convergence: indexed by (N_d_idx, seed_idx, qres_idx)
    n_qres = len(QUAD_RESOLUTIONS)
    quad_conv_arr = np.full((n_nd, n_seeds, n_qres), np.nan)

    # ── Main loop ─────────────────────────────────────────────────────────
    total_runs = n_nd * n_seeds
    run_count = 0

    for i_nd, N_d in enumerate(N_D_VALUES):
        for i_s, seed in enumerate(SEEDS):
            run_count += 1
            t1 = time.time()
            print(f"\n--- [{run_count}/{total_runs}] N_d={N_d}, seed={seed} ---")

            try:
                result = run_single(N_d, seed, box, space, target_grid, N_p,
                                    eval_pts, m_true_vals)
            except Exception as e:
                print(f"  *** FAILED: {e}")
                chol_failed_arr[i_nd, i_s] = True
                continue

            elapsed = time.time() - t1
            chol_failed_arr[i_nd, i_s] = result['cholesky_failed']
            cond_GGT_arr[i_nd, i_s]    = result['cond_GGT']
            cond_reg_arr[i_nd, i_s]    = result['cond_reg']

            eigval_spectra_GGT[(N_d, seed)] = result['eigvals_GGT']
            eigval_spectra_reg[(N_d, seed)] = result['eigvals_reg']

            for i_q, qres in enumerate(QUAD_RESOLUTIONS):
                quad_conv_arr[i_nd, i_s, i_q] = result['quad_errors'][qres]

            if not result['cholesky_failed']:
                recon_err_arr[i_nd, i_s]  = result['rel_recon_err']
                prop_err_arr[i_nd, i_s]   = result['rms_prop_err']
                prop_std_arr[i_nd, i_s]   = result['mean_prop_std']
                X_norm_arr[i_nd, i_s]     = result['X_norm']
                data_resid_arr[i_nd, i_s] = result['rel_data_residual']
                noise_std_arr[i_nd, i_s]  = result['noise_std']

                print(f"  cond(GG*)       = {result['cond_GGT']:.2e}")
                print(f"  cond(GG*+C_D)   = {result['cond_reg']:.2e}")
                print(f"  ||X||           = {result['X_norm']:.4f}")
                print(f"  recon err       = {result['rel_recon_err']:.4f}")
                print(f"  prop RMS err    = {result['rms_prop_err']:.6f}")
                print(f"  mean prop std   = {result['mean_prop_std']:.6f}")
                print(f"  data residual   = {result['rel_data_residual']:.6f}")
            else:
                print(f"  *** Cholesky FAILED (matrix not SPD)")
                print(f"  cond(GG*)       = {result['cond_GGT']:.2e}")
                print(f"  cond(GG*+C_D)   = {result['cond_reg']:.2e}")

            print(f"  [{elapsed:.1f}s]")

    # ── Save results ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Saving results...")

    # Pack eigenvalue spectra into padded arrays
    max_nd = max(N_D_VALUES)
    eigvals_GGT_padded = np.full((n_nd, n_seeds, max_nd), np.nan)
    eigvals_reg_padded = np.full((n_nd, n_seeds, max_nd), np.nan)
    for i_nd, N_d in enumerate(N_D_VALUES):
        for i_s, seed in enumerate(SEEDS):
            key = (N_d, seed)
            if key in eigval_spectra_GGT:
                ev = eigval_spectra_GGT[key]
                eigvals_GGT_padded[i_nd, i_s, :len(ev)] = ev
            if key in eigval_spectra_reg:
                ev = eigval_spectra_reg[key]
                eigvals_reg_padded[i_nd, i_s, :len(ev)] = ev

    np.savez(
        OUT_DIR / "sweep_results.npz",
        N_d_values=np.array(N_D_VALUES),
        seeds=np.array(SEEDS),
        quad_resolutions=np.array(QUAD_RESOLUTIONS),
        cond_GGT=cond_GGT_arr,
        cond_reg=cond_reg_arr,
        recon_err=recon_err_arr,
        prop_err=prop_err_arr,
        prop_std=prop_std_arr,
        X_norm=X_norm_arr,
        data_residual=data_resid_arr,
        noise_std=noise_std_arr,
        chol_failed=chol_failed_arr,
        eigvals_GGT=eigvals_GGT_padded,
        eigvals_reg=eigvals_reg_padded,
        quad_conv=quad_conv_arr,
    )
    print(f"  Saved: {OUT_DIR / 'sweep_results.npz'}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("Generating plots...")

    fig_w, fig_h = 10, 6
    colors = plt.cm.viridis(np.linspace(0, 1, n_nd))
    seed_markers = ['o', 's', '^']

    # --- 1. Eigenvalue spectra (seed=42 only, all N_d) ---
    fig, axes = plt.subplots(1, 2, figsize=(fig_w * 1.5, fig_h))
    for i_nd, N_d in enumerate(N_D_VALUES):
        key = (N_d, SEEDS[0])
        if key in eigval_spectra_GGT:
            ev = eigval_spectra_GGT[key]
            ev_pos = ev[ev > 0]
            axes[0].semilogy(range(1, len(ev_pos)+1), ev_pos,
                           'o-', color=colors[i_nd], ms=3, label=f'N_d={N_d}')
        if key in eigval_spectra_reg:
            ev = eigval_spectra_reg[key]
            ev_pos = ev[ev > 0]
            axes[1].semilogy(range(1, len(ev_pos)+1), ev_pos,
                           'o-', color=colors[i_nd], ms=3, label=f'N_d={N_d}')
    axes[0].set_title(r'Eigenvalues of $GG^*$')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Eigenvalue')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title(r'Eigenvalues of $GG^* + C_D$')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Eigenvalue')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eigenvalue_spectra.png", dpi=150)
    plt.close()
    print(f"  Saved: eigenvalue_spectra.png")

    # --- 2. Condition numbers vs N_d ---
    fig, axes = plt.subplots(1, 2, figsize=(fig_w * 1.5, fig_h))
    for i_s, seed in enumerate(SEEDS):
        valid = ~np.isnan(cond_GGT_arr[:, i_s])
        axes[0].semilogy(np.array(N_D_VALUES)[valid], cond_GGT_arr[valid, i_s],
                        f'{seed_markers[i_s]}-', label=f'seed={seed}')
        valid = ~np.isnan(cond_reg_arr[:, i_s])
        axes[1].semilogy(np.array(N_D_VALUES)[valid], cond_reg_arr[valid, i_s],
                        f'{seed_markers[i_s]}-', label=f'seed={seed}')
    axes[0].set_title(r'$\mathrm{cond}(GG^*)$ vs $N_d$')
    axes[0].set_xlabel(r'$N_d$')
    axes[0].set_ylabel('Condition number')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title(r'$\mathrm{cond}(GG^* + C_D)$ vs $N_d$')
    axes[1].set_xlabel(r'$N_d$')
    axes[1].set_ylabel('Condition number')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "condition_numbers.png", dpi=150)
    plt.close()
    print(f"  Saved: condition_numbers.png")

    # --- 3. Quadrature convergence (seed=42, all N_d) ---
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for i_nd, N_d in enumerate(N_D_VALUES):
        vals = quad_conv_arr[i_nd, 0, :]  # seed=42
        valid = ~np.isnan(vals)
        if valid.any():
            ax.semilogy(np.array(QUAD_RESOLUTIONS)[valid], vals[valid],
                       'o-', color=colors[i_nd], label=f'N_d={N_d}')
    ax.set_title(r'Quadrature convergence: $\|G_{res} - G_{ref}\|_F / \|G_{ref}\|_F$')
    ax.set_xlabel('Quadrature resolution (per axis)')
    ax.set_ylabel('Relative Frobenius error')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "quad_convergence.png", dpi=150)
    plt.close()
    print(f"  Saved: quad_convergence.png")

    # --- 4. Reconstruction quality vs N_d ---
    fig, axes = plt.subplots(1, 2, figsize=(fig_w * 1.5, fig_h))
    for i_s, seed in enumerate(SEEDS):
        valid = ~np.isnan(recon_err_arr[:, i_s])
        axes[0].plot(np.array(N_D_VALUES)[valid], recon_err_arr[valid, i_s],
                    f'{seed_markers[i_s]}-', label=f'seed={seed}')
        valid = ~np.isnan(data_resid_arr[:, i_s])
        axes[1].semilogy(np.array(N_D_VALUES)[valid], data_resid_arr[valid, i_s],
                        f'{seed_markers[i_s]}-', label=f'seed={seed}')
    axes[0].set_title(r'Relative reconstruction error vs $N_d$')
    axes[0].set_xlabel(r'$N_d$')
    axes[0].set_ylabel(r'$\|m_{hat} - m_{true}\|_2 / \|m_{true}\|_2$')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title(r'Relative data residual vs $N_d$')
    axes[1].set_xlabel(r'$N_d$')
    axes[1].set_ylabel(r'$\|d_{hat} - \tilde{d}\|_2 / \|\tilde{d}\|_2$')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "reconstruction_quality.png", dpi=150)
    plt.close()
    print(f"  Saved: reconstruction_quality.png")

    # --- 5. Property inference quality vs N_d ---
    fig, axes = plt.subplots(1, 3, figsize=(fig_w * 2, fig_h))
    for i_s, seed in enumerate(SEEDS):
        valid = ~np.isnan(prop_err_arr[:, i_s])
        axes[0].plot(np.array(N_D_VALUES)[valid], prop_err_arr[valid, i_s],
                    f'{seed_markers[i_s]}-', label=f'seed={seed}')
        valid = ~np.isnan(prop_std_arr[:, i_s])
        axes[1].plot(np.array(N_D_VALUES)[valid], prop_std_arr[valid, i_s],
                    f'{seed_markers[i_s]}-', label=f'seed={seed}')
        valid = ~np.isnan(X_norm_arr[:, i_s])
        axes[2].semilogy(np.array(N_D_VALUES)[valid], X_norm_arr[valid, i_s],
                        f'{seed_markers[i_s]}-', label=f'seed={seed}')
    axes[0].set_title(r'Property RMS error vs $N_d$')
    axes[0].set_xlabel(r'$N_d$')
    axes[0].set_ylabel('RMS error')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title(r'Mean property std vs $N_d$')
    axes[1].set_xlabel(r'$N_d$')
    axes[1].set_ylabel(r'Mean $\sigma_p$')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].set_title(r'Solution operator norm $\|X\|$ vs $N_d$')
    axes[2].set_xlabel(r'$N_d$')
    axes[2].set_ylabel(r'$\|X\|_2$')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "property_quality.png", dpi=150)
    plt.close()
    print(f"  Saved: property_quality.png")

    # --- 6. Combined summary: cond number vs reconstruction error ---
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for i_s, seed in enumerate(SEEDS):
        valid = ~np.isnan(cond_reg_arr[:, i_s]) & ~np.isnan(recon_err_arr[:, i_s])
        ax.scatter(cond_reg_arr[valid, i_s], recon_err_arr[valid, i_s],
                  marker=seed_markers[i_s], c=np.array(N_D_VALUES)[valid.ravel() if valid.ndim > 1 else valid],
                  cmap='viridis', s=60, edgecolors='k', linewidths=0.5,
                  label=f'seed={seed}')
        # Annotate each point with N_d
        for i_nd in range(n_nd):
            if valid[i_nd]:
                ax.annotate(f'{N_D_VALUES[i_nd]}',
                          (cond_reg_arr[i_nd, i_s], recon_err_arr[i_nd, i_s]),
                          fontsize=7, ha='left', va='bottom')
    ax.set_xscale('log')
    ax.set_title(r'Reconstruction error vs $\mathrm{cond}(GG^* + C_D)$')
    ax.set_xlabel(r'$\mathrm{cond}(GG^* + C_D)$')
    ax.set_ylabel(r'Relative reconstruction error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cond_vs_error.png", dpi=150)
    plt.close()
    print(f"  Saved: cond_vs_error.png")

    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"SWEEP COMPLETE — {total:.1f}s total")
    print(f"Results in: {OUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
