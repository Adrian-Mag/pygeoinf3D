"""
sola_convergence_test.py — SOLA property-estimate convergence across data density
==================================================================================
Sweeps N_d ∈ {10, 50, 100, 500, 1000} using the same sensitivity-kernel pool
(first N_d kernels from a fixed random pool of 1000) and the same true model.

The PRIMARY metric is the SOLA property error:
  prop_error = ||X d_tilde - T m_true|| / ||T m_true||
where X = compute_sola_matrix(solver_T, noise_cov) is the SOLA estimator matrix.
This is the correct convergence diagnostic for SOLA inversion — it should
decrease monotonically with N_d.

The data_misfit ||G m_hat - d_tilde||/||d_tilde|| is also reported but is
EXPECTED to grow with N_d.  It equals σ² ||q||/||d|| (regularisation residual
of the minimum-norm model), NOT a numerical error.  See notes in the code.

IMPORTANT — sigma_target vs sigma_sensitivity:
  Target kernels must have sigma_target ≥ sigma_sensitivity (the SMALLEST
  sensitivity kernel sigma, here 0.12) for property estimates to converge.
  If sigma_target << sigma_sensitivity, SOLA is attempting sub-resolution
  estimation: signal_error still converges to 0, but noise amplification
  ||X||_F grows, dominating the error at large N_d.
  The demo sola_noise.py uses sigma_target=0.08 intentionally (for
  fine-resolution targets with known high uncertainty), but for a
  CONVERGENCE TEST we use sigma_target=0.14 (≥ sigma_sensitivity).

For each N_d the script saves:
  • convergence_results/recon_Nd_{N_d}.png     — z-slice images (true vs recon)
  • convergence_results/prop_Nd_{N_d}.png       — true vs estimated properties
  • convergence_results/convergence.csv         — full numeric table

Usage
-----
    conda activate inferences2
    python pygeoinf3D/rough_work/sola_convergence_test.py [--quad-resolution 20]
"""

import argparse
import sys
import time
import csv
from pathlib import Path

# ─── path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]          # …/Inferences

for repo_root in [WORKSPACE_ROOT / "pygeoinf", WORKSPACE_ROOT / "pygeoinf3D"]:
    s = str(repo_root)
    if repo_root.exists() and s not in sys.path:
        sys.path.insert(0, s)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pygeoinf3d import BoxDomain, Lebesgue3D, Function3D
from pygeoinf3d.providers.function_providers import GaussianProvider3D
from pygeoinf3d.solvers import DenseVolumeSolaSolver

# ─── CLI ─────────────────────────────────────────────────────────────────────
_p = argparse.ArgumentParser()
_p.add_argument("--quad-resolution", type=int, default=20,
                help="Quadrature grid points per axis (default 20). "
                     "Use 20 for N_d=10000+; higher values need more RAM.")
_p.add_argument("--target-sigma",    type=float, default=0.14,
                help="Target kernel sigma (default 0.14). Must be >= smallest "
                     "sensitivity sigma (0.12) for convergent property estimates.")
_args = _p.parse_args()

# ─── configuration ────────────────────────────────────────────────────────────
ND_VALUES       = [10, 50, 100, 500, 1000]
ND_MAX          = max(ND_VALUES)
QUAD_RESOLUTION = _args.quad_resolution
SIGMA_TARGET    = _args.target_sigma
MEMORY_BUDGET   = 4 * 1024**3            # 4 GB — generous on europa
SIGMA_RANGE     = (0.12, 0.20)
NOISE_FRACTION  = 0.05
VIZ_RESOLUTION  = 40
N_SLICES        = 7

OUT_DIR  = SCRIPT_DIR / "convergence_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "convergence.csv"

print(f"Q={QUAD_RESOLUTION} (N_q={QUAD_RESOLUTION**3})  sigma_target={SIGMA_TARGET}")
print(f"Memory budget = {MEMORY_BUDGET/1024**3:.0f} GB")
if SIGMA_TARGET < min(SIGMA_RANGE):
    print(f"WARNING: sigma_target={SIGMA_TARGET} < sigma_sensitivity_min={min(SIGMA_RANGE)}.")
    print("  Sub-resolution targeting: noise amplification will grow with N_d.")
    print("  Set --target-sigma >= 0.12 for convergent property estimates.")
print()

# ─── fixed geometry ──────────────────────────────────────────────────────────
box   = BoxDomain(np.zeros(3), np.ones(3))
space = Lebesgue3D(box)

# 8×8×8 target grid.  10×10×10 with broad sigma causes near-singular Gram.
target_grid = np.array([[x, y, z]
                         for x in np.linspace(0.1, 0.9, 8)
                         for y in np.linspace(0.1, 0.9, 8)
                         for z in np.linspace(0.1, 0.9, 8)])
N_p = len(target_grid)   # 512

# ─── random kernel pool (fixed seed) ─────────────────────────────────────────
np.random.seed(42)
centers_pool = np.random.uniform(0.1, 0.9, size=(ND_MAX, 3))
sigmas_pool  = np.random.uniform(*SIGMA_RANGE, size=(ND_MAX, 3))

# ─── true model (same 3-Gaussian structure as sola_noise.py) ─────────────────
def _blob(x, center, sigma):
    diff = x - center[None, :]
    return np.exp(-0.5 * np.sum((diff / sigma) ** 2, axis=1))

def true_model_fn(x):
    x = np.atleast_2d(x)
    return (
        + 1.5 * _blob(x, np.array([0.30, 0.30, 0.50]), 0.12)
        + 1.0 * _blob(x, np.array([0.70, 0.65, 0.55]), 0.10)
        - 0.8 * _blob(x, np.array([0.55, 0.40, 0.45]), 0.08)
    ).squeeze()

m_true = Function3D(box, evaluate_callable=true_model_fn)

# ─── target solver (shared across all N_d) ────────────────────────────────────
# regularization=1e-6 stabilises Gram for tightly-packed overlapping targets.
print("Building target solver (shared)…")
import warnings
fp_T     = GaussianProvider3D(box, centers=target_grid, sigmas=SIGMA_TARGET, normalized=True)
with warnings.catch_warnings(): warnings.simplefilter("ignore")
solver_T = DenseVolumeSolaSolver(box, fp_T, resolution=QUAD_RESOLUTION,
                                  memory_budget_bytes=MEMORY_BUDGET,
                                  regularization=1e-6)
T_op = solver_T.forward_operator(space)
p_true = T_op(m_true)          # (N_p,) — "true" properties at quadrature resolution
print(f"  Target operator: M → R^{N_p}  (sigma_target={SIGMA_TARGET})")
print(f"  ||p_true|| = {np.linalg.norm(p_true):.4f}\n")

# ─── visualisation helpers ────────────────────────────────────────────────────
_eval_xs = np.linspace(0, 1, VIZ_RESOLUTION)
_XX, _YY = np.meshgrid(_eval_xs, _eval_xs, indexing='ij')

def _eval_slices(fn, z_levels):
    slices = []
    for z in z_levels:
        pts = np.column_stack([_XX.ravel(), _YY.ravel(), np.full(VIZ_RESOLUTION**2, z)])
        slices.append(fn(pts).reshape(VIZ_RESOLUTION, VIZ_RESOLUTION))
    return slices

# ─── main sweep ──────────────────────────────────────────────────────────────
csv_rows = []

for N_d in ND_VALUES:
    print(f"{'='*60}")
    print(f"  N_d = {N_d}")
    print(f"{'='*60}")
    t0 = time.time()
    sys.stdout.flush()

    # ── build forward operator ────────────────────────────────────────────────
    fp_G = GaussianProvider3D(box, centers=centers_pool[:N_d], sigmas=sigmas_pool[:N_d],
                               normalized=True)
    with warnings.catch_warnings(): warnings.simplefilter("ignore")
    solver_G = DenseVolumeSolaSolver(box, fp_G, resolution=QUAD_RESOLUTION,
                                      memory_budget_bytes=MEMORY_BUDGET)
    G = solver_G.forward_operator(space)

    # ── synthetic data ────────────────────────────────────────────────────────
    np.random.seed(7)
    d_bar     = G(m_true)
    noise_std = NOISE_FRACTION * np.abs(d_bar).max()
    noise_vec = np.random.normal(0.0, noise_std, size=N_d)
    d_tilde   = d_bar + noise_vec
    snr       = np.linalg.norm(d_bar) / np.linalg.norm(noise_vec)
    print(f"  noise_std={noise_std:.5f}  SNR={snr:.2f}")
    sys.stdout.flush()

    # ── SOLA reconstruction (minimum-norm model) ──────────────────────────────
    t_solve = time.time()
    with warnings.catch_warnings(): warnings.simplefilter("ignore")
    m_hat = solver_G.reconstruct(d_tilde, noise_cov=noise_std)
    t_solve = time.time() - t_solve
    print(f"  reconstruct: {t_solve:.2f}s")
    sys.stdout.flush()

    # ── data misfit (expected to GROW with N_d — see docstring) ──────────────
    d_hat       = G(m_hat)
    data_misfit = float(np.linalg.norm(d_hat - d_tilde) / np.linalg.norm(d_tilde))
    # data_misfit = σ²‖q‖/‖d‖ — regularisation residual, NOT a solver error.
    # It grows with N_d because more near-degenerate kernels contribute larger
    # "unresolvable" components to d that are handled by C_D, not by the model.
    print(f"  data_misfit = {data_misfit:.4f}  [expected to grow — see docstring]")
    sys.stdout.flush()

    # ── SOLA property estimates (PRIMARY CONVERGENCE METRIC) ──────────────────
    t_sola = time.time()
    with warnings.catch_warnings(): warnings.simplefilter("ignore")
    X = solver_G.compute_sola_matrix(solver_T, noise_cov=noise_std)
    t_sola = time.time() - t_sola

    p_hat      = X @ d_tilde               # (N_p,) property estimates
    p_signal   = X @ d_bar                 # noiseless signal component
    p_noise    = X @ noise_vec             # noise component

    prop_error  = float(np.linalg.norm(p_hat    - p_true) / np.linalg.norm(p_true))
    signal_error= float(np.linalg.norm(p_signal - p_true) / np.linalg.norm(p_true))
    noise_error = float(np.linalg.norm(p_noise)           / np.linalg.norm(p_true))
    p_std       = noise_std * np.sqrt(np.einsum('ij,ij->i', X, X))  # σ * ||X[k,:]||
    mean_uncert = float(np.mean(p_std))

    print(f"  prop_error  = {prop_error:.4f}  (‖Xd̃−Tm_true‖/‖Tm_true‖)  [should DECREASE]")
    print(f"    signal_err= {signal_error:.4f}  noise_err={noise_error:.4f}  mean_σ={mean_uncert:.4f}")
    print(f"  sola_matrix : {t_sola:.2f}s")
    sys.stdout.flush()

    wall_time = time.time() - t0
    print(f"  wall_time   = {wall_time:.1f}s")
    sys.stdout.flush()

    csv_rows.append(dict(
        N_d=N_d,
        prop_error=prop_error,
        signal_error=signal_error,
        noise_error=noise_error,
        mean_uncert=mean_uncert,
        data_misfit=data_misfit,
        noise_std=noise_std,
        snr=snr,
        wall_time_s=wall_time,
    ))

    # ── reconstruction z-slices (true vs m_hat) ───────────────────────────────
    z_levels    = np.linspace(0.1, 0.9, N_SLICES)
    sl_true     = _eval_slices(m_true, z_levels)
    sl_hat      = _eval_slices(m_hat,  z_levels)
    vmax        = max(abs(s).max() for s in sl_true + sl_hat)

    fig, axes = plt.subplots(2, N_SLICES, figsize=(3*N_SLICES, 6), squeeze=False)
    fig.suptitle(f"N_d={N_d}  noise_std={noise_std:.4f}  data_misfit={data_misfit:.3f}"
                 f"  prop_error={prop_error:.3f}", fontsize=11)
    for j, z in enumerate(z_levels):
        kw = dict(origin='lower', extent=[0,1,0,1], vmin=-vmax, vmax=vmax,
                  cmap='RdBu_r', aspect='equal', interpolation='bilinear')
        axes[0, j].imshow(sl_true[j].T, **kw); axes[0,j].set_title(f"true z={z:.2f}", fontsize=8); axes[0,j].axis('off')
        im = axes[1, j].imshow(sl_hat[j].T, **kw)
        axes[1, j].set_title(f"recon z={z:.2f}", fontsize=8); axes[1,j].axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, label='model value')
    out_png = OUT_DIR / f"recon_Nd_{N_d:04d}.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] {out_png.name}")

    # ── property estimate plot: true vs estimated ─────────────────────────────
    # Show properties at z=0.5 slice (all target points with z≈0.5)
    z_mid   = target_grid[:, 2]
    z_lo, z_hi = np.percentile(z_mid, [40, 60])
    z_mask  = (z_mid >= z_lo) & (z_mid <= z_hi)
    idx     = np.where(z_mask)[0]
    if len(idx) > 0:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.plot(idx, p_true[idx],   'k-',  lw=2, label='p_true (quadrature)')
        ax2.plot(idx, p_hat[idx],    'r--', lw=1.5, label='p_hat (SOLA estimate)')
        ax2.fill_between(idx,
                         p_hat[idx] - 2*p_std[idx],
                         p_hat[idx] + 2*p_std[idx],
                         alpha=0.25, color='r', label='±2σ uncertainty')
        ax2.set_title(f"N_d={N_d}  prop_error={prop_error:.3f}  signal={signal_error:.3f}  noise={noise_error:.3f}")
        ax2.set_xlabel("target index (z≈0.5 slice)")
        ax2.set_ylabel("property value")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        out_prop = OUT_DIR / f"prop_Nd_{N_d:04d}.png"
        fig2.savefig(str(out_prop), dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"  [saved] {out_prop.name}")
    print()

# ─── write CSV ───────────────────────────────────────────────────────────────
fieldnames = ['N_d', 'prop_error', 'signal_error', 'noise_error', 'mean_uncert',
              'data_misfit', 'noise_std', 'snr', 'wall_time_s']
with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"[saved] {CSV_PATH}")

# ─── summary table ────────────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"Q={QUAD_RESOLUTION}  sigma_target={SIGMA_TARGET}")
print(f"{'N_d':>6}  {'prop_err':>9}  {'sig_err':>8}  {'noise_err':>9}  {'data_misfit':>11}  {'time':>7}")
print("-"*80)
for row in csv_rows:
    print(f"  {row['N_d']:>4}  {row['prop_error']:>9.4f}  {row['signal_error']:>8.4f}"
          f"  {row['noise_error']:>9.4f}  {row['data_misfit']:>11.4f}  {row['wall_time_s']:>6.1f}s")
print("="*80)
print()
print("prop_err  = ||X d_tilde - T m_true|| / ||T m_true||  ← primary metric (should ↓)")
print("sig_err   = ||X d_bar   - T m_true|| / ||T m_true||  ← bias  (monotone ↓)")
print("noise_err = ||X e_noise||              / ||T m_true||  ← noise (↓ asymptotically)")
print("data_misfit = ||G m_hat - d_tilde|| / ||d_tilde||     ← regularisation residual (expected ↑)")
print("\nDone.")
