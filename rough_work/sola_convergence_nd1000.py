#!/usr/bin/env python
"""
sola_convergence_nd1000.py — run only the N_d=1000 case of the convergence test.
This script is identical to sola_convergence_test.py but only runs N_d=1000,
for use on europa where the full sweep would require more memory for that case.
"""

import sys
import time
import csv
from pathlib import Path

SCRIPT_DIR     = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]

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

# ─── configuration ────────────────────────────────────────────────────────────
ND_VALUES       = [1000]              # only N_d=1000
ND_MAX          = 1000
QUAD_RESOLUTION = 20
MEMORY_BUDGET   = 6 * 1024**3        # 6 GB on europa
SIGMA_RANGE     = (0.12, 0.20)
NOISE_FRACTION  = 0.05
VIZ_RESOLUTION  = 40
N_SLICES        = 7

OUT_DIR = SCRIPT_DIR / "convergence_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DIR / "convergence_nd1000.csv"

print("sola_convergence_nd1000.py: starting N_d=1000 run")
print(f"  Output dir: {OUT_DIR}")
sys.stdout.flush()

# ─── fixed geometry ──────────────────────────────────────────────────────────
box   = BoxDomain(np.zeros(3), np.ones(3))
space = Lebesgue3D(box)

target_grid = np.array([[x, y, z]
                         for x in np.linspace(0.1, 0.9, 10)
                         for y in np.linspace(0.1, 0.9, 10)
                         for z in np.linspace(0.1, 0.9, 10)])
N_p = len(target_grid)

# ─── random kernel pool (same seed as main script) ────────────────────────────
np.random.seed(42)
centers_pool = np.random.uniform(0.1, 0.9, size=(ND_MAX, 3))
sigmas_pool  = np.random.uniform(*SIGMA_RANGE, size=(ND_MAX, 3))

# ─── true model ──────────────────────────────────────────────────────────────
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

# ─── target solver ────────────────────────────────────────────────────────────
print("Building target solver…")
sys.stdout.flush()
fp_T     = GaussianProvider3D(box, centers=target_grid, sigmas=0.08, normalized=True)
solver_T = DenseVolumeSolaSolver(box, fp_T, resolution=QUAD_RESOLUTION,
                                  memory_budget_bytes=MEMORY_BUDGET)
T_op     = solver_T.forward_operator(space)
print(f"  Target operator: {space.__class__.__name__} → R^{N_p}")
sys.stdout.flush()

# ─── approximation helpers ────────────────────────────────────────────────────
_eval_xs = np.linspace(0, 1, VIZ_RESOLUTION)
_XX, _YY = np.meshgrid(_eval_xs, _eval_xs, indexing='ij')

_Q  = 30
_qx = np.linspace(0, 1, _Q)
_QX, _QY, _QZ = np.meshgrid(_qx, _qx, _qx, indexing='ij')
_q_pts = np.column_stack([_QX.ravel(), _QY.ravel(), _QZ.ravel()])
_dV    = (1.0 / _Q) ** 3

def _l2_norm(fn):
    vals = fn(_q_pts)
    return float(np.sqrt(np.sum(vals**2) * _dV))

m_true_l2 = _l2_norm(m_true)
print(f"‖m_true‖_L2 ≈ {m_true_l2:.4f}")
sys.stdout.flush()

# ─── main sweep ──────────────────────────────────────────────────────────────
csv_rows = []
N_d = 1000
print(f"\n{'='*60}")
print(f"  N_d = {N_d}")
print(f"{'='*60}")
sys.stdout.flush()
t0 = time.time()

centers_G = centers_pool[:N_d]
sigmas_G  = sigmas_pool[:N_d]

print("  Building forward operator…")
sys.stdout.flush()
fp_G     = GaussianProvider3D(box, centers=centers_G, sigmas=sigmas_G, normalized=True)
solver_G = DenseVolumeSolaSolver(box, fp_G, resolution=QUAD_RESOLUTION,
                                  memory_budget_bytes=MEMORY_BUDGET)
G = solver_G.forward_operator(space)
print(f"  Forward operator built  [{time.time()-t0:.1f}s]")
sys.stdout.flush()

np.random.seed(7)
d_bar     = G(m_true)
noise_std = NOISE_FRACTION * np.abs(d_bar).max()
d_tilde   = d_bar + np.random.normal(0.0, noise_std, size=N_d)
snr       = np.linalg.norm(d_bar) / np.linalg.norm(d_tilde - d_bar)
C_D       = noise_std ** 2 * np.eye(N_d)

print(f"  noise_std = {noise_std:.5f}   SNR = {snr:.2f}")
sys.stdout.flush()

print("  Solving…")
sys.stdout.flush()
t_solve = time.time()
m_hat = solver_G.reconstruct(d_tilde, noise_cov=C_D)
t_solve = time.time() - t_solve
print(f"  reconstruct: {t_solve:.2f}s")
sys.stdout.flush()

d_hat       = G(m_hat)
data_misfit = float(np.linalg.norm(d_hat - d_tilde) / np.linalg.norm(d_tilde))
diff_fn     = Function3D(box, evaluate_callable=lambda x, m=m_hat: m(x) - true_model_fn(x))
model_misfit = _l2_norm(diff_fn) / m_true_l2
wall_time    = time.time() - t0

print(f"  data misfit  = {data_misfit:.4f}")
print(f"  model misfit = {model_misfit:.4f}")
print(f"  wall time    = {wall_time:.1f}s")
sys.stdout.flush()

csv_rows.append(dict(
    N_d=N_d, data_misfit=data_misfit, model_misfit_l2=model_misfit,
    noise_std=noise_std, snr=snr, wall_time_s=wall_time,
))

# ─── slices for visualisation ────────────────────────────────────────────────
z_levels = np.linspace(0.1, 0.9, N_SLICES)
pts_list = [
    np.column_stack([_XX.ravel(), _YY.ravel(), np.full(VIZ_RESOLUTION**2, z)])
    for z in z_levels
]

def _eval_slices(fn):
    return [fn(pts).reshape(VIZ_RESOLUTION, VIZ_RESOLUTION) for pts in pts_list]

slices_true = _eval_slices(m_true)
slices_hat  = _eval_slices(m_hat)
vmax = max(abs(s).max() for s in slices_true + slices_hat)
vmin = -vmax

fig, axes = plt.subplots(2, N_SLICES, figsize=(3 * N_SLICES, 6), squeeze=False)
fig.suptitle(f"N_d = {N_d}   noise_std={noise_std:.4f}   data misfit={data_misfit:.3f}"
             f"   model misfit={model_misfit:.3f}", fontsize=11)
for j, z in enumerate(z_levels):
    kw = dict(origin='lower', extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax,
              cmap='RdBu_r', aspect='equal', interpolation='bilinear')
    axes[0, j].imshow(slices_true[j].T, **kw)
    axes[0, j].set_title(f"true  z={z:.2f}", fontsize=8)
    axes[0, j].axis('off')
    im = axes[1, j].imshow(slices_hat[j].T, **kw)
    axes[1, j].set_title(f"recon z={z:.2f}", fontsize=8)
    axes[1, j].axis('off')
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, label='model value')

out_png = OUT_DIR / "recon_Nd_1000.png"
fig.savefig(str(out_png), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  [saved] {out_png.name}")

# ─── CSV ─────────────────────────────────────────────────────────────────────
fieldnames = ['N_d', 'data_misfit', 'model_misfit_l2', 'noise_std', 'snr', 'wall_time_s']
with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"[saved] {CSV_PATH}")
print("\nDone.")
sys.stdout.flush()
