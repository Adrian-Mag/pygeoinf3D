"""
collect_small_nd_misfits.py — compute misfits for N_d in {10, 50, 100, 500} locally
and write combined convergence.csv merging with the N_d=1000 result from europa.

Usage:
    conda activate inferences3
    python pygeoinf3D/rough_work/collect_small_nd_misfits.py
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

from pygeoinf3d import BoxDomain, Lebesgue3D, Function3D
from pygeoinf3d.providers.function_providers import GaussianProvider3D
from pygeoinf3d.solvers import DenseVolumeSolaSolver

ND_VALUES       = [10, 50, 100, 500]
ND_MAX          = 1000
QUAD_RESOLUTION = 20
MEMORY_BUDGET   = 2 * 1024**3   # 2 GB — conservative for local machine
SIGMA_RANGE     = (0.12, 0.20)
NOISE_FRACTION  = 0.05

OUT_DIR  = SCRIPT_DIR / "convergence_results"
CSV_ND1K = OUT_DIR / "convergence_nd1000.csv"
CSV_OUT  = OUT_DIR / "convergence.csv"

box   = BoxDomain(np.zeros(3), np.ones(3))
space = Lebesgue3D(box)

target_grid = np.array([[x, y, z]
                         for x in np.linspace(0.1, 0.9, 10)
                         for y in np.linspace(0.1, 0.9, 10)
                         for z in np.linspace(0.1, 0.9, 10)])
N_p = len(target_grid)

np.random.seed(42)
centers_pool = np.random.uniform(0.1, 0.9, size=(ND_MAX, 3))
sigmas_pool  = np.random.uniform(*SIGMA_RANGE, size=(ND_MAX, 3))

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

csv_rows = []

for N_d in ND_VALUES:
    print(f"\nN_d = {N_d}")
    t0 = time.time()

    centers_G = centers_pool[:N_d]
    sigmas_G  = sigmas_pool[:N_d]
    fp_G      = GaussianProvider3D(box, centers=centers_G, sigmas=sigmas_G, normalized=True)
    solver_G  = DenseVolumeSolaSolver(box, fp_G, resolution=QUAD_RESOLUTION,
                                       memory_budget_bytes=MEMORY_BUDGET)
    G = solver_G.forward_operator(space)

    np.random.seed(7)
    d_bar     = G(m_true)
    noise_std = NOISE_FRACTION * np.abs(d_bar).max()
    d_tilde   = d_bar + np.random.normal(0.0, noise_std, size=N_d)
    snr       = np.linalg.norm(d_bar) / np.linalg.norm(d_tilde - d_bar)
    C_D       = noise_std ** 2 * np.eye(N_d)

    m_hat = solver_G.reconstruct(d_tilde, noise_cov=C_D)
    d_hat = G(m_hat)
    data_misfit  = float(np.linalg.norm(d_hat - d_tilde) / np.linalg.norm(d_tilde))
    diff_fn      = Function3D(box, evaluate_callable=lambda x, m=m_hat: m(x) - true_model_fn(x))
    model_misfit = _l2_norm(diff_fn) / m_true_l2
    wall_time    = time.time() - t0

    print(f"  data misfit  = {data_misfit:.4f}")
    print(f"  model misfit = {model_misfit:.4f}")
    print(f"  wall time    = {wall_time:.1f}s")

    csv_rows.append(dict(
        N_d=N_d, data_misfit=data_misfit, model_misfit_l2=model_misfit,
        noise_std=noise_std, snr=snr, wall_time_s=wall_time,
    ))

# ─── load N_d=1000 from europa CSV ────────────────────────────────────────────
if CSV_ND1K.exists():
    with open(CSV_ND1K) as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_rows.append({k: (int(v) if k == 'N_d' else float(v)) for k, v in row.items()})
    print(f"\nAppended N_d=1000 from {CSV_ND1K.name}")
else:
    print(f"\nWARNING: {CSV_ND1K} not found — N_d=1000 row missing")

# sort by N_d
csv_rows.sort(key=lambda r: r['N_d'])

# ─── write combined CSV ───────────────────────────────────────────────────────
fieldnames = ['N_d', 'data_misfit', 'model_misfit_l2', 'noise_std', 'snr', 'wall_time_s']
with open(CSV_OUT, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"\n[saved] {CSV_OUT}")

print("\n" + "="*65)
print(f"{'N_d':>8}  {'data misfit':>12}  {'model misfit':>13}  {'snr':>7}  {'time':>8}")
print("-"*65)
for row in csv_rows:
    print(f"  {row['N_d']:>6}  {row['data_misfit']:>12.4f}  {row['model_misfit_l2']:>13.4f}"
          f"  {row['snr']:>7.2f}  {row['wall_time_s']:>7.1f}s")
print("="*65)
