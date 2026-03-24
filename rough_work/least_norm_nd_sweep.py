"""
least_norm_nd_sweep.py — Least-norm reconstruction quality vs N_d with NormalModesProvider3D
============================================================================================

Sweeps N_d ∈ N_D_VALUES and SEEDS random seeds (5 by default).
For each (N_d, seed) pair:

  1. Build N_d NormalModesProvider3D sensitivity kernels (seeded by `seed`).
  2. Compute the Gram matrix GG* and its condition number.
  3. Generate clean data d_bar = G m_true and noisy data d = d_bar + ε,  ε ~ N(0, σ²I).
     σ is fixed to NOISE_FRACTION * max|d_bar| for the FIRST N_d in the sweep
     and held constant across all N_d to keep the noise level apples-to-apples.
  4. Compute the least-norm reconstruction m_hat = G*(GG* + σ²I)^{-1} d.
  5. Evaluate relative L2 error ||m_hat - m_true||₂ / ||m_true||₂ on a uniform grid.
  6. Also report noise amplification ||G*(GG* + σ²I)^{-2} G||_F (via trace).

Outputs to rough_work/least_norm_sweep_results/:
  sweep_results.npz       — all arrays
  sweep_results.csv       — human-readable numbers
  recon_error.png         — recon error vs N_d (one line per seed, mean + band)
  condition_numbers.png   — cond(GG*) and cond(GG*+σI) vs N_d
  eigenvalue_spectra.png  — GG* eigenvalue decay for each N_d (seed-averaged)
  noise_amplification.png — ||X||_F vs N_d

Usage
-----
    conda activate inferences2
    cd ~/data/PhD/Inferences
    python pygeoinf3D/rough_work/least_norm_nd_sweep.py               \\
        [--quad-resolution 20] [--noise-fraction 0.05]

For a quick smoke test on a small range:
    python pygeoinf3D/rough_work/least_norm_nd_sweep.py --quick
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

SCRIPT_DIR     = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]          # …/Inferences

for repo_root in [WORKSPACE_ROOT / "pygeoinf", WORKSPACE_ROOT / "pygeoinf3D"]:
    _s = str(repo_root)
    if repo_root.exists() and _s not in sys.path:
        sys.path.insert(0, _s)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve

from pygeoinf3d import BoxDomain, Lebesgue3D, Function3D
from pygeoinf3d.providers.function_providers import NormalModesProvider3D
from pygeoinf3d.solvers import DenseVolumeSolaSolver

# ── CLI ───────────────────────────────────────────────────────────────────────
_p = argparse.ArgumentParser()
_p.add_argument("--quad-resolution", type=int, default=20,
                help="Quadrature grid points per axis (default 20).")
_p.add_argument("--noise-fraction",  type=float, default=0.05,
                help="Noise std as fraction of max|d_bar| (default 0.05).")
_p.add_argument("--quick", action="store_true",
                help="Smoke test: only N_d in {10, 50, 100} and 2 seeds.")
_p.add_argument("--memory-gb", type=float, default=8.0,
                help="Memory budget in GB for K cache (default 8).")
_args = _p.parse_args()

# ── Configuration ─────────────────────────────────────────────────────────────
if _args.quick:
    N_D_VALUES   = [10, 50, 100]
    SEEDS        = [42, 123]
else:
    N_D_VALUES   = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
    SEEDS        = [42, 123, 7, 999, 31415]

QUAD_RESOLUTION = _args.quad_resolution
NOISE_FRACTION  = _args.noise_fraction
EVAL_GRID_RES   = 30        # per-axis for reconstruction grid eval
MEMORY_BYTES    = int(_args.memory_gb * 1024**3)

FREQ_RANGE             = (1.0, 4.0)   # NormalModesProvider3D frequency range
GAUSSIAN_WIDTH_RANGE   = (5.0, 30.0)  # % of interval length

OUT_DIR = SCRIPT_DIR / "least_norm_sweep_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Least-norm sweep: NormalModesProvider3D kernels")
print(f"  N_D_VALUES       : {N_D_VALUES}")
print(f"  SEEDS            : {SEEDS}")
print(f"  quad resolution  : {QUAD_RESOLUTION}  (N_q = {QUAD_RESOLUTION**3})")
print(f"  noise fraction   : {NOISE_FRACTION}")
print(f"  eval grid res    : {EVAL_GRID_RES}  (N_eval = {EVAL_GRID_RES**3})")
print(f"  memory budget    : {_args.memory_gb:.0f} GB")
print("=" * 70)

# ── Domain & evaluation grid ──────────────────────────────────────────────────
box   = BoxDomain(np.zeros(3), np.ones(3))
space = Lebesgue3D(box)

_ax = np.linspace(0.05, 0.95, EVAL_GRID_RES)
_xx, _yy, _zz = np.meshgrid(_ax, _ax, _ax, indexing='ij')
EVAL_PTS = np.column_stack([_xx.ravel(), _yy.ravel(), _zz.ravel()])  # (N_eval, 3)
EVAL_W   = ((0.9 / EVAL_GRID_RES) ** 3) * np.ones(len(EVAL_PTS))    # uniform weights

# ── True model ────────────────────────────────────────────────────────────────
def _blob(pts, center, sigma):
    diff = pts - center[np.newaxis, :]
    return np.exp(-0.5 * np.sum((diff / sigma) ** 2, axis=1))

def true_model_fn(x):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    v = (
        + 1.5 * _blob(x, np.array([0.30, 0.30, 0.50]), 0.12)
        + 1.0 * _blob(x, np.array([0.70, 0.65, 0.55]), 0.10)
        - 0.8 * _blob(x, np.array([0.55, 0.40, 0.45]), 0.08)
    )
    return v.squeeze()

m_true      = Function3D(box, evaluate_callable=true_model_fn)
m_true_grid = true_model_fn(EVAL_PTS)         # (N_eval,)
m_true_norm = np.sqrt(np.dot(m_true_grid**2, EVAL_W))

print(f"\nTrue model L2 norm (grid approx) = {m_true_norm:.4f}")

# Noise std is determined PER (seed, N_d) case as:
#   σ = NOISE_FRACTION * (||d_bar||_2 / sqrt(N_d))
# This is the "per-measurement RMS noise as fraction of per-measurement RMS signal"
# and ensures a meaningful SNR across all (seed, N_d) combinations.
# The σ values are reported in the CSV so comparisons remain valid.
print(f"\nNoise std will be computed per (seed, N_d) as: σ = {NOISE_FRACTION} × rms(d_bar)")

# ── Storage ───────────────────────────────────────────────────────────────────
# keyed by (seed, nd)
results = {}   # dict of metric dicts

# max N_d for eigenvalue storage  (store all eigs for the two largest N_d per seed)
STORE_EIGS_FOR = sorted(N_D_VALUES)[-3:]

# ── Main sweep ────────────────────────────────────────────────────────────────
n_total     = len(SEEDS) * len(N_D_VALUES)
n_completed = 0

for seed in SEEDS:
    rng_noise = np.random.default_rng(seed + 777)   # separate noise RNG

    for nd in N_D_VALUES:
        t0 = time.perf_counter()
        key = (seed, nd)

        print(f"\n[{n_completed+1}/{n_total}]  seed={seed}  N_d={nd}")

        # ── Build provider + solver ─────────────────────────────────────────
        fp = NormalModesProvider3D(
            box, random_state=seed, dim=nd,
            freq_range=FREQ_RANGE,
            gaussian_width_percent_range=GAUSSIAN_WIDTH_RANGE
        )
        solver = DenseVolumeSolaSolver(
            box, fp, resolution=QUAD_RESOLUTION,
            memory_budget_bytes=MEMORY_BYTES
        )
        G_op = solver.forward_operator(space)
        t_build = time.perf_counter() - t0
        print(f"  solver built in {t_build:.1f}s")

        # ── Gram matrix metrics ─────────────────────────────────────────────
        gram = solver.gram()                            # (nd, nd)
        eigvals = np.linalg.eigvalsh(gram)              # ascending
        eigvals = eigvals[::-1]                         # descending
        eigvals = np.maximum(eigvals, 0.0)              # clip numerical noise
        cond_gram = float(eigvals[0] / eigvals[-1]) if eigvals[-1] > 0 else np.inf

        print(f"  cond(GG*)={cond_gram:.2e}")

        # ── Data and reconstruction ─────────────────────────────────────────
        d_bar = G_op(m_true)

        # Per-(seed,nd) noise: σ = NOISE_FRACTION × rms(d_bar)
        gram_trace = float(np.trace(gram))
        rms_d      = float(np.linalg.norm(d_bar) / np.sqrt(nd))
        sigma      = NOISE_FRACTION * rms_d
        # Ensure a numeric floor so the Cholesky in the solver is stable
        sigma      = max(sigma, 1e-10 * np.sqrt(gram_trace / nd))
        noise      = rng_noise.normal(0.0, sigma, size=nd)
        d_noisy    = d_bar + noise

        snr = float(np.linalg.norm(d_bar) / np.linalg.norm(noise))
        print(f"  SNR = {snr:.2f}  σ = {sigma:.3e}")

        # Regularised matrix for condition number
        reg_gram = gram + sigma**2 * np.eye(nd)
        reg_eigs = np.linalg.eigvalsh(reg_gram)[::-1]
        cond_reg = float(reg_eigs[0] / reg_eigs[-1]) if reg_eigs[-1] > 0 else np.inf
        print(f"  cond(GG*+σ²I)={cond_reg:.2e}")

        # noise_cov takes σ (std dev), not σ² (variance)
        effective_cov = sigma

        m_hat = solver.reconstruct(d_noisy, noise_cov=effective_cov)
        t_recon = time.perf_counter() - t0 - t_build

        # ── Reconstruction quality ──────────────────────────────────────────
        m_hat_grid = m_hat(EVAL_PTS)                    # (N_eval,)
        err_grid   = m_hat_grid - m_true_grid
        recon_err  = float(np.sqrt(np.dot(err_grid**2, EVAL_W)) / m_true_norm)

        # Clean-data error (noiseless)
        d_clean     = d_bar.copy()
        m_hat_clean = solver.reconstruct(d_clean, noise_cov=effective_cov)
        m_hat_clean_grid = m_hat_clean(EVAL_PTS)
        err_clean  = m_hat_clean_grid - m_true_grid
        recon_err_clean = float(np.sqrt(np.dot(err_clean**2, EVAL_W)) / m_true_norm)

        print(f"  recon_err (noisy)  = {recon_err:.4f}")
        print(f"  recon_err (clean)  = {recon_err_clean:.4f}")

        # ── Data misfit of least-norm solution ──────────────────────────────
        d_hat   = G_op(m_hat)
        misfit  = float(np.linalg.norm(d_hat - d_noisy) / np.linalg.norm(d_noisy))

        # ── Noise amplification ||G*(GG*+σ²I)^{-2}G||_F via trace ──────────
        # = trace[G G*(GG*+σ²I)^{-2} G G*]
        # = trace[Λ² (Λ+σ²)^{-2}] where Λ = eigvals of GG* (pre-computed above)
        # noise amplification uses sigma (std dev), so regularisation is sigma^2
        noise_amp = float(np.sum((eigvals / (eigvals + sigma**2))**2))

        # ── Solution norm ||q||₂ (regularisation norm of coefficients) ──────
        try:
            q_noiseless = cho_solve(cho_factor(reg_gram), d_bar)
        except np.linalg.LinAlgError:
            q_noiseless = np.linalg.lstsq(reg_gram, d_bar, rcond=None)[0]
        sol_norm = float(np.linalg.norm(q_noiseless))

        # ── Relative bias (resolution error): how much error is signal-only ─
        # bias = ||T m_hat_clean - T m_true|| / ||T m_true|| for T = identity
        bias = recon_err_clean

        t_total = time.perf_counter() - t0
        print(f"  misfit={misfit:.4f}  noise_amp={noise_amp:.4e}  "
              f"sol_norm={sol_norm:.4e}  [{t_total:.1f}s total]")

        results[key] = dict(
            seed=seed, nd=nd,
            cond_gram=cond_gram,
            cond_reg=cond_reg,
            eigvals=(eigvals if nd in STORE_EIGS_FOR else None),
            recon_err=recon_err,
            recon_err_clean=recon_err_clean,
            bias=bias,
            misfit=misfit,
            noise_amp=noise_amp,
            sol_norm=sol_norm,
            sigma=float(sigma),
            snr=snr,
            t_total=t_total,
        )
        n_completed += 1

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = OUT_DIR / "sweep_results.csv"
with csv_path.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=[
        "seed", "nd",
        "recon_err", "recon_err_clean", "bias",
        "misfit", "noise_amp", "sol_norm",
        "cond_gram", "cond_reg", "sigma", "snr", "t_total"
    ])
    w.writeheader()
    for (seed, nd), m in sorted(results.items()):
        w.writerow({k: m[k] for k in w.fieldnames})
print(f"\nCSV saved to {csv_path}")

# ── Save NPZ ──────────────────────────────────────────────────────────────────
npz_data = {}
for (seed, nd), m in results.items():
    prefix = f"s{seed}_nd{nd}"
    npz_data[f"{prefix}_recon_err"] = np.float64(m["recon_err"])
    npz_data[f"{prefix}_recon_err_clean"] = np.float64(m["recon_err_clean"])
    npz_data[f"{prefix}_cond_gram"] = np.float64(m["cond_gram"])
    npz_data[f"{prefix}_cond_reg"]  = np.float64(m["cond_reg"])
    npz_data[f"{prefix}_noise_amp"] = np.float64(m["noise_amp"])
    if m["eigvals"] is not None:
        npz_data[f"{prefix}_eigvals"] = m["eigvals"]
np.savez(OUT_DIR / "sweep_results.npz", **npz_data)
print(f"NPZ saved to {OUT_DIR / 'sweep_results.npz'}")

# ── Plotting ──────────────────────────────────────────────────────────────────
nd_arr  = np.array(N_D_VALUES)
PALETTE = plt.cm.tab10.colors

def _to_matrix(key, seeds=SEEDS, nds=N_D_VALUES):
    """Extract (n_seeds × n_nds) matrix for metric `key`."""
    return np.array([[results[(s, nd)][key] for nd in nds] for s in seeds])


# ── 1. Reconstruction error vs N_d ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Least-norm reconstruction quality  (NormalModesProvider3D kernels)", y=1.01)

for ax, metric, label in [
    (axes[0], "recon_err",       "Relative L² error  (noisy data)"),
    (axes[1], "recon_err_clean", "Relative L² error  (clean data — bias only)"),
]:
    mat = _to_matrix(metric)
    mean = mat.mean(axis=0)
    lo   = mat.min(axis=0)
    hi   = mat.max(axis=0)

    for i, s in enumerate(SEEDS):
        row = mat[i]
        ax.plot(nd_arr, row, "o--", color=PALETTE[i], alpha=0.5,
                label=f"seed={s}")
    ax.fill_between(nd_arr, lo, hi, alpha=0.15, color="grey", label="min-max band")
    ax.plot(nd_arr, mean, "k-o", lw=2, label="mean")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N_d  (number of observations)"); ax.set_ylabel(label)
    ax.set_title(label)
    ax.legend(fontsize=7); ax.grid(True, which="both", ls=":")

plt.tight_layout()
plt.savefig(OUT_DIR / "recon_error.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved recon_error.png")


# ── 2. Condition numbers vs N_d ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Gram matrix condition numbers", y=1.01)

for ax, metric, label in [
    (axes[0], "cond_gram", "cond(GG*)"),
    (axes[1], "cond_reg",  "cond(GG* + σ²I)   (σ per-case)"),
]:
    mat = _to_matrix(metric)
    for i, s in enumerate(SEEDS):
        ax.plot(nd_arr, mat[i], "o--", color=PALETTE[i], alpha=0.6, label=f"seed={s}")
    ax.plot(nd_arr, mat.mean(axis=0), "k-o", lw=2, label="mean")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N_d"); ax.set_ylabel(label); ax.set_title(label)
    ax.legend(fontsize=7); ax.grid(True, which="both", ls=":")

plt.tight_layout()
plt.savefig(OUT_DIR / "condition_numbers.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved condition_numbers.png")


# ── 3. Noise amplification vs N_d ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
mat = _to_matrix("noise_amp")
for i, s in enumerate(SEEDS):
    ax.plot(nd_arr, mat[i], "o--", color=PALETTE[i], alpha=0.6, label=f"seed={s}")
ax.plot(nd_arr, mat.mean(axis=0), "k-o", lw=2, label="mean")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("N_d"); ax.set_ylabel("Σ [λᵢ/(λᵢ+σ²)]²  (noise amplification)")
ax.set_title("Noise amplification of least-norm estimator")
ax.legend(fontsize=7); ax.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig(OUT_DIR / "noise_amplification.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved noise_amplification.png")


# ── 4. Eigenvalue spectra for largest stored N_d ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
seed0 = SEEDS[0]
for nd in STORE_EIGS_FOR:
    eigs = results[(seed0, nd)].get("eigvals")
    if eigs is not None:
        ax.semilogy(np.arange(1, len(eigs)+1), eigs, label=f"N_d={nd}")
ax.set_xlabel("Eigenvalue index"); ax.set_ylabel("Eigenvalue of GG*")
ax.set_title(f"GG* eigenvalue spectra  (seed={seed0})")
ax.legend(); ax.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig(OUT_DIR / "eigenvalue_spectra.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved eigenvalue_spectra.png")


# ── 5. Summary table ─────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print(f"{'nd':<8}{'recon_err(mean)':<20}{'recon_err_clean(mean)':<24}"
      f"{'cond_gram(mean)':<20}{'noise_amp(mean)':<18}")
print("-" * 90)
for nd in N_D_VALUES:
    vals = [results[(s, nd)] for s in SEEDS]
    print(f"{nd:<8}"
          f"{np.mean([v['recon_err']       for v in vals]):<20.4f}"
          f"{np.mean([v['recon_err_clean'] for v in vals]):<24.4f}"
          f"{np.mean([v['cond_gram']       for v in vals]):<20.2e}"
          f"{np.mean([v['noise_amp']       for v in vals]):<18.2e}")
print("=" * 90)
print("\nDone.")
