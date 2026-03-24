"""sola_scale_benchmark.py — Benchmark DenseVolumeSolaSolver.solve() across N_d.

Measures wall time (build + solve), peak memory, and solution quality
(linear-system residual) for increasing numbers of data kernels.

Usage
-----
    conda run -n inferences3 python pygeoinf3D/rough_work/sola_scale_benchmark.py
    conda run -n inferences3 python pygeoinf3D/rough_work/sola_scale_benchmark.py \\
        --n-vals 100 500 1000 --memory-budget-gb 2.0

Results are saved to pygeoinf3D/rough_work/sola_scale_benchmark_results.txt.
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

# ── Path setup (allow running without editable install) ──────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _SCRIPT_DIR.parents[1]
for _repo in [_WORKSPACE_ROOT / "pygeoinf", _WORKSPACE_ROOT / "pygeoinf3D"]:
    _repo_str = str(_repo)
    if _repo.exists() and _repo_str not in sys.path:
        sys.path.insert(0, _repo_str)

from pygeoinf3d import BoxDomain                                     # noqa: E402
from pygeoinf3d.providers.function_providers import GaussianProvider3D  # noqa: E402
from pygeoinf3d.solvers import DenseVolumeSolaSolver                  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_N_VALS = [100, 500, 1000, 5000, 10000]
_RESOLUTION     = (15, 15, 15)   # 3375 quadrature points
_SIGMA          = 0.05
_NOISE_STD      = 0.1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--n-vals",
        nargs="+",
        type=int,
        default=_DEFAULT_N_VALS,
        metavar="N",
        help="list of N_d values to benchmark (default: 100 500 1000 5000 10000)",
    )
    p.add_argument(
        "--memory-budget-gb",
        type=float,
        default=0.5,
        metavar="GB",
        help="memory budget for K cache in GB (default: 0.5)",
    )
    return p.parse_args()


def _benchmark_one(
    n_d: int,
    rng: np.random.Generator,
    memory_budget_bytes: int,
) -> dict:
    """Run one benchmark for a given N_d and return a result dict."""
    box = BoxDomain(np.zeros(3), np.ones(3))
    centers = rng.uniform(0.1, 0.9, size=(n_d, 3))
    provider = GaussianProvider3D(box, centers, sigmas=_SIGMA)

    # ── Build: construct solver (includes Gram matrix + Cholesky) ────────────
    tracemalloc.start()
    t0 = time.perf_counter()
    solver = DenseVolumeSolaSolver(
        box,
        provider,
        resolution=_RESOLUTION,
        memory_budget_bytes=memory_budget_bytes,
    )
    build_time = time.perf_counter() - t0
    snap_after_build = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Peak memory during build (sum of top stats)
    stats = snap_after_build.statistics("lineno")
    peak_mb = sum(s.size for s in stats) / 1024**2

    # ── Solve: time a single solve call ──────────────────────────────────────
    d = rng.standard_normal(n_d)
    tracemalloc.start()
    t1 = time.perf_counter()
    q = solver.solve(d, noise_std=_NOISE_STD)
    solve_time = time.perf_counter() - t1
    snap_after_solve = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Update peak memory to include solve phase
    stats2 = snap_after_solve.statistics("lineno")
    peak_mb_total = max(peak_mb, sum(s.size for s in stats2) / 1024**2)

    # ── Solution quality: residual of the linear system (G + σ²I)q = d ──────
    #    G = K diag(w) K^T is stored in solver._gram_matrix (if K fits in budget)
    residual = float("nan")
    if solver._gram_matrix is not None:
        G = solver._gram_matrix                     # shape (N_d, N_d)
        r = G @ q + _NOISE_STD**2 * q - d          # should be ~0
        residual = float(np.linalg.norm(r) / (np.linalg.norm(d) + 1e-300))

    return {
        "n_d": n_d,
        "build_s": build_time,
        "solve_s": solve_time,
        "peak_mb": peak_mb_total,
        "residual": residual,
    }


def _format_table(rows: list[dict]) -> str:
    header = (
        f"{'N_d':>8} | {'Build(s)':>10} | {'Solve(s)':>10} | "
        f"{'Peak RAM(MB)':>13} | {'Residual':>12}"
    )
    sep = "-" * 9 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 15 + "+" + "-" * 14
    lines = [header, sep]
    for r in rows:
        res_str = f"{r['residual']:.2e}" if not np.isnan(r["residual"]) else "N/A (streaming)"
        lines.append(
            f"{r['n_d']:>8} | {r['build_s']:>10.3f} | {r['solve_s']:>10.4f} | "
            f"{r['peak_mb']:>13.1f} | {res_str:>12}"
        )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    memory_budget_bytes = int(args.memory_budget_gb * 1024**3)
    n_vals = sorted(set(args.n_vals))

    rng = np.random.default_rng(42)

    print(f"DenseVolumeSolaSolver benchmark — resolution={_RESOLUTION}, "
          f"sigma={_SIGMA}, noise_std={_NOISE_STD}")
    print(f"Memory budget: {args.memory_budget_gb:.1f} GB  |  N_d values: {n_vals}\n")

    rows = []
    for n_d in n_vals:
        # Quick skip if projected time is too long (>5 min) based on first probe
        if rows:
            last = rows[-1]
            # Rough O(N_d²) scaling for build, O(N_d) for solve
            scale = (n_d / last["n_d"]) ** 2
            projected_build = last["build_s"] * scale
            if projected_build > 300:
                print(f"N_d={n_d:>6}: skipped — projected build time "
                      f"~{projected_build:.0f}s (>5 min)")
                continue

        print(f"N_d={n_d:>6}: running ...", end=" ", flush=True)
        row = _benchmark_one(n_d, rng, memory_budget_bytes)
        rows.append(row)
        res_str = (f"{row['residual']:.2e}" if not np.isnan(row["residual"])
                   else "N/A (streaming)")
        print(
            f"build={row['build_s']:.3f}s  solve={row['solve_s']:.4f}s  "
            f"RAM={row['peak_mb']:.1f} MB  residual={res_str}"
        )

    print()
    table = _format_table(rows)
    print(table)

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = Path(__file__).with_name("sola_scale_benchmark_results.txt")
    with open(out_path, "w") as f:
        f.write(f"DenseVolumeSolaSolver benchmark\n")
        f.write(f"resolution={_RESOLUTION}  sigma={_SIGMA}  noise_std={_NOISE_STD}\n")
        f.write(f"memory_budget={args.memory_budget_gb:.1f} GB\n\n")
        f.write(table + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
