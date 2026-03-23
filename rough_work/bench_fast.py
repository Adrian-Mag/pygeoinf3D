"""Benchmark SpectralGramSolver at various N_PATHS."""
import numpy as np
import time
import sys

sys.path.insert(0, "/home/adrian/PhD/Inferences/pygeoinf")
sys.path.insert(0, "/home/adrian/PhD/Inferences/pygeoinf3D")

from pygeoinf3d import BoxDomain, Lebesgue3D, Sobolev3D, BesselSobolevBox, BoxLaplacianSpectrumProvider, Path3D, Function3D
from pygeoinf3d.spaces.forms import PathIntegralFunctional
from pygeoinf3d.solvers.spectral_gram import SpectralGramSolver

box = BoxDomain(np.zeros(3), np.ones(3))
base_space = Lebesgue3D(box)
TRUNCATION = (8, 8, 8)
provider = BoxLaplacianSpectrumProvider(box, truncation=TRUNCATION, bc_type="neumann")
W = BesselSobolevBox(base_space, provider, shift=1.0, order=1.75)
W_inv = BesselSobolevBox(base_space, provider, shift=1.0, order=-1.75)
sobolev = Sobolev3D(base_space, W, W_inv)
lo, hi = box.bounding_box().low, box.bounding_box().high

centres = [(0.25, 0.25, 0.25), (0.75, 0.50, 0.75), (0.50, 0.75, 0.25)]
amplitudes = [1.0, 0.8, 0.6]
scales_m = [0.15, 0.12, 0.18]

def true_model_callable(x):
    x = np.atleast_2d(x)
    result = np.zeros(x.shape[0])
    for c, a, s in zip(centres, amplitudes, scales_m):
        diff = x - np.array(c)
        result += a * np.exp(-0.5 * np.sum(diff**2, axis=1) / s**2)
    return result

u_true = Function3D(box, evaluate_callable=true_model_callable, name="true model")

def generate_paths(n_paths, rng):
    face_labels = ["x_min", "x_max", "y_min", "y_max", "z_min"]
    face_indices = rng.choice(5, size=n_paths, p=[0.2]*5)
    sources = np.empty((n_paths, 3))
    for k, fi in enumerate(face_indices):
        f = face_labels[fi]
        if f == "x_min":
            sources[k] = [lo[0], rng.uniform(lo[1], hi[1]), rng.uniform(lo[2], hi[2])]
        elif f == "x_max":
            sources[k] = [hi[0], rng.uniform(lo[1], hi[1]), rng.uniform(lo[2], hi[2])]
        elif f == "y_min":
            sources[k] = [rng.uniform(lo[0], hi[0]), lo[1], rng.uniform(lo[2], hi[2])]
        elif f == "y_max":
            sources[k] = [rng.uniform(lo[0], hi[0]), hi[1], rng.uniform(lo[2], hi[2])]
        else:
            sources[k] = [rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1]), lo[2]]
    receivers = np.column_stack([
        rng.uniform(lo[0], hi[0], size=n_paths),
        rng.uniform(lo[1], hi[1], size=n_paths),
        np.full(n_paths, hi[2]),
    ])
    return [Path3D.from_polyline(np.array([sources[i], receivers[i]]), domain=box) for i in range(n_paths)]

# Evaluation grid for convergence
grid_1d = np.linspace(0.05, 0.95, 15)
xx, yy, zz = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing="ij")
eval_pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
u_true_grid = true_model_callable(eval_pts)

header = f"{'N':>5} | {'build':>8} | {'solve':>8} | {'total':>8} | {'resid':>10} | {'L2_err':>10}"
print(header)
print("-" * len(header))
sys.stdout.flush()

for n_paths in [10, 20, 40, 60, 100, 200, 500]:
    rng = np.random.default_rng(42)
    paths = generate_paths(n_paths, rng)
    forms = [PathIntegralFunctional(sobolev, p, n_quad=300) for p in paths]

    sigma = 0.01  # small noise

    t0 = time.perf_counter()
    solver = SpectralGramSolver(sobolev, forms)
    t_build = time.perf_counter() - t0

    d = solver.forward(u_true)
    d_noisy = d + rng.normal(0, sigma, n_paths)

    t0 = time.perf_counter()
    u_hat = solver.solve(d_noisy)
    t_solve = time.perf_counter() - t0

    t_total = t_build + t_solve

    d_recon = solver.forward(u_hat)
    resid = np.linalg.norm(d_recon - d_noisy)
    u_hat_grid = u_hat(eval_pts)
    l2_err = np.sqrt(np.mean((u_hat_grid - u_true_grid)**2))

    print(f"{n_paths:>5} | {t_build:>7.3f}s | {t_solve:>7.4f}s | {t_total:>7.3f}s | {resid:>10.4e} | {l2_err:>10.4f}")
    sys.stdout.flush()

print("\nDone.")
