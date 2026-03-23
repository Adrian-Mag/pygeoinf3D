"""End-to-end minimum-norm inversion with path-integral measurements.

Demonstrates:
  1.  Build a Sobolev3D space with BesselSobolevBox weight.
  2.  Define path-integral measurement forms.
  3.  Build SOLAOperator3D with Sobolev3D domain.
  4.  Run LinearMinimumNormInversion from pygeoinf.
  5.  Verify the adjoint works and the inversion residual is small.
"""

import numpy as np
from pygeoinf import (
    EuclideanSpace, LinearForwardProblem,
    LinearMinimumNormInversion, EigenSolver,
)
from pygeoinf3d import BoxDomain, Lebesgue3D, Function3D, SOLAOperator3D
from pygeoinf3d.providers.tensor_product import BoxLaplacianSpectrumProvider
from pygeoinf3d.operators.bessel_sobolev_box import BesselSobolevBox
from pygeoinf3d.spaces.sobolev import Sobolev3D
from pygeoinf3d.spaces.forms import PathIntegralFunctional
from pygeoinf3d.core.paths import Path3D

np.random.seed(42)

# --- 1. Domain and Sobolev space ---
box = BoxDomain(np.zeros(3), np.ones(3))
base_space = Lebesgue3D(box)
provider = BoxLaplacianSpectrumProvider(box, truncation=(6, 6, 6), bc_type="neumann")
W     = BesselSobolevBox(base_space, provider, shift=1.0, order=4.0)
W_inv = BesselSobolevBox(base_space, provider, shift=1.0, order=-4.0)
sobolev = Sobolev3D(base_space, W, W_inv)

# --- 2. True model (sum of Gaussians) ---
def true_model_callable(x):
    x = np.asarray(x, dtype=float)
    single = x.ndim == 1
    pts = x[None, :] if single else x
    c1 = np.array([0.3, 0.5, 0.5])
    c2 = np.array([0.7, 0.5, 0.5])
    val = (1.5 * np.exp(-np.sum((pts - c1)**2, axis=1) / (2 * 0.1**2))
           - 1.0 * np.exp(-np.sum((pts - c2)**2, axis=1) / (2 * 0.12**2)))
    return float(val[0]) if single else val

u_true = Function3D(box, evaluate_callable=true_model_callable, name="true model")

# --- 3. Path-integral measurement forms (6 paths) ---
n_paths = 6
waypoints = [
    (np.array([0.1, float(i) / (n_paths - 1) * 0.8 + 0.1, 0.5]),
     np.array([0.9, float(i) / (n_paths - 1) * 0.8 + 0.1, 0.5]))
    for i in range(n_paths)
]
paths = [Path3D.from_polyline(np.array([a, b])) for a, b in waypoints]
forms = [PathIntegralFunctional(base_space, p, n_quad=100) for p in paths]
G = SOLAOperator3D(sobolev, EuclideanSpace(n_paths), forms)

# --- 4. Synthetic data ---
d = G(u_true)
print("Data:", np.round(d, 4))

# --- 5. Minimum-norm inversion ---
forward_problem = LinearForwardProblem(G)
inversion = LinearMinimumNormInversion(forward_problem)
inverse_op = inversion.minimum_norm_operator(
    EigenSolver(galerkin=True, rtol=1e-10),
    minimum_damping=1e-6,
)
u_hat = inverse_op(d)
print("Residual ||Gu_hat - d||:", np.round(np.linalg.norm(G(u_hat) - d), 6))

# --- 6. Quick sanity check at centre ---
centre = np.array([[0.5, 0.5, 0.5]])
print(f"u_true at centre: {u_true(centre)[0]:.4f}")
print(f"u_hat  at centre: {u_hat(centre)[0]:.4f}")
