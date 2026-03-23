"""Tests for SpectralGramSolver with VolumeKernel3D forms (Phase 4).

Verifies:
1. L[i,k] for volume forms matches direct quadrature reference
2. Gram matrix for pure volume stack matches direct computation
3. Mixed path+volume stack: forward() matches direct form evaluations
4. Router recommends SpectralGramSolver for Sobolev3D + VolumeKernel3D stack
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from pygeoinf3d import (
    BoxDomain,
    Lebesgue3D,
    Sobolev3D,
    BesselSobolevBox,
    BoxLaplacianSpectrumProvider,
    Path3D,
    Function3D,
)
from pygeoinf3d.spaces.forms import PathIntegralFunctional, VolumeKernel3D
from pygeoinf3d.solvers.spectral_gram import SpectralGramSolver
from pygeoinf3d.solvers.router import SolverRouter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def unit_box():
    return BoxDomain(np.zeros(3), np.ones(3))


@pytest.fixture
def sobolev_space(unit_box):
    base_space = Lebesgue3D(unit_box)
    provider = BoxLaplacianSpectrumProvider(
        unit_box, truncation=(4, 4, 4), bc_type="neumann",
    )
    W = BesselSobolevBox(base_space, provider, shift=1.0, order=1.75)
    W_inv = BesselSobolevBox(base_space, provider, shift=1.0, order=-1.75)
    return Sobolev3D(base_space, W, W_inv)


@pytest.fixture
def gaussian_kernels(unit_box, sobolev_space):
    """3 Gaussian kernel Function3D objects on the unit box with known centres."""
    centres = np.array([
        [0.2, 0.2, 0.2],
        [0.5, 0.5, 0.5],
        [0.8, 0.8, 0.8],
    ])
    sigma = 0.2
    fns = []
    for c in centres:
        c_local = c.copy()

        def make_kernel(centre):
            def kernel_callable(x):
                x = np.asarray(x, dtype=float)
                if x.ndim == 1:
                    return float(np.exp(-0.5 * np.sum((x - centre) ** 2) / sigma ** 2))
                return np.exp(-0.5 * np.sum((x - centre) ** 2, axis=1) / sigma ** 2)
            return kernel_callable

        fns.append(Function3D(unit_box, evaluate_callable=make_kernel(c_local)))
    return fns


@pytest.fixture
def volume_forms(sobolev_space, gaussian_kernels):
    return [VolumeKernel3D(sobolev_space, k) for k in gaussian_kernels]


@pytest.fixture
def path_forms(unit_box, sobolev_space):
    """3 random straight-line paths and their PathIntegralFunctional forms."""
    rng = np.random.default_rng(7)
    n_paths = 3
    sources = rng.uniform(0.05, 0.95, size=(n_paths, 3))
    receivers = rng.uniform(0.05, 0.95, size=(n_paths, 3))
    paths = [
        Path3D.from_polyline(np.array([sources[i], receivers[i]]), domain=unit_box)
        for i in range(n_paths)
    ]
    return [PathIntegralFunctional(sobolev_space, p, n_quad=50) for p in paths]


# ---------------------------------------------------------------------------
# Test 1: L[i,k] matches direct quadrature
# ---------------------------------------------------------------------------


class TestSpectralGramVolumeInteractionMatrix:

    def test_spectral_interaction_coefficients_for_volume_form_match_direct_quadrature(
        self, unit_box, sobolev_space, volume_forms
    ):
        """L[i,k] from solver matches sum_j kernel_i(pts[j]) * Phi_k(pts[j]) * w[j]."""
        resolution = 12
        solver = SpectralGramSolver(sobolev_space, volume_forms, resolution=resolution)

        # Direct reference: build L manually
        pts = unit_box.structured_mesh(resolution, location="cell_centers")
        w = unit_box.volume_weights(resolution, location="cell_centers")

        inv_w = sobolev_space.inverse_weight
        provider = inv_w._provider
        Phi = provider.evaluate_all_eigenfunctions(pts)  # (N, n_pts)

        n_forms = len(volume_forms)
        N = provider.dim
        L_ref = np.empty((n_forms, N))
        for i, form in enumerate(volume_forms):
            k_vals = form.kernel(pts)  # (n_pts,)
            L_ref[i, :] = (k_vals * w) @ Phi.T  # (N,)

        npt.assert_allclose(solver._L, L_ref, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Test 2: Gram matrix for pure volume stack matches direct computation
# ---------------------------------------------------------------------------


class TestSpectralGramVolumeGramMatrix:

    def test_spectral_gram_solver_volume_stack_gram_matches_direct_quadrature(
        self, unit_box, sobolev_space, volume_forms
    ):
        """Gram matrix from solver matches L_ref @ diag(nu) @ L_ref.T."""
        resolution = 12
        solver = SpectralGramSolver(sobolev_space, volume_forms, resolution=resolution)

        pts = unit_box.structured_mesh(resolution, location="cell_centers")
        w = unit_box.volume_weights(resolution, location="cell_centers")

        inv_w = sobolev_space.inverse_weight
        provider = inv_w._provider
        Phi = provider.evaluate_all_eigenfunctions(pts)
        nu = solver.eigenvalues  # (N,)

        n_forms = len(volume_forms)
        N = provider.dim
        L_ref = np.empty((n_forms, N))
        for i, form in enumerate(volume_forms):
            k_vals = form.kernel(pts)
            L_ref[i, :] = (k_vals * w) @ Phi.T

        G_ref = L_ref * nu[np.newaxis, :] @ L_ref.T
        npt.assert_allclose(solver.gram_matrix, G_ref, rtol=1e-8, atol=1e-12)


# ---------------------------------------------------------------------------
# Test 3: Mixed path + volume stack — forward() matches direct evaluation
# ---------------------------------------------------------------------------


class TestSpectralGramMixedStack:

    def test_spectral_gram_solver_supports_mixed_path_and_volume_stack(
        self, sobolev_space, path_forms, volume_forms, unit_box
    ):
        """forward(f) on mixed stack matches direct form evaluations."""
        forms = path_forms + volume_forms
        solver = SpectralGramSolver(sobolev_space, forms, resolution=12)

        # Smooth test function
        def f_call(x):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                return float(np.exp(-5.0 * np.sum((x - 0.5) ** 2)))
            return np.exp(-5.0 * np.sum((x - 0.5) ** 2, axis=1))

        f = Function3D(unit_box, evaluate_callable=f_call, name="test")

        d_fast = solver.forward(f)
        d_direct = np.array([form(f) for form in forms], dtype=float)

        npt.assert_allclose(d_fast, d_direct, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 4: Router recommends SpectralGramSolver for Sobolev3D + VolumeKernel3D
# ---------------------------------------------------------------------------


class TestSolverRouterVolumeOnSobolev:

    def test_router_accepts_sobolev_volume_stack(self, sobolev_space, volume_forms):
        """Router recommends SpectralGramSolver for Sobolev3D + all VolumeKernel3D."""
        router = SolverRouter()
        rec = router.recommend(sobolev_space, volume_forms)
        assert rec.solver_spec is not None
        assert rec.solver_spec.name == "SpectralGramSolver"

    def test_router_accepts_sobolev_mixed_stack(
        self, sobolev_space, path_forms, volume_forms
    ):
        """Router recommends SpectralGramSolver for Sobolev3D + mixed forms."""
        forms = path_forms + volume_forms
        router = SolverRouter()
        rec = router.recommend(sobolev_space, forms)
        assert rec.solver_spec is not None
        assert rec.solver_spec.name == "SpectralGramSolver"
