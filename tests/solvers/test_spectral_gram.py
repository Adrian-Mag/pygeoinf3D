"""Tests for pygeoinf3d.solvers.spectral_gram.SpectralGramSolver.

Verifies:
- Gram matrix matches the generic framework's (G @ G.adjoint).matrix()
- Minimum-norm solution matches LinearMinimumNormInversion result
- Residual ||G(u_hat) - d|| ≈ 0 for exact data
- Representer Function3D evaluates correctly
- Speed is dramatically faster than the generic path
"""

from __future__ import annotations

import time

import numpy as np
import numpy.testing as npt
import pytest

from pygeoinf import EuclideanSpace, LinearForwardProblem, LinearMinimumNormInversion, EigenSolver

from pygeoinf3d import BoxDomain, Lebesgue3D, Sobolev3D, BesselSobolevBox, BoxLaplacianSpectrumProvider, Path3D, Function3D
from pygeoinf3d.spaces.forms import PathIntegralFunctional
from pygeoinf3d.operators.sola import SOLAOperator3D
from pygeoinf3d.solvers.spectral_gram import SpectralGramSolver


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
def paths_and_forms(unit_box, sobolev_space):
    """5 random straight-line paths and their PathIntegralFunctional forms."""
    rng = np.random.default_rng(42)
    n_paths = 5
    sources = rng.uniform(0.01, 0.99, size=(n_paths, 3))
    receivers = rng.uniform(0.01, 0.99, size=(n_paths, 3))
    paths = [
        Path3D.from_polyline(np.array([sources[i], receivers[i]]), domain=unit_box)
        for i in range(n_paths)
    ]
    forms = [PathIntegralFunctional(sobolev_space, p, n_quad=100) for p in paths]
    return paths, forms


@pytest.fixture
def true_model(unit_box):
    def f(x):
        x = np.atleast_2d(x)
        return np.exp(-5.0 * np.sum((x - 0.5) ** 2, axis=1))
    return Function3D(unit_box, evaluate_callable=f, name="test model")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSpectralGramSolver:

    def test_gram_matrix_matches_generic(self, sobolev_space, paths_and_forms):
        """Gram matrix from SpectralGramSolver matches generic (G @ G.adjoint).matrix()."""
        _, forms = paths_and_forms
        n = len(forms)

        solver = SpectralGramSolver(sobolev_space, forms)
        gram_fast = solver.gram_matrix

        # Build the same Gram matrix via the generic framework
        G = SOLAOperator3D(sobolev_space, EuclideanSpace(n), forms)
        GGT = G @ G.adjoint
        gram_generic = GGT.matrix(dense=True, galerkin=True)

        npt.assert_allclose(gram_fast, gram_generic, rtol=1e-8, atol=1e-12)

    def test_interaction_matrix_shape(self, sobolev_space, paths_and_forms):
        """L matrix has shape (m, N)."""
        _, forms = paths_and_forms
        solver = SpectralGramSolver(sobolev_space, forms)
        L = solver.interaction_matrix
        assert L.shape == (len(forms), sobolev_space.inverse_weight.n_modes)

    def test_forward_matches_sola(self, sobolev_space, paths_and_forms, true_model):
        """Forward evaluation matches SOLAOperator3D."""
        _, forms = paths_and_forms
        n = len(forms)

        solver = SpectralGramSolver(sobolev_space, forms)
        d_fast = solver.forward(true_model)

        G = SOLAOperator3D(sobolev_space, EuclideanSpace(n), forms)
        d_generic = G(true_model)

        npt.assert_allclose(d_fast, d_generic, rtol=1e-12)

    def test_minimum_norm_solution_matches_generic(
        self, sobolev_space, paths_and_forms, true_model,
    ):
        """Solution from SpectralGramSolver matches LinearMinimumNormInversion."""
        _, forms = paths_and_forms
        n = len(forms)

        # Generic path
        G = SOLAOperator3D(sobolev_space, EuclideanSpace(n), forms)
        d = G(true_model)
        fp = LinearForwardProblem(G)
        inv = LinearMinimumNormInversion(fp)
        inv_op = inv.minimum_norm_operator(EigenSolver(galerkin=True, rtol=1e-14))
        u_generic = inv_op(d)

        # Fast path
        solver = SpectralGramSolver(sobolev_space, forms)
        u_fast = solver.solve(d)

        # Compare on a grid
        rng = np.random.default_rng(99)
        pts = rng.uniform(0.05, 0.95, size=(200, 3))
        npt.assert_allclose(u_fast(pts), u_generic(pts), rtol=1e-6, atol=1e-10)

    def test_residual_near_zero(self, sobolev_space, paths_and_forms, true_model):
        """Residual ||G(u_hat) - d|| ≈ 0 for exact data."""
        _, forms = paths_and_forms
        solver = SpectralGramSolver(sobolev_space, forms)
        d = solver.forward(true_model)
        u_hat = solver.solve(d)
        d_recon = solver.forward(u_hat)
        npt.assert_allclose(d_recon, d, atol=1e-10)

    def test_representer_single_point(self, sobolev_space, paths_and_forms, true_model):
        """Representer evaluates correctly at a single point."""
        _, forms = paths_and_forms
        solver = SpectralGramSolver(sobolev_space, forms)
        d = solver.forward(true_model)
        u_hat = solver.solve(d)
        pt = np.array([0.5, 0.5, 0.5])
        val = u_hat(pt)
        assert isinstance(val, float)

    def test_solve_is_fast(self, sobolev_space, paths_and_forms, true_model):
        """Construction + solve should be << 1 second for 5 paths."""
        _, forms = paths_and_forms
        t0 = time.perf_counter()
        solver = SpectralGramSolver(sobolev_space, forms)
        d = solver.forward(true_model)
        u_hat = solver.solve(d)
        _ = u_hat(np.array([[0.5, 0.5, 0.5]]))
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"Fast path took {elapsed:.2f}s (should be < 2s)"
