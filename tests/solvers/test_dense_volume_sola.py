"""Tests for DenseVolumeSolaSolver (Phase 3).

Verifies:
1. Gram matrix from DenseVolumeSolaSolver matches generic SOLAOperator3D Gram
2. Resolving coefficients match direct solve via generic Gram matrix
3. Forward map matches SOLAOperator3D forward evaluation
4. SolverRouter recommends DenseVolumeSolaSolver for BoxDomain+GaussianProvider3D+Lebesgue3D

Mathematical background
-----------------------
For m VolumeKernel3D forms {ℓ_i | ℓ_i(f) = ∫ k_i · f dV} on Lebesgue3D:
- Gram matrix  G[i,j] = ⟨k_i, k_j⟩ = ∫ k_i k_j dV ≈ K @ diag(w) @ K^T
- Forward map  d[i] = ℓ_i(f) = K[i,:] · (w * f_vals)
- Resolve      q satisfying G @ q = cross, where cross[i] = ∫ k_i k_target dV
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from pygeoinf import EuclideanSpace

from pygeoinf3d import BoxDomain, Lebesgue3D, Function3D
from pygeoinf3d.providers.function_providers.gaussian import GaussianProvider3D
from pygeoinf3d.spaces.forms import VolumeKernel3D
from pygeoinf3d.operators.sola import SOLAOperator3D
from pygeoinf3d.solvers.dense_volume_sola import DenseVolumeSolaSolver
from pygeoinf3d.solvers.router import SolverRouter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_box():
    return BoxDomain(np.zeros(3), np.ones(3))


@pytest.fixture
def lebesgue_space(unit_box):
    return Lebesgue3D(unit_box)


@pytest.fixture
def gaussian_provider(unit_box):
    """3 Gaussian kernels at different positions, sigma=0.2."""
    centers = np.array([
        [0.2, 0.2, 0.2],
        [0.5, 0.5, 0.5],
        [0.8, 0.8, 0.8],
    ])
    return GaussianProvider3D(unit_box, centers, sigmas=0.2)


@pytest.fixture
def sola_solver(unit_box, gaussian_provider):
    """DenseVolumeSolaSolver with resolution=20 to match domain.integrate_volume default."""
    return DenseVolumeSolaSolver(unit_box, gaussian_provider, resolution=20)


@pytest.fixture
def sola_forms(lebesgue_space, gaussian_provider):
    """Corresponding VolumeKernel3D forms on Lebesgue3D."""
    return [
        VolumeKernel3D(lebesgue_space, gaussian_provider.get_function(i))
        for i in range(gaussian_provider.dim)
    ]


@pytest.fixture
def generic_sola_operator(lebesgue_space, sola_forms):
    """SOLAOperator3D wrapping the same kernels."""
    n = len(sola_forms)
    return SOLAOperator3D(lebesgue_space, EuclideanSpace(n), sola_forms)


@pytest.fixture
def test_function(unit_box):
    """Smooth test function: exp(-3||x - 0.5||²)."""
    def f(x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return float(np.exp(-3.0 * np.sum((x - 0.5) ** 2)))
        return np.exp(-3.0 * np.sum((x - 0.5) ** 2, axis=1))

    return Function3D(unit_box, evaluate_callable=f, name="test_function")


# ---------------------------------------------------------------------------
# Test 1: Gram matrix matches generic SOLAOperator3D
# ---------------------------------------------------------------------------

class TestDenseVolumeSolaGram:

    def test_gram_matches_generic_sola_operator(
        self, sola_solver, generic_sola_operator
    ):
        """DenseVolumeSolaSolver.gram() must equal the Gram matrix of SOLAOperator3D.

        Both use resolution=20 cell-center quadrature for  G[i,j] = ∫ k_i k_j dV,
        so the results must agree to machine precision (<< rtol=1e-8).
        """
        gram_fast = sola_solver.gram()

        # Generic path: (A @ A†).matrix(dense=True, galerkin=True)
        GGT = generic_sola_operator @ generic_sola_operator.adjoint
        gram_generic = GGT.matrix(dense=True, galerkin=True)

        npt.assert_allclose(gram_fast, gram_generic, rtol=1e-8, atol=1e-14)

    def test_gram_is_symmetric(self, sola_solver):
        """Gram matrix must be symmetric."""
        G = sola_solver.gram()
        npt.assert_allclose(G, G.T, atol=1e-14)

    def test_gram_is_positive_definite(self, sola_solver):
        """Gram matrix must have all positive eigenvalues."""
        G = sola_solver.gram()
        eigvals = np.linalg.eigvalsh(G)
        assert np.all(eigvals > 0), f"Non-positive eigenvalues: {eigvals}"


# ---------------------------------------------------------------------------
# Test 2: Resolving coefficients match generic Gram solve
# ---------------------------------------------------------------------------

class TestDenseVolumeSolaResolve:

    def test_resolving_coefficients_match_direct_solve(self, sola_solver):
        """resolve(target_row) must equal np.linalg.solve(Gram, cross).

        Verifies the Cholesky factorisation is correct.
        """
        K = sola_solver._backend.interaction_matrix()
        w = sola_solver._backend._weights

        # Use first kernel's quadrature values as the target
        target_row = K[0, :]
        cross = K @ (w * target_row)

        q_fast = sola_solver.resolve(target_row)
        q_direct = np.linalg.solve(sola_solver.gram(), cross)

        npt.assert_allclose(q_fast, q_direct, rtol=1e-10, atol=1e-14)

    def test_cross_gram_matches_generic_form_evaluation(
        self, sola_solver, sola_forms, gaussian_provider
    ):
        """K @ (w * target_row) must equal [form_i(target_fn) for form_i in forms].

        Confirms that the DenseVolumeSolaSolver's cross-gram computation
        is numerically consistent with VolumeKernel3D.__call__ at resolution=20.
        """
        K = sola_solver._backend.interaction_matrix()
        w = sola_solver._backend._weights

        # Use the last kernel as the target (independent of the first)
        target_idx = len(sola_forms) - 1
        target_fn = gaussian_provider.get_function(target_idx)
        target_row = K[target_idx, :]

        # Fast cross-gram: K @ (w * target_row)
        cross_fast = K @ (w * target_row)

        # Generic cross-gram: form_i(target_fn) = ∫ k_i * target_fn dV
        cross_generic = np.array(
            [form(target_fn) for form in sola_forms], dtype=float
        )

        # Both use resolution=20 quadrature → should match to near machine eps
        npt.assert_allclose(cross_fast, cross_generic, rtol=1e-10, atol=1e-14)

    def test_resolving_satisfies_normal_equations(self, sola_solver):
        """G @ q = cross must hold exactly for the resolved coefficients."""
        K = sola_solver._backend.interaction_matrix()
        w = sola_solver._backend._weights

        target_row = K[1, :]  # use second kernel row as target
        q = sola_solver.resolve(target_row)
        cross = K @ (w * target_row)
        G = sola_solver.gram()

        residual = G @ q - cross
        npt.assert_allclose(residual, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 3: Forward map matches SOLAOperator3D
# ---------------------------------------------------------------------------

class TestDenseVolumeSolaForward:

    def test_forward_matches_sola_operator(
        self, sola_solver, generic_sola_operator, test_function
    ):
        """DenseVolumeSolaSolver.forward(f) must equal SOLAOperator3D(f).

        Both compute d[i] = ∫ k_i f dV at resolution=20, so results must agree
        to near machine precision.
        """
        d_fast = sola_solver.forward(test_function)
        d_generic = generic_sola_operator(test_function)

        npt.assert_allclose(d_fast, d_generic, rtol=1e-10, atol=1e-14)

    def test_forward_shape(self, sola_solver, test_function, gaussian_provider):
        """forward(f) returns array of shape (n_kernels,)."""
        d = sola_solver.forward(test_function)
        assert d.shape == (gaussian_provider.dim,)

    def test_forward_is_linear_in_f(self, sola_solver, unit_box):
        """Forward map is linear: forward(a·f + b·g) = a·forward(f) + b·forward(g)."""
        def f_call(x):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                return float(x[0])
            return x[:, 0]

        def g_call(x):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                return float(x[1])
            return x[:, 1]

        f = Function3D(unit_box, evaluate_callable=f_call)
        g = Function3D(unit_box, evaluate_callable=g_call)
        a, b = 3.0, -2.0

        def fg_call(x):
            return a * f_call(x) + b * g_call(x)

        fg = Function3D(unit_box, evaluate_callable=fg_call)

        d_f = sola_solver.forward(f)
        d_g = sola_solver.forward(g)
        d_fg = sola_solver.forward(fg)

        npt.assert_allclose(d_fg, a * d_f + b * d_g, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# Test 4: SolverRouter recommendation
# ---------------------------------------------------------------------------

class TestSolverRouterDenseVolumeSola:

    def test_router_recommends_dense_volume_sola_for_box_gaussian_stack(
        self, unit_box, lebesgue_space, gaussian_provider, sola_forms
    ):
        """SolverRouter.recommend with provider=GaussianProvider3D on Lebesgue3D+BoxDomain
        must recommend DenseVolumeSolaSolver.
        """
        router = SolverRouter()
        rec = router.recommend(lebesgue_space, sola_forms, provider=gaussian_provider)
        assert rec.solver_spec is not None, (
            f"Expected DenseVolumeSolaSolver recommendation, got reason: {rec.reason!r}"
        )
        assert rec.solver_spec.name == "DenseVolumeSolaSolver", (
            f"Expected DenseVolumeSolaSolver, got {rec.solver_spec.name!r}"
        )

    def test_router_still_recommends_sola_without_provider(
        self, lebesgue_space, sola_forms
    ):
        """Without provider kwarg, existing SOLAOperator3D routing must be unchanged."""
        router = SolverRouter()
        rec = router.recommend(lebesgue_space, sola_forms)
        assert rec.solver_spec is not None
        assert rec.solver_spec.name == "SOLAOperator3D"

    def test_router_dense_volume_sola_recommendation_has_reason(
        self, unit_box, lebesgue_space, gaussian_provider, sola_forms
    ):
        """Recommendation must include a non-empty reason string."""
        router = SolverRouter()
        rec = router.recommend(lebesgue_space, sola_forms, provider=gaussian_provider)
        assert isinstance(rec.reason, str) and len(rec.reason) > 0
