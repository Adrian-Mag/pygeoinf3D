"""Integration tests for DenseVolumeSolaSolver.forward_operator (Phase 5).

Verifies that the precomputed forward matrix can be exposed as a
pygeoinf.LinearOperator that integrates cleanly with the abstract
Hilbert-space framework, enabling GaussianMeasure.affine_mapping.

Tests
-----
1. forward_operator returns a LinearOperator with correct domain/codomain.
2. affine_mapping through forward_operator produces a valid GaussianMeasure;
   its covariance matches the expected matrix formula (G²).
3. (A @ A*) equals the Gram matrix — verifies the adjoint mapping uses the
   cached K matrix consistently.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from pygeoinf import EuclideanSpace, LinearOperator, GaussianMeasure

from pygeoinf3d import BoxDomain, Lebesgue3D, Function3D
from pygeoinf3d.providers.function_providers.gaussian import GaussianProvider3D
from pygeoinf3d.spaces.forms import VolumeKernel3D
from pygeoinf3d.operators.sola import SOLAOperator3D
from pygeoinf3d.solvers.dense_volume_sola import DenseVolumeSolaSolver


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
    """DenseVolumeSolaSolver with resolution=20."""
    return DenseVolumeSolaSolver(unit_box, gaussian_provider, resolution=20)


@pytest.fixture
def sola_forms(lebesgue_space, gaussian_provider):
    """VolumeKernel3D forms for each kernel."""
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
    """Smooth test function: exp(-3‖x - 0.5‖²)."""
    def f(x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return float(np.exp(-3.0 * np.sum((x - 0.5) ** 2)))
        return np.exp(-3.0 * np.sum((x - 0.5) ** 2, axis=1))
    return Function3D(unit_box, evaluate_callable=f, name="test_function")


# ---------------------------------------------------------------------------
# Test 1: forward_operator returns a LinearOperator with correct structure
# ---------------------------------------------------------------------------

class TestForwardOperatorStructure:

    def test_returns_linear_operator(self, sola_solver, lebesgue_space):
        """forward_operator must return a pygeoinf.LinearOperator."""
        op = sola_solver.forward_operator(lebesgue_space)
        assert isinstance(op, LinearOperator)

    def test_domain_is_function_space(self, sola_solver, lebesgue_space):
        """Operator domain must be exactly the supplied function space."""
        op = sola_solver.forward_operator(lebesgue_space)
        assert op.domain is lebesgue_space

    def test_codomain_has_correct_dim(self, sola_solver, lebesgue_space):
        """Codomain must be EuclideanSpace with dim == n_kernels."""
        op = sola_solver.forward_operator(lebesgue_space)
        assert op.codomain.dim == sola_solver.n_kernels

    def test_application_matches_solver_forward(
        self, sola_solver, lebesgue_space, test_function
    ):
        """op(f) must give the same result as solver.forward(f)."""
        op = sola_solver.forward_operator(lebesgue_space)
        result_op = np.asarray(op(test_function))
        result_direct = sola_solver.forward(test_function)
        npt.assert_allclose(result_op, result_direct, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Test 2: affine_mapping through forward_operator
# ---------------------------------------------------------------------------

class TestAffineMappingIntegration:

    def test_affine_mapping_returns_gaussian_measure(
        self, sola_solver, lebesgue_space, sola_forms, generic_sola_operator
    ):
        """prior.affine_mapping(forward_operator) must return a GaussianMeasure."""
        # Covariance = A* @ A  (SPD operator on Lebesgue3D)
        covariance = generic_sola_operator.adjoint @ generic_sola_operator
        prior = GaussianMeasure(covariance=covariance, expectation=lebesgue_space.zero)

        A_fwd = sola_solver.forward_operator(lebesgue_space)
        data_measure = prior.affine_mapping(operator=A_fwd)

        assert isinstance(data_measure, GaussianMeasure)

    def test_affine_mapping_domain_is_euclidean(
        self, sola_solver, lebesgue_space, generic_sola_operator
    ):
        """Pushed-forward measure must live in EuclideanSpace(n_kernels)."""
        covariance = generic_sola_operator.adjoint @ generic_sola_operator
        prior = GaussianMeasure(covariance=covariance, expectation=lebesgue_space.zero)

        A_fwd = sola_solver.forward_operator(lebesgue_space)
        data_measure = prior.affine_mapping(operator=A_fwd)

        assert isinstance(data_measure.domain, EuclideanSpace)
        assert data_measure.domain.dim == sola_solver.n_kernels

    def test_affine_mapping_expectation_is_zero(
        self, sola_solver, lebesgue_space, generic_sola_operator
    ):
        """With zero prior mean, data measure expectation should also be zero."""
        covariance = generic_sola_operator.adjoint @ generic_sola_operator
        prior = GaussianMeasure(covariance=covariance, expectation=lebesgue_space.zero)

        A_fwd = sola_solver.forward_operator(lebesgue_space)
        data_measure = prior.affine_mapping(operator=A_fwd)

        npt.assert_allclose(
            data_measure.expectation,
            np.zeros(sola_solver.n_kernels),
            atol=1e-12,
        )

    def test_affine_mapping_covariance_equals_gram_squared(
        self, sola_solver, lebesgue_space, generic_sola_operator
    ):
        """Data covariance must equal G² (solver.gram() @ solver.gram()).

        When covariance = A_sola* @ A_sola and forward = A_fwd = same map,
        data_cov = A_fwd @ (A_sola* @ A_sola) @ A_fwd* = G @ G.
        """
        covariance = generic_sola_operator.adjoint @ generic_sola_operator
        prior = GaussianMeasure(covariance=covariance, expectation=lebesgue_space.zero)

        A_fwd = sola_solver.forward_operator(lebesgue_space)
        data_measure = prior.affine_mapping(operator=A_fwd)

        G = sola_solver.gram()
        expected = G @ G  # G²

        # data_measure.covariance : EuclideanSpace(m) → EuclideanSpace(m)
        data_cov_matrix = data_measure.covariance.matrix(dense=True, galerkin=True)

        npt.assert_allclose(data_cov_matrix, expected, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 3: (A @ A*) matches the Gram matrix exactly
# ---------------------------------------------------------------------------

class TestForwardOperatorGramStructure:

    def test_A_Astar_equals_gram(self, sola_solver, lebesgue_space):
        """(A @ A*).matrix(dense=True) must equal solver.gram() exactly.

        This verifies that the adjoint mapping returns a Function3D whose
        values at the quadrature points are K^T @ d, so that the full
        composition K @ diag(w) @ K^T recovers the precomputed Gram matrix.
        """
        A_fwd = sola_solver.forward_operator(lebesgue_space)
        AAt = A_fwd @ A_fwd.adjoint  # EuclideanSpace(m) → EuclideanSpace(m)

        gram_via_operator = AAt.matrix(dense=True, galerkin=True)
        gram_direct = sola_solver.gram()

        npt.assert_allclose(gram_via_operator, gram_direct, rtol=1e-8, atol=1e-14)

    def test_forward_operator_different_calls_are_independent(
        self, sola_solver, lebesgue_space
    ):
        """Two calls to forward_operator return independent objects with same result."""
        A1 = sola_solver.forward_operator(lebesgue_space)
        A2 = sola_solver.forward_operator(lebesgue_space)

        assert A1 is not A2  # separate objects
        assert A1.domain is A2.domain  # same domain
