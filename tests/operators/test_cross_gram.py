"""Tests for SpectralGramSolver._build_L_matrix (Phase 1 — extraction).

Verifies:
- Pure path-form stack: _build_L_matrix returns (L, nu) matching solver attributes
- Pure volume-form stack: same
- Mixed path+volume stack: same
- Non-spectral inverse_weight: raises TypeError
"""

from __future__ import annotations

from unittest.mock import MagicMock

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
def path_forms(unit_box, sobolev_space):
    rng = np.random.default_rng(99)
    sources = rng.uniform(0.05, 0.95, size=(3, 3))
    receivers = rng.uniform(0.05, 0.95, size=(3, 3))
    paths = [
        Path3D.from_polyline(np.array([sources[i], receivers[i]]), domain=unit_box)
        for i in range(3)
    ]
    return [PathIntegralFunctional(sobolev_space, p, n_quad=40) for p in paths]


@pytest.fixture
def vol_forms(unit_box, sobolev_space):
    centres = np.array([[0.2, 0.2, 0.2], [0.7, 0.7, 0.7]])
    sigma = 0.2

    def make_kernel(centre):
        def fn(x):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                return float(np.exp(-0.5 * np.sum((x - centre) ** 2) / sigma ** 2))
            return np.exp(-0.5 * np.sum((x - centre) ** 2, axis=1) / sigma ** 2)
        return fn

    kernels = [Function3D(unit_box, evaluate_callable=make_kernel(c)) for c in centres]
    return [VolumeKernel3D(sobolev_space, k) for k in kernels]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildLMatrix:

    def test_path_forms_matches_interaction_matrix(self, sobolev_space, path_forms):
        solver = SpectralGramSolver(sobolev_space, path_forms)
        L, nu = SpectralGramSolver._build_L_matrix(sobolev_space, path_forms)
        npt.assert_allclose(L, solver.interaction_matrix, rtol=1e-12)
        npt.assert_allclose(nu, solver.eigenvalues, rtol=1e-12)

    def test_vol_forms_matches_interaction_matrix(self, sobolev_space, vol_forms):
        solver = SpectralGramSolver(sobolev_space, vol_forms, resolution=15)
        L, nu = SpectralGramSolver._build_L_matrix(sobolev_space, vol_forms, resolution=15)
        npt.assert_allclose(L, solver.interaction_matrix, rtol=1e-12)
        npt.assert_allclose(nu, solver.eigenvalues, rtol=1e-12)

    def test_mixed_forms_matches_interaction_matrix(
        self, sobolev_space, path_forms, vol_forms
    ):
        mixed = [path_forms[0], vol_forms[0]]
        solver = SpectralGramSolver(sobolev_space, mixed, resolution=15)
        L, nu = SpectralGramSolver._build_L_matrix(sobolev_space, mixed, resolution=15)
        npt.assert_allclose(L, solver.interaction_matrix, rtol=1e-12)
        npt.assert_allclose(nu, solver.eigenvalues, rtol=1e-12)

    def test_rejects_non_spectral_sobolev(self, unit_box, path_forms):
        base_space = Lebesgue3D(unit_box)
        mock_w = MagicMock()
        mock_w.domain = base_space
        mock_w.codomain = base_space
        non_spectral_sobolev = Sobolev3D(base_space, mock_w, mock_w)
        with pytest.raises(TypeError):
            SpectralGramSolver._build_L_matrix(non_spectral_sobolev, path_forms)

    def test_rejects_unsupported_form_type(self, sobolev_space):
        """A form that is neither PathIntegralFunctional nor VolumeKernel3D must raise TypeError."""
        mock_form = MagicMock()
        # Ensure it is NOT a PathIntegralFunctional or VolumeKernel3D instance
        mock_form.__class__ = type("UnsupportedForm", (), {})
        with pytest.raises(TypeError, match="unsupported type"):
            SpectralGramSolver._build_L_matrix(sobolev_space, [mock_form])


# ---------------------------------------------------------------------------
# Phase 2: cross-Gram fast path via SOLAOperator3D.__matmul__
# ---------------------------------------------------------------------------

from pygeoinf import EuclideanSpace
from pygeoinf.linear_operators import MatrixLinearOperator
from pygeoinf3d.operators.sola import SOLAOperator3D


@pytest.fixture
def T_op(sobolev_space, vol_forms):
    """SOLAOperator3D with 2 volume-kernel forms (targets)."""
    codomain = EuclideanSpace(len(vol_forms))
    return SOLAOperator3D(sobolev_space, codomain, vol_forms)


@pytest.fixture
def G_op(sobolev_space, path_forms):
    """SOLAOperator3D with 3 path-integral forms (measurements)."""
    codomain = EuclideanSpace(len(path_forms))
    return SOLAOperator3D(sobolev_space, codomain, path_forms)


class TestCrossGramFastPath:

    def test_T_at_G_adjoint_returns_matrix_linear_operator(self, T_op, G_op):
        result = T_op @ G_op.adjoint
        assert isinstance(result, MatrixLinearOperator)

    def test_G_at_T_adjoint_returns_matrix_linear_operator(self, T_op, G_op):
        result = G_op @ T_op.adjoint
        assert isinstance(result, MatrixLinearOperator)

    def test_cross_gram_value_matches_slow_path(self, T_op, G_op, sobolev_space):
        fast_op = T_op @ G_op.adjoint
        fast_mat = fast_op.matrix(dense=True, galerkin=True)

        # Slow reference: column-by-column evaluation
        n_d = G_op.codomain.dim
        slow_cols = []
        for j in range(n_d):
            e_j = G_op.codomain.basis_vector(j)
            col_j = T_op(G_op.adjoint(e_j))
            slow_cols.append(np.asarray(col_j, dtype=float))
        slow_mat = np.column_stack(slow_cols)  # shape (N_p, N_d)

        np.testing.assert_allclose(fast_mat, slow_mat, rtol=1e-5)

    def test_self_gram_still_uses_existing_fast_path(self, G_op):
        result = G_op @ G_op.adjoint
        assert isinstance(result, MatrixLinearOperator)

    def test_cross_gram_is_cached(self, T_op, G_op):
        op1 = T_op @ G_op.adjoint
        op2 = T_op @ G_op.adjoint
        assert op1 is op2

    def test_lebesgue_domain_skips_fast_path(self, unit_box, vol_forms):
        """T@G.adjoint on Lebesgue3D (non-Sobolev) must not use cross-gram fast path.

        T_l2 is not G_l2, so the self-gram path also does not fire.
        The result should be a generic composition, not a MatrixLinearOperator.
        """
        lebesgue = Lebesgue3D(unit_box)
        from pygeoinf3d.spaces.forms import VolumeKernel3D
        from pygeoinf3d.core.functions import Function3D

        centres = np.array([[0.2, 0.2, 0.2], [0.7, 0.7, 0.7]])
        sigma = 0.2

        def make_kernel(centre):
            def fn(x):
                x = np.asarray(x, dtype=float)
                if x.ndim == 1:
                    return float(np.exp(-0.5 * np.sum((x - centre) ** 2) / sigma ** 2))
                return np.exp(-0.5 * np.sum((x - centre) ** 2, axis=1) / sigma ** 2)
            return fn

        kernels_l2 = [
            VolumeKernel3D(lebesgue, Function3D(unit_box, evaluate_callable=make_kernel(c)))
            for c in centres
        ]
        T_l2 = SOLAOperator3D(lebesgue, EuclideanSpace(2), kernels_l2)
        # Separate instance with same kernels but different SOLAOperator3D object
        G_l2 = SOLAOperator3D(lebesgue, EuclideanSpace(2), kernels_l2)

        result = T_l2 @ G_l2.adjoint
        # Cross-gram path requires Sobolev3D domain; Lebesgue3D should not trigger it.
        # Self-gram path requires other._adjoint_base is self (T_l2 is not G_l2).
        # So the result should be a generic (non-matrix) composition.
        assert not isinstance(result, MatrixLinearOperator)
