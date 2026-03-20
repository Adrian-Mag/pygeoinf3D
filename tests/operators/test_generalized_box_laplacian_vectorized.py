"""Tests for the vectorized GeneralizedBoxLaplacian._apply (Phase 3).

Covers:
- Applying to an eigenfunction returns mu_k * phi_k pointwise
- Vectorized _apply matches the reference loop implementation
- Single-point evaluation returns a scalar float
- Batch-point evaluation returns shape (n,)
- power=0 (identity) returns a function matching the original
"""

from __future__ import annotations

import pytest
import numpy as np
import numpy.testing as npt

from pygeoinf3d.geometries.box.domain import BoxDomain
from pygeoinf3d.providers.tensor_product import BoxLaplacianSpectrumProvider
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.operators.generalized_box_laplacian import GeneralizedBoxLaplacian
from pygeoinf3d.core.functions import Function3D


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_box():
    return BoxDomain(np.zeros(3), np.ones(3))


@pytest.fixture
def space(unit_box):
    return Lebesgue3D(unit_box)


@pytest.fixture
def dirichlet_provider(unit_box):
    return BoxLaplacianSpectrumProvider(
        unit_box, truncation=(3, 3, 3), bc_type='dirichlet'
    )


@pytest.fixture
def neumann_provider(unit_box):
    return BoxLaplacianSpectrumProvider(
        unit_box, truncation=(3, 3, 3), bc_type='neumann'
    )


@pytest.fixture
def test_pts():
    rng = np.random.default_rng(42)
    return rng.uniform(0.05, 0.95, size=(30, 3))


# ---------------------------------------------------------------------------
# TestVectorizedApply
# ---------------------------------------------------------------------------

class TestVectorizedApply:
    """Vectorized _apply produces correct results."""

    def test_apply_vectorized_matches_eigenfunction(
        self, space, dirichlet_provider, test_pts
    ):
        """Applying the operator to eigenfunction phi_k should return mu_k * phi_k.

        For any eigenfunction phi_k of the operator, A(phi_k) = mu_k * phi_k
        by the spectral theorem.
        """
        op = GeneralizedBoxLaplacian(
            space, dirichlet_provider, power=2, alpha=1.0, quad_resolution=25
        )
        for k in range(min(4, op.n_modes)):
            mu_k = op.get_eigenvalue(k)
            phi_k = op.get_eigenfunction(k)
            result = op(phi_k)
            # Pointwise check: (A phi_k)(x) ≈ mu_k * phi_k(x)
            npt.assert_allclose(
                result(test_pts),
                mu_k * phi_k(test_pts),
                rtol=1e-2,
                err_msg=f"Eigenfunction application failed for mode k={k}",
            )

    def test_apply_vectorized_matches_loop(
        self, space, dirichlet_provider, test_pts
    ):
        """Vectorized _apply and the reference loop produce the same result.

        We compute the reference result by hand using the loop formula:
          c_k = <f, phi_k>_L2  (quadrature)
          result = sum_k mu_k * c_k * phi_k(x)
        and compare to the operator output.
        """
        op = GeneralizedBoxLaplacian(
            space, dirichlet_provider, power=1, alpha=1.0, quad_resolution=20
        )

        # Define a smooth, non-eigenfunction test function
        def f_callable(x):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                return float(np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]) * x[2])
            return np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]) * x[:, 2]

        f = Function3D(op.domain.function_domain, evaluate_callable=f_callable)

        # Apply via the operator
        result_op = op(f)

        # Compute reference via explicit loop
        pts_q = op._pts
        w_q = op._weights
        f_vals = np.asarray(f(pts_q), dtype=float)

        ref_vals = np.zeros(len(test_pts))
        for k in range(op.n_modes):
            mu_k = op.get_eigenvalue(k)
            phi_k = op.get_eigenfunction(k)
            phi_vals = np.asarray(phi_k(pts_q), dtype=float)
            c_k = float(np.dot(f_vals * phi_vals, w_q))
            ref_vals += mu_k * c_k * np.asarray(phi_k(test_pts), dtype=float)

        npt.assert_allclose(result_op(test_pts), ref_vals, rtol=1e-8)

    def test_apply_vectorized_single_point(self, space, dirichlet_provider):
        """Result evaluated at a single 3D point returns a Python float."""
        op = GeneralizedBoxLaplacian(
            space, dirichlet_provider, power=1, alpha=1.0, quad_resolution=15
        )
        phi0 = op.get_eigenfunction(0)
        result = op(phi0)

        single_pt = np.array([0.3, 0.4, 0.5])
        val = result(single_pt)
        assert isinstance(val, float), (
            f"Expected float for single-point evaluation, got {type(val)}"
        )

    def test_apply_vectorized_batch_points(self, space, dirichlet_provider, test_pts):
        """Result evaluated at a batch of 3D points returns shape (n,)."""
        op = GeneralizedBoxLaplacian(
            space, dirichlet_provider, power=1, alpha=1.0, quad_resolution=15
        )
        phi0 = op.get_eigenfunction(0)
        result = op(phi0)

        vals = result(test_pts)
        assert isinstance(vals, np.ndarray), (
            f"Expected ndarray for batch evaluation, got {type(vals)}"
        )
        assert vals.shape == (len(test_pts),), (
            f"Expected shape ({len(test_pts)},), got {vals.shape}"
        )

    def test_apply_vectorized_identity(
        self, space, dirichlet_provider, test_pts
    ):
        """power=0 (identity operator) applied to phi_k returns phi_k values.

        For power=0, all operator eigenvalues are 1.0, so the spectral
        projection is exactly the identity on the truncated basis.
        """
        op = GeneralizedBoxLaplacian(
            space, dirichlet_provider, power=0, quad_resolution=25
        )
        phi0 = dirichlet_provider.get_eigenfunction(0)
        result = op(phi0)
        # The truncated identity should reproduce phi0 to good accuracy
        npt.assert_allclose(
            result(test_pts),
            phi0(test_pts),
            rtol=1e-2,
            err_msg="power=0 should reproduce the eigenfunction",
        )
