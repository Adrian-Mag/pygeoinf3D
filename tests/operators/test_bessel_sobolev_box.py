"""Tests for BesselSobolevBox spectral operator (Phase 1).

Covers:
- Constructor validation (shift > 0, box domain identity)
- Eigenvalue formula: mu_k = (shift + lambda_k) ** (order / 2)
- Identity semantics for order=0 (all operator eigenvalues == 1.0)
- Positive and negative order both work through the unified formula
- SpectralOperator3D / SpectralPowerBase3D interface satisfaction
- Apply-to-eigenfunction correctness (eigenvalue-scale check)
"""

from __future__ import annotations

import pytest
import numpy as np
import numpy.testing as npt

from pygeoinf3d.geometries.box.domain import BoxDomain
from pygeoinf3d.providers.tensor_product import TensorProductProvider
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.operators.bessel_sobolev_box import BesselSobolevBox
from pygeoinf3d.operators.base import SpectralOperator3D
from pygeoinf3d.operators.spectral_power_base import SpectralPowerBase3D


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
    return TensorProductProvider(
        unit_box, truncation=(3, 3, 3), bc_type='dirichlet'
    )


@pytest.fixture
def neumann_provider(unit_box):
    return TensorProductProvider(
        unit_box, truncation=(3, 3, 3), bc_type='neumann'
    )


# ---------------------------------------------------------------------------
# TestConstructorValidation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    """BesselSobolevBox validates shift > 0 and matching box domain."""

    def test_construct_order2_shift1(self, space, dirichlet_provider):
        op = BesselSobolevBox(space, dirichlet_provider, shift=1.0, order=2.0)
        assert op is not None

    def test_construct_order0_shift1(self, space, dirichlet_provider):
        op = BesselSobolevBox(space, dirichlet_provider, shift=1.0, order=0.0)
        assert op is not None

    def test_construct_order_half_shift1(self, space, dirichlet_provider):
        op = BesselSobolevBox(space, dirichlet_provider, shift=1.0, order=0.5)
        assert op is not None

    def test_construct_order_negative_shift1(self, space, dirichlet_provider):
        op = BesselSobolevBox(space, dirichlet_provider, shift=1.0, order=-1.0)
        assert op is not None

    def test_construct_order_negative_float_shift1(
        self, space, dirichlet_provider
    ):
        op = BesselSobolevBox(space, dirichlet_provider, shift=1.0, order=-1.5)
        assert op is not None

    def test_construct_neumann_provider(self, space, neumann_provider):
        op = BesselSobolevBox(space, neumann_provider, shift=0.5, order=1.0)
        assert op is not None

    def test_raises_shift_zero(self, space, dirichlet_provider):
        with pytest.raises(ValueError, match="shift"):
            BesselSobolevBox(space, dirichlet_provider, shift=0.0, order=1.0)

    def test_raises_shift_negative(self, space, dirichlet_provider):
        with pytest.raises(ValueError, match="shift"):
            BesselSobolevBox(space, dirichlet_provider, shift=-1.0, order=1.0)

    def test_raises_shift_nan(self, space, dirichlet_provider):
        with pytest.raises(ValueError, match="finite"):
            BesselSobolevBox(
                space,
                dirichlet_provider,
                shift=np.nan,
                order=1.0,
            )

    def test_raises_shift_inf(self, space, dirichlet_provider):
        with pytest.raises(ValueError, match="finite"):
            BesselSobolevBox(
                space,
                dirichlet_provider,
                shift=np.inf,
                order=1.0,
            )

    def test_raises_domain_mismatch(self, space, dirichlet_provider):
        """Provider box differs from space domain → ValueError."""
        other_box = BoxDomain(np.zeros(3), 2.0 * np.ones(3))
        other_provider = TensorProductProvider(
            other_box, truncation=(3, 3, 3), bc_type='dirichlet'
        )
        with pytest.raises(ValueError):
            BesselSobolevBox(space, other_provider, shift=1.0, order=1.0)

    def test_shift_attribute_stored(self, space, dirichlet_provider):
        op = BesselSobolevBox(space, dirichlet_provider, shift=2.5, order=1.0)
        assert op.shift == pytest.approx(2.5)

    def test_order_attribute_stored(self, space, dirichlet_provider):
        op = BesselSobolevBox(space, dirichlet_provider, shift=1.0, order=3.0)
        assert op.order == pytest.approx(3.0)

    def test_is_spectral_operator3d(self, space, dirichlet_provider):
        op = BesselSobolevBox(space, dirichlet_provider, shift=1.0, order=1.0)
        assert isinstance(op, SpectralOperator3D)

    def test_is_spectral_power_base3d(self, space, dirichlet_provider):
        op = BesselSobolevBox(space, dirichlet_provider, shift=1.0, order=1.0)
        assert isinstance(op, SpectralPowerBase3D)

    def test_domain_is_space(self, space, dirichlet_provider):
        op = BesselSobolevBox(space, dirichlet_provider, shift=1.0, order=1.0)
        assert op.domain is space

    def test_codomain_is_space(self, space, dirichlet_provider):
        op = BesselSobolevBox(space, dirichlet_provider, shift=1.0, order=1.0)
        assert op.codomain is space

    def test_n_modes_equals_provider_dim(self, space, dirichlet_provider):
        op = BesselSobolevBox(space, dirichlet_provider, shift=1.0, order=1.0)
        assert op.n_modes == dirichlet_provider.dim

    def test_repr_includes_shift_and_order(self, space, dirichlet_provider):
        op = BesselSobolevBox(space, dirichlet_provider, shift=1.5, order=2.0)
        r = repr(op)
        assert "1.5" in r
        assert "2.0" in r


# ---------------------------------------------------------------------------
# TestEigenvalueSemantics
# ---------------------------------------------------------------------------

class TestEigenvalueSemantics:
    """Operator eigenvalue = (shift + lambda_k) ** (order / 2)."""

    def test_order0_all_eigenvalues_one(self, space, dirichlet_provider):
        """order=0: mu_k = (shift + lambda_k)^0 = 1.0 for every mode."""
        op = BesselSobolevBox(space, dirichlet_provider, shift=1.0, order=0.0)
        for i in range(op.n_modes):
            npt.assert_allclose(op.get_eigenvalue(i), 1.0, rtol=1e-12)

    def test_order0_shift_does_not_matter(self, space, dirichlet_provider):
        """order=0 gives identity regardless of shift value."""
        for shift in [0.1, 1.0, 10.0]:
            op = BesselSobolevBox(
                space,
                dirichlet_provider,
                shift=shift,
                order=0.0,
            )
            for i in range(op.n_modes):
                npt.assert_allclose(op.get_eigenvalue(i), 1.0, rtol=1e-12)

    def test_order2_eigenvalue_equals_shift_plus_lambda(
        self, space, dirichlet_provider
    ):
        """order=2: mu_k = (shift + lambda_k)^1 = shift + lambda_k."""
        shift = 1.0
        op = BesselSobolevBox(
            space,
            dirichlet_provider,
            shift=shift,
            order=2.0,
        )
        for i in range(op.n_modes):
            lam_k = dirichlet_provider.get_eigenvalue(i)
            npt.assert_allclose(
                op.get_eigenvalue(i),
                shift + lam_k,
                rtol=1e-12,
            )

    def test_order1_eigenvalue_equals_sqrt(self, space, dirichlet_provider):
        """order=1 gives the square root of shift plus eigenvalue."""
        shift = 2.0
        op = BesselSobolevBox(
            space,
            dirichlet_provider,
            shift=shift,
            order=1.0,
        )
        for i in range(op.n_modes):
            lam_k = dirichlet_provider.get_eigenvalue(i)
            npt.assert_allclose(
                op.get_eigenvalue(i), np.sqrt(shift + lam_k), rtol=1e-12
            )

    def test_order_half_eigenvalue_formula(self, space, dirichlet_provider):
        """order=0.5: mu_k = (shift + lambda_k)^0.25."""
        shift = 1.0
        op = BesselSobolevBox(
            space,
            dirichlet_provider,
            shift=shift,
            order=0.5,
        )
        for i in range(op.n_modes):
            lam_k = dirichlet_provider.get_eigenvalue(i)
            npt.assert_allclose(
                op.get_eigenvalue(i), (shift + lam_k) ** 0.25, rtol=1e-12
            )

    def test_order_minus1_eigenvalue_inverse_sqrt(
        self, space, dirichlet_provider
    ):
        """order=-1: mu_k = (shift + lambda_k)^(-1/2)."""
        shift = 1.0
        op = BesselSobolevBox(
            space,
            dirichlet_provider,
            shift=shift,
            order=-1.0,
        )
        for i in range(op.n_modes):
            lam_k = dirichlet_provider.get_eigenvalue(i)
            npt.assert_allclose(
                op.get_eigenvalue(i), (shift + lam_k) ** (-0.5), rtol=1e-12
            )

    def test_order_minus2_eigenvalue_inverse(self, space, dirichlet_provider):
        """order=-2 gives the reciprocal of shift plus eigenvalue."""
        shift = 0.5
        op = BesselSobolevBox(
            space,
            dirichlet_provider,
            shift=shift,
            order=-2.0,
        )
        for i in range(op.n_modes):
            lam_k = dirichlet_provider.get_eigenvalue(i)
            npt.assert_allclose(
                op.get_eigenvalue(i), 1.0 / (shift + lam_k), rtol=1e-12
            )

    def test_negative_order_neumann_zero_mode_no_singularity(
        self, space, neumann_provider
    ):
        """Positive shift prevents singularity for Neumann negative order."""
        shift = 1.0
        op = BesselSobolevBox(space, neumann_provider, shift=shift, order=-2.0)
        for i in range(op.n_modes):
            mu_k = op.get_eigenvalue(i)
            assert np.isfinite(mu_k), (
                f"Mode {i} eigenvalue is non-finite: {mu_k}"
            )
            assert mu_k > 0, f"Mode {i} eigenvalue is non-positive: {mu_k}"

    def test_eigenvalue_positive_for_positive_order(
        self, space, dirichlet_provider
    ):
        """All eigenvalues stay positive when shift is positive."""
        op = BesselSobolevBox(space, dirichlet_provider, shift=0.1, order=3.0)
        for i in range(op.n_modes):
            assert op.get_eigenvalue(i) > 0.0

    def test_eigenvalue_custom_shift_formula(self, space, dirichlet_provider):
        """Custom shift: formula is (shift + lambda_k)^(order/2)."""
        shift = 3.7
        order = 1.5
        op = BesselSobolevBox(
            space,
            dirichlet_provider,
            shift=shift,
            order=order,
        )
        for i in range(min(5, op.n_modes)):
            lam_k = dirichlet_provider.get_eigenvalue(i)
            expected = (shift + lam_k) ** (order / 2.0)
            npt.assert_allclose(op.get_eigenvalue(i), expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# TestApplySemantics
# ---------------------------------------------------------------------------

class TestApplySemantics:
    """BesselSobolevBox._apply scales each eigenfunction by its mu_k."""

    def _make_op(self, space, provider, shift, order, quad_resolution=20):
        return BesselSobolevBox(
            space,
            provider,
            shift=shift,
            order=order,
            quad_resolution=quad_resolution,
        )

    def test_apply_to_eigenfunction_scales_by_eigenvalue(
        self, space, dirichlet_provider
    ):
        """Applying to phi_k should return mu_k * phi_k pointwise."""
        shift = 1.0
        order = 2.0
        op = self._make_op(space, dirichlet_provider, shift=shift, order=order)

        # Use the first eigenfunction
        idx = 0
        phi0 = dirichlet_provider.get_eigenfunction(idx)
        mu0 = op.get_eigenvalue(idx)

        result = op(phi0)

        # Sample at a set of interior test points
        test_pts = np.array([
            [0.25, 0.25, 0.25],
            [0.5,  0.5,  0.5],
            [0.75, 0.25, 0.5],
        ])
        result_vals = np.asarray(result(test_pts), dtype=float)
        expected_vals = mu0 * np.asarray(phi0(test_pts), dtype=float)

        npt.assert_allclose(result_vals, expected_vals, rtol=1e-4, atol=1e-6)

    def test_apply_order0_identity(self, space, dirichlet_provider):
        """order=0 should act as the identity operator."""
        op = self._make_op(space, dirichlet_provider, shift=1.0, order=0.0)

        phi0 = dirichlet_provider.get_eigenfunction(0)
        result = op(phi0)

        test_pts = np.array([
            [0.3, 0.3, 0.3],
            [0.6, 0.4, 0.2],
        ])
        result_vals = np.asarray(result(test_pts), dtype=float)
        original_vals = np.asarray(phi0(test_pts), dtype=float)

        npt.assert_allclose(result_vals, original_vals, rtol=1e-4, atol=1e-6)

    def test_apply_negative_order_scales_correctly(
        self, space, dirichlet_provider
    ):
        """Negative order uses the same unified spectral formula."""
        shift = 2.0
        order = -1.0
        op = self._make_op(space, dirichlet_provider, shift=shift, order=order)

        idx = 1
        phi1 = dirichlet_provider.get_eigenfunction(idx)
        mu1 = op.get_eigenvalue(idx)

        result = op(phi1)

        test_pts = np.array([
            [0.4, 0.4, 0.4],
            [0.6, 0.6, 0.6],
        ])
        result_vals = np.asarray(result(test_pts), dtype=float)
        expected_vals = mu1 * np.asarray(phi1(test_pts), dtype=float)

        npt.assert_allclose(result_vals, expected_vals, rtol=1e-4, atol=1e-6)

    def test_apply_positive_and_negative_order_are_inverse(
        self, space, dirichlet_provider
    ):
        """order=s followed by order=-s on an eigenfunction recovers it."""
        np.random.seed(42)
        shift = 1.0
        s = 1.5
        op_pos = self._make_op(space, dirichlet_provider, shift=shift, order=s)
        op_neg = self._make_op(
            space,
            dirichlet_provider,
            shift=shift,
            order=-s,
        )

        phi0 = dirichlet_provider.get_eigenfunction(0)
        result = op_neg(op_pos(phi0))

        test_pts = np.array([
            [0.25, 0.25, 0.25],
            [0.5,  0.5,  0.5],
            [0.75, 0.5,  0.25],
        ])
        result_vals = np.asarray(result(test_pts), dtype=float)
        original_vals = np.asarray(phi0(test_pts), dtype=float)

        npt.assert_allclose(result_vals, original_vals, rtol=1e-3, atol=1e-5)
