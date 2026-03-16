"""Tests for BesselSobolevBox spectral operator (Phases 1-2).

Covers:
- Constructor validation (shift > 0, box domain identity)
- Eigenvalue formula: mu_k = (shift + lambda_k) ** (order / 2)
- Identity semantics for order=0 (all operator eigenvalues == 1.0)
- Positive and negative order both work through the unified formula
- SpectralOperator3D / SpectralPowerBase3D interface satisfaction
- Apply-to-eigenfunction correctness (eigenvalue-scale check)
- Ladder semantics: B(-s)∘B(s)=I, B(s1)∘B(s2)=B(s1+s2)
- Self-adjointness: <Bf, g> = <f, Bg> in L²
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


# ---------------------------------------------------------------------------
# TestLadderSemantics
# ---------------------------------------------------------------------------

class TestLadderSemantics:
    """Verify operator action, self-adjointness and ladder composition."""

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_op(space, provider, shift, order, quad_resolution=20):
        return BesselSobolevBox(
            space,
            provider,
            shift=shift,
            order=order,
            quad_resolution=quad_resolution,
        )

    @staticmethod
    def _quad_inner(f, g, box, res=20):
        """Numerical L² inner product of two Function3D objects."""
        pts = box.structured_mesh(res)
        weights = box.volume_weights(res)
        f_vals = np.asarray(f(pts), dtype=float)
        g_vals = np.asarray(g(pts), dtype=float)
        return float(np.dot(f_vals * g_vals, weights))

    # -----------------------------------------------------------------------
    # Tests
    # -----------------------------------------------------------------------

    def test_apply_all_eigenfunctions(
        self, space, dirichlet_provider
    ):
        """Apply to each eigenfunction gives mu_k * phi_k pointwise.

        Checks order=1 and order=2 for the first 4 modes.
        """
        test_pts = np.array([
            [0.25, 0.25, 0.25],
            [0.5,  0.5,  0.5],
            [0.75, 0.25, 0.5],
        ])
        for order in [1.0, 2.0]:
            op = self._make_op(
                space, dirichlet_provider, shift=1.0, order=order
            )
            n_check = min(4, op.n_modes)
            for idx in range(n_check):
                phi_k = dirichlet_provider.get_eigenfunction(idx)
                mu_k = op.get_eigenvalue(idx)
                result = op(phi_k)
                result_vals = np.asarray(result(test_pts), dtype=float)
                expected_vals = mu_k * np.asarray(
                    phi_k(test_pts), dtype=float
                )
                npt.assert_allclose(
                    result_vals,
                    expected_vals,
                    rtol=1e-4,
                    atol=1e-6,
                    err_msg=(
                        f"order={order}, mode={idx}: "
                        "B*phi_k != mu_k*phi_k"
                    ),
                )

    def test_self_adjointness_spectral(
        self, space, dirichlet_provider
    ):
        """<B f, g> = <f, B g> in L² for smooth test functions."""
        np.random.seed(7)
        shift = 1.0
        order = 1.5
        op = self._make_op(
            space, dirichlet_provider, shift=shift, order=order
        )
        n = dirichlet_provider.dim
        # Build two functions as random linear combinations of eigenfunctions
        c_f = np.random.randn(n)
        c_g = np.random.randn(n)

        # f = sum_k c_f[k] * phi_k
        eigenfuncs = [
            dirichlet_provider.get_eigenfunction(k) for k in range(n)
        ]

        def make_combo(coeffs):
            snapshot = list(zip(coeffs, eigenfuncs))

            def _eval(x):
                x = np.asarray(x, dtype=float)
                total = np.zeros(
                    len(x) if x.ndim == 2 else ()
                )
                for c, phi in snapshot:
                    total = total + c * np.asarray(phi(x), dtype=float)
                return total

            from pygeoinf3d.core.functions import Function3D
            return Function3D(
                space.function_domain, evaluate_callable=_eval
            )

        f = make_combo(c_f)
        g = make_combo(c_g)

        box = space.function_domain
        bf_dot_g = self._quad_inner(op(f), g, box)
        f_dot_bg = self._quad_inner(f, op(g), box)

        npt.assert_allclose(
            bf_dot_g,
            f_dot_bg,
            rtol=1e-4,
            atol=1e-8,
            err_msg="Self-adjointness failed: <Bf,g> != <f,Bg>",
        )

    def test_ladder_composition_identity(
        self, space, dirichlet_provider
    ):
        """B(order=s) then B(order=-s) recovers the input function."""
        test_pts = np.array([
            [0.3, 0.3, 0.3],
            [0.5, 0.5, 0.5],
            [0.7, 0.4, 0.3],
        ])
        # Build a sum of the first 3 eigenfunctions
        n_modes = min(3, dirichlet_provider.dim)
        coeffs = np.array([1.0, 0.5, -0.3])[:n_modes]
        eigenfuncs = [
            dirichlet_provider.get_eigenfunction(k) for k in range(n_modes)
        ]

        def original_vals(pts):
            total = np.zeros(len(pts))
            for c, phi in zip(coeffs, eigenfuncs):
                total = total + c * np.asarray(phi(pts), dtype=float)
            return total

        for s in [0.5, 1.0, 2.0]:
            op_pos = self._make_op(
                space, dirichlet_provider, shift=1.0, order=s
            )
            op_neg = self._make_op(
                space, dirichlet_provider, shift=1.0, order=-s
            )
            # Build function object to pass to operators
            snap = list(zip(coeffs, eigenfuncs))

            def _eval_f(x, _snap=snap):
                x = np.asarray(x, dtype=float)
                t = np.zeros(len(x) if x.ndim == 2 else ())
                for c, phi in _snap:
                    t = t + c * np.asarray(phi(x), dtype=float)
                return t

            from pygeoinf3d.core.functions import Function3D
            f = Function3D(
                space.function_domain, evaluate_callable=_eval_f
            )
            result = op_neg(op_pos(f))
            result_vals = np.asarray(result(test_pts), dtype=float)
            expected = original_vals(test_pts)
            npt.assert_allclose(
                result_vals,
                expected,
                rtol=1e-3,
                atol=1e-5,
                err_msg=f"Ladder identity failed for s={s}",
            )

    def test_ladder_composition_additive(
        self, space, dirichlet_provider
    ):
        """B(s1) ∘ B(s2) = B(s1+s2) in the spectral sense."""
        s1, s2 = 1.0, 0.5
        shift = 1.0
        op_s1 = self._make_op(
            space, dirichlet_provider, shift=shift, order=s1
        )
        op_s2 = self._make_op(
            space, dirichlet_provider, shift=shift, order=s2
        )
        op_sum = self._make_op(
            space, dirichlet_provider, shift=shift, order=s1 + s2
        )
        test_pts = np.array([
            [0.25, 0.25, 0.25],
            [0.5,  0.5,  0.5],
            [0.75, 0.25, 0.5],
        ])
        n_check = min(3, dirichlet_provider.dim)
        for idx in range(n_check):
            phi_k = dirichlet_provider.get_eigenfunction(idx)
            # B(s1) ∘ B(s2) applied to phi_k
            composed_vals = np.asarray(
                op_s1(op_s2(phi_k))(test_pts), dtype=float
            )
            # B(s1+s2) applied to phi_k
            direct_vals = np.asarray(op_sum(phi_k)(test_pts), dtype=float)
            npt.assert_allclose(
                composed_vals,
                direct_vals,
                rtol=1e-3,
                atol=1e-6,
                err_msg=(
                    f"Additive composition failed for mode {idx}"
                ),
            )

    def test_output_domain_is_space_domain(
        self, space, dirichlet_provider
    ):
        """Function3D returned by _apply has correct domain."""
        op = self._make_op(
            space, dirichlet_provider, shift=1.0, order=1.0
        )
        phi0 = dirichlet_provider.get_eigenfunction(0)
        result = op(phi0)
        assert result.domain is space.function_domain

    def test_apply_matches_manual_spectral_sum(
        self, space, dirichlet_provider
    ):
        """Operator output matches hand-computed spectral expansion."""
        np.random.seed(13)
        shift = 2.0
        order = 1.0
        op = self._make_op(
            space, dirichlet_provider, shift=shift, order=order
        )
        n = dirichlet_provider.dim
        # Random coefficients for the input function
        c_in = np.random.randn(n)
        eigenfuncs = [
            dirichlet_provider.get_eigenfunction(k) for k in range(n)
        ]
        mu = [op.get_eigenvalue(k) for k in range(n)]

        test_pts = np.array([
            [0.2, 0.3, 0.4],
            [0.5, 0.5, 0.5],
            [0.8, 0.2, 0.6],
        ])

        # Build input function
        snap_in = list(zip(c_in, eigenfuncs))

        def _eval_f(x, _snap=snap_in):
            x = np.asarray(x, dtype=float)
            t = np.zeros(len(x) if x.ndim == 2 else ())
            for c, phi in _snap:
                t = t + c * np.asarray(phi(x), dtype=float)
            return t

        from pygeoinf3d.core.functions import Function3D
        f = Function3D(space.function_domain, evaluate_callable=_eval_f)

        # Apply operator
        result = op(f)
        result_vals = np.asarray(result(test_pts), dtype=float)

        # Manual sum: sum_k mu_k * c_k * phi_k(x)
        manual = np.zeros(len(test_pts))
        for c_k, phi_k, mu_k in zip(c_in, eigenfuncs, mu):
            manual += mu_k * c_k * np.asarray(phi_k(test_pts), dtype=float)

        npt.assert_allclose(
            result_vals,
            manual,
            rtol=1e-4,
            atol=1e-8,
            err_msg="Operator output differs from manual spectral sum",
        )

    def test_neumann_all_modes_finite(
        self, space, neumann_provider
    ):
        """Negative order + shift=1 keeps all eigenvalues finite."""
        shift = 1.0
        order = -2.0
        op = self._make_op(
            space, neumann_provider, shift=shift, order=order
        )
        # Check eigenvalues
        for i in range(op.n_modes):
            mu_k = op.get_eigenvalue(i)
            assert np.isfinite(mu_k), (
                f"Eigenvalue {i} is non-finite: {mu_k}"
            )
            assert mu_k > 0.0, (
                f"Eigenvalue {i} is non-positive: {mu_k}"
            )
        # Check apply on the zero-mode eigenfunction
        phi0 = neumann_provider.get_eigenfunction(0)
        result = op(phi0)
        test_pts = np.array([[0.5, 0.5, 0.5]])
        val = np.asarray(result(test_pts), dtype=float)
        assert np.all(np.isfinite(val)), (
            f"Apply result is non-finite: {val}"
        )
