"""Phase 1 tests for GeneralizedBoxLaplacian integer-power semantics.

Covers:
- Constructor validation (integer power, positive alpha, box identity)
- Eigenvalue formula: (alpha * lambda_k)^power for all integer powers
- Negative-power pseudoinverse suppression of Neumann zero modes
- Identity semantics for power=0 (all operator eigenvalues == 1.0)
- Eigenvalue-level equivalence with BoxLaplacian (power=1) and
  InverseBoxLaplacian pseudoinverse (power=-1)
- SpectralOperator3D interface satisfaction (n_modes, get_eigenfunction, ...)
- Apply-semantics tests (xfail: Phase 2 _apply implementation pending)

Phase 1 does NOT implement _apply; tests requiring actual function evaluation
are decorated with @pytest.mark.xfail(strict=False) to document intent
without blocking the suite.
"""

from __future__ import annotations

import pytest
import numpy as np
import numpy.testing as npt

from pygeoinf3d.geometries.box.domain import BoxDomain
from pygeoinf3d.providers.tensor_product import TensorProductProvider
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.operators.laplacian_box import BoxLaplacian
from pygeoinf3d.operators.inverse_laplacian_box import InverseBoxLaplacian
from pygeoinf3d.operators.generalized_box_laplacian import GeneralizedBoxLaplacian
from pygeoinf3d.operators.base import SpectralOperator3D


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
    """GeneralizedBoxLaplacian validates integer power and positive alpha."""

    def test_construct_power1_default_alpha(self, space, dirichlet_provider):
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=1)
        assert op is not None

    def test_construct_power_neg1_neumann(self, space, neumann_provider):
        op = GeneralizedBoxLaplacian(space, neumann_provider, power=-1)
        assert op is not None

    def test_construct_power0(self, space, dirichlet_provider):
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=0)
        assert op is not None

    def test_construct_power2(self, space, dirichlet_provider):
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=2)
        assert op is not None

    def test_construct_power_neg2(self, space, dirichlet_provider):
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=-2)
        assert op is not None

    def test_construct_custom_alpha(self, space, dirichlet_provider):
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=1, alpha=2.5)
        assert op is not None

    def test_raises_non_integer_power_float(self, space, dirichlet_provider):
        with pytest.raises((TypeError, ValueError)):
            GeneralizedBoxLaplacian(space, dirichlet_provider, power=0.5)

    def test_raises_non_integer_power_neg_float(self, space, dirichlet_provider):
        with pytest.raises((TypeError, ValueError)):
            GeneralizedBoxLaplacian(space, dirichlet_provider, power=-0.5)

    def test_raises_non_integer_power_pi(self, space, dirichlet_provider):
        with pytest.raises((TypeError, ValueError)):
            GeneralizedBoxLaplacian(space, dirichlet_provider, power=float(np.pi))

    def test_raises_non_positive_alpha_zero(self, space, dirichlet_provider):
        with pytest.raises(ValueError, match="alpha"):
            GeneralizedBoxLaplacian(space, dirichlet_provider, power=1, alpha=0.0)

    def test_raises_non_positive_alpha_negative(self, space, dirichlet_provider):
        with pytest.raises(ValueError, match="alpha"):
            GeneralizedBoxLaplacian(space, dirichlet_provider, power=1, alpha=-1.0)

    def test_raises_domain_mismatch(self, unit_box, space, dirichlet_provider):
        """Provider box differs from space domain → ValueError."""
        other_box = BoxDomain(np.zeros(3), 2.0 * np.ones(3))
        other_provider = TensorProductProvider(
            other_box, truncation=(3, 3, 3), bc_type='dirichlet'
        )
        with pytest.raises(ValueError):
            GeneralizedBoxLaplacian(space, other_provider, power=1)

    def test_power_attribute_stored(self, space, dirichlet_provider):
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=2)
        assert op.power == 2

    def test_alpha_attribute_stored(self, space, dirichlet_provider):
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=1, alpha=3.0)
        assert op.alpha == pytest.approx(3.0)

    def test_is_spectral_operator3d(self, space, dirichlet_provider):
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=1)
        assert isinstance(op, SpectralOperator3D)

    def test_domain_is_space(self, space, dirichlet_provider):
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=1)
        assert op.domain == space

    def test_codomain_is_space(self, space, dirichlet_provider):
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=1)
        assert op.codomain == space

    def test_n_modes_equals_provider_dim(self, space, dirichlet_provider):
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=1)
        assert op.n_modes == dirichlet_provider.dim

    def test_repr_includes_power_and_alpha(self, space, dirichlet_provider):
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=2, alpha=1.5)
        r = repr(op)
        assert "power=2" in r
        assert "1.5" in r


# ---------------------------------------------------------------------------
# TestEigenvalueSemantics: mu_k = (alpha * lambda_k)^power
# ---------------------------------------------------------------------------

class TestEigenvalueSemantics:
    """Operator eigenvalue = (alpha * lambda_k)^power (positive power cases)."""

    def test_power1_alpha1_eigenvalue_equals_laplacian_ev(
        self, space, dirichlet_provider
    ):
        """power=1, alpha=1: mu_k = lambda_k."""
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=1, alpha=1.0)
        for i in range(4):
            lam_k = dirichlet_provider.get_eigenvalue(i)
            npt.assert_allclose(op.get_eigenvalue(i), lam_k, rtol=1e-12)

    def test_power1_alpha2_eigenvalue_scaled(
        self, space, dirichlet_provider
    ):
        """power=1, alpha=2: mu_k = 2 * lambda_k."""
        alpha = 2.0
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=1, alpha=alpha)
        for i in range(4):
            lam_k = dirichlet_provider.get_eigenvalue(i)
            npt.assert_allclose(op.get_eigenvalue(i), alpha * lam_k, rtol=1e-12)

    def test_power2_eigenvalue_squared(self, space, dirichlet_provider):
        """power=2, alpha=1: mu_k = lambda_k^2."""
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=2, alpha=1.0)
        for i in range(4):
            lam_k = dirichlet_provider.get_eigenvalue(i)
            npt.assert_allclose(op.get_eigenvalue(i), lam_k ** 2, rtol=1e-12)

    def test_power2_custom_alpha_eigenvalue(self, space, dirichlet_provider):
        """power=2, alpha=3: mu_k = (3 * lambda_k)^2."""
        alpha = 3.0
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=2, alpha=alpha)
        for i in range(4):
            lam_k = dirichlet_provider.get_eigenvalue(i)
            npt.assert_allclose(
                op.get_eigenvalue(i), (alpha * lam_k) ** 2, rtol=1e-12
            )

    def test_power3_eigenvalue_cubed(self, space, dirichlet_provider):
        """power=3, alpha=1: mu_k = lambda_k^3."""
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=3, alpha=1.0)
        for i in range(4):
            lam_k = dirichlet_provider.get_eigenvalue(i)
            npt.assert_allclose(op.get_eigenvalue(i), lam_k ** 3, rtol=1e-12)

    def test_power_neg1_dirichlet_eigenvalue_inverted(
        self, space, dirichlet_provider
    ):
        """power=-1, alpha=1, Dirichlet: mu_k = 1/lambda_k (no zero modes)."""
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=-1, alpha=1.0)
        for i in range(4):
            lam_k = dirichlet_provider.get_eigenvalue(i)
            npt.assert_allclose(op.get_eigenvalue(i), 1.0 / lam_k, rtol=1e-12)

    def test_power_neg2_dirichlet_eigenvalue(self, space, dirichlet_provider):
        """power=-2, alpha=1, Dirichlet: mu_k = 1/lambda_k^2."""
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=-2, alpha=1.0)
        for i in range(4):
            lam_k = dirichlet_provider.get_eigenvalue(i)
            npt.assert_allclose(op.get_eigenvalue(i), 1.0 / lam_k ** 2, rtol=1e-12)

    def test_power_neg1_custom_alpha_eigenvalue(self, space, dirichlet_provider):
        """power=-1, alpha=2: mu_k = 1 / (2 * lambda_k)."""
        alpha = 2.0
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=-1, alpha=alpha)
        for i in range(4):
            lam_k = dirichlet_provider.get_eigenvalue(i)
            npt.assert_allclose(
                op.get_eigenvalue(i), 1.0 / (alpha * lam_k), rtol=1e-12
            )

    def test_get_eigenfunction_forwarded_from_provider(
        self, unit_box, space, dirichlet_provider
    ):
        """Eigenfunctions are forwarded unchanged from the provider."""
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=1)
        rng = np.random.default_rng(42)
        pts = rng.uniform(0, 1, size=(50, 3))
        for i in range(3):
            phi_op = op.get_eigenfunction(i)
            phi_prov = dirichlet_provider.get_eigenfunction(i)
            npt.assert_allclose(phi_op(pts), phi_prov(pts), rtol=1e-12)

    def test_neumann_positive_power_zero_mode_eigenvalue_is_zero(
        self, space, neumann_provider
    ):
        """Neumann mode 0 has lambda_0=0; (1*0)^2 = 0 for positive power."""
        op = GeneralizedBoxLaplacian(space, neumann_provider, power=2, alpha=1.0)
        npt.assert_allclose(op.get_eigenvalue(0), 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# TestNegativePowerZeroModeSuppression
# ---------------------------------------------------------------------------

class TestNegativePowerZeroModeSuppression:
    """Neumann zero mode (lambda_0 ≈ 0) is suppressed to 0 for power < 0."""

    def test_neumann_mode0_suppressed_power_neg1(self, space, neumann_provider):
        """power=-1, Neumann: mode 0 operator eigenvalue = 0 (pseudoinverse)."""
        op = GeneralizedBoxLaplacian(space, neumann_provider, power=-1)
        assert op.get_eigenvalue(0) == pytest.approx(0.0)

    def test_neumann_mode0_suppressed_power_neg2(self, space, neumann_provider):
        """power=-2, Neumann: mode 0 operator eigenvalue = 0 (pseudoinverse)."""
        op = GeneralizedBoxLaplacian(space, neumann_provider, power=-2)
        assert op.get_eigenvalue(0) == pytest.approx(0.0)

    def test_neumann_mode1_inverted_power_neg1(self, space, neumann_provider):
        """power=-1, Neumann: mode 1 mu_1 = 1/lambda_1 (non-zero mode)."""
        op = GeneralizedBoxLaplacian(space, neumann_provider, power=-1)
        lam1 = neumann_provider.get_eigenvalue(1)
        npt.assert_allclose(op.get_eigenvalue(1), 1.0 / lam1, rtol=1e-12)

    def test_neumann_mode2_inverted_power_neg1(self, space, neumann_provider):
        """power=-1, Neumann: mode 2 mu_2 = 1/lambda_2."""
        op = GeneralizedBoxLaplacian(space, neumann_provider, power=-1)
        lam2 = neumann_provider.get_eigenvalue(2)
        npt.assert_allclose(op.get_eigenvalue(2), 1.0 / lam2, rtol=1e-12)

    def test_dirichlet_no_zero_modes_power_neg1(self, space, dirichlet_provider):
        """Dirichlet has no zero modes; all power=-1 eigenvalues are finite."""
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=-1)
        for i in range(op.n_modes):
            assert abs(op.get_eigenvalue(i)) > 0, (
                f"Mode {i} eigenvalue unexpectedly zero in Dirichlet negative-power op"
            )

    def test_custom_eigenvalue_tol_triggers_suppression(
        self, unit_box, space
    ):
        """A custom eigenvalue_tol can suppress a nominally small eigenvalue."""
        # Use a Dirichlet box where all eigenvalues are >= 3*pi^2 ~ 29.6.
        # Setting tol=100 suppresses all modes (all lambda_k < 100 at truncation=1).
        provider = TensorProductProvider(
            unit_box, truncation=(1, 1, 1), bc_type='dirichlet'
        )
        op = GeneralizedBoxLaplacian(
            space, provider, power=-1, eigenvalue_tol=100.0
        )
        # lambda_0 = 3*pi^2 ≈ 29.6 < 100 → suppressed
        assert op.get_eigenvalue(0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestIdentityPowerSemantics
# ---------------------------------------------------------------------------

class TestIdentityPowerSemantics:
    """power=0 is the identity: all operator eigenvalues equal exactly 1.0."""

    def test_dirichlet_all_eigenvalues_unity(self, space, dirichlet_provider):
        op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=0)
        for i in range(op.n_modes):
            npt.assert_allclose(
                op.get_eigenvalue(i), 1.0, rtol=1e-12,
                err_msg=f"Mode {i}: expected eigenvalue 1.0 for power=0",
            )

    def test_neumann_all_eigenvalues_unity_including_zero_mode(
        self, space, neumann_provider
    ):
        """Even the zero-eigenvalue Neumann mode maps to 1.0 for power=0."""
        op = GeneralizedBoxLaplacian(space, neumann_provider, power=0)
        for i in range(op.n_modes):
            npt.assert_allclose(
                op.get_eigenvalue(i), 1.0, rtol=1e-12,
                err_msg=f"Mode {i}: expected eigenvalue 1.0 for power=0",
            )

    def test_power0_alpha_irrelevant(self, space, dirichlet_provider):
        """alpha value does not affect power=0 eigenvalues (anything^0 = 1)."""
        for alpha in [0.1, 1.0, 5.0, 100.0]:
            op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=0, alpha=alpha)
            for i in range(op.n_modes):
                npt.assert_allclose(
                    op.get_eigenvalue(i), 1.0, rtol=1e-12,
                    err_msg=f"alpha={alpha}, mode {i}: expected 1.0",
                )


# ---------------------------------------------------------------------------
# TestEquivalenceWithExistingOperators (eigenvalue level)
# ---------------------------------------------------------------------------

class TestEquivalenceWithExistingOperators:
    """Eigenvalue-level equivalence: power=1 ≡ BoxLaplacian, power=-1 ≡ InverseBoxLaplacian.

    These tests verify that the Phase 3 refactoring will be semantically
    correct: GeneralizedBoxLaplacian with the right power matches the
    eigenvalues of the dedicated operator wrappers exactly.
    """

    def test_power1_eigenvalues_match_boxlaplacian(
        self, space, dirichlet_provider
    ):
        """power=1, alpha=1, Dirichlet: eigenvalues == BoxLaplacian eigenvalues."""
        gen_op = GeneralizedBoxLaplacian(
            space, dirichlet_provider, power=1, alpha=1.0
        )
        box_lap = BoxLaplacian(space, dirichlet_provider)
        for i in range(4):
            npt.assert_allclose(
                gen_op.get_eigenvalue(i),
                box_lap.get_eigenvalue(i),
                rtol=1e-12,
                err_msg=(
                    f"Mode {i}: GeneralizedBoxLaplacian(power=1) vs BoxLaplacian mismatch"
                ),
            )

    def test_power1_neumann_eigenvalues_match_boxlaplacian(
        self, space, neumann_provider
    ):
        """power=1, alpha=1, Neumann: eigenvalues == BoxLaplacian eigenvalues."""
        gen_op = GeneralizedBoxLaplacian(
            space, neumann_provider, power=1, alpha=1.0
        )
        box_lap = BoxLaplacian(space, neumann_provider)
        for i in range(4):
            npt.assert_allclose(
                gen_op.get_eigenvalue(i),
                box_lap.get_eigenvalue(i),
                rtol=1e-12,
            )

    def test_power_neg1_eigenvalues_match_inverse_boxlaplacian_dirichlet(
        self, space, dirichlet_provider
    ):
        """power=-1, alpha=1, Dirichlet: eigenvalues == InverseBoxLaplacian eigenvalues."""
        gen_op = GeneralizedBoxLaplacian(
            space, dirichlet_provider, power=-1, alpha=1.0
        )
        inv_lap = InverseBoxLaplacian(
            space, dirichlet_provider,
            null_space_handling='pseudoinverse',
        )
        for i in range(4):
            npt.assert_allclose(
                gen_op.get_eigenvalue(i),
                inv_lap.get_eigenvalue(i),
                rtol=1e-12,
                err_msg=(
                    f"Mode {i}: GeneralizedBoxLaplacian(power=-1) vs "
                    f"InverseBoxLaplacian(pseudoinverse) mismatch"
                ),
            )

    def test_power_neg1_eigenvalues_match_inverse_boxlaplacian_neumann(
        self, space, neumann_provider
    ):
        """power=-1, alpha=1, Neumann: eigenvalues == InverseBoxLaplacian pseudoinverse."""
        gen_op = GeneralizedBoxLaplacian(
            space, neumann_provider, power=-1, alpha=1.0
        )
        inv_lap = InverseBoxLaplacian(
            space, neumann_provider,
            null_space_handling='pseudoinverse',
        )
        for i in range(4):
            npt.assert_allclose(
                gen_op.get_eigenvalue(i),
                inv_lap.get_eigenvalue(i),
                rtol=1e-12,
                err_msg=(
                    f"Mode {i}: GeneralizedBoxLaplacian(power=-1, Neumann) vs "
                    f"InverseBoxLaplacian(pseudoinverse, Neumann) mismatch"
                ),
            )

    def test_boxlaplacian_eigenvalue0_is_3pi2(self, unit_box, space, dirichlet_provider):
        """Regression: BoxLaplacian first Dirichlet eigenvalue = 3*pi^2 on [0,1]^3.

        Known analytical value: each 1D Dirichlet mode has eigenvalue (n*pi/L)^2
        with n=1, L=1.  The trivariate mode (1,1,1) has eigenvalue 3*pi^2.
        Phase 3 must preserve this when BoxLaplacian is refactored as a wrapper.
        """
        box_lap = BoxLaplacian(space, dirichlet_provider)
        npt.assert_almost_equal(
            box_lap.get_eigenvalue(0), 3.0 * np.pi ** 2, decimal=8
        )

    def test_power1_first_eigenvalue_is_3pi2(self, space, dirichlet_provider):
        """GeneralizedBoxLaplacian(power=1) first eigenvalue = 3*pi^2 on [0,1]^3."""
        gen_op = GeneralizedBoxLaplacian(
            space, dirichlet_provider, power=1, alpha=1.0
        )
        npt.assert_almost_equal(
            gen_op.get_eigenvalue(0), 3.0 * np.pi ** 2, decimal=8
        )


# ---------------------------------------------------------------------------
# TestApplySemantics — Phase 2 pending (xfail)
# ---------------------------------------------------------------------------

class TestApplySemantics:
    """Full function-application semantics — Phase 2 _apply implementation pending.

    These tests document the intended apply-level behavior and serve as the
    acceptance criteria for Phase 2.  They are marked xfail because
    GeneralizedBoxLaplacian._apply raises NotImplementedError in Phase 1.
    """

    @pytest.mark.xfail(
        raises=NotImplementedError,
        reason="Phase 2 pending: GeneralizedBoxLaplacian._apply not yet implemented",
        strict=False,
    )
    def test_power1_apply_matches_boxlaplacian(
        self, unit_box, space, dirichlet_provider
    ):
        """GeneralizedBoxLaplacian(power=1)(phi_0) ≈ BoxLaplacian(phi_0) pointwise."""
        gen_op = GeneralizedBoxLaplacian(
            space, dirichlet_provider, power=1, alpha=1.0
        )
        box_lap = BoxLaplacian(space, dirichlet_provider)
        phi0 = dirichlet_provider.get_eigenfunction(0)
        rng = np.random.default_rng(123)
        pts = rng.uniform(0, 1, size=(50, 3))
        npt.assert_allclose(gen_op(phi0)(pts), box_lap(phi0)(pts), rtol=1e-3)

    @pytest.mark.xfail(
        raises=NotImplementedError,
        reason="Phase 2 pending: GeneralizedBoxLaplacian._apply not yet implemented",
        strict=False,
    )
    def test_power0_apply_is_identity(
        self, unit_box, space, dirichlet_provider
    ):
        """GeneralizedBoxLaplacian(power=0)(phi_0) ≈ phi_0 pointwise."""
        gen_op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=0)
        phi0 = dirichlet_provider.get_eigenfunction(0)
        rng = np.random.default_rng(456)
        pts = rng.uniform(0, 1, size=(50, 3))
        npt.assert_allclose(gen_op(phi0)(pts), phi0(pts), rtol=1e-3)

    @pytest.mark.xfail(
        raises=NotImplementedError,
        reason="Phase 2 pending: GeneralizedBoxLaplacian._apply not yet implemented",
        strict=False,
    )
    def test_power_neg1_apply_matches_inverse_boxlaplacian(
        self, unit_box, space, dirichlet_provider
    ):
        """GeneralizedBoxLaplacian(power=-1)(phi_0) ≈ InverseBoxLaplacian(phi_0)."""
        gen_op = GeneralizedBoxLaplacian(
            space, dirichlet_provider, power=-1, alpha=1.0
        )
        inv_lap = InverseBoxLaplacian(space, dirichlet_provider)
        phi0 = dirichlet_provider.get_eigenfunction(0)
        rng = np.random.default_rng(789)
        pts = rng.uniform(0, 1, size=(50, 3))
        npt.assert_allclose(gen_op(phi0)(pts), inv_lap(phi0)(pts), rtol=1e-3)

    @pytest.mark.xfail(
        raises=NotImplementedError,
        reason="Phase 2 pending: GeneralizedBoxLaplacian._apply not yet implemented",
        strict=False,
    )
    def test_power_neg1_neumann_zero_mode_maps_to_zero_function(
        self, unit_box, space, neumann_provider
    ):
        """Applying power=-1 to the Neumann zero mode returns the zero function."""
        gen_op = GeneralizedBoxLaplacian(space, neumann_provider, power=-1)
        phi0 = neumann_provider.get_eigenfunction(0)   # constant, lambda_0=0
        result = gen_op(phi0)
        rng = np.random.default_rng(999)
        pts = rng.uniform(0, 1, size=(50, 3))
        npt.assert_allclose(result(pts), 0.0, atol=1e-6)

    @pytest.mark.xfail(
        raises=NotImplementedError,
        reason="Phase 2 pending: GeneralizedBoxLaplacian._apply not yet implemented",
        strict=False,
    )
    def test_power2_apply_eigenmode_scaling(
        self, unit_box, space, dirichlet_provider
    ):
        """GeneralizedBoxLaplacian(power=2)(phi_0) ≈ lambda_0^2 * phi_0."""
        gen_op = GeneralizedBoxLaplacian(
            space, dirichlet_provider, power=2, alpha=1.0
        )
        phi0 = dirichlet_provider.get_eigenfunction(0)
        lam0 = dirichlet_provider.get_eigenvalue(0)
        result = gen_op(phi0)
        rng = np.random.default_rng(111)
        pts = rng.uniform(0, 1, size=(50, 3))
        npt.assert_allclose(result(pts), (lam0 ** 2) * phi0(pts), rtol=1e-3)
