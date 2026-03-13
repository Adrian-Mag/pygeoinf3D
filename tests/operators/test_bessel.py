"""Tests for pygeoinf3d.operators.bessel.BesselOperator3D.

Covers:
- Construction and basic properties
- Eigenpair scaling: BesselOp(e_k) = scale_k * e_k
- Inverse consistency: bessel_inverse(bessel_direct(c)) ≈ c
- Power=0 is identity
- Self-adjointness (diagonal operator)
- Convenience constructors bessel_direct / bessel_inverse
"""

from __future__ import annotations

import pytest
import numpy as np
import numpy.testing as npt

from pygeoinf.linear_operators import LinearOperator

from pygeoinf3d.geometries.box.domain import BoxDomain
from pygeoinf3d.providers.tensor_product import TensorProductProvider
from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.spaces.sobolev import Sobolev3D, SpectralSobolev3D
from pygeoinf3d.operators.bessel import (
    BesselOperator3D,
    SpectralBesselOperator3D,
    bessel_direct,
    bessel_inverse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_box():
    return BoxDomain(np.zeros(3), np.ones(3))


@pytest.fixture
def dir_provider(unit_box):
    return TensorProductProvider(
        unit_box, truncation=(3, 3, 3), bc_type='dirichlet'
    )


@pytest.fixture
def sobolev(dir_provider):
    return SpectralSobolev3D(dir_provider, s=2.0, kappa=1.0)


@pytest.fixture
def op_direct(sobolev):
    return SpectralBesselOperator3D(sobolev, power=sobolev.s / 2)


@pytest.fixture
def op_inverse(sobolev):
    return SpectralBesselOperator3D(sobolev, power=-sobolev.s / 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vector(n: int, k: int) -> np.ndarray:
    e = np.zeros(n)
    e[k] = 1.0
    return e


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestSpectralBesselOperator3DConstruction:

    def test_is_linear_operator(self, op_direct):
        assert isinstance(op_direct, LinearOperator)

    def test_domain_is_sobolev(self, op_direct, sobolev):
        assert op_direct.domain is sobolev

    def test_codomain_is_sobolev(self, op_direct, sobolev):
        assert op_direct.codomain is sobolev

    def test_power_property(self, sobolev):
        op = SpectralBesselOperator3D(sobolev, power=0.75)
        assert op.power == pytest.approx(0.75)

    def test_negative_power(self, sobolev):
        op = SpectralBesselOperator3D(sobolev, power=-1.5)
        assert op.power == pytest.approx(-1.5)

    def test_direct_power(self, sobolev):
        op = SpectralBesselOperator3D(sobolev, power=sobolev.s / 2)
        assert op.power == pytest.approx(sobolev.s / 2)

    def test_inverse_power(self, sobolev):
        op = SpectralBesselOperator3D(sobolev, power=-sobolev.s / 2)
        assert op.power == pytest.approx(-sobolev.s / 2)


# ---------------------------------------------------------------------------
# Eigenpair scaling
# ---------------------------------------------------------------------------

class TestEigenpairScaling:

    def test_unit_vector_scaled(self, sobolev):
        """BesselOperator3D with power p applied to e_k gives scale_k * e_k."""
        power = 1.5
        op = SpectralBesselOperator3D(sobolev, power=power)
        lams = sobolev.eigenvalues
        kappa = sobolev.kappa

        for k in range(sobolev.dim):
            ek = _unit_vector(sobolev.dim, k)
            Bek = op(ek)
            expected_scale = (kappa**2 + lams[k]) ** power
            # Only entry k should be non-zero
            npt.assert_allclose(Bek[k], expected_scale, rtol=1e-12)
            Bek_others = np.delete(Bek, k)
            npt.assert_allclose(Bek_others, 0.0, atol=1e-15)

    def test_direct_scales_first_mode(self, op_direct, sobolev):
        lam0 = sobolev.eigenvalues[0]
        kappa = sobolev.kappa
        expected = (kappa**2 + lam0) ** (sobolev.s / 2)
        e0 = _unit_vector(sobolev.dim, 0)
        result = op_direct(e0)
        npt.assert_allclose(result[0], expected, rtol=1e-12)

    def test_inverse_scales_first_mode(self, op_inverse, sobolev):
        lam0 = sobolev.eigenvalues[0]
        kappa = sobolev.kappa
        expected = (kappa**2 + lam0) ** (-sobolev.s / 2)
        e0 = _unit_vector(sobolev.dim, 0)
        result = op_inverse(e0)
        npt.assert_allclose(result[0], expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Inverse consistency
# ---------------------------------------------------------------------------

class TestInverseConsistency:

    def test_direct_then_inverse_is_identity(
        self, op_direct, op_inverse, sobolev
    ):
        """bessel_inverse(bessel_direct(c)) == c."""
        rng = np.random.default_rng(42)
        c = rng.standard_normal(sobolev.dim)
        c_back = op_inverse(op_direct(c))
        npt.assert_allclose(c_back, c, rtol=1e-12)

    def test_inverse_then_direct_is_identity(
        self, op_direct, op_inverse, sobolev
    ):
        """bessel_direct(bessel_inverse(c)) == c."""
        rng = np.random.default_rng(43)
        c = rng.standard_normal(sobolev.dim)
        c_back = op_direct(op_inverse(c))
        npt.assert_allclose(c_back, c, rtol=1e-12)

    def test_low_mode_inverse_consistency(self, sobolev):
        """On each unit vector e_k, B^{-1} * B = identity."""
        op = SpectralBesselOperator3D(sobolev, power=sobolev.s / 2)
        op_inv = SpectralBesselOperator3D(sobolev, power=-sobolev.s / 2)
        for k in range(sobolev.dim):
            ek = _unit_vector(sobolev.dim, k)
            npt.assert_allclose(op_inv(op(ek)), ek, rtol=1e-12,
                                err_msg=f"Roundtrip failed at mode k={k}")

    def test_continuous_factories_reject_spectral_space(self, sobolev):
        with pytest.raises(TypeError, match="continuous-first Sobolev3D"):
            bessel_direct(sobolev)
        with pytest.raises(TypeError, match="continuous-first Sobolev3D"):
            bessel_inverse(sobolev)


# ---------------------------------------------------------------------------
# Power = 0 is identity
# ---------------------------------------------------------------------------

class TestPowerZeroIdentity:

    def test_power_zero_returns_unchanged(self, sobolev):
        op = SpectralBesselOperator3D(sobolev, power=0.0)
        rng = np.random.default_rng(50)
        c = rng.standard_normal(sobolev.dim)
        result = op(c)
        npt.assert_allclose(result, c, rtol=1e-12)


# ---------------------------------------------------------------------------
# Self-adjointness
# ---------------------------------------------------------------------------

class TestSelfAdjointness:

    def test_self_adjoint_dot_product(self, op_direct, sobolev):
        """<B c1, c2> (component dot) == <c1, B c2>."""
        rng = np.random.default_rng(60)
        c1 = rng.standard_normal(sobolev.dim)
        c2 = rng.standard_normal(sobolev.dim)
        lhs = np.dot(op_direct(c1), c2)
        rhs = np.dot(c1, op_direct(c2))
        npt.assert_allclose(lhs, rhs, rtol=1e-12)

    def test_adjoint_mapping_same_as_forward(self, sobolev):
        """The adjoint of BesselOperator3D is itself (self-adjoint)."""
        op = SpectralBesselOperator3D(sobolev, power=1.0)
        rng = np.random.default_rng(61)
        c = rng.standard_normal(sobolev.dim)
        # adjoint_mapping was set to self._apply, so op.adjoint(c) == op(c)
        npt.assert_allclose(op.adjoint(c), op(c), rtol=1e-12)


# ---------------------------------------------------------------------------
# Neumann space (contains zero eigenvalue for the constant mode)
# ---------------------------------------------------------------------------

class TestNeumannSpaceBessel:

    def test_neumann_direct_all_modes(self, unit_box):
        """Dirichlet has lam_0 > 0; Neumann constant mode has lam_0 = 0."""
        provider = TensorProductProvider(unit_box, truncation=(2, 2, 2),
                                         bc_type='neumann')
        space = SpectralSobolev3D(provider, s=1.0, kappa=1.5)
        op = SpectralBesselOperator3D(space, power=space.s / 2)
        # constant mode: lam_0 = 0, so scale = (kappa^2)^(s/2) = kappa^s
        expected_scale_0 = space.kappa ** space.s
        e0 = _unit_vector(space.dim, 0)
        npt.assert_allclose(op(e0)[0], expected_scale_0, rtol=1e-12)


# ---------------------------------------------------------------------------
# Degeneracy: negative power with zero (kappa^2 + lambda) (review fix #3)
# ---------------------------------------------------------------------------

class TestBesselNegativePowerDegeneracy:

    def test_neumann_kappa0_negative_power_raises(self, unit_box):
        """Neumann + kappa=0 + s=0 (valid Sobolev) +
        negative power must raise."""
        provider = TensorProductProvider(
            unit_box, truncation=(2, 2, 2), bc_type='neumann'
        )
        # s=0 avoids the Sobolev3D degeneracy check; but negative power is bad
        space = SpectralSobolev3D(provider, s=0.0, kappa=0.0)
        with pytest.raises(ValueError, match="not invertible"):
            SpectralBesselOperator3D(space, power=-1.0)

    def test_neumann_kappa0_large_negative_power_raises(self, unit_box):
        provider = TensorProductProvider(
            unit_box, truncation=(2, 2, 2), bc_type='neumann'
        )
        space = SpectralSobolev3D(provider, s=0.0, kappa=0.0)
        with pytest.raises(ValueError, match="not invertible"):
            SpectralBesselOperator3D(space, power=-0.5)

    def test_neumann_kappa0_positive_power_ok(self, unit_box):
        """Positive power with kappa=0 Neumann is fine (0^p = 0 for p>0)."""
        provider = TensorProductProvider(
            unit_box, truncation=(2, 2, 2), bc_type='neumann'
        )
        space = SpectralSobolev3D(provider, s=0.0, kappa=0.0)
        op = SpectralBesselOperator3D(space, power=1.0)  # should not raise
        assert op.power == pytest.approx(1.0)

    def test_neumann_kappa_positive_negative_power_ok(self, unit_box):
        """Neumann + kappa>0 + negative power is fine (all bases > 0)."""
        provider = TensorProductProvider(
            unit_box, truncation=(2, 2, 2), bc_type='neumann'
        )
        space = SpectralSobolev3D(provider, s=1.0, kappa=1.0)
        op = SpectralBesselOperator3D(space, power=-1.0)  # should not raise
        assert op.power == pytest.approx(-1.0)

    def test_dirichlet_kappa0_negative_power_ok(self, unit_box):
        """Dirichlet + kappa=0 has no zero eigenvalue;
        negative power is fine."""
        provider = TensorProductProvider(
            unit_box, truncation=(2, 2, 2), bc_type='dirichlet'
        )
        space = SpectralSobolev3D(provider, s=1.0, kappa=0.0)
        op = SpectralBesselOperator3D(space, power=-1.0)  # should not raise
        assert op.power == pytest.approx(-1.0)


# ===========================================================================
# New continuous-first BesselOperator3D tests (Function3D elements)
# ===========================================================================


@pytest.fixture
def cont_sobolev(dir_provider):
    """Continuous-first Sobolev3D (s=2.0, kappa=1.0)."""
    return Sobolev3D(dir_provider, s=2.0, kappa=1.0)


@pytest.fixture
def cont_op_direct(cont_sobolev):
    return bessel_direct(cont_sobolev)


@pytest.fixture
def cont_op_inverse(cont_sobolev):
    return bessel_inverse(cont_sobolev)


class TestBesselOperator3DNew:
    """BesselOperator3D on continuous-first Sobolev3D (Function3D elements)."""

    def test_is_linear_operator(self, cont_op_direct):
        assert isinstance(cont_op_direct, LinearOperator)

    def test_domain_is_sobolev3d(self, cont_op_direct):
        assert isinstance(cont_op_direct.domain, Sobolev3D)

    def test_apply_returns_function3d(self, cont_op_direct, dir_provider):
        phi0 = dir_provider.get_eigenfunction(0)
        result = cont_op_direct(phi0)
        assert isinstance(result, Function3D)

    def test_direct_convenience_power(self, cont_sobolev):
        op = bessel_direct(cont_sobolev)
        assert op.power == pytest.approx(cont_sobolev.s / 2)

    def test_inverse_convenience_power(self, cont_sobolev):
        op = bessel_inverse(cont_sobolev)
        assert op.power == pytest.approx(-cont_sobolev.s / 2)

    def test_eigenmode_scaling(self, cont_sobolev, dir_provider, unit_box):
        """op(phi_0) ~= scale_0 * phi_0 pointwise."""
        op = BesselOperator3D(cont_sobolev, power=1.5)
        lam0 = dir_provider.get_eigenvalue(0)
        scale0 = (cont_sobolev.kappa ** 2 + lam0) ** 1.5
        phi0 = dir_provider.get_eigenfunction(0)
        result = op(phi0)
        pts = unit_box.structured_mesh(8)
        npt.assert_allclose(result(pts), scale0 * phi0(pts), rtol=5e-3)

    def test_inverse_consistency(
        self, cont_sobolev, cont_op_direct, cont_op_inverse,
        dir_provider, unit_box
    ):
        """bessel_inverse(bessel_direct(phi0)) ~= phi0 pointwise."""
        phi0 = dir_provider.get_eigenfunction(0)
        result = cont_op_inverse(cont_op_direct(phi0))
        pts = unit_box.structured_mesh(8)
        npt.assert_allclose(result(pts), phi0(pts), rtol=5e-3)

    def test_power_zero_is_identity(
        self, cont_sobolev, dir_provider, unit_box
    ):
        op = BesselOperator3D(cont_sobolev, power=0.0)
        phi0 = dir_provider.get_eigenfunction(0)
        result = op(phi0)
        pts = unit_box.structured_mesh(8)
        npt.assert_allclose(result(pts), phi0(pts), rtol=1e-3)

    def test_negative_power_neumann_kappa0_raises(self, unit_box):
        """kappa=0 Neumann constant mode has zero base.

        Negative power must raise.
        """
        provider = TensorProductProvider(
            unit_box, truncation=(2, 2, 2), bc_type='neumann'
        )
        space = Sobolev3D(provider, s=0.0, kappa=0.0)
        with pytest.raises(ValueError, match="not invertible"):
            BesselOperator3D(space, power=-1.0)
