"""Tests for pygeoinf3d.operators.inverse_laplacian_box.InverseBoxLaplacian.

Covers:
- Construction and validation (pseudoinverse, shift, strict modes)
- Eigenfunction/inverse-eigenvalue pairs: A(phi_k) ≈ (1/lambda_k) * phi_k
- Round-trip: InverseBoxLaplacian(BoxLaplacian(phi_k)) ≈ phi_k
- Self-adjointness: <Af, g> ≈ <f, Ag>
- Zero-mode suppression in Neumann pseudoinverse mode
"""

from __future__ import annotations

import pytest
import numpy as np
import numpy.testing as npt

from pygeoinf.linear_operators import LinearOperator

from pygeoinf3d.geometries.box.domain import BoxDomain
from pygeoinf3d.providers.tensor_product import TensorProductProvider
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.operators.laplacian_box import BoxLaplacian
from pygeoinf3d.operators.inverse_laplacian_box import InverseBoxLaplacian
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
    return TensorProductProvider(
        unit_box,
        truncation=(3, 3, 3),
        bc_type='dirichlet',
    )


@pytest.fixture
def neumann_provider(unit_box):
    return TensorProductProvider(
        unit_box,
        truncation=(3, 3, 3),
        bc_type='neumann',
    )


@pytest.fixture
def inv_lap_dirichlet(space, dirichlet_provider):
    return InverseBoxLaplacian(space, dirichlet_provider, quad_resolution=30)


@pytest.fixture
def inv_lap_neumann_pseudo(space, neumann_provider):
    return InverseBoxLaplacian(
        space,
        neumann_provider,
        null_space_handling='pseudoinverse',
        quad_resolution=30,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QUAD_RES = 30


def _ip(box, f, g, res=_QUAD_RES):
    """L² inner product <f, g> via box quadrature."""
    pts = box.structured_mesh(res)
    w = box.volume_weights(res)
    return float(np.dot(f(pts) * g(pts), w))


def _build_combo(provider, box, n_modes=4, seed=42):
    """Random linear combination of the first n_modes eigenfunctions."""
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(size=n_modes)
    phi_list = [provider.get_eigenfunction(i) for i in range(n_modes)]

    def combo(x):
        x = np.asarray(x)
        total = np.zeros(len(x)) if x.ndim == 2 else 0.0
        for c, phi in zip(coeffs, phi_list):
            total = total + c * phi(x)
        return total

    return Function3D(box, evaluate_callable=combo)


# ---------------------------------------------------------------------------
# TestInverseBoxLaplacianConstruction
# ---------------------------------------------------------------------------

class TestInverseBoxLaplacianConstruction:

    def test_construct_dirichlet(self, inv_lap_dirichlet):
        assert inv_lap_dirichlet is not None

    def test_construct_neumann_pseudoinverse(self, inv_lap_neumann_pseudo):
        assert inv_lap_neumann_pseudo is not None

    def test_construct_shift(self, space, dirichlet_provider):
        op = InverseBoxLaplacian(
            space, dirichlet_provider,
            null_space_handling='shift',
            shift=2.0,
        )
        assert op is not None

    def test_construct_strict_dirichlet_ok(self, space, dirichlet_provider):
        """Dirichlet has no zero eigenvalues; strict mode should succeed."""
        op = InverseBoxLaplacian(
            space, dirichlet_provider,
            null_space_handling='strict',
        )
        assert op is not None

    def test_construct_strict_neumann_raises(self, space, neumann_provider):
        """Neumann has λ_0 = 0; strict mode must raise ValueError."""
        with pytest.raises(ValueError, match="near-zero"):
            InverseBoxLaplacian(
                space, neumann_provider,
                null_space_handling='strict',
            )

    def test_construct_shift_nonpositive_raises_zero(
        self, space, dirichlet_provider
    ):
        """shift=0 must raise ValueError."""
        with pytest.raises(ValueError, match="shift"):
            InverseBoxLaplacian(
                space, dirichlet_provider,
                null_space_handling='shift',
                shift=0.0,
            )

    def test_construct_shift_nonpositive_raises_negative(
        self, space, dirichlet_provider
    ):
        """shift < 0 must raise ValueError."""
        with pytest.raises(ValueError, match="shift"):
            InverseBoxLaplacian(
                space, dirichlet_provider,
                null_space_handling='shift',
                shift=-1.0,
            )

    def test_is_linear_operator(self, inv_lap_dirichlet):
        assert isinstance(inv_lap_dirichlet, LinearOperator)

    def test_domain_is_lebesgue3d(self, inv_lap_dirichlet, space):
        assert inv_lap_dirichlet.domain == space

    def test_codomain_is_lebesgue3d(self, inv_lap_dirichlet, space):
        assert inv_lap_dirichlet.codomain == space


# ---------------------------------------------------------------------------
# TestInverseBoxLaplacianEigenpairs
# ---------------------------------------------------------------------------

class TestInverseBoxLaplacianEigenpairs:
    """A(phi_k) ≈ (1/lambda_k) * phi_k at sample points."""

    _TOL = dict(rtol=1e-3, atol=1e-3)

    def _eval_pts(self, box, n=200):
        rng = np.random.default_rng(2025)
        return rng.uniform(box.bounding_box().low, box.bounding_box().high, size=(n, 3))

    def test_eigenfunction_inverse_eigenvalue_dirichlet_mode0(
        self, unit_box, inv_lap_dirichlet, dirichlet_provider
    ):
        phi0 = dirichlet_provider.get_eigenfunction(0)
        lam0 = dirichlet_provider.get_eigenvalue(0)
        result = inv_lap_dirichlet(phi0)
        pts = self._eval_pts(unit_box)
        npt.assert_allclose(
            result(pts), (1.0 / lam0) * phi0(pts), **self._TOL
        )

    def test_eigenfunction_inverse_eigenvalue_dirichlet_mode1(
        self, unit_box, inv_lap_dirichlet, dirichlet_provider
    ):
        phi1 = dirichlet_provider.get_eigenfunction(1)
        lam1 = dirichlet_provider.get_eigenvalue(1)
        result = inv_lap_dirichlet(phi1)
        pts = self._eval_pts(unit_box)
        npt.assert_allclose(
            result(pts), (1.0 / lam1) * phi1(pts), **self._TOL
        )

    def test_eigenfunction_inverse_eigenvalue_dirichlet_mode2(
        self, unit_box, inv_lap_dirichlet, dirichlet_provider
    ):
        phi2 = dirichlet_provider.get_eigenfunction(2)
        lam2 = dirichlet_provider.get_eigenvalue(2)
        result = inv_lap_dirichlet(phi2)
        pts = self._eval_pts(unit_box)
        npt.assert_allclose(
            result(pts), (1.0 / lam2) * phi2(pts), **self._TOL
        )

    def test_eigenfunction_inverse_eigenvalue_neumann_mode1(
        self, unit_box, inv_lap_neumann_pseudo, neumann_provider
    ):
        """Mode 1 is non-zero; A(phi_1) ≈ (1/lambda_1) * phi_1."""
        phi1 = neumann_provider.get_eigenfunction(1)
        lam1 = neumann_provider.get_eigenvalue(1)
        result = inv_lap_neumann_pseudo(phi1)
        pts = self._eval_pts(unit_box)
        npt.assert_allclose(
            result(pts), (1.0 / lam1) * phi1(pts), **self._TOL
        )

    def test_eigenfunction_inverse_eigenvalue_neumann_mode2(
        self, unit_box, inv_lap_neumann_pseudo, neumann_provider
    ):
        """Mode 2 is non-zero; A(phi_2) ≈ (1/lambda_2) * phi_2."""
        phi2 = neumann_provider.get_eigenfunction(2)
        lam2 = neumann_provider.get_eigenvalue(2)
        result = inv_lap_neumann_pseudo(phi2)
        pts = self._eval_pts(unit_box)
        npt.assert_allclose(
            result(pts), (1.0 / lam2) * phi2(pts), **self._TOL
        )

    def test_zero_mode_suppressed_neumann_pseudoinverse(
        self, unit_box, inv_lap_neumann_pseudo, neumann_provider
    ):
        """Constant mode (lambda_0 = 0) must map to zero in pseudoinverse."""
        phi0 = neumann_provider.get_eigenfunction(0)
        result = inv_lap_neumann_pseudo(phi0)
        pts = self._eval_pts(unit_box)
        npt.assert_allclose(result(pts), 0.0, atol=1e-6)

    def test_shift_mode_eigenvalue_scaling_mode0(
        self, unit_box, space, dirichlet_provider
    ):
        """Shift mode: A(phi_k) ≈ (1/(lambda_k + shift)) * phi_k."""
        shift = 5.0
        op = InverseBoxLaplacian(
            space, dirichlet_provider,
            null_space_handling='shift',
            shift=shift,
            quad_resolution=30,
        )
        phi0 = dirichlet_provider.get_eigenfunction(0)
        lam0 = dirichlet_provider.get_eigenvalue(0)
        result = op(phi0)
        pts = self._eval_pts(unit_box)
        npt.assert_allclose(
            result(pts), (1.0 / (lam0 + shift)) * phi0(pts), **self._TOL
        )

    def test_shift_mode_eigenvalue_scaling_mode1(
        self, unit_box, space, dirichlet_provider
    ):
        shift = 5.0
        op = InverseBoxLaplacian(
            space, dirichlet_provider,
            null_space_handling='shift',
            shift=shift,
            quad_resolution=30,
        )
        phi1 = dirichlet_provider.get_eigenfunction(1)
        lam1 = dirichlet_provider.get_eigenvalue(1)
        result = op(phi1)
        pts = self._eval_pts(unit_box)
        npt.assert_allclose(
            result(pts), (1.0 / (lam1 + shift)) * phi1(pts), **self._TOL
        )


# ---------------------------------------------------------------------------
# TestInverseBoxLaplacianRoundTrip
# ---------------------------------------------------------------------------

class TestInverseBoxLaplacianRoundTrip:
    """(InverseLap ∘ Lap)(phi_k) ≈ phi_k."""

    _TOL = dict(rtol=1e-2, atol=1e-2)

    def _eval_pts(self, box, n=200):
        rng = np.random.default_rng(7777)
        return rng.uniform(box.bounding_box().low, box.bounding_box().high, size=(n, 3))

    def test_round_trip_dirichlet_mode0(
        self, unit_box, space, dirichlet_provider
    ):
        lap = BoxLaplacian(space, dirichlet_provider, quad_resolution=30)
        inv_lap = InverseBoxLaplacian(space, dirichlet_provider, quad_resolution=30)
        phi0 = dirichlet_provider.get_eigenfunction(0)
        g = inv_lap(lap(phi0))
        pts = self._eval_pts(unit_box)
        npt.assert_allclose(g(pts), phi0(pts), **self._TOL)

    def test_round_trip_dirichlet_mode1(
        self, unit_box, space, dirichlet_provider
    ):
        lap = BoxLaplacian(space, dirichlet_provider, quad_resolution=30)
        inv_lap = InverseBoxLaplacian(space, dirichlet_provider, quad_resolution=30)
        phi1 = dirichlet_provider.get_eigenfunction(1)
        g = inv_lap(lap(phi1))
        pts = self._eval_pts(unit_box)
        npt.assert_allclose(g(pts), phi1(pts), **self._TOL)

    def test_round_trip_neumann_pseudoinverse_mode1(
        self, unit_box, space, neumann_provider
    ):
        """Non-zero mode k=1 round-trips correctly; zero mode is projected out."""
        lap = BoxLaplacian(space, neumann_provider, quad_resolution=30)
        inv_lap = InverseBoxLaplacian(
            space, neumann_provider,
            null_space_handling='pseudoinverse',
            quad_resolution=30,
        )
        phi1 = neumann_provider.get_eigenfunction(1)
        g = inv_lap(lap(phi1))
        pts = self._eval_pts(unit_box)
        npt.assert_allclose(g(pts), phi1(pts), **self._TOL)


# ---------------------------------------------------------------------------
# TestInverseBoxLaplacianSelfAdjoint
# ---------------------------------------------------------------------------

class TestInverseBoxLaplacianSelfAdjoint:
    """<Af, g>_L2 ≈ <f, Ag>_L2."""

    def test_self_adjoint_dirichlet(self, unit_box, inv_lap_dirichlet, dirichlet_provider):
        f = _build_combo(dirichlet_provider, unit_box, n_modes=4, seed=11)
        g = _build_combo(dirichlet_provider, unit_box, n_modes=4, seed=22)
        lhs = _ip(unit_box, inv_lap_dirichlet(f), g)
        rhs = _ip(unit_box, f, inv_lap_dirichlet(g))
        npt.assert_almost_equal(lhs, rhs, decimal=4)

    def test_self_adjoint_neumann_pseudoinverse(
        self, unit_box, inv_lap_neumann_pseudo, neumann_provider
    ):
        # Use modes 1+ only to avoid the null-space mode
        f = _build_combo(neumann_provider, unit_box, n_modes=4, seed=33)
        g = _build_combo(neumann_provider, unit_box, n_modes=4, seed=44)
        lhs = _ip(unit_box, inv_lap_neumann_pseudo(f), g)
        rhs = _ip(unit_box, f, inv_lap_neumann_pseudo(g))
        npt.assert_almost_equal(lhs, rhs, decimal=4)
