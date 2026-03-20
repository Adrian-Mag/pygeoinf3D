"""Tests for pygeoinf3d.operators.laplacian_box.BoxLaplacian.

Covers:
- Construction from BoxLaplacianSpectrumProvider and Lebesgue3D space
- Is a LinearOperator between Lebesgue3D spaces
- Eigenfunction eigenpair test: L(phi_k) ≈ lambda_k * phi_k  (pointwise)
- Low-mode regression: known eigenvalues for Dirichlet+Neumann on unit box
- Self-adjointness smoke test: <Lf, g> ≈ <f, Lg>
"""

from __future__ import annotations

import pytest
import numpy as np
import numpy.testing as npt

from pygeoinf.linear_operators import LinearOperator

from pygeoinf3d.geometries.box.domain import BoxDomain
from pygeoinf3d.providers.tensor_product import BoxLaplacianSpectrumProvider
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.operators.laplacian_box import BoxLaplacian
from pygeoinf3d.operators.base import SpectralOperator3D
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
        unit_box,
        truncation=(3, 3, 3),
        bc_type='dirichlet',
    )


@pytest.fixture
def neumann_provider(unit_box):
    return BoxLaplacianSpectrumProvider(
        unit_box,
        truncation=(3, 3, 3),
        bc_type='neumann',
    )


@pytest.fixture
def laplacian_dirichlet(space, dirichlet_provider):
    return BoxLaplacian(space, dirichlet_provider)


@pytest.fixture
def laplacian_neumann(space, neumann_provider):
    return BoxLaplacian(space, neumann_provider)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestBoxLaplacianConstruction:

    def test_construct_dirichlet(self, laplacian_dirichlet):
        assert laplacian_dirichlet is not None

    def test_construct_neumann(self, laplacian_neumann):
        assert laplacian_neumann is not None

    def test_is_linear_operator(self, laplacian_dirichlet):
        assert isinstance(laplacian_dirichlet, LinearOperator)

    def test_domain_is_lebesgue3d(self, laplacian_dirichlet, space):
        assert laplacian_dirichlet.domain == space

    def test_codomain_is_lebesgue3d(self, laplacian_dirichlet, space):
        assert laplacian_dirichlet.codomain == space


# ---------------------------------------------------------------------------
# Eigenpair tests
# ---------------------------------------------------------------------------

_EVAL_RES = 20        # resolution for pointwise evaluation comparison
_QUAD_RES = 30        # resolution for L^2 integration checks


def _pts_interior(box, n=200):
    """Fixed interior sample points for pointwise tests."""
    rng = np.random.default_rng(1234)
    return rng.uniform(
        box.bounding_box().low,
        box.bounding_box().high,
        size=(n, 3),
    )


def _l2_relative_diff(box, f, g, res=_QUAD_RES):
    """||f - g||_L2 / ||g||_L2 using box quadrature."""
    pts = box.structured_mesh(res)
    w = box.volume_weights(res)
    diff = f(pts) - g(pts)
    num = float(np.sqrt(np.dot(diff ** 2, w)))
    den = float(np.sqrt(np.dot(g(pts) ** 2, w))) + 1e-14
    return num / den


class TestDirichletEigenpairs:

    def test_first_mode_eigenvalue(self, unit_box, space, dirichlet_provider):
        """L(phi_0) should equal lambda_0 * phi_0."""
        lap = BoxLaplacian(space, dirichlet_provider)
        phi0 = dirichlet_provider.get_eigenfunction(0)
        ev0 = dirichlet_provider.get_eigenvalue(0)
        Lphi0 = lap(phi0)
        expected_f = Function3D(
            unit_box,
            evaluate_callable=lambda x: ev0 * phi0(x),
        )
        rel_err = _l2_relative_diff(unit_box, Lphi0, expected_f)
        assert rel_err < 0.05, f"Relative L2 error = {rel_err:.4f}"

    def test_second_mode_eigenvalue(self, unit_box, space, dirichlet_provider):
        lap = BoxLaplacian(space, dirichlet_provider)
        phi1 = dirichlet_provider.get_eigenfunction(1)
        ev1 = dirichlet_provider.get_eigenvalue(1)
        Lphi1 = lap(phi1)
        expected_f = Function3D(
            unit_box,
            evaluate_callable=lambda x: ev1 * phi1(x),
        )
        rel_err = _l2_relative_diff(unit_box, Lphi1, expected_f)
        assert rel_err < 0.05, f"Relative L2 error = {rel_err:.4f}"

    def test_known_first_eigenvalue_value(self, unit_box, dirichlet_provider):
        # On [0,1]^3 with Dirichlet, lambda_0 = 3*pi^2
        ev0 = dirichlet_provider.get_eigenvalue(0)
        npt.assert_almost_equal(ev0, 3.0 * np.pi ** 2, decimal=8)


class TestNeumannEigenpairs:

    def test_first_mode_is_zero_eigenvalue(
        self,
        unit_box,
        space,
        neumann_provider,
    ):
        """L(phi_0) = 0 (constant mode, eigenvalue = 0)."""
        lap = BoxLaplacian(space, neumann_provider)
        phi0 = neumann_provider.get_eigenfunction(0)  # constant
        Lphi0 = lap(phi0)
        pts = unit_box.structured_mesh(_EVAL_RES)
        vals = Lphi0(pts)
        npt.assert_allclose(vals, 0.0, atol=1e-3)

    def test_second_mode_eigenvalue(self, unit_box, space, neumann_provider):
        lap = BoxLaplacian(space, neumann_provider)
        phi1 = neumann_provider.get_eigenfunction(1)
        ev1 = neumann_provider.get_eigenvalue(1)
        Lphi1 = lap(phi1)
        expected_f = Function3D(
            unit_box,
            evaluate_callable=lambda x: ev1 * phi1(x),
        )
        rel_err = _l2_relative_diff(unit_box, Lphi1, expected_f)
        assert rel_err < 0.05, f"Relative L2 error = {rel_err:.4f}"


# ---------------------------------------------------------------------------
# Self-adjointness smoke test
# ---------------------------------------------------------------------------

class TestSelfAdjointness:

    def _build_combo(self, provider, box, n_modes=3):
        """Build f = sum_{i<n} a_i phi_i with fixed random coefficients."""
        rng = np.random.default_rng(999)
        coeffs = rng.normal(size=n_modes)
        phi_list = [provider.get_eigenfunction(i) for i in range(n_modes)]

        def combo(x):
            total = np.zeros(len(x)) if x.ndim == 2 else 0.0
            for c, phi in zip(coeffs, phi_list):
                total = total + c * phi(x)
            return total

        return Function3D(box, evaluate_callable=combo)

    def _ip(self, box, f, g, res=_QUAD_RES):
        pts = box.structured_mesh(res)
        w = box.volume_weights(res)
        return float(np.dot(f(pts) * g(pts), w))

    def test_dirichlet_self_adjoint(self, unit_box, space, dirichlet_provider):
        lap = BoxLaplacian(space, dirichlet_provider)
        f = self._build_combo(dirichlet_provider, unit_box, n_modes=3)
        g = self._build_combo(dirichlet_provider, unit_box, n_modes=3)
        lf_g = self._ip(unit_box, lap(f), g)
        f_lg = self._ip(unit_box, f, lap(g))
        npt.assert_almost_equal(lf_g, f_lg, decimal=3)

    def test_neumann_self_adjoint(self, unit_box, space, neumann_provider):
        lap = BoxLaplacian(space, neumann_provider)
        f = self._build_combo(neumann_provider, unit_box, n_modes=3)
        g = self._build_combo(neumann_provider, unit_box, n_modes=3)
        lf_g = self._ip(unit_box, lap(f), g)
        f_lg = self._ip(unit_box, f, lap(g))
        npt.assert_almost_equal(lf_g, f_lg, decimal=3)


# ---------------------------------------------------------------------------
# SpectralOperator3D interface
# ---------------------------------------------------------------------------

class TestBoxLaplacianSpectralInterface:
    """BoxLaplacian satisfies the SpectralOperator3D protocol."""

    def test_is_spectral_operator(self, laplacian_dirichlet):
        assert isinstance(laplacian_dirichlet, SpectralOperator3D)

    def test_n_modes_equals_provider_dim(self, laplacian_dirichlet, dirichlet_provider):
        assert laplacian_dirichlet.n_modes == dirichlet_provider.dim

    def test_get_eigenvalue_matches_provider(self, laplacian_dirichlet, dirichlet_provider):
        for i in range(4):
            assert laplacian_dirichlet.get_eigenvalue(i) == dirichlet_provider.get_eigenvalue(i)

    def test_get_eigenfunction_matches_provider(self, unit_box, laplacian_dirichlet, dirichlet_provider):
        rng = np.random.default_rng(42)
        pts = rng.uniform(0, 1, size=(50, 3))
        for i in range(3):
            phi_op = laplacian_dirichlet.get_eigenfunction(i)
            phi_prov = dirichlet_provider.get_eigenfunction(i)
            npt.assert_allclose(phi_op(pts), phi_prov(pts), rtol=1e-12)

    def test_neumann_is_spectral_operator(self, laplacian_neumann):
        assert isinstance(laplacian_neumann, SpectralOperator3D)


# ---------------------------------------------------------------------------
# Regression: positive-operator convention (for Phase 3 wrapper verification)
# ---------------------------------------------------------------------------

class TestBoxLaplacianPositiveOperatorConvention:
    """Regression tests pinning the positive-operator (-Delta) convention.

    BoxLaplacian represents (-alpha*Delta) with alpha=1. These tests document
    the exact eigenvalue formula a Phase 3 refactoring to
    GeneralizedBoxLaplacian(power=1, alpha=1) must match.
    """

    def test_dirichlet_eigenvalue0_is_3pi2_on_unit_box(
        self, unit_box, space, dirichlet_provider
    ):
        """On [0,1]^3 Dirichlet, lambda_0 = 3*pi^2 (analytical)."""
        lap = BoxLaplacian(space, dirichlet_provider)
        npt.assert_almost_equal(lap.get_eigenvalue(0), 3.0 * np.pi ** 2, decimal=8)

    def test_all_dirichlet_eigenvalues_strictly_positive(
        self, space, dirichlet_provider
    ):
        """All Dirichlet eigenvalues are strictly positive (no null space)."""
        lap = BoxLaplacian(space, dirichlet_provider)
        for i in range(lap.n_modes):
            assert lap.get_eigenvalue(i) > 0, f"Mode {i} eigenvalue not positive"

    def test_neumann_zero_mode_eigenvalue_is_zero(
        self, space, neumann_provider
    ):
        """Neumann mode 0 eigenvalue = 0 (constant function in null space)."""
        lap = BoxLaplacian(space, neumann_provider)
        npt.assert_allclose(lap.get_eigenvalue(0), 0.0, atol=1e-12)

    def test_neumann_nonzero_modes_positive(self, space, neumann_provider):
        """Neumann modes 1+ are strictly positive."""
        lap = BoxLaplacian(space, neumann_provider)
        for i in range(1, lap.n_modes):
            assert lap.get_eigenvalue(i) > 0, f"Neumann mode {i} eigenvalue not positive"


# ---------------------------------------------------------------------------
# Phase 3 wrapper-equivalence tests
# ---------------------------------------------------------------------------

class TestBoxLaplacianWrapperEquivalence:
    """BoxLaplacian must be a GeneralizedBoxLaplacian subclass (Phase 3)."""

    def test_box_laplacian_is_generalized_subclass(self, laplacian_dirichlet):
        from pygeoinf3d.operators.generalized_box_laplacian import GeneralizedBoxLaplacian
        assert isinstance(laplacian_dirichlet, GeneralizedBoxLaplacian)

    def test_box_laplacian_power_is_1(self, laplacian_dirichlet):
        assert laplacian_dirichlet.power == 1

    def test_box_laplacian_alpha_is_1(self, laplacian_dirichlet):
        assert laplacian_dirichlet.alpha == 1.0

    def test_box_laplacian_apply_matches_generalized(self, unit_box, space, dirichlet_provider):
        from pygeoinf3d.operators.generalized_box_laplacian import GeneralizedBoxLaplacian
        lap = BoxLaplacian(space, dirichlet_provider, quad_resolution=30)
        gen = GeneralizedBoxLaplacian(
            space, dirichlet_provider, power=1, alpha=1.0, quad_resolution=30
        )
        phi0 = dirichlet_provider.get_eigenfunction(0)
        pts = np.random.default_rng(99).uniform(0, 1, size=(50, 3))
        npt.assert_allclose(lap(phi0)(pts), gen(phi0)(pts), rtol=1e-10)

    def test_box_laplacian_repr_format(self, laplacian_dirichlet):
        r = repr(laplacian_dirichlet)
        assert r.startswith("BoxLaplacian(")
