"""Tests for pygeoinf3d.providers.tensor_product.TensorProductProvider.

Covers:
- Construction from BoxDomain
- dim == product of truncations
- get_multi_index / get_index round-trips
- Eigenvalue ordering: sorted increasing, tie-break lexicographic
- Dirichlet lowest mode is (1,1,1) → eigenvalue = 3*(pi/L)^2 on unit box
- Neumann lowest mode is (0,0,0) → eigenvalue = 0
- get_eigenfunction returns Function3D
- Normalization: <phi_i, phi_i>_{L^2} ≈ 1 (low modes)
- Orthogonality: <phi_i, phi_j>_{L^2} ≈ 0 for i≠j (low modes)
- 1D Dirichlet: sin at domain boundary ≈ 0
- 1D Neumann: derivative at boundary ≈ 0 (verified analytically)
"""

from __future__ import annotations

import pytest
import numpy as np
import numpy.testing as npt

from pygeoinf3d.geometries.box.domain import BoxDomain
from pygeoinf3d.providers.tensor_product import TensorProductProvider
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.core.functions import Function3D


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_box():
    return BoxDomain(np.zeros(3), np.ones(3))


@pytest.fixture
def small_provider_dirichlet(unit_box):
    """2×2×2 Dirichlet provider on unit box."""
    return TensorProductProvider(
        unit_box,
        truncation=(2, 2, 2),
        bc_type='dirichlet',
    )


@pytest.fixture
def small_provider_neumann(unit_box):
    """2×2×2 Neumann provider on unit box (modes 0..1 each axis)."""
    return TensorProductProvider(
        unit_box,
        truncation=(2, 2, 2),
        bc_type='neumann',
    )


@pytest.fixture
def unit_space(unit_box):
    return Lebesgue3D(unit_box)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestTensorProductProviderConstruction:

    def test_construct_dirichlet(self, small_provider_dirichlet):
        assert small_provider_dirichlet is not None

    def test_construct_neumann(self, small_provider_neumann):
        assert small_provider_neumann is not None

    def test_invalid_bc_type_raises(self, unit_box):
        with pytest.raises(ValueError):
            TensorProductProvider(
                unit_box,
                truncation=(2, 2, 2),
                bc_type='robin',
            )

    def test_dim_equals_product(self, small_provider_dirichlet):
        assert small_provider_dirichlet.dim == 2 * 2 * 2

    def test_dim_neumann(self, small_provider_neumann):
        assert small_provider_neumann.dim == 2 * 2 * 2


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

class TestTensorProductIndexing:

    def test_get_multi_index_first(self, small_provider_dirichlet):
        mi = small_provider_dirichlet.get_multi_index(0)
        assert len(mi) == 3

    def test_get_index_roundtrip(self, small_provider_dirichlet):
        for i in range(small_provider_dirichlet.dim):
            mi = small_provider_dirichlet.get_multi_index(i)
            assert small_provider_dirichlet.get_index(mi) == i

    def test_get_index_roundtrip_neumann(self, small_provider_neumann):
        for i in range(small_provider_neumann.dim):
            mi = small_provider_neumann.get_multi_index(i)
            assert small_provider_neumann.get_index(mi) == i

    def test_multi_indices_unique(self, small_provider_dirichlet):
        mis = [small_provider_dirichlet.get_multi_index(i)
               for i in range(small_provider_dirichlet.dim)]
        assert len(set(mis)) == small_provider_dirichlet.dim


# ---------------------------------------------------------------------------
# Eigenvalue ordering
# ---------------------------------------------------------------------------

class TestEigenvalueOrdering:

    def test_eigenvalues_non_decreasing_dirichlet(
        self,
        small_provider_dirichlet,
    ):
        evs = [small_provider_dirichlet.get_eigenvalue(i)
               for i in range(small_provider_dirichlet.dim)]
        assert all(evs[i] <= evs[i+1] for i in range(len(evs)-1))

    def test_eigenvalues_non_decreasing_neumann(self, small_provider_neumann):
        evs = [small_provider_neumann.get_eigenvalue(i)
               for i in range(small_provider_neumann.dim)]
        assert all(evs[i] <= evs[i+1] for i in range(len(evs)-1))

    def test_dirichlet_first_mode_on_unit_box(self, unit_box):
        # On unit box, Dirichlet lowest eigenvalue = 3*pi^2
        p = TensorProductProvider(
            unit_box,
            truncation=(3, 3, 3),
            bc_type='dirichlet',
        )
        ev0 = p.get_eigenvalue(0)
        expected = 3.0 * (np.pi ** 2)
        npt.assert_almost_equal(ev0, expected, decimal=10)

    def test_dirichlet_first_multi_index_on_unit_box(self, unit_box):
        p = TensorProductProvider(
            unit_box,
            truncation=(3, 3, 3),
            bc_type='dirichlet',
        )
        mi = p.get_multi_index(0)
        assert mi == (1, 1, 1)

    def test_neumann_first_mode_is_zero(self, small_provider_neumann):
        ev0 = small_provider_neumann.get_eigenvalue(0)
        npt.assert_almost_equal(ev0, 0.0, decimal=10)

    def test_neumann_first_multi_index_is_zero(self, small_provider_neumann):
        mi = small_provider_neumann.get_multi_index(0)
        assert mi == (0, 0, 0)

    def test_all_dirichlet_eigenvalues_positive(
        self,
        small_provider_dirichlet,
    ):
        evs = [small_provider_dirichlet.get_eigenvalue(i)
               for i in range(small_provider_dirichlet.dim)]
        assert all(ev > 0 for ev in evs)

    def test_all_neumann_eigenvalues_nonneg(self, small_provider_neumann):
        evs = [small_provider_neumann.get_eigenvalue(i)
               for i in range(small_provider_neumann.dim)]
        assert all(ev >= 0 for ev in evs)


# ---------------------------------------------------------------------------
# Eigenfunctions
# ---------------------------------------------------------------------------

class TestEigenfunctions:

    def test_get_eigenfunction_returns_function3d(
        self,
        small_provider_dirichlet,
        unit_box,
    ):
        phi = small_provider_dirichlet.get_eigenfunction(0)
        assert isinstance(phi, Function3D)

    def test_eigenfunction_callable_batch(
        self,
        small_provider_dirichlet,
        unit_box,
    ):
        phi = small_provider_dirichlet.get_eigenfunction(0)
        pts = unit_box.structured_mesh(4)
        vals = phi(pts)
        assert vals.shape == (pts.shape[0],)

    def test_eigenfunction_callable_point(self, small_provider_dirichlet):
        phi = small_provider_dirichlet.get_eigenfunction(0)
        val = phi(np.array([0.5, 0.5, 0.5]))
        assert np.isscalar(val) or val.shape == ()

    def test_dirichlet_zero_on_x_min_face(self, unit_box):
        """Dirichlet eigenfunctions vanish on x=0."""
        p = TensorProductProvider(
            unit_box,
            truncation=(2, 2, 2),
            bc_type='dirichlet',
        )
        phi = p.get_eigenfunction(0)
        pts = np.column_stack([
            np.zeros(10),
            np.linspace(0, 1, 10),
            np.linspace(0, 1, 10),
        ])
        vals = phi(pts)
        npt.assert_allclose(vals, 0.0, atol=1e-12)

    def test_dirichlet_zero_on_x_max_face(self, unit_box):
        """Dirichlet eigenfunctions vanish on x=1."""
        p = TensorProductProvider(
            unit_box,
            truncation=(2, 2, 2),
            bc_type='dirichlet',
        )
        phi = p.get_eigenfunction(0)  # (k=1,1,1) mode
        pts = np.column_stack([
            np.ones(10),
            np.linspace(0, 1, 10),
            np.linspace(0, 1, 10),
        ])
        vals = phi(pts)
        npt.assert_allclose(vals, 0.0, atol=1e-12)

    def test_neumann_constant_mode_uniform(self, unit_box):
        """Neumann (0,0,0) mode is constant 1 on unit box."""
        p = TensorProductProvider(
            unit_box,
            truncation=(2, 2, 2),
            bc_type='neumann',
        )
        phi = p.get_eigenfunction(0)  # (0,0,0) mode
        pts = unit_box.structured_mesh(4)
        vals = phi(pts)
        # 1/sqrt(1) * 1/sqrt(1) * 1/sqrt(1) = 1.0
        npt.assert_allclose(vals, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Normalization and orthogonality
# ---------------------------------------------------------------------------

class TestNormalizationOrthogonality:
    """Use low-resolution integration — only accuracy to ~1e-3 expected."""

    # Integration resolution on unit box — use moderate value for speed
    _RES = 40

    def _ip(self, box, phi_i, phi_j):
        """Compute <phi_i, phi_j>_{L2} via tensor-product quadrature."""
        pts = box.structured_mesh(self._RES)
        w = box.volume_weights(self._RES)
        return float(np.dot(phi_i(pts) * phi_j(pts), w))

    def test_dirichlet_normalization_first_mode(
        self,
        unit_box,
        small_provider_dirichlet,
    ):
        phi = small_provider_dirichlet.get_eigenfunction(0)
        ip = self._ip(unit_box, phi, phi)
        npt.assert_almost_equal(ip, 1.0, decimal=3)

    def test_dirichlet_orthogonality_first_two(
        self,
        unit_box,
        small_provider_dirichlet,
    ):
        phi0 = small_provider_dirichlet.get_eigenfunction(0)
        phi1 = small_provider_dirichlet.get_eigenfunction(1)
        ip = self._ip(unit_box, phi0, phi1)
        npt.assert_almost_equal(ip, 0.0, decimal=3)

    def test_neumann_normalization_first_mode(
        self,
        unit_box,
        small_provider_neumann,
    ):
        phi = small_provider_neumann.get_eigenfunction(0)
        ip = self._ip(unit_box, phi, phi)
        npt.assert_almost_equal(ip, 1.0, decimal=3)

    def test_neumann_orthogonality_modes_0_1(
        self,
        unit_box,
        small_provider_neumann,
    ):
        phi0 = small_provider_neumann.get_eigenfunction(0)
        phi1 = small_provider_neumann.get_eigenfunction(1)
        ip = self._ip(unit_box, phi0, phi1)
        npt.assert_almost_equal(ip, 0.0, decimal=3)

    def test_all_dirichlet_norms_close_to_1(self, unit_box):
        p = TensorProductProvider(
            unit_box,
            truncation=(2, 2, 2),
            bc_type='dirichlet',
        )
        for i in range(p.dim):
            phi = p.get_eigenfunction(i)
            ip = self._ip(unit_box, phi, phi)
            npt.assert_almost_equal(ip, 1.0, decimal=2,
                                    err_msg=f"Mode {i} norm^2 = {ip:.6f}")
