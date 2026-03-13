"""Tests for pygeoinf3d.geometries.box.domain.BoxDomain.

Covers:
- Construction from low/high arrays and from AxisAlignedBoundingBox
- Membership/contains
- Properties: center, extents, volume, coordinate_system
- structured_mesh: shape, location='cell_centers' and 'nodes'
- volume_weights: shape and sum equals volume
- integrate_volume: constant and linear polynomial accuracy
- boundary and boundary_components (six labeled faces)
- sample_interior: shape and membership
"""

from __future__ import annotations

import pytest
import numpy as np
import numpy.testing as npt

from pygeoinf3d.core.config import AxisAlignedBoundingBox, CoordinateSystem
from pygeoinf3d.core.boundary import Boundary3D
from pygeoinf3d.geometries.box.domain import BoxDomain


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_box():
    """Unit box [0,1]^3."""
    return BoxDomain(np.zeros(3), np.ones(3))


@pytest.fixture
def shifted_box():
    """Box [1, 2] x [2, 5] x [-1, 1]."""
    return BoxDomain(np.array([1.0, 2.0, -1.0]), np.array([2.0, 5.0, 1.0]))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestBoxDomainConstruction:

    def test_construct_from_arrays(self, unit_box):
        assert unit_box is not None

    def test_construct_from_bbox(self):
        bbox = AxisAlignedBoundingBox(np.zeros(3), np.ones(3))
        dom = BoxDomain(bbox)
        assert dom is not None

    def test_construct_from_bbox_matches_arrays(self):
        bbox = AxisAlignedBoundingBox(np.zeros(3), np.ones(3))
        d1 = BoxDomain(np.zeros(3), np.ones(3))
        d2 = BoxDomain(bbox)
        npt.assert_array_equal(d1.bounding_box().low, d2.bounding_box().low)
        npt.assert_array_equal(d1.bounding_box().high, d2.bounding_box().high)

    def test_invalid_low_gt_high_raises(self):
        with pytest.raises(ValueError):
            BoxDomain(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 1.0]))

    def test_dim_is_3(self, unit_box):
        assert unit_box.dim == 3

    def test_coordinate_system_cartesian(self, unit_box):
        assert unit_box.coordinate_system == CoordinateSystem.CARTESIAN


# ---------------------------------------------------------------------------
# Geometry properties
# ---------------------------------------------------------------------------

class TestBoxDomainProperties:

    def test_center(self, unit_box):
        npt.assert_array_almost_equal(unit_box.center, [0.5, 0.5, 0.5])

    def test_center_shifted(self, shifted_box):
        npt.assert_array_almost_equal(shifted_box.center, [1.5, 3.5, 0.0])

    def test_extents(self, unit_box):
        npt.assert_array_almost_equal(unit_box.extents, [1.0, 1.0, 1.0])

    def test_extents_shifted(self, shifted_box):
        npt.assert_array_almost_equal(shifted_box.extents, [1.0, 3.0, 2.0])

    def test_volume(self, unit_box):
        npt.assert_almost_equal(unit_box.volume, 1.0)

    def test_volume_shifted(self, shifted_box):
        npt.assert_almost_equal(shifted_box.volume, 1.0 * 3.0 * 2.0)


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------

class TestBoxDomainContains:

    def test_interior_point(self, unit_box):
        assert unit_box.contains(np.array([0.5, 0.5, 0.5]))

    def test_corner_point(self, unit_box):
        assert unit_box.contains(np.zeros(3))
        assert unit_box.contains(np.ones(3))

    def test_outside_point_x(self, unit_box):
        assert not unit_box.contains(np.array([-0.1, 0.5, 0.5]))

    def test_outside_point_y(self, unit_box):
        assert not unit_box.contains(np.array([0.5, 1.1, 0.5]))

    def test_outside_point_z(self, unit_box):
        assert not unit_box.contains(np.array([0.5, 0.5, -0.01]))


# ---------------------------------------------------------------------------
# sample_interior
# ---------------------------------------------------------------------------

class TestBoxDomainSampleInterior:

    def test_shape(self, unit_box):
        pts = unit_box.sample_interior(50, rng=0)
        assert pts.shape == (50, 3)

    def test_all_inside(self, unit_box):
        pts = unit_box.sample_interior(200, rng=42)
        for p in pts:
            assert unit_box.contains(p)

    def test_rng_reproducible(self, unit_box):
        p1 = unit_box.sample_interior(10, rng=7)
        p2 = unit_box.sample_interior(10, rng=7)
        npt.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# structured_mesh
# ---------------------------------------------------------------------------

class TestBoxDomainStructuredMesh:

    def test_shape_cell_centers_uniform(self, unit_box):
        pts = unit_box.structured_mesh(4)
        assert pts.shape == (4 * 4 * 4, 3)

    def test_shape_cell_centers_tuple(self, unit_box):
        pts = unit_box.structured_mesh((2, 3, 4))
        assert pts.shape == (2 * 3 * 4, 3)

    def test_shape_nodes_uniform(self, unit_box):
        pts = unit_box.structured_mesh(5, location='nodes')
        assert pts.shape == (5 * 5 * 5, 3)

    def test_shape_nodes_tuple(self, unit_box):
        pts = unit_box.structured_mesh((2, 3, 4), location='nodes')
        assert pts.shape == (2 * 3 * 4, 3)

    def test_cell_centers_all_inside(self, unit_box):
        pts = unit_box.structured_mesh(4)
        for p in pts:
            assert unit_box.contains(p)

    def test_nodes_include_boundary(self, unit_box):
        pts = unit_box.structured_mesh(3, location='nodes')
        x_vals = np.unique(np.round(pts[:, 0], 10))
        # For 3 nodes: 0.0, 0.5, 1.0
        npt.assert_array_almost_equal(sorted(x_vals), [0.0, 0.5, 1.0])

    def test_unknown_location_raises(self, unit_box):
        with pytest.raises(ValueError):
            unit_box.structured_mesh(4, location='invalid')

    def test_column_bounds_xyz(self, shifted_box):
        pts = shifted_box.structured_mesh(4)
        # cell centers should be strictly inside
        assert np.all(pts[:, 0] > shifted_box.bounding_box().low[0])
        assert np.all(pts[:, 0] < shifted_box.bounding_box().high[0])


# ---------------------------------------------------------------------------
# volume_weights
# ---------------------------------------------------------------------------

class TestBoxDomainVolumeWeights:

    def test_shape_matches_mesh(self, unit_box):
        res = 4
        pts = unit_box.structured_mesh(res)
        w = unit_box.volume_weights(res)
        assert w.shape == (pts.shape[0],)

    def test_sum_equals_volume(self, unit_box):
        w = unit_box.volume_weights(8)
        npt.assert_almost_equal(w.sum(), unit_box.volume, decimal=12)

    def test_sum_equals_volume_shifted(self, shifted_box):
        w = shifted_box.volume_weights((4, 5, 6))
        npt.assert_almost_equal(w.sum(), shifted_box.volume, decimal=12)

    def test_all_positive(self, unit_box):
        w = unit_box.volume_weights(5)
        assert np.all(w > 0)

    def test_nodes_sum_equals_volume(self, unit_box):
        w = unit_box.volume_weights(6, location='nodes')
        npt.assert_almost_equal(w.sum(), unit_box.volume, decimal=12)


# ---------------------------------------------------------------------------
# integrate_volume
# ---------------------------------------------------------------------------

class TestBoxDomainIntegrate:

    def test_constant_one(self, unit_box):
        result = unit_box.integrate_volume(lambda x: np.ones(len(x)))
        npt.assert_almost_equal(result, 1.0, decimal=10)

    def test_constant_three(self, shifted_box):
        result = shifted_box.integrate_volume(lambda x: 3.0 * np.ones(len(x)))
        npt.assert_almost_equal(result, 3.0 * shifted_box.volume, decimal=10)

    def test_linear_x(self, unit_box):
        # ∫_0^1 ∫_0^1 ∫_0^1 x dx dy dz = 0.5
        result = unit_box.integrate_volume(lambda x: x[:, 0])
        npt.assert_almost_equal(result, 0.5, decimal=4)

    def test_linear_y(self, unit_box):
        result = unit_box.integrate_volume(lambda x: x[:, 1])
        npt.assert_almost_equal(result, 0.5, decimal=4)

    def test_linear_xyz(self, unit_box):
        # ∫_[0,1]^3 x*y*z = 0.125
        result = unit_box.integrate_volume(
            lambda x: x[:, 0] * x[:, 1] * x[:, 2]
        )
        npt.assert_almost_equal(result, 0.125, decimal=4)


# ---------------------------------------------------------------------------
# boundary and boundary_components
# ---------------------------------------------------------------------------

class TestBoxDomainBoundary:

    def test_boundary_is_boundary3d(self, unit_box):
        assert isinstance(unit_box.boundary, Boundary3D)

    def test_boundary_components_keys(self, unit_box):
        keys = set(unit_box.boundary_components.keys())
        assert keys == {'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'}

    def test_all_components_are_boundary3d(self, unit_box):
        for v in unit_box.boundary_components.values():
            assert isinstance(v, Boundary3D)

    def test_face_x_min_membership(self, unit_box):
        face = unit_box.boundary_components['x_min']
        assert face.contains(np.array([0.0, 0.5, 0.5]))
        assert not face.contains(np.array([0.5, 0.5, 0.5]))

    def test_face_x_max_membership(self, unit_box):
        face = unit_box.boundary_components['x_max']
        assert face.contains(np.array([1.0, 0.5, 0.5]))
        assert not face.contains(np.array([0.5, 0.5, 0.5]))

    def test_face_z_min_membership(self, unit_box):
        face = unit_box.boundary_components['z_min']
        assert face.contains(np.array([0.5, 0.5, 0.0]))
        assert not face.contains(np.array([0.5, 0.5, 0.5]))

    def test_boundary_point_in_overall(self, unit_box):
        bnd = unit_box.boundary
        assert bnd.contains(np.array([0.0, 0.5, 0.5]))
        assert bnd.contains(np.array([1.0, 0.0, 0.8]))
        assert not bnd.contains(np.array([0.5, 0.5, 0.5]))
