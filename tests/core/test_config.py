"""Tests for pygeoinf3d.core.config.

Classes: AxisAlignedBoundingBox, CoordinateSystem.
"""
import pytest
import numpy as np

from pygeoinf3d.core.config import AxisAlignedBoundingBox, CoordinateSystem


class TestAxisAlignedBoundingBox:
    def test_creation_stores_arrays(self):
        low = np.array([0.0, 0.0, 0.0])
        high = np.array([1.0, 2.0, 3.0])
        bb = AxisAlignedBoundingBox(low=low, high=high)
        np.testing.assert_array_equal(bb.low, low)
        np.testing.assert_array_equal(bb.high, high)

    def test_center(self):
        bb = AxisAlignedBoundingBox(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([2.0, 4.0, 6.0]),
        )
        np.testing.assert_array_equal(bb.center, np.array([1.0, 2.0, 3.0]))

    def test_extents(self):
        bb = AxisAlignedBoundingBox(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([2.0, 4.0, 6.0]),
        )
        np.testing.assert_array_equal(bb.extents, np.array([2.0, 4.0, 6.0]))

    def test_contains_interior_point(self):
        bb = AxisAlignedBoundingBox(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
        )
        assert bb.contains(np.array([0.5, 0.5, 0.5]))

    def test_contains_boundary_point(self):
        bb = AxisAlignedBoundingBox(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
        )
        assert bb.contains(np.array([0.0, 0.0, 0.0]))
        assert bb.contains(np.array([1.0, 1.0, 1.0]))

    def test_contains_exterior_point(self):
        bb = AxisAlignedBoundingBox(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
        )
        assert not bb.contains(np.array([1.5, 0.5, 0.5]))
        assert not bb.contains(np.array([-0.1, 0.5, 0.5]))

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            AxisAlignedBoundingBox(
                low=np.array([0.0, 0.0]),
                high=np.array([1.0, 1.0, 1.0]),
            )

    def test_invalid_low_greater_than_high_raises(self):
        with pytest.raises(ValueError):
            AxisAlignedBoundingBox(
                low=np.array([1.0, 0.0, 0.0]),
                high=np.array([0.0, 1.0, 1.0]),
            )

    def test_non_square_bounding_box(self):
        bb = AxisAlignedBoundingBox(
            low=np.array([-5.0, -2.0, 0.0]),
            high=np.array([5.0, 3.0, 10.0]),
        )
        np.testing.assert_array_equal(bb.extents, np.array([10.0, 5.0, 10.0]))


class TestCoordinateSystem:
    def test_cartesian_exists(self):
        assert CoordinateSystem.CARTESIAN is not None

    def test_spherical_exists(self):
        assert CoordinateSystem.SPHERICAL is not None

    def test_cylindrical_exists(self):
        assert CoordinateSystem.CYLINDRICAL is not None

    def test_values_are_distinct(self):
        vals = {
            CoordinateSystem.CARTESIAN,
            CoordinateSystem.SPHERICAL,
            CoordinateSystem.CYLINDRICAL,
        }
        assert len(vals) == 3
