import numpy as np
import pytest

from pygeoinf import EuclideanSpace, Ball
from pygeoinf.subsets import Complement

from pygeoinf3d.core.config import AxisAlignedBoundingBox
from pygeoinf3d.core.region import Region3D


def _make_unit_ball_region() -> Region3D:
    """Helper: unit ball centered at origin in EuclideanSpace(3)."""
    space = EuclideanSpace(3)
    center = np.zeros(3)
    ball = Ball(space, center, 1.0)
    bbox = AxisAlignedBoundingBox(
        low=np.array([-1.0, -1.0, -1.0]),
        high=np.array([1.0, 1.0, 1.0]),
    )
    return Region3D(ball, bounding_box=bbox)


class TestRegion3DConstruction:
    def test_ambient_space_is_R3(self):
        r = _make_unit_ball_region()
        assert r.ambient_space == EuclideanSpace(3)

    def test_subset_is_stored(self):
        space = EuclideanSpace(3)
        ball = Ball(space, np.zeros(3), 1.0)
        bbox = AxisAlignedBoundingBox(
            low=np.full(3, -1.0), high=np.full(3, 1.0)
        )
        r = Region3D(ball, bounding_box=bbox)
        assert r.subset is ball

    def test_bounding_box_stored(self):
        r = _make_unit_ball_region()
        assert isinstance(r.bounding_box(), AxisAlignedBoundingBox)

    def test_bounding_box_can_be_inferred_from_convex_subset(self):
        space = EuclideanSpace(3)
        ball = Ball(space, np.zeros(3), 1.0)

        region = Region3D(ball)

        np.testing.assert_allclose(region.bounding_box().low, -np.ones(3))
        np.testing.assert_allclose(region.bounding_box().high, np.ones(3))

    def test_explicit_bounding_box_overrides_inference(self):
        space = EuclideanSpace(3)
        ball = Ball(space, np.zeros(3), 1.0)
        bbox = AxisAlignedBoundingBox(
            low=np.array([-2.0, -2.0, -2.0]),
            high=np.array([2.0, 2.0, 2.0]),
        )

        region = Region3D(ball, bounding_box=bbox)

        assert region.bounding_box() is bbox

    def test_rejects_subset_in_wrong_dimension(self):
        """Region3D must reject subsets defined in spaces other than R^3."""
        space2d = EuclideanSpace(2)
        disk = Ball(space2d, np.zeros(2), 1.0)
        with pytest.raises(ValueError, match=r"EuclideanSpace\(3\)"):
            Region3D(
                disk,
                bounding_box=AxisAlignedBoundingBox(
                    low=np.full(3, -1.0), high=np.full(3, 1.0)
                ),
            )

    def test_requires_explicit_bounding_box_for_non_convex_subset(self):
        space = EuclideanSpace(3)
        ball = Ball(space, np.zeros(3), 1.0)
        non_convex_subset = Complement(ball)

        with pytest.raises(ValueError, match="explicit bounding_box"):
            Region3D(non_convex_subset)


class TestRegion3DContains:
    def test_contains_origin(self):
        r = _make_unit_ball_region()
        assert r.contains(np.array([0.0, 0.0, 0.0]))

    def test_contains_interior_point(self):
        r = _make_unit_ball_region()
        assert r.contains(np.array([0.5, 0.0, 0.0]))

    def test_does_not_contain_exterior_point(self):
        r = _make_unit_ball_region()
        assert not r.contains(np.array([2.0, 0.0, 0.0]))

    def test_does_not_contain_far_point(self):
        r = _make_unit_ball_region()
        assert not r.contains(np.array([10.0, 10.0, 10.0]))


class TestRegion3DBoundarySubset:
    def test_boundary_subset_not_none(self):
        r = _make_unit_ball_region()
        b = r.boundary_subset
        assert b is not None

    def test_boundary_subset_is_sphere(self):
        """The boundary of a Ball is a Sphere in pygeoinf."""
        from pygeoinf.subsets import Sphere
        r = _make_unit_ball_region()
        assert isinstance(r.boundary_subset, Sphere)

    def test_boundary_point_is_on_sphere(self):
        """A point on the unit sphere should be on the boundary subset."""
        r = _make_unit_ball_region()
        sphere = r.boundary_subset
        on_sphere = np.array([1.0, 0.0, 0.0])
        assert sphere.is_element(on_sphere)
