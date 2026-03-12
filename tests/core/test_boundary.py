"""Tests for pygeoinf3d.core.boundary: Boundary3D ABC."""
import pytest
import numpy as np

from pygeoinf import EuclideanSpace, Ball

from pygeoinf3d.core.region import Region3D
from pygeoinf3d.core.config import AxisAlignedBoundingBox
from pygeoinf3d.core.domain_base import Domain3D
from pygeoinf3d.core.boundary import Boundary3D


# ---------------------------------------------------------------------------
# Minimal concrete implementations for testing
# ---------------------------------------------------------------------------

class _SphereBoundary(Boundary3D):
    """Boundary of the unit ball: the unit sphere."""

    def contains(self, point: np.ndarray) -> bool:
        return bool(abs(np.linalg.norm(point) - 1.0) < 1e-6)


class _ConcreteDomain(Domain3D):
    def sample_interior(self, n, *, rng=None):
        raise NotImplementedError

    def integrate_volume(self, f, *, method=None, **kwargs):
        raise NotImplementedError

    @property
    def boundary(self) -> "_SphereBoundary":
        return _SphereBoundary(self)


def _make_domain() -> _ConcreteDomain:
    space = EuclideanSpace(3)
    ball = Ball(space, np.zeros(3), 1.0)
    bbox = AxisAlignedBoundingBox(
        low=np.full(3, -1.0), high=np.full(3, 1.0)
    )
    region = Region3D(ball, bounding_box=bbox)
    return _ConcreteDomain(region)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBoundary3DIsAbstract:
    def test_cannot_instantiate_directly(self):
        domain = _make_domain()
        with pytest.raises(TypeError):
            Boundary3D(domain)  # type: ignore[abstract]


class TestConcreteBoundary3D:
    def setup_method(self):
        self.domain = _make_domain()
        self.boundary = _SphereBoundary(self.domain)

    def test_parent_domain(self):
        assert self.boundary.parent_domain is self.domain

    def test_ambient_space_is_R3(self):
        assert self.boundary.ambient_space == EuclideanSpace(3)

    def test_dim_is_2(self):
        assert self.boundary.dim == 2

    def test_contains_point_on_sphere(self):
        on_sphere = np.array([1.0, 0.0, 0.0])
        assert self.boundary.contains(on_sphere)

    def test_does_not_contain_interior_point(self):
        interior = np.array([0.0, 0.0, 0.0])
        assert not self.boundary.contains(interior)

    def test_does_not_contain_exterior_point(self):
        exterior = np.array([2.0, 0.0, 0.0])
        assert not self.boundary.contains(exterior)

    def test_ambient_space_matches_domain(self):
        assert self.boundary.ambient_space == self.domain.ambient_space
