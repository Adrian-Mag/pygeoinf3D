"""Tests for pygeoinf3d.core.domain_base.

Domain3D and StructuredDomain3D ABCs.
"""
import pytest
import numpy as np
from typing import Any

from pygeoinf import EuclideanSpace, Ball

from pygeoinf3d.core.region import Region3D
from pygeoinf3d.core.config import AxisAlignedBoundingBox, CoordinateSystem
from pygeoinf3d.core.domain_base import Domain3D, StructuredDomain3D
from pygeoinf3d.core.boundary import Boundary3D


# ---------------------------------------------------------------------------
# Minimal concrete implementations for testing
# ---------------------------------------------------------------------------

class _ConcreteBoundary(Boundary3D):
    """Trivial boundary that never contains any point."""
    def contains(self, point: np.ndarray) -> bool:
        return False


class _ConcreteDomain(Domain3D):
    """Minimal concrete Domain3D for testing the ABC contract."""

    def sample_interior(self, n: int, *, rng: Any = None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        return rng.uniform(-1.0, 1.0, size=(n, 3))

    def integrate_volume(self, f, *, method=None, **kwargs) -> float:
        raise NotImplementedError

    @property
    def boundary(self) -> Boundary3D:
        return _ConcreteBoundary(self)


class _ConcreteStructuredDomain(StructuredDomain3D):
    """Minimal concrete StructuredDomain3D for testing the ABC contract."""

    def sample_interior(self, n: int, *, rng: Any = None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        return rng.uniform(-1.0, 1.0, size=(n, 3))

    def integrate_volume(self, f, *, method=None, **kwargs) -> float:
        raise NotImplementedError

    @property
    def boundary(self) -> Boundary3D:
        return _ConcreteBoundary(self)

    @property
    def coordinate_system(self) -> CoordinateSystem:
        return CoordinateSystem.CARTESIAN

    def structured_mesh(
        self, resolution: Any, *, location: str = "cell_centers"
    ) -> Any:
        raise NotImplementedError

    def volume_weights(self, resolution: Any, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @property
    def boundary_components(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unit_ball_region() -> Region3D:
    space = EuclideanSpace(3)
    ball = Ball(space, np.zeros(3), 1.0)
    bbox = AxisAlignedBoundingBox(
        low=np.full(3, -1.0), high=np.full(3, 1.0)
    )
    return Region3D(ball, bounding_box=bbox)


# ---------------------------------------------------------------------------
# Tests for Domain3D
# ---------------------------------------------------------------------------

class TestDomain3DIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Domain3D(_make_unit_ball_region())  # type: ignore[abstract]


class TestConcreteDomain3D:
    def setup_method(self):
        self.region = _make_unit_ball_region()
        self.domain = _ConcreteDomain(self.region)

    def test_dim_is_3(self):
        assert self.domain.dim == 3

    def test_ambient_space_is_R3(self):
        assert self.domain.ambient_space == EuclideanSpace(3)

    def test_region_attribute(self):
        assert self.domain.region is self.region

    def test_contains_interior_point(self):
        assert self.domain.contains(np.array([0.0, 0.0, 0.0]))

    def test_does_not_contain_exterior_point(self):
        assert not self.domain.contains(np.array([2.0, 0.0, 0.0]))

    def test_contains_delegates_to_region(self):
        # Both domain.contains and region.contains should agree
        pt_in = np.array([0.3, 0.2, 0.1])
        pt_out = np.array([3.0, 0.0, 0.0])
        assert self.domain.contains(pt_in) == self.region.contains(pt_in)
        assert self.domain.contains(pt_out) == self.region.contains(pt_out)

    def test_bounding_box_returns_aabb(self):
        bb = self.domain.bounding_box()
        assert isinstance(bb, AxisAlignedBoundingBox)

    def test_sample_interior_shape(self):
        pts = self.domain.sample_interior(20, rng=42)
        assert pts.shape == (20, 3)

    def test_sample_interior_different_rng_seeds(self):
        pts1 = self.domain.sample_interior(5, rng=0)
        pts2 = self.domain.sample_interior(5, rng=1)
        assert not np.allclose(pts1, pts2)

    def test_boundary_is_boundary3d(self):
        b = self.domain.boundary
        assert isinstance(b, Boundary3D)


# ---------------------------------------------------------------------------
# Tests for StructuredDomain3D
# ---------------------------------------------------------------------------

class TestStructuredDomain3DIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            StructuredDomain3D(_make_unit_ball_region())  # type: ignore


class TestConcreteStructuredDomain3D:
    def setup_method(self):
        self.region = _make_unit_ball_region()
        self.domain = _ConcreteStructuredDomain(self.region)

    def test_is_domain3d_subclass(self):
        assert isinstance(self.domain, Domain3D)

    def test_coordinate_system(self):
        assert self.domain.coordinate_system == CoordinateSystem.CARTESIAN

    def test_dim_is_3(self):
        assert self.domain.dim == 3

    def test_boundary_components_is_dict(self):
        assert isinstance(self.domain.boundary_components, dict)

    def test_structured_mesh_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.domain.structured_mesh(10)

    def test_volume_weights_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.domain.volume_weights(10)

    def test_optional_integrate_on_mesh_raises(self):
        with pytest.raises(NotImplementedError):
            self.domain.integrate_volume_on_mesh(np.array([1.0, 2.0]))

    def test_optional_sample_boundary_raises(self):
        with pytest.raises(NotImplementedError):
            self.domain.sample_boundary(5)

    def test_optional_outward_normal_raises(self):
        with pytest.raises(NotImplementedError):
            self.domain.outward_normal(np.ones((3, 3)))
