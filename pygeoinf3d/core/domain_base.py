"""Abstract base classes for computational 3D domains.

The domain hierarchy separates geometric semantics (`Region3D`) from
computational machinery:

    Region3D  →  Domain3D  →  StructuredDomain3D  →  (BoxDomain, BallDomain, …)

`Domain3D` is **not** a `HilbertSpace`; function spaces are built *on top* of
domains and inherit from `pygeoinf.HilbertSpace` separately.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, TYPE_CHECKING

import numpy as np
from pygeoinf.hilbert_space import EuclideanSpace

from .region import Region3D
from .config import AxisAlignedBoundingBox, CoordinateSystem

if TYPE_CHECKING:
    from .boundary import Boundary3D


# Single shared instance – dimension never changes.
_R3 = EuclideanSpace(3)


class Domain3D(ABC):
    """Abstract base class for a computational 3D domain.

    A `Domain3D` wraps a `Region3D` and provides the computational geometry
    hooks needed by function spaces, operators, and samplers.  It deliberately
    does **not** inherit from `HilbertSpace`.

    The physical dimension is fixed at 3.

    Subclasses must implement:
        - `sample_interior`
        - `integrate_volume`
        - `boundary` (property)

    Args:
        region: The `Region3D` defining the geometric shape.
    """

    def __init__(self, region: Region3D) -> None:
        self._region = region

    # ------------------------------------------------------------------
    # Concrete properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Physical dimension of the domain (always 3)."""
        return 3

    @property
    def ambient_space(self) -> EuclideanSpace:
        """The ambient `EuclideanSpace(3)`."""
        return _R3

    @property
    def region(self) -> Region3D:
        """The `Region3D` providing geometric membership."""
        return self._region

    def contains(self, point: np.ndarray) -> bool:
        """Test whether *point* lies inside the domain.

        Delegates to `self.region.contains`.

        Args:
            point: Array of shape ``(3,)``.

        Returns:
            ``True`` if the point belongs to the domain region.
        """
        return self._region.contains(point)

    def bounding_box(self) -> AxisAlignedBoundingBox:
        """Return the bounding box of the domain.

        Delegates to the underlying region's bounding box by default.
        Subclasses may override for tighter or analytically known bounds.

        Returns:
            An `AxisAlignedBoundingBox` enclosing the domain.
        """
        return self._region.bounding_box()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def sample_interior(self, n: int, *, rng: Any = None) -> np.ndarray:
        """Sample *n* interior points.

        Args:
            n: Number of sample points to generate.
            rng: Optional random state accepted by `np.random.default_rng`.

        Returns:
            Array of shape ``(n, 3)``.
        """

    @abstractmethod
    def integrate_volume(
        self,
        f: Callable,
        *,
        method: Optional[str] = None,
        **kwargs: Any,
    ) -> float:
        """Integrate a scalar callable over the domain volume.

        Args:
            f: Callable with signature ``f(x)`` where ``x`` has shape
                ``(3,)`` or ``(n, 3)`` depending on the method.
            method: Optional integration method hint (e.g. ``'monte_carlo'``).
            **kwargs: Additional arguments forwarded to the integration method.

        Returns:
            Approximate integral value as a float.
        """

    @property
    @abstractmethod
    def boundary(self) -> "Boundary3D":
        """The computational boundary object for this domain."""


class StructuredDomain3D(Domain3D):
    """Abstract base class for domains with strong numerical structure.

    `StructuredDomain3D` extends `Domain3D` for geometries that admit a
    natural coordinate system, a tensor-product or spectral mesh structure,
    or analytically known quadrature weights.  Examples: boxes (Cartesian
    tensor-product), balls (spherical coordinates), cylinders (cylindrical
    coordinates).

    In addition to the `Domain3D` abstract interface, subclasses must
    implement:
        - `coordinate_system` (property)
        - `structured_mesh`
        - `volume_weights`
        - `boundary_components` (property)

    Optional fast-path stubs are provided with `NotImplementedError` as the
    default; concrete subclasses should override them for performance.
    """

    # ------------------------------------------------------------------
    # Additional abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def coordinate_system(self) -> CoordinateSystem:
        """The native coordinate system used by this domain."""

    @abstractmethod
    def structured_mesh(
        self,
        resolution: Any,
        *,
        location: str = "cell_centers",
    ) -> Any:
        """Return a geometry-appropriate mesh or grid.

        Args:
            resolution: Domain-specific resolution parameter.  For box domains
                this is typically an integer (uniform) or a 3-tuple.  For ball
                domains it is a ``(n_radial, l_max)`` pair.
            location: Node placement strategy.  Supported values are
                ``'cell_centers'`` (default) and ``'nodes'``.

        Returns:
            Domain-specific mesh object; the concrete type and layout are
            documented in each subclass.
        """

    @abstractmethod
    def volume_weights(self, resolution: Any, **kwargs: Any) -> np.ndarray:
        """Return quadrature weights compatible with `structured_mesh`.

        Args:
            resolution: Must match the resolution passed to `structured_mesh`.
            **kwargs: Additional method-specific arguments.

        Returns:
            Array of integration weights; the shape and ordering match the
            output of `structured_mesh(resolution)`.
        """

    @property
    @abstractmethod
    def boundary_components(self) -> dict:
        """Named boundary pieces as a ``{label: Boundary3D}`` mapping.

        Concrete subclasses populate this dict to expose labelled surfaces
        (e.g. ``'x_min'``, ``'x_max'``, ``'outer'``).  May be empty at the
        skeleton stage.

        Returns:
            Dict mapping string labels to `Boundary3D` objects.
        """

    # ------------------------------------------------------------------
    # Optional fast-path stubs
    # ------------------------------------------------------------------

    def integrate_volume_on_mesh(
        self,
        values: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """Fast-path volume integral using pre-computed mesh values.

        Args:
            values: Function values evaluated at mesh nodes, shape ``(N,)``.
            weights: Quadrature weights, shape ``(N,)``.  If ``None`` the
                method should fall back to uniform weights; the default stub
                raises `NotImplementedError`.

        Returns:
            Approximate integral.

        Raises:
            NotImplementedError: Until overridden by a concrete subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented"
            " integrate_volume_on_mesh."
        )

    def sample_boundary(self, n: int, *, rng: Any = None) -> np.ndarray:
        """Sample *n* points on the boundary surface.

        Args:
            n: Number of sample points.
            rng: Optional random state.

        Returns:
            Array of shape ``(n, 3)``.

        Raises:
            NotImplementedError: Until overridden by a concrete subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented sample_boundary."
        )

    def outward_normal(
        self,
        points: np.ndarray,
        *,
        component: Optional[str] = None,
    ) -> np.ndarray:
        """Return outward unit normals at boundary *points*.

        Args:
            points: Array of shape ``(n, 3)`` giving boundary point locations.
            component: Named boundary component; if ``None`` normals are
                computed for the global boundary.

        Returns:
            Array of shape ``(n, 3)`` of unit outward normals.

        Raises:
            NotImplementedError: Until overridden by a concrete subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented outward_normal."
        )
