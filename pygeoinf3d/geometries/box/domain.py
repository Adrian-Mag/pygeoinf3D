"""Axis-aligned box domain for 3D tensor-product computations.

Provides :class:`BoxDomain`, a concrete
:class:`~pygeoinf3d.core.domain_base.StructuredDomain3D` for
axis-aligned rectangular boxes in $\\mathbb{R}^3$. It supports
Cartesian tensor-product meshing, exact-weight quadrature, and
exposes six labeled face boundaries.

Design
------
- Uniform all-Cartesian structure:
    ``coordinate_system == CoordinateSystem.CARTESIAN``.
- ``structured_mesh`` returns flattened ``(N, 3)`` arrays in C-order
    matching ``np.meshgrid(..., indexing='ij')``.
- ``volume_weights`` produces cell-volume weights for ``cell_centers`` and
  trapezoidal weights for ``nodes``; both sum exactly to the box volume.
- ``integrate_volume`` defaults to tensor-product midpoint quadrature
    on ``cell_centers``, which is exact for all trilinear functions.
- Boundary faces are exposed as :class:`_BoxFaceBoundary` objects; the full
  boundary is :class:`_BoxBoundary` (union of all six faces).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.subsets import Subset

from pygeoinf3d.core.config import AxisAlignedBoundingBox, CoordinateSystem
from pygeoinf3d.core.region import Region3D
from pygeoinf3d.core.domain_base import StructuredDomain3D
from pygeoinf3d.core.boundary import Boundary3D


_R3 = EuclideanSpace(3)


# ---------------------------------------------------------------------------
# Internal Subset for the box region
# ---------------------------------------------------------------------------

class _BoxSubset(Subset):
    """A pygeoinf.Subset representing a closed axis-aligned box.

    Used internally to wrap an :class:`AxisAlignedBoundingBox` so that
    :class:`Region3D` can accept it.

    Args:
        low: Lower corner, shape ``(3,)``.
        high: Upper corner, shape ``(3,)``.
    """

    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        super().__init__(_R3)
        self._low = np.asarray(low, dtype=float)
        self._high = np.asarray(high, dtype=float)

    def is_element(self, x: np.ndarray, /, *, rtol: float = 1e-6) -> bool:
        x = np.asarray(x, dtype=float)
        tol = rtol * np.maximum(1.0, np.abs(self._high - self._low))
        return bool(
            np.all(x >= self._low - tol) and np.all(x <= self._high + tol)
        )

    @property
    def boundary(self) -> "_BoxSurfaceSubset":
        return _BoxSurfaceSubset(self._low, self._high)


class _BoxSurfaceSubset(Subset):
    """The surface $\\partial B$ of an axis-aligned box — union of six faces.

    Used as the ``boundary`` of :class:`_BoxSubset`.  A point lies on the
    surface if it is on **any** face and inside the box.

    Args:
        low: Lower corner, shape ``(3,)``.
        high: Upper corner, shape ``(3,)``.
    """

    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        super().__init__(_R3)
        self._low = low
        self._high = high

    def is_element(self, x: np.ndarray, /, *, rtol: float = 1e-6) -> bool:
        x = np.asarray(x, dtype=float)
        ext = np.maximum(1.0, self._high - self._low)
        tol = rtol * ext
        in_box = np.all(x >= self._low - tol) and np.all(x <= self._high + tol)
        if not in_box:
            return False
        on_face = (
            any(abs(x[i] - self._low[i]) <= tol[i] for i in range(3))
            or any(abs(x[i] - self._high[i]) <= tol[i] for i in range(3))
        )
        return bool(on_face)

    @property
    def boundary(self) -> "Subset":
        from pygeoinf.subsets import EmptySet
        return EmptySet(_R3)


# ---------------------------------------------------------------------------
# Boundary3D implementations for BoxDomain
# ---------------------------------------------------------------------------

class _BoxFaceBoundary(Boundary3D):
    """One axis-aligned face of a :class:`BoxDomain`.

    A face is characterized by a fixed axis and a fixed coordinate value.
    Points on the face must satisfy ``x[axis] ≈ coord_value`` and lie
    within the box in the remaining two axes.

    Args:
        parent_domain: The owning :class:`BoxDomain`.
        axis: 0, 1, or 2 (for x, y, or z respectively).
        coord_value: Fixed coordinate value on this face.
    """

    def __init__(
        self,
        parent_domain: "BoxDomain",
        axis: int,
        coord_value: float,
    ) -> None:
        super().__init__(parent_domain)
        self._box_domain = parent_domain
        self._axis = axis
        self._coord_value = float(coord_value)

    def contains(self, point: np.ndarray, *, rtol: float = 1e-8) -> bool:
        """Test whether *point* lies on this face.

        Args:
            point: Array of shape ``(3,)``.
            rtol: Relative tolerance for the face-coordinate equality test.

        Returns:
            ``True`` if the point is on the face and inside the box.
        """
        p = np.asarray(point, dtype=float)
        bbox = self._box_domain.bounding_box()
        scale = max(
            1.0,
            abs(bbox.high[self._axis] - bbox.low[self._axis]),
        )
        if abs(p[self._axis] - self._coord_value) > rtol * scale:
            return False
        # Must also be inside the box in the orthogonal directions
        for i in range(3):
            if i == self._axis:
                continue
            lo, hi = bbox.low[i], bbox.high[i]
            s = max(1.0, hi - lo)
            if p[i] < lo - rtol * s or p[i] > hi + rtol * s:
                return False
        return True


class _BoxBoundary(Boundary3D):
    """The full surface $\\partial B$ of a :class:`BoxDomain`.

    A point belongs if it lies on **any** of the six faces.

    Args:
        parent_domain: The owning :class:`BoxDomain`.
    """

    def __init__(self, parent_domain: "BoxDomain") -> None:
        super().__init__(parent_domain)
        self._box_domain = parent_domain

    def contains(self, point: np.ndarray, *, rtol: float = 1e-8) -> bool:
        """Test whether *point* lies on the box boundary.

        Args:
            point: Array of shape ``(3,)``.
            rtol: Relative tolerance for face membership.

        Returns:
            ``True`` if the point lies on any face of the box.
        """
        return any(
            face.contains(point, rtol=rtol)
            for face in self._box_domain.boundary_components.values()
        )


# ---------------------------------------------------------------------------
# Resolution parsing
# ---------------------------------------------------------------------------

def _parse_resolution(
    resolution: Union[int, Tuple[int, int, int]]
) -> Tuple[int, int, int]:
    """Convert scalar or 3-tuple resolution to ``(nx, ny, nz)``."""
    if isinstance(resolution, (int, np.integer)):
        n = int(resolution)
        return (n, n, n)
    nx, ny, nz = resolution
    return int(nx), int(ny), int(nz)


def _trap_weights(a: float, b: float, n: int) -> np.ndarray:
    """Trapezoidal quadrature weights on *n* nodes in ``[a, b]``.

    For *n* nodes (including endpoints) the trapezoidal rule assigns
    ``h/2`` to the endpoints and ``h`` to interior nodes, where
    ``h = (b-a)/(n-1)``.
    The weights sum exactly to ``b - a``.

    Args:
        a: Left endpoint.
        b: Right endpoint.
        n: Number of nodes (must be >= 2).

    Returns:
        Array of shape ``(n,)`` summing to ``b - a``.

    Raises:
        ValueError: If ``n < 2``.
    """
    if n < 2:
        raise ValueError(f"Trapezoidal weights require n >= 2; got n={n}.")
    w = np.ones(n)
    w[0] = 0.5
    w[-1] = 0.5
    h = (b - a) / (n - 1)
    return w * h


# ---------------------------------------------------------------------------
# BoxDomain
# ---------------------------------------------------------------------------

class BoxDomain(StructuredDomain3D):
    """Axis-aligned rectangular box domain with tensor-product structure.

    ``BoxDomain`` is a concrete
    :class:`~pygeoinf3d.core.domain_base.StructuredDomain3D` for the
    closed box $[a_x, b_x] \\times [a_y, b_y] \\times [a_z, b_z]$.

    Parameters
    ----------
    low_or_bbox:
        Either the lower corner as an array of shape ``(3,)`` when *high*
        is also provided, or an
        :class:`~pygeoinf3d.core.config.AxisAlignedBoundingBox` when *high*
        is omitted.
    high:
        Upper corner, shape ``(3,)``.  Required when *low_or_bbox* is an array;
        ignored when *low_or_bbox* is an ``AxisAlignedBoundingBox``.

    Raises
    ------
    ValueError
        If the box is invalid (``low > high`` in any coordinate) or if the
        wrong combination of arguments is supplied.

    Examples
    --------
    Construction from arrays::

        dom = BoxDomain(np.zeros(3), np.ones(3))

    Construction from bounding box::

        bbox = AxisAlignedBoundingBox(np.zeros(3), np.ones(3))
        dom = BoxDomain(bbox)
    """

    def __init__(
        self,
        low_or_bbox: Union[np.ndarray, AxisAlignedBoundingBox],
        high: Optional[np.ndarray] = None,
    ) -> None:
        if isinstance(low_or_bbox, AxisAlignedBoundingBox):
            bbox = low_or_bbox
        else:
            if high is None:
                raise ValueError(
                    "BoxDomain requires a 'high' array when 'low_or_bbox' is "
                    "an array, not an AxisAlignedBoundingBox."
                )
            bbox = AxisAlignedBoundingBox(
                np.asarray(low_or_bbox, dtype=float),
                np.asarray(high, dtype=float),
            )

        self._bbox = bbox
        box_subset = _BoxSubset(bbox.low, bbox.high)
        region = Region3D(box_subset, bounding_box=bbox)
        super().__init__(region)

        # Boundary objects — lazily initialised
        self._boundary_obj: Optional[_BoxBoundary] = None
        self._boundary_components_obj: Optional[
            dict[str, _BoxFaceBoundary]
        ] = None

    # ------------------------------------------------------------------
    # StructuredDomain3D abstract interface
    # ------------------------------------------------------------------

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """Cartesian coordinate system for the box."""
        return CoordinateSystem.CARTESIAN

    def structured_mesh(
        self,
        resolution: Union[int, Tuple[int, int, int]],
        *,
        location: str = "cell_centers",
    ) -> np.ndarray:
        """Return a tensor-product Cartesian grid over the box.

        Parameters
        ----------
        resolution:
            Either a single integer *n* (gives ``n×n×n`` grid) or a
            3-tuple ``(nx, ny, nz)``.
        location:
            ``'cell_centers'`` (default) — midpoints of uniform cells,
            strictly interior to the box.
            ``'nodes'`` — uniformly spaced nodes including the boundary.

        Returns
        -------
        ndarray of shape ``(N, 3)``:
            Flattened grid points in C/``'ij'`` ordering
            (x varies most slowly).

        Raises
        ------
        ValueError
            If *location* is not ``'cell_centers'`` or ``'nodes'``.
        """
        nx, ny, nz = _parse_resolution(resolution)
        lo, hi = self._bbox.low, self._bbox.high

        if location == "cell_centers":
            x = np.linspace(lo[0], hi[0], nx + 1)
            x = 0.5 * (x[:-1] + x[1:])
            y = np.linspace(lo[1], hi[1], ny + 1)
            y = 0.5 * (y[:-1] + y[1:])
            z = np.linspace(lo[2], hi[2], nz + 1)
            z = 0.5 * (z[:-1] + z[1:])
        elif location == "nodes":
            x = np.linspace(lo[0], hi[0], nx)
            y = np.linspace(lo[1], hi[1], ny)
            z = np.linspace(lo[2], hi[2], nz)
        else:
            raise ValueError(
                f"Unknown location '{location}': "
                "expected 'cell_centers' or 'nodes'."
            )

        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    def volume_weights(
        self,
        resolution: Union[int, Tuple[int, int, int]],
        **kwargs: Any,
    ) -> np.ndarray:
        """Return quadrature weights for :meth:`structured_mesh`.

        The weights sum **exactly** to the box volume.

        Parameters
        ----------
        resolution:
            Must match the value passed to :meth:`structured_mesh`.
        location:
            Passed via *kwargs*.  ``'cell_centers'`` (default) gives piecewise-
            constant midpoint weights; ``'nodes'`` gives trapezoidal weights.

        Returns
        -------
        ndarray of shape ``(N,)``:
            Positive weights summing to ``volume``.
        """
        nx, ny, nz = _parse_resolution(resolution)
        location = kwargs.get("location", "cell_centers")
        lo, hi = self._bbox.low, self._bbox.high

        if location == "cell_centers":
            wx = np.full(nx, (hi[0] - lo[0]) / nx)
            wy = np.full(ny, (hi[1] - lo[1]) / ny)
            wz = np.full(nz, (hi[2] - lo[2]) / nz)
        elif location == "nodes":
            if nx < 2 or ny < 2 or nz < 2:
                raise ValueError(
                    "Trapezoidal weights require at least 2 nodes per axis."
                )
            wx = _trap_weights(lo[0], hi[0], nx)
            wy = _trap_weights(lo[1], hi[1], ny)
            wz = _trap_weights(lo[2], hi[2], nz)
        else:
            raise ValueError(
                f"Unknown location '{location}': "
                "expected 'cell_centers' or 'nodes'."
            )

        ww = np.einsum("i,j,k->ijk", wx, wy, wz)
        return ww.ravel()

    @property
    def boundary_components(self) -> dict[str, _BoxFaceBoundary]:
        """Six labeled face boundaries of the box.

        Returns
        -------
        dict mapping label → :class:`_BoxFaceBoundary`:
            Keys: ``'x_min'``, ``'x_max'``, ``'y_min'``, ``'y_max'``,
            ``'z_min'``, ``'z_max'``.
        """
        if self._boundary_components_obj is None:
            lo, hi = self._bbox.low, self._bbox.high
            self._boundary_components_obj = {
                "x_min": _BoxFaceBoundary(self, 0, lo[0]),
                "x_max": _BoxFaceBoundary(self, 0, hi[0]),
                "y_min": _BoxFaceBoundary(self, 1, lo[1]),
                "y_max": _BoxFaceBoundary(self, 1, hi[1]),
                "z_min": _BoxFaceBoundary(self, 2, lo[2]),
                "z_max": _BoxFaceBoundary(self, 2, hi[2]),
            }
        return self._boundary_components_obj

    # ------------------------------------------------------------------
    # Domain3D abstract interface
    # ------------------------------------------------------------------

    def sample_interior(self, n: int, *, rng: Any = None) -> np.ndarray:
        """Sample *n* uniformly random interior points.

        Parameters
        ----------
        n:
            Number of points.
        rng:
            Optional seed or ``np.random.Generator`` instance.

        Returns
        -------
        ndarray of shape ``(n, 3)``:
            Points strictly inside the box (open-interval sampling).
        """
        rng = np.random.default_rng(rng)
        return rng.uniform(self._bbox.low, self._bbox.high, size=(n, 3))

    def integrate_volume(
        self,
        f: Any,
        *,
        method: Optional[str] = None,
        **kwargs: Any,
    ) -> float:
        """Integrate scalar callable *f* over the box.

        Uses tensor-product midpoint (cell-center) quadrature by default.
        The callable must accept a ``(N, 3)`` array and return ``(N,)`` values.

        Parameters
        ----------
        f:
            Callable ``f(x)`` on ``(N, 3)`` input returning ``(N,)`` values.
        resolution:
            Passed via *kwargs*.  Defaults to ``20`` in each dimension.
        location:
            Passed via *kwargs*.  Defaults to ``'cell_centers'``.

        Returns
        -------
        float:
            Approximate integral.
        """
        resolution = kwargs.pop("resolution", 20)
        location = kwargs.pop("location", "cell_centers")
        pts = self.structured_mesh(resolution, location=location)
        weights = self.volume_weights(resolution, location=location)
        vals = np.asarray(f(pts), dtype=float).ravel()
        return float(np.dot(vals, weights))

    @property
    def boundary(self) -> _BoxBoundary:
        """The full surface of the box (union of all six faces)."""
        if self._boundary_obj is None:
            self._boundary_obj = _BoxBoundary(self)
        return self._boundary_obj

    # ------------------------------------------------------------------
    # Concrete fast-path overrides
    # ------------------------------------------------------------------

    def sample_boundary(self, n: int, *, rng: Any = None) -> np.ndarray:
        """Sample *n* points uniformly on the box boundary surface.

        Distributes samples proportionally to each face's area so that the
        distribution is surface-area-uniform over the full boundary.

        Parameters
        ----------
        n:
            Number of boundary points to sample.
        rng:
            Optional random state accepted by ``np.random.default_rng``.

        Returns
        -------
        ndarray of shape ``(n, 3)``:
            Points on the six faces of the box.
        """
        rng = np.random.default_rng(rng)
        lo, hi = self._bbox.low, self._bbox.high
        Lx = hi[0] - lo[0]
        Ly = hi[1] - lo[1]
        Lz = hi[2] - lo[2]

        # Areas of the six faces (two per axis pair)
        areas = np.array([
            Ly * Lz,   # x_min
            Ly * Lz,   # x_max
            Lx * Lz,   # y_min
            Lx * Lz,   # y_max
            Lx * Ly,   # z_min
            Lx * Ly,   # z_max
        ])
        probs = areas / areas.sum()

        # Allocate samples per face via multinomial draw
        face_counts = rng.multinomial(n, probs)

        parts = []

        # face 0: x_min (x = lo[0])
        k = face_counts[0]
        if k > 0:
            parts.append(np.column_stack([
                np.full(k, lo[0]),
                rng.uniform(lo[1], hi[1], k),
                rng.uniform(lo[2], hi[2], k),
            ]))

        # face 1: x_max (x = hi[0])
        k = face_counts[1]
        if k > 0:
            parts.append(np.column_stack([
                np.full(k, hi[0]),
                rng.uniform(lo[1], hi[1], k),
                rng.uniform(lo[2], hi[2], k),
            ]))

        # face 2: y_min (y = lo[1])
        k = face_counts[2]
        if k > 0:
            parts.append(np.column_stack([
                rng.uniform(lo[0], hi[0], k),
                np.full(k, lo[1]),
                rng.uniform(lo[2], hi[2], k),
            ]))

        # face 3: y_max (y = hi[1])
        k = face_counts[3]
        if k > 0:
            parts.append(np.column_stack([
                rng.uniform(lo[0], hi[0], k),
                np.full(k, hi[1]),
                rng.uniform(lo[2], hi[2], k),
            ]))

        # face 4: z_min (z = lo[2])
        k = face_counts[4]
        if k > 0:
            parts.append(np.column_stack([
                rng.uniform(lo[0], hi[0], k),
                rng.uniform(lo[1], hi[1], k),
                np.full(k, lo[2]),
            ]))

        # face 5: z_max (z = hi[2])
        k = face_counts[5]
        if k > 0:
            parts.append(np.column_stack([
                rng.uniform(lo[0], hi[0], k),
                rng.uniform(lo[1], hi[1], k),
                np.full(k, hi[2]),
            ]))

        if not parts:
            return np.empty((0, 3))
        return np.vstack(parts)

    def outward_normal(
        self,
        points: np.ndarray,
        *,
        component: Optional[str] = None,
    ) -> np.ndarray:
        """Return outward unit normals at boundary *points*.

        For each point the nearest face is identified and its outward
        unit normal is returned.  If a named *component* is given, all
        points are assigned that face's normal without distance testing.

        .. note::
            This method is designed for **boundary points**.  For an interior
            point the returned normal is that of the nearest face, which is
            geometrically valid but semantically undefined.  Callers should
            ensure *points* lie on the boundary (e.g. obtained from
            :meth:`sample_boundary`).

        Parameters
        ----------
        points:
            Array of shape ``(n, 3)`` of boundary point locations.
        component:
            Named face: one of ``'x_min'``, ``'x_max'``, ``'y_min'``,
            ``'y_max'``, ``'z_min'``, ``'z_max'``.  If ``None``, the
            face is inferred per-point from the minimum distance.

        Returns
        -------
        ndarray of shape ``(n, 3)``:
            Outward unit normals.

        Raises
        ------
        ValueError
            If *component* is not a recognised face label.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=float))

        if pts.shape[0] == 0:
            return np.empty((0, 3))

        # Outward unit normals for each of the six faces
        _face_normals = np.array([
            [-1., 0., 0.],  # x_min
            [+1., 0., 0.],  # x_max
            [0., -1., 0.],  # y_min
            [0., +1., 0.],  # y_max
            [0.,  0., -1.],  # z_min
            [0.,  0., +1.],  # z_max
        ])
        _face_labels = [
            'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max',
        ]

        if component is not None:
            if component not in _face_labels:
                raise ValueError(
                    f"Unknown boundary component {component!r}. "
                    f"Expected one of {_face_labels!r}."
                )
            idx = _face_labels.index(component)
            return np.tile(_face_normals[idx], (len(pts), 1))

        # General case: find the face with smallest unsigned distance.
        lo, hi = self._bbox.low, self._bbox.high
        # Unsigned distances to each face
        dists = np.column_stack([
            np.abs(pts[:, 0] - lo[0]),  # x_min
            np.abs(pts[:, 0] - hi[0]),  # x_max
            np.abs(pts[:, 1] - lo[1]),  # y_min
            np.abs(pts[:, 1] - hi[1]),  # y_max
            np.abs(pts[:, 2] - lo[2]),  # z_min
            np.abs(pts[:, 2] - hi[2]),  # z_max
        ])  # shape (n, 6)
        closest_face = np.argmin(dists, axis=1)  # (n,), lowest index wins ties
        return _face_normals[closest_face]

    def integrate_volume_on_mesh(
        self, values: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> float:
        """Dot product of mesh values with weights.

        Parameters
        ----------
        values:
            Function values at mesh nodes, shape ``(N,)``.
        weights:
            Quadrature weights, shape ``(N,)``.  Must be provided.

        Returns
        -------
        float:
            The weighted sum.

        Raises
        ------
        ValueError
            If *weights* is ``None``.
        """
        if weights is None:
            raise ValueError(
                "integrate_volume_on_mesh requires explicit 'weights'."
            )
        return float(np.dot(np.asarray(values).ravel(), weights))

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def center(self) -> np.ndarray:
        """Midpoint of the box, shape ``(3,)``."""
        return self._bbox.center

    @property
    def extents(self) -> np.ndarray:
        """Side lengths of the box, shape ``(3,)``."""
        return self._bbox.extents

    @property
    def volume(self) -> float:
        """Volume of the box: $L_x L_y L_z$."""
        return float(np.prod(self._bbox.extents))

    def bounding_box(self) -> AxisAlignedBoundingBox:
        """Return the axis-aligned bounding box for this domain."""
        return self._bbox

    def contains(self, point: np.ndarray) -> bool:
        """Test whether *point* lies inside (or on the boundary of) the box.

        Parameters
        ----------
        point:
            Array of shape ``(3,)``.

        Returns
        -------
        bool:
            ``True`` if the point belongs to the closed box.
        """
        p = np.asarray(point, dtype=float)
        return bool(
            np.all(p >= self._bbox.low) and np.all(p <= self._bbox.high)
        )

    def __repr__(self) -> str:
        lo = self._bbox.low
        hi = self._bbox.high
        return (
            f"BoxDomain(low=[{lo[0]:.4g}, {lo[1]:.4g}, {lo[2]:.4g}], "
            f"high=[{hi[0]:.4g}, {hi[1]:.4g}, {hi[2]:.4g}])"
        )
