"""Region3D: Thin geometric wrapper around a pygeoinf Subset in
EuclideanSpace(3).

`Region3D` is the pure-semantics layer of pygeoinf3D.  It guarantees that a
`Subset` is defined over `EuclideanSpace(3)` and provides a stable interface
for membership testing, boundary access, and spatial bounds — without any
numerical or computational machinery.
"""

from __future__ import annotations

import numpy as np

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.subsets import ConvexSubset, Subset

from .config import AxisAlignedBoundingBox


class Region3D:
    """A thin semantic wrapper around a `Subset` in $\\mathbb{R}^3$.

    `Region3D` separates pure geometric membership semantics from the
    computational concerns handled by `Domain3D`.  The wrapped subset
    **must** be defined on `EuclideanSpace(3)`.

    Args:
        subset: A `pygeoinf.Subset` whose `.domain` is `EuclideanSpace(3)`.
        bounding_box: An axis-aligned box enclosing the region.

    Raises:
        ValueError: If the subset's domain is not `EuclideanSpace(3)`.
    """

    def __init__(
        self,
        subset: Subset,
        *,
        bounding_box: AxisAlignedBoundingBox | None = None,
    ) -> None:
        if (
            not isinstance(subset.domain, EuclideanSpace)
            or subset.domain.dim != 3
        ):
            raise ValueError(
                "Region3D requires a Subset defined on EuclideanSpace(3); "
                f"got domain={subset.domain!r}."
            )
        self._subset = subset
        self._bounding_box = (
            bounding_box
            if bounding_box is not None
            else self._infer_bounding_box(subset)
        )

    @staticmethod
    def _infer_bounding_box(subset: Subset) -> AxisAlignedBoundingBox:
        r"""Infer a finite axis-aligned bounding box when possible.

        For a closed convex subset with support function $h_S$, the coordinate
        bounds satisfy

        $\max_{x \in S} x_i = h_S(e_i)$ and
        $\min_{x \in S} x_i = -h_S(-e_i)$.

        If a support function is unavailable, this method falls back to the
        subset's `directional_bound` implementation when the subset is a
        `ConvexSubset`.  Inference fails for non-convex subsets or when any
        coordinate direction is unbounded.
        """
        if not isinstance(subset, ConvexSubset):
            raise ValueError(
                "Region3D requires an explicit bounding_box for non-convex "
                "subsets."
            )

        space = subset.domain
        lows = np.empty(3, dtype=float)
        highs = np.empty(3, dtype=float)
        support_function = subset.support_function

        for axis in range(3):
            direction = space.basis_vector(axis)
            negative_direction = space.multiply(-1.0, direction)

            if support_function is not None:
                high = float(support_function(direction))
                low = -float(support_function(negative_direction))
            else:
                _, high = subset.directional_bound(direction)
                _, low_on_negative = subset.directional_bound(negative_direction)
                low = -float(low_on_negative)

            if not np.isfinite(high) or not np.isfinite(low):
                raise ValueError(
                    "Region3D could not infer a finite bounding_box because "
                    "the subset is unbounded in at least one coordinate "
                    f"direction; axis={axis}, low={low}, high={high}."
                )

            lows[axis] = low
            highs[axis] = high

        return AxisAlignedBoundingBox(low=lows, high=highs)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def subset(self) -> Subset:
        """The wrapped `Subset` object."""
        return self._subset

    @property
    def ambient_space(self) -> EuclideanSpace:
        """Returns `EuclideanSpace(3)`, the ambient space of the region."""
        return self._subset.domain  # type: ignore[return-value]

    def bounding_box(self) -> AxisAlignedBoundingBox:
        """The axis-aligned box enclosing this region."""
        return self._bounding_box

    def contains(self, point: np.ndarray) -> bool:
        """Test whether *point* belongs to this region.

        Args:
            point: Array of shape ``(3,)``.

        Returns:
            ``True`` if the point is an element of the underlying subset.
        """
        return bool(self._subset.is_element(np.asarray(point, dtype=float)))

    @property
    def boundary_subset(self) -> Subset:
        """The geometric boundary $\\partial\\Omega$ as a `Subset`.

        Delegates to `subset.boundary`.  The concrete type depends on the
        underlying subset (e.g. a `Ball` returns a `Sphere`).
        """
        return self._subset.boundary
