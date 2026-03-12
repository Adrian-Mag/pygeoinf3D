"""Configuration types and shared enumerations for pygeoinf3D.

This module provides lightweight, dependency-free helpers used across the
package: the `AxisAlignedBoundingBox` dataclass for spatial bounds and the
`CoordinateSystem` enum used by `StructuredDomain3D` subclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class CoordinateSystem(Enum):
    """Coordinate systems supported by structured 3D domains.

    Used by `StructuredDomain3D.coordinate_system` to declare the natural
    parametric coordinates of a geometry.

    Members:
        CARTESIAN: Standard $(x, y, z)$ Cartesian coordinates.
        SPHERICAL: Spherical $(r, \\theta, \\phi)$ coordinates.
        CYLINDRICAL: Cylindrical $(r, \\phi, z)$ coordinates.
    """

    CARTESIAN = auto()
    SPHERICAL = auto()
    CYLINDRICAL = auto()


@dataclass
class AxisAlignedBoundingBox:
    """An axis-aligned bounding box in $\\mathbb{R}^3$.

    Stores the lower and upper corners of a rectangular box aligned with the
    coordinate axes.  Used by `Region3D` and `Domain3D` to provide finite
    spatial bounds for sampling and plotting.

    Attributes:
        low: Lower corner of the box, shape ``(3,)``.
        high: Upper corner of the box, shape ``(3,)``.

    Raises:
        ValueError: On construction if shapes are not ``(3,)`` or if any
            component of ``low`` strictly exceeds the corresponding component
            of ``high``.
    """

    low: np.ndarray
    high: np.ndarray

    def __post_init__(self) -> None:
        self.low = np.asarray(self.low, dtype=float)
        self.high = np.asarray(self.high, dtype=float)
        if self.low.shape != (3,) or self.high.shape != (3,):
            raise ValueError(
                "AxisAlignedBoundingBox requires arrays of shape (3,); "
                f"got low={self.low.shape}, high={self.high.shape}."
            )
        if np.any(self.low > self.high):
            raise ValueError(
                "All components of low must be <= the corresponding "
                f"components of high; got low={self.low}, high={self.high}."
            )

    @property
    def center(self) -> np.ndarray:
        """Midpoint of the bounding box, shape ``(3,)``."""
        return 0.5 * (self.low + self.high)

    @property
    def extents(self) -> np.ndarray:
        """Side lengths of the bounding box, shape ``(3,)``."""
        return self.high - self.low

    def contains(self, point: np.ndarray) -> bool:
        """Test whether *point* lies inside or on the boundary of the box.

        Args:
            point: Array of shape ``(3,)``.

        Returns:
            ``True`` if ``low[i] <= point[i] <= high[i]`` for all ``i``.
        """
        p = np.asarray(point, dtype=float)
        return bool(np.all(p >= self.low) and np.all(p <= self.high))
