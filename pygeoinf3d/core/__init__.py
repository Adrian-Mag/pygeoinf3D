"""
pygeoinf3d.core — Core geometry contracts and abstractions.

Exports:
    AxisAlignedBoundingBox: Axis-aligned bounding box dataclass.
    CoordinateSystem: Enum of supported coordinate systems.
    Region3D: Thin wrapper around a pygeoinf Subset in EuclideanSpace(3).
    Domain3D: Abstract computational 3D domain.
    StructuredDomain3D: Abstract domain with natural grid/coordinate structure.
    Boundary3D: Abstract first-class boundary object.
"""

from .config import AxisAlignedBoundingBox, CoordinateSystem
from .region import Region3D
from .domain_base import Domain3D, StructuredDomain3D
from .boundary import Boundary3D
from .functions import Function3D

__all__ = [
    "AxisAlignedBoundingBox",
    "CoordinateSystem",
    "Region3D",
    "Domain3D",
    "StructuredDomain3D",
    "Boundary3D",
    "Function3D",
]
