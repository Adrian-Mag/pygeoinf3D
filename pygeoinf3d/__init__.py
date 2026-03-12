"""
pygeoinf3d — Concrete 3D function spaces and operators
for geophysical inference.

Built on top of pygeoinf using EuclideanSpace(3) and Subset
for geometry semantics.
"""

from .core.config import AxisAlignedBoundingBox, CoordinateSystem
from .core.region import Region3D
from .core.domain_base import Domain3D, StructuredDomain3D
from .core.boundary import Boundary3D
from .core.functions import Function3D
from .spaces.lebesgue import Lebesgue3D
from .spaces.forms import VolumeKernel3D

__all__ = [
    "AxisAlignedBoundingBox",
    "CoordinateSystem",
    "Region3D",
    "Domain3D",
    "StructuredDomain3D",
    "Boundary3D",
    "Function3D",
    "Lebesgue3D",
    "VolumeKernel3D",
]
