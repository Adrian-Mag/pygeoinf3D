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

# Phase 3 — Box geometry, tensor-product provider, spectral Laplacian
from .geometries.box.domain import BoxDomain
from .providers.tensor_product import TensorProductProvider
from .operators.laplacian_box import BoxLaplacian

# Phase 4 — Sobolev space and Bessel operators on box domains
from .spaces.sobolev import Sobolev3D, SpectralSobolev3D
from .spaces.forms import SobolevKernel3D
from .operators.bessel import (
    BesselOperator3D,
    SpectralBesselOperator3D,
    bessel_direct,
    bessel_inverse,
)

__all__ = [
    # Core geometry
    "AxisAlignedBoundingBox",
    "CoordinateSystem",
    "Region3D",
    "Domain3D",
    "StructuredDomain3D",
    "Boundary3D",
    # Functions and spaces
    "Function3D",
    "Lebesgue3D",
    "VolumeKernel3D",
    # Phase 3 — Box geometry MVP
    "BoxDomain",
    "TensorProductProvider",
    "BoxLaplacian",
    # Phase 4 — Sobolev H^s spaces and Bessel operators
    "Sobolev3D",
    "SobolevKernel3D",
    "SpectralSobolev3D",
    "BesselOperator3D",
    "SpectralBesselOperator3D",
    "bessel_direct",
    "bessel_inverse",
]
