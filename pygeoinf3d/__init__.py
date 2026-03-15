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
from .providers.base import (
    FunctionProvider3D,
    FunctionToKernelAdapter,
    KernelProvider3D,
)
from .providers.tensor_product import TensorProductProvider
from .operators.base import SpectralOperator3D
from .operators.laplacian_box import BoxLaplacian
from .operators.inverse_laplacian_box import InverseBoxLaplacian
from .operators.generalized_box_laplacian import GeneralizedBoxLaplacian

# SOLA operator — Phases 1-4: volume-kernel operator core, API locked
from .operators.sola import SOLAOperator3D

# Sampling — KL expansion sampler for Gaussian measures
from .sampling import KLSampler3D

# Lebesgue coefficient architecture Phase 1 — representation boundary
# Lebesgue coefficient architecture Phase 2 — Gram-aware backends
from .representations import (
    CoefficientRepresentation3D,
    LebesgueCoefficientBackend,
    LebesgueRepresentation3D,
    OrthonormalProjectionBackend,
    GramProjectionBackend,
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
    "FunctionProvider3D",
    "KernelProvider3D",
    "FunctionToKernelAdapter",
    "TensorProductProvider",
    "SpectralOperator3D",
    "BoxLaplacian",
    "InverseBoxLaplacian",
    "GeneralizedBoxLaplacian",
    # SOLA operator — Phases 1-4
    "SOLAOperator3D",
    # Sampling
    "KLSampler3D",
    # Phase 4 — Representation boundary
    "CoefficientRepresentation3D",
    "LebesgueRepresentation3D",
    # Phase 2 (backends) — Gram-aware coefficient infrastructure
    "LebesgueCoefficientBackend",
    "OrthonormalProjectionBackend",
    "GramProjectionBackend",
]
