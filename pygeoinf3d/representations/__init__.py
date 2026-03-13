"""
pygeoinf3d.representations — Finite-dimensional coefficient representations.

This package provides the interface and concrete objects for attaching a
finite-dimensional analysis/synthesis layer to continuous 3D function spaces
without polluting the primal space definition.

Architecture (see active-plans/pygeoinf3d-lebesgue-coefficient-architecture-plan.md):

* :class:`CoefficientRepresentation3D` — abstract base for all representations.
* :class:`LebesgueRepresentation3D`    — concrete representation for L² spaces.

Phase roadmap:
    Phase 1 (current): Representation boundary and API contracts.
    Phase 2: Gram-aware backends (OrthonormalProjectionBackend, GramProjectionBackend).
    Phase 3: TensorProductProvider integration.
    Phase 4: BoxLaplacian refactor to consume shared representations.
"""

from .base import CoefficientRepresentation3D
from .lebesgue import LebesgueRepresentation3D

__all__ = [
    "CoefficientRepresentation3D",
    "LebesgueRepresentation3D",
]
