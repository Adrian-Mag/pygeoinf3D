"""
pygeoinf3d.representations — Finite-dimensional coefficient representations.

This package provides the interface and concrete objects for attaching a
finite-dimensional analysis/synthesis layer to continuous 3D function spaces
without polluting the primal space definition.

Architecture (see active-plans/pygeoinf3d-lebesgue-coefficient-architecture-plan.md):

* :class:`CoefficientRepresentation3D` — abstract base for all representations.
* :class:`LebesgueRepresentation3D`    — concrete representation for L² spaces.
* :class:`LebesgueCoefficientBackend`  — abstract numerical backend interface.
* :class:`OrthonormalProjectionBackend` — backend for orthonormal families.
* :class:`GramProjectionBackend`       — backend for general non-orthogonal families.

Phase roadmap:
    Phase 1 (complete): Representation boundary and API contracts.
    Phase 2 (current): Gram-aware backends + induced coefficient metric.
    Phase 3: TensorProductProvider integration.
    Phase 4: BoxLaplacian refactor to consume shared representations.
"""

from .base import CoefficientRepresentation3D, LebesgueCoefficientBackend
from .lebesgue import LebesgueRepresentation3D
from .backends import OrthonormalProjectionBackend, GramProjectionBackend

__all__ = [
    "CoefficientRepresentation3D",
    "LebesgueCoefficientBackend",
    "LebesgueRepresentation3D",
    "OrthonormalProjectionBackend",
    "GramProjectionBackend",
]
