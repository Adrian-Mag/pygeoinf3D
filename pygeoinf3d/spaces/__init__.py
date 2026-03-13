"""
pygeoinf3d.spaces -- Function spaces on 3D domains.

Exports:
    Lebesgue3D: L2 Hilbert space on a Domain3D with Function3D elements.
    VolumeKernel3D: Integration-backed LinearForm for Lebesgue3D pairings.
    Sobolev3D: H^s space with Function3D elements (continuous-first).
    SobolevKernel3D: Spectral-projection-backed LinearForm for Sobolev3D duals.
    SpectralSobolev3D: Coefficient-vector H^s space (secondary spectral API).
"""

from .lebesgue import Lebesgue3D
from .forms import VolumeKernel3D, SobolevKernel3D
from .sobolev import Sobolev3D, SpectralSobolev3D

__all__ = [
    "Lebesgue3D",
    "VolumeKernel3D",
    "Sobolev3D",
    "SobolevKernel3D",
    "SpectralSobolev3D",
]
