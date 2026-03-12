"""
pygeoinf3d.spaces — Function spaces on 3D domains.

Exports:
    Lebesgue3D: L² Hilbert space on a Domain3D with pairing-first design.
    VolumeKernel3D: Integration-backed LinearForm for Lebesgue3D pairings.
"""

from .lebesgue import Lebesgue3D
from .forms import VolumeKernel3D

__all__ = [
    "Lebesgue3D",
    "VolumeKernel3D",
]
