"""Sampling utilities for pygeoinf3d.

Currently provides:
  - KLSampler3D: Karhunen-Loève expansion sampler for any SpectralOperator3D
"""

from .kl_sampler import KLSampler3D

__all__ = ["KLSampler3D"]
