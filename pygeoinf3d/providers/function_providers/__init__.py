"""Volumetric kernel function providers.

This sub-package collects user-facing
:class:`~pygeoinf3d.providers.base.FunctionProvider3D`
implementations that yield :class:`~pygeoinf3d.core.functions.Function3D`
objects suitable for wrapping into
:class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` via
:class:`~pygeoinf3d.providers.base.FunctionToKernelAdapter`.

Layout convention
-----------------
Each concrete provider family lives in its own module:

* ``gaussian.py``   — axis-aligned Gaussian kernel providers
* ``bump.py``       — :math:`C^\\infty` axis-separable compactly-
    supported bumps
* ``boxcar.py``     — axis-aligned box-car (indicator) providers

All provider classes inherit from :class:`FunctionProvider3D`
(re-exported here for subclass convenience).

Public API
----------
:class:`~pygeoinf3d.providers.base.FunctionProvider3D`
    Abstract base re-exported so subclasses can import it from here.
:class:`GaussianProvider3D`
    Axis-aligned anisotropic Gaussian functions.
:class:`BumpProvider3D`
    Axis-separable :math:`C^\\infty` compactly-supported bump functions.
:class:`BoxcarProvider3D`
    Axis-aligned box-car (indicator) functions.
"""

from pygeoinf3d.providers.base import FunctionProvider3D
from pygeoinf3d.providers.function_providers.gaussian import GaussianProvider3D
from pygeoinf3d.providers.function_providers.bump import BumpProvider3D
from pygeoinf3d.providers.function_providers.boxcar import BoxcarProvider3D

__all__: list[str] = [
    "FunctionProvider3D",
    "GaussianProvider3D",
    "BumpProvider3D",
    "BoxcarProvider3D",
]
