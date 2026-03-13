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

* ``gaussian.py``   — axis-aligned Gaussian kernel providers (Phase 2)
* ``bump.py``       — Wendland/compactly-supported bump providers (Phase 2)
* ``boxcar.py``     — axis-aligned box-car providers (Phase 2)

All provider classes inherit from :class:`FunctionProvider3D`
(re-exported here for subclass convenience).

Public API
----------
:class:`~pygeoinf3d.providers.base.FunctionProvider3D`
    Abstract base re-exported so Phase-2 implementors can import it from
    here.
"""

from pygeoinf3d.providers.base import FunctionProvider3D

__all__: list[str] = [
    "FunctionProvider3D",
]
