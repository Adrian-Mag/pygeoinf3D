"""pygeoinf3d providers package.

Public API
----------
:class:`~pygeoinf3d.providers.base.FunctionProvider3D`
    Abstract base for indexed function providers.
:class:`~pygeoinf3d.providers.base.KernelProvider3D`
    Abstract base for indexed kernel providers.
:class:`~pygeoinf3d.providers.base.FunctionToKernelAdapter`
    Adapter wrapping a Function3D provider into a KernelProvider3D.
:class:`~pygeoinf3d.providers.tensor_product.TensorProductProvider`
    Spectral eigenbasis provider for box domains.
"""

from pygeoinf3d.providers.base import (
    FunctionProvider3D,
    FunctionToKernelAdapter,
    KernelProvider3D,
)
from pygeoinf3d.providers.tensor_product import TensorProductProvider

__all__ = [
    "FunctionProvider3D",
    "KernelProvider3D",
    "FunctionToKernelAdapter",
    "TensorProductProvider",
]
