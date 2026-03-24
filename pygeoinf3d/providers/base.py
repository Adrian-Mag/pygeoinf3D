"""Abstract bases and adapters for indexed providers.

Defines :class:`FunctionProvider3D`, the minimal protocol for objects that
supply :class:`~pygeoinf3d.core.functions.Function3D` instances on indexed
demand, and :class:`KernelProvider3D` for
:class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` instances.

Also provides :class:`FunctionToKernelAdapter`, a small adapter that wraps
any :class:`FunctionProvider3D` (e.g.
:class:`~pygeoinf3d.providers.tensor_product.BoxLaplacianSpectrumProvider`) into a
:class:`KernelProvider3D` by binding each produced
:class:`~pygeoinf3d.core.functions.Function3D` to a given
:class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D`.

Design notes
------------
* ``FunctionProvider3D`` only requires ``dim`` and ``get_function``.
* ``KernelProvider3D`` only requires ``dim`` and ``get_kernel``.
* ``FunctionToKernelAdapter`` does *not* validate space/provider compatibility
  eagerly.  A mismatch (wrong domain identity) surfaces as a ``ValueError``
  inside :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` on the first call
  to ``get_kernel``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pygeoinf3d.core.functions import Function3D
    from pygeoinf3d.spaces.forms import VolumeKernel3D
    from pygeoinf3d.spaces.lebesgue import Lebesgue3D


class FunctionProvider3D(ABC):
    """Abstract base for objects that produce functions on demand.

    Implementors must expose a total count (``dim``) and a factory method
    ``get_function(i)`` that returns the *i*-th function without requiring
    all functions to be built in advance.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Total number of functions this provider can produce."""

    @abstractmethod
    def get_function(self, i: int) -> "Function3D":
        """Return the *i*-th :class:`~pygeoinf3d.core.functions.Function3D`."""

    def evaluate_all(self, points: np.ndarray) -> np.ndarray:
        """Evaluate all functions at the given points.

        Parameters
        ----------
        points : ndarray of shape ``(n_pts, 3)``
            Evaluation points.

        Returns
        -------
        ndarray of shape ``(dim, n_pts)``
            Row *i* contains the values of the *i*-th function at all
            *points*.  Subclasses may override this default with a
            vectorized implementation for better performance.
        """
        points = np.asarray(points, dtype=float)
        rows = [self.get_function(i)(points) for i in range(self.dim)]
        return np.vstack(rows)

    def evaluate_batch(self, i_start: int, i_end: int, points: np.ndarray) -> np.ndarray:
        """Evaluate functions ``i_start:i_end`` at the given points.

        Parameters
        ----------
        i_start, i_end : int
            Half-open row range ``[i_start, i_end)``.
        points : ndarray of shape ``(n_pts, 3)``

        Returns
        -------
        ndarray of shape ``(i_end - i_start, n_pts)``
            Default implementation calls :meth:`evaluate_all` and slices rows.
            Subclasses should override for efficiency.
        """
        return self.evaluate_all(points)[i_start:i_end]


class KernelProvider3D(ABC):
    """Abstract base for objects that produce kernels on demand.

    Implementors must expose a total count (``dim``) and a factory method
    ``get_kernel(i)`` that returns the *i*-th kernel without requiring all
    kernels to be built in advance.

    This ABC carries no storage and makes no assumptions about how kernels
    are generated — spectral modes, data-driven kernels, etc. are all
    compatible.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Total number of kernels this provider can produce."""

    @abstractmethod
    def get_kernel(self, i: int) -> "VolumeKernel3D":
        """Return the *i*-th :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D`.

        Parameters
        ----------
        i:
            Flat index in ``range(dim)``.

        Returns
        -------
        VolumeKernel3D
            The kernel for index *i*.

        Raises
        ------
        IndexError
            If *i* is out of ``range(dim)``.
        """


class FunctionToKernelAdapter(KernelProvider3D):
    """Adapts a *Function3D* provider into a :class:`KernelProvider3D`.

    Wraps any object that exposes ``dim: int`` and
    ``get_function(i) -> Function3D`` (e.g.
    :class:`~pygeoinf3d.providers.tensor_product.BoxLaplacianSpectrumProvider`)
    by binding each produced function to a
    :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D`
    space to form a :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D`.

    Parameters
    ----------
    function_provider:
        Any object with ``dim: int`` and
        ``get_function(i: int) -> Function3D``.  Typically a
        :class:`~pygeoinf3d.providers.tensor_product.BoxLaplacianSpectrumProvider`.
    space:
        The :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` to which each
        produced function is bound.  Its ``function_domain`` must be the
        same :class:`~pygeoinf3d.core.domain_base.Domain3D` instance that
        *function_provider* uses internally — this is validated lazily by
        :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` at first access.

    Notes
    -----
    Compatibility between *function_provider* and *space* is not checked at
    construction time.  A domain mismatch raises ``ValueError`` on the first
    call to :meth:`get_kernel` when ``VolumeKernel3D`` validates
    ``kernel.domain is space.function_domain``.
    """

    def __init__(
        self,
        function_provider: "FunctionProvider3D",
        space: "Lebesgue3D",
    ) -> None:
        self._function_provider = function_provider
        self._space = space

    @property
    def dim(self) -> int:
        """Number of kernels — delegates to the wrapped provider's ``dim``."""
        return self._function_provider.dim

    def get_kernel(self, i: int) -> "VolumeKernel3D":
        """Return the *i*-th kernel by wrapping ``get_function(i)``.

        Calls ``function_provider.get_function(i)`` and wraps the
        result in a :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` bound
        to the space given at construction.

        Parameters
        ----------
        i:
            Flat index in ``range(dim)``.

        Returns
        -------
        VolumeKernel3D
            A freshly constructed kernel (caching is the caller's
            responsibility — see
            :class:`~pygeoinf3d.operators.sola.SOLAOperator3D`.

        Raises
        ------
        IndexError
            Propagated from ``function_provider.get_function(i)`` if
            *i* is out of range.
        ValueError
            If the produced function's domain does not match
            ``space.function_domain`` (raised by ``VolumeKernel3D``).
        """
        # Import here to avoid circular imports at module level.
        from pygeoinf3d.spaces.forms import VolumeKernel3D  # noqa: PLC0415

        fn = self._function_provider.get_function(i)
        return VolumeKernel3D(self._space, fn)

    def __repr__(self) -> str:
        return (
            f"FunctionToKernelAdapter("
            f"provider={self._function_provider!r}, "
            f"space={self._space!r})"
        )
