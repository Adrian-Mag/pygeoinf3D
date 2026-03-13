"""SOLA-style operator for 3D volume kernels.

Provides :class:`SOLAOperator3D`, a :class:`~pygeoinf.LinearOperator`
subclass that stacks volume kernels into a finite-dimensional output.
All three mappings (forward, dual, adjoint) are implemented
basis-free on the domain side.

Mathematical definition
-----------------------
Given $m$ kernels $k_1, \\ldots, k_m$ on the function space
$V = L^2(\\Omega)$ and an $m$-dimensional codomain $W$, the operator is

.. math::

    A : V \\to W, \\quad (Af)_i = \\ell_i(f), \\quad i = 1, \\ldots, m

where each $\\ell_i$ is induced by a
:class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` kernel $k_i \\in V$:

.. math::

    \\ell_i(f)
    = \\int_\\Omega k_i(\\mathbf{x})\\, f(\\mathbf{x})\\,\\mathrm{d}V.

Dual mapping
------------
The dual $A^* : W^* \\to V^*$ is given by

.. math::

    A^*(y') = \\sum_{i=1}^m c_i\\, \\ell_i,
    \\qquad c_i = (y')_i,

where $c = $ ``codomain.dual.to_components(y')``.
The result is a :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` whose
kernel is $\\sum_i c_i k_i$.  No component machinery on the domain side is
used.

Adjoint mapping
---------------
The adjoint $A^\\dagger : W \\to V$ is computed as

.. math::

    A^\\dagger(y) = R_V^{-1}\\bigl(A^*(R_W(y))\\bigr),

where $R_W$ is the Riesz map of the codomain (``codomain.to_dual``) and
$R_V^{-1}$ is its domain inverse (``domain.from_dual``).  This is correct
for any finite-dimensional Hilbert codomain, including
:class:`~pygeoinf.MassWeightedHilbertSpace`.

Design constraints
------------------
* The forward map does **not** call
  :meth:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D.to_components` or
  :meth:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D.from_components` on the
  domain side.  Each measurement is a pairing-first dual element evaluated
  directly.
* :meth:`~pygeoinf.LinearOperator.from_linear_forms` is **not** used.
* ``codomain.from_components`` controls how the raw scalar kernel
    evaluations are
  assembled into the output element.
* Kernels may alternatively be provided via a
  :class:`~pygeoinf3d.providers.base.KernelProvider3D` for lazy,
  cached, indexed access (Phase 3).

Usage
-----
Sequence-backed (eager)::

    from pygeoinf3d.operators.sola import SOLAOperator3D
    from pygeoinf3d.spaces.forms import VolumeKernel3D
    from pygeoinf import EuclideanSpace

    space = Lebesgue3D(some_domain)
    codomain = EuclideanSpace(n)
    kernels = [VolumeKernel3D(space, k_i) for k_i in kernel_functions]
    A = SOLAOperator3D(space, codomain, kernels)
    y = A(f)   # element of codomain; np.ndarray for EuclideanSpace

Provider-backed (lazy + cached)::

    from pygeoinf3d.providers.base import FunctionToKernelAdapter
    from pygeoinf3d.providers.tensor_product import TensorProductProvider

    provider = TensorProductProvider(box, truncation=4, bc_type='dirichlet')
    adapter = FunctionToKernelAdapter(provider, space)
    codomain = EuclideanSpace(adapter.dim)
    A = SOLAOperator3D(space, codomain, adapter)
    y = A(f)
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from pygeoinf.linear_operators import LinearOperator

from pygeoinf3d.providers.base import KernelProvider3D
from pygeoinf3d.spaces.forms import VolumeKernel3D
from pygeoinf3d.spaces.lebesgue import Lebesgue3D


class SOLAOperator3D(LinearOperator):
    """Forward mapping defined by stacked volume kernels.

    Evaluates each :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` on
    the input function and assembles the scalar outputs into a codomain
    element via ``codomain.from_components``.

    The *kernels* argument accepts either:

    * A **sequence** of :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D`
      objects (eager, sequence-backed mode); or
    * A :class:`~pygeoinf3d.providers.base.KernelProvider3D` instance
      (lazy, provider-backed mode).  Kernels are fetched on demand via
      ``provider.get_kernel(i)`` and cached for stability.

    Parameters
    ----------
    domain:
        The :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` on which the
        kernels are defined.
    codomain:
        A finite-dimensional :class:`~pygeoinf.HilbertSpace` with
        ``codomain.dim == len(kernels)`` (or ``codomain.dim == provider.dim``
        in provider-backed mode).  Only ``from_components`` is exercised by
        the forward mapping.
    kernels:
        Either a sequence of :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D`
        objects or a :class:`~pygeoinf3d.providers.base.KernelProvider3D`.

    Raises
    ------
    TypeError
        If *domain* is not a :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D`.
    ValueError
        If kernel count does not match ``codomain.dim``.
    TypeError
        If *kernels* is a sequence and any element is not a
        :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` (checked eagerly
        for sequences only).
    ValueError
        If *kernels* is a sequence and any kernel is defined on a different
        ``Lebesgue3D`` instance than *domain* (identity check, sequence
        mode only).

    Notes
    -----
    The forward, dual, and adjoint mappings are all basis-free on the
    domain side.  No call to
    :meth:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D.to_components` or
    :meth:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D.from_components` is
    made.  The adjoint is correct for any finite-dimensional Hilbert
    codomain including :class:`~pygeoinf.MassWeightedHilbertSpace`.

    In provider-backed mode, per-kernel validation is deferred to first
    access: ``VolumeKernel3D`` will raise if the provider produces a kernel
    incompatible with *domain*.
    """

    def __init__(
        self,
        domain: Lebesgue3D,
        codomain,
        kernels: Union[Sequence[VolumeKernel3D], KernelProvider3D],
    ) -> None:
        if not isinstance(domain, Lebesgue3D):
            raise TypeError(
                f"SOLAOperator3D requires a Lebesgue3D domain, "
                f"got {type(domain).__name__!r}."
            )

        if isinstance(kernels, KernelProvider3D):
            # ---- Provider-backed path (lazy + cached) ----
            if kernels.dim == 0:
                raise ValueError(
                    "SOLAOperator3D requires at least one kernel "
                    "(codomain.dim must be positive)."
                )
            if kernels.dim != codomain.dim:
                raise ValueError(
                    f"Number of kernels ({kernels.dim}) must equal "
                    f"codomain.dim ({codomain.dim})."
                )
            self._provider: KernelProvider3D | None = kernels
            self._kernels: list[VolumeKernel3D] | None = None
            self._n_kernels: int = kernels.dim
        else:
            # ---- Sequence-backed path (eager validation) ----
            kernels_list = list(kernels)

            if not kernels_list:
                raise ValueError(
                    "SOLAOperator3D requires at least one kernel "
                    "(codomain.dim must be positive)."
                )
            if len(kernels_list) != codomain.dim:
                raise ValueError(
                    f"Number of kernels ({len(kernels_list)}) must equal "
                    f"codomain.dim ({codomain.dim})."
                )
            for i, kernel in enumerate(kernels_list):
                if not isinstance(kernel, VolumeKernel3D):
                    raise TypeError(
                        f"kernels[{i}] is not a VolumeKernel3D "
                        f"(got {type(kernel).__name__!r})."
                    )
                if kernel.domain is not domain:
                    raise ValueError(
                        f"kernels[{i}].domain is not the same Lebesgue3D "
                        "instance as the operator domain (identity check).  "
                        "Build each VolumeKernel3D with the exact same "
                        "Lebesgue3D object passed as 'domain' here."
                    )
            self._provider = None
            self._kernels = kernels_list
            self._n_kernels = len(kernels_list)

        # Cache for provider-backed mode; unused in sequence-backed mode.
        self._kernel_cache: dict[int, VolumeKernel3D] = {}

        super().__init__(
            domain,
            codomain,
            self._forward,
            dual_mapping=self._dual_mapping,
            adjoint_mapping=self._adjoint_mapping,
        )

    # ------------------------------------------------------------------
    # Kernel access (with caching for provider-backed mode)
    # ------------------------------------------------------------------

    def _get_kernel(self, i: int) -> VolumeKernel3D:
        """Return the *i*-th kernel, using the cache in provider-backed mode.

        In sequence-backed mode returns ``self._kernels[i]`` directly.
        In provider-backed mode, checks ``self._kernel_cache`` first,
        calls ``self._provider.get_kernel(i)`` on a miss, then caches
        and returns the result.

        Args:
            i: Index in ``range(self._n_kernels)``.

        Returns:
            The :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` at index *i*.
        """
        if self._provider is None:
            # Sequence-backed: direct access, no caching needed.
            return self._kernels[i]  # type: ignore[index]
        if i not in self._kernel_cache:
            self._kernel_cache[i] = self._provider.get_kernel(i)
        return self._kernel_cache[i]

    # ------------------------------------------------------------------
    # Internal mappings
    # ------------------------------------------------------------------

    def _forward(self, f) -> object:
        """Evaluate each kernel on *f*.

        Return the realised codomain element.

        Args:
            f: A :class:`~pygeoinf3d.core.functions.Function3D` from the
               domain space.

        Returns:
            An element of the codomain, constructed via
            ``codomain.from_components(raw_components)`` where
            ``raw_components[i] = kernels[i](f)``.
        """
        raw = np.array(
            [self._get_kernel(i)(f) for i in range(self._n_kernels)],
            dtype=float,
        )
        return self.codomain.from_components(raw)

    def _dual_mapping(self, yp) -> VolumeKernel3D:
        """Dual map $A^*: W^* \\to V^*$.

        Extracts codomain-dual coordinates and returns the corresponding
        weighted sum of the kernel-induced forms as a
        :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D`.

        Args:
            yp: An element of the codomain dual space (a
                :class:`~pygeoinf.LinearForm` on the codomain).

        Returns:
            A :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` equal to
            $\\sum_i c_i \\ell_i$ where $c_i$ are the dual coordinates of *yp*.
        """
        c = self.codomain.dual.to_components(yp)
        result = float(c[0]) * self._get_kernel(0)
        for i in range(1, self._n_kernels):
            result = result + float(c[i]) * self._get_kernel(i)
        return result

    def _adjoint_mapping(self, y):
        """Adjoint map $A^\\dagger: W \\to V$.

        Applies the codomain Riesz map, the dual mapping, then the domain
        inverse-Riesz map.

        Args:
            y: An element of the codomain.

        Returns:
            A :class:`~pygeoinf3d.core.functions.Function3D` in the domain.
        """
        yp = self.codomain.to_dual(y)
        xp = self._dual_mapping(yp)
        return self.domain.from_dual(xp)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def kernels(self) -> list[VolumeKernel3D]:
        """The list of volume kernels.

        In sequence-backed mode returns a copy of the stored list.
        In provider-backed mode, realizes all kernels on demand (with caching)
        and returns them as a list.  Repeated calls return identity-equal
        kernel objects because the cache is shared.

        Returns a new list object each call, so external mutation is safe.
        """
        return [self._get_kernel(i) for i in range(self._n_kernels)]
