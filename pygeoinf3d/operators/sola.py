"""SOLA-style measurement operator for 3D volume-kernel measurements.

Provides :class:`SOLAOperator3D`, a :class:`~pygeoinf.LinearOperator`
subclass that stacks volume-kernel measurements into a finite-dimensional
output.  The forward map is basis-free on the domain side and uses
``codomain.from_components`` on the output side.

Mathematical definition
-----------------------
Given $m$ measurements $\\ell_1, \\ldots, \\ell_m$ on the function space
$V = L^2(\\Omega)$ and an $m$-dimensional codomain $W$, the operator is

.. math::

    A : V \\to W, \\quad (Af)_i = \\ell_i(f), \\quad i = 1, \\ldots, m

where each $\\ell_i$ is a :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D`
with kernel $k_i \\in V$:

.. math::

    \\ell_i(f) = \\int_\\Omega k_i(\\mathbf{x})\\, f(\\mathbf{x})\\,\\mathrm{d}V.

Phase limitations
-----------------
**Phase 1 only: forward mapping.**  Dual and adjoint mappings are deferred
to Phase 2.  The :class:`~pygeoinf.LinearOperator` base-class fallback for
dual/adjoint (numerical matrix assembly from component enumeration) is
inherited and is valid only for very low-dimensional codomains.

Design constraints
------------------
* The forward map does **not** call
  :meth:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D.to_components` or
  :meth:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D.from_components` on the
  domain side.  Each measurement is a pairing-first dual element evaluated
  directly.
* :meth:`~pygeoinf.LinearOperator.from_linear_forms` is **not** used.
* ``codomain.from_components`` controls how the raw scalar measurements are
  assembled into the output element.

Usage
-----
::

    from pygeoinf3d.operators.sola import SOLAOperator3D
    from pygeoinf3d.spaces.forms import VolumeKernel3D
    from pygeoinf import EuclideanSpace

    space = Lebesgue3D(some_domain)
    codomain = EuclideanSpace(n)
    kernels = [VolumeKernel3D(space, k_i) for k_i in kernel_functions]
    A = SOLAOperator3D(space, codomain, kernels)
    y = A(f)   # element of codomain; np.ndarray for EuclideanSpace
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pygeoinf.linear_operators import LinearOperator

from pygeoinf3d.spaces.forms import VolumeKernel3D
from pygeoinf3d.spaces.lebesgue import Lebesgue3D


class SOLAOperator3D(LinearOperator):
    """Forward mapping defined by stacked volume-kernel measurements.

    Evaluates each :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` on
    the input function and assembles the scalar outputs into a codomain
    element via ``codomain.from_components``.

    Parameters
    ----------
    domain:
        The :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` on which the
        measurements are defined.
    codomain:
        A finite-dimensional :class:`~pygeoinf.HilbertSpace` with
        ``codomain.dim == len(measurements)``.  Only ``from_components``
        is exercised by the Phase-1 forward mapping.
    measurements:
        A sequence of :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D`
        objects, each defined on *domain*.

    Raises
    ------
    TypeError
        If *domain* is not a :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D`.
    ValueError
        If ``len(measurements) != codomain.dim``.
    TypeError
        If any element of *measurements* is not a
        :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D`.
    ValueError
        If any measurement is defined on a different ``Lebesgue3D`` instance
        than *domain* (identity check).

    Notes
    -----
    **Phase 1** provides the forward mapping only.  Explicit dual and
    adjoint mappings are deliberately omitted and will be implemented in
    Phase 2.  The :class:`~pygeoinf.LinearOperator` super-class numerical
    fallback for dual/adjoint (slow matrix assembly via component
    enumeration) is inherited but is not intended for production use.

    The forward map does **not** touch
    :meth:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D.to_components` or
    :meth:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D.from_components` on
    the domain side.
    """

    def __init__(
        self,
        domain: Lebesgue3D,
        codomain,
        measurements: Sequence[VolumeKernel3D],
    ) -> None:
        if not isinstance(domain, Lebesgue3D):
            raise TypeError(
                f"SOLAOperator3D requires a Lebesgue3D domain, "
                f"got {type(domain).__name__!r}."
            )

        measurements = list(measurements)

        if len(measurements) != codomain.dim:
            raise ValueError(
                f"Number of measurements ({len(measurements)}) must equal "
                f"codomain.dim ({codomain.dim})."
            )

        for i, m in enumerate(measurements):
            if not isinstance(m, VolumeKernel3D):
                raise TypeError(
                    f"measurements[{i}] is not a VolumeKernel3D "
                    f"(got {type(m).__name__!r})."
                )
            if m.domain is not domain:
                raise ValueError(
                    f"measurements[{i}].domain is not the same Lebesgue3D "
                    "instance as the operator domain (identity check).  "
                    "Build each VolumeKernel3D with the exact same Lebesgue3D "
                    "object passed as 'domain' here."
                )

        self._measurements: list[VolumeKernel3D] = measurements

        # Phase 1: forward mapping only.
        # Dual and adjoint mappings are intentionally deferred to Phase 2.
        # The LinearOperator base class will fall back to slow numerical
        # dual/adjoint if called, which is safe for small codomains only.
        super().__init__(
            domain,
            codomain,
            self._forward,
            # dual_mapping and adjoint_mapping intentionally omitted (Phase 1).
        )

    # ------------------------------------------------------------------
    # Internal forward mapping
    # ------------------------------------------------------------------

    def _forward(self, f) -> object:
        """Evaluate each measurement on *f* and return the realised codomain element.

        Args:
            f: A :class:`~pygeoinf3d.core.functions.Function3D` from the
               domain space.

        Returns:
            An element of the codomain, constructed via
            ``codomain.from_components(raw_components)`` where
            ``raw_components[i] = measurements[i](f)``.
        """
        raw = np.array(
            [m(f) for m in self._measurements],
            dtype=float,
        )
        return self.codomain.from_components(raw)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def measurements(self) -> list[VolumeKernel3D]:
        """The list of volume-kernel measurements.

        Returns a copy so external mutation cannot affect the operator.
        """
        return list(self._measurements)
