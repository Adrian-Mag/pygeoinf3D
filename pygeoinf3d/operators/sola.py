"""SOLA-style measurement operator for 3D volume-kernel measurements.

Provides :class:`SOLAOperator3D`, a :class:`~pygeoinf.LinearOperator`
subclass that stacks volume-kernel measurements into a finite-dimensional
output.  All three mappings (forward, dual, adjoint) are implemented
basis-free on the domain side.

Mathematical definition
-----------------------
Given $m$ measurements $\\ell_1, \\ldots, \\ell_m$ on the function space
$V = L^2(\\Omega)$ and an $m$-dimensional codomain $W$, the operator is

.. math::

    A : V \\to W, \\quad (Af)_i = \\ell_i(f), \\quad i = 1, \\ldots, m

where each $\\ell_i$ is a :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D`
with kernel $k_i \\in V$:

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
    The forward, dual, and adjoint mappings are all basis-free on the
    domain side.  No call to
    :meth:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D.to_components` or
    :meth:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D.from_components` is
    made.  The adjoint is correct for any finite-dimensional Hilbert
    codomain including :class:`~pygeoinf.MassWeightedHilbertSpace`.
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

        if not measurements:
            raise ValueError(
                "SOLAOperator3D requires at least one measurement "
                "(codomain.dim must be positive)."
            )

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

        super().__init__(
            domain,
            codomain,
            self._forward,
            dual_mapping=self._dual_mapping,
            adjoint_mapping=self._adjoint_mapping,
        )

    # ------------------------------------------------------------------
    # Internal mappings
    # ------------------------------------------------------------------

    def _forward(self, f) -> object:
        """Evaluate each measurement on *f*.

        Return the realised codomain element.

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

    def _dual_mapping(self, yp) -> VolumeKernel3D:
        """Dual map $A^*: W^* \\to V^*$.

        Extracts codomain-dual coordinates and returns the corresponding
        weighted sum of the measurement forms as a
        :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D`.

        Args:
            yp: An element of the codomain dual space (a
                :class:`~pygeoinf.LinearForm` on the codomain).

        Returns:
            A :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` equal to
            $\\sum_i c_i \\ell_i$ where $c_i$ are the dual coordinates of *yp*.
        """
        c = self.codomain.dual.to_components(yp)
        result = float(c[0]) * self._measurements[0]
        for i in range(1, len(self._measurements)):
            result = result + float(c[i]) * self._measurements[i]
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
    def measurements(self) -> list[VolumeKernel3D]:
        """The list of volume-kernel measurements.

        Returns a copy so external mutation cannot affect the operator.
        """
        return list(self._measurements)
