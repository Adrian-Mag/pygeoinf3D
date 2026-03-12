"""L² (Lebesgue) function space on a 3D domain.

Provides :class:`Lebesgue3D`, a concrete :class:`~pygeoinf.HilbertSpace`
whose vectors are :class:`~pygeoinf3d.core.functions.Function3D` objects and
whose inner product is the standard L² volume integral:

.. math::

    \\langle f,\\, g \\rangle_{L^2(\\Omega)}
    = \\int_\\Omega f(\\mathbf{x})\\,g(\\mathbf{x})\\,\\mathrm{d}V

Design notes
------------
*Pairing-first.*
  :meth:`to_dual` is the primary route into dual space.  The inner
  product is derived from it via ``inner_product(f, g) = to_dual(f)(g)``,
  which is the :class:`~pygeoinf.HilbertSpace` default.

*Basis-free.*
  ``dim == 0`` signals that no finite-dimensional basis exists in Phase 2.
  ``to_components`` and ``from_components`` raise :exc:`NotImplementedError`.
  The ``zero`` property and all arithmetic helpers are overridden so they
  never fall through to the basis-machinery defaults in
  :class:`~pygeoinf.HilbertSpace`.

*Conservative ``from_dual``.*
  Only :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` objects whose
  ``domain`` is *this* exact space instance are accepted.  Any other object
  is rejected with a clear error.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pygeoinf import HilbertSpace

from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.spaces.forms import VolumeKernel3D


class Lebesgue3D(HilbertSpace):
    """Concrete L² Hilbert space on a :class:`~pygeoinf3d.core.domain_base.Domain3D`.

    Parameters
    ----------
    domain:
        A :class:`~pygeoinf3d.core.domain_base.Domain3D` providing the
        geometric region and integration back-end.

    Examples
    --------
    ::

        dom = SomeDomain3D(...)
        space = Lebesgue3D(dom)

        f = Function3D(dom, evaluate_callable=lambda x: x[:, 0] if x.ndim == 2 else float(x[0]))
        g = Function3D(dom, evaluate_callable=lambda x: x[:, 1] if x.ndim == 2 else float(x[1]))

        # Pairing (dual evaluation)
        fp = space.to_dual(f)   # VolumeKernel3D
        value = fp(g)           # ∫ f·g dV

        # Inner product (derived from pairing)
        ip = space.inner_product(f, g)  # identical to fp(g)
    """

    def __init__(self, domain) -> None:
        self._function_domain = domain

    # ------------------------------------------------------------------
    # HilbertSpace abstract interface
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Dimension of the finite component representation.

        Returns 0 in Phase 2 (no basis); signals basis-free mode.
        """
        return 0

    @property
    def function_domain(self):
        """The :class:`~pygeoinf3d.core.domain_base.Domain3D` this space is built on."""
        return self._function_domain

    def to_dual(self, f: Function3D) -> VolumeKernel3D:
        """Map a primal function to its Riesz dual.

        Returns a :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` that
        evaluates ``g ↦ ∫_Ω f(x) g(x) dV``.

        Parameters
        ----------
        f:
            A :class:`~pygeoinf3d.core.functions.Function3D` in this space.

        Returns
        -------
        VolumeKernel3D
            Dual element representing *f*.

        Raises
        ------
        ValueError
            If *f*'s domain is not the same object as this space's
            ``function_domain``.
        """
        if f.domain is not self._function_domain:
            raise ValueError(
                "Lebesgue3D.to_dual: f.domain does not match this space's "
                "function_domain (identity check).  "
                "f must be a Function3D built on the same Domain3D instance."
            )
        return VolumeKernel3D(self, f)

    def from_dual(self, xp: Any) -> Function3D:
        """Recover a primal function from a dual element.

        Only :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` objects whose
        ``domain`` is *this* space instance are accepted.  All other dual
        elements are rejected.

        Parameters
        ----------
        xp:
            A dual element.  Must be a ``VolumeKernel3D`` produced by
            ``self.to_dual(f)`` for some *f*.

        Returns
        -------
        Function3D
            The primal representer.

        Raises
        ------
        TypeError
            If *xp* is not a ``VolumeKernel3D``.
        ValueError
            If *xp* was produced by a different ``Lebesgue3D`` instance.
        """
        if not isinstance(xp, VolumeKernel3D):
            raise TypeError(
                f"Lebesgue3D.from_dual requires a VolumeKernel3D, "
                f"got {type(xp).__name__!r}.  "
                "Generic LinearForm objects are not representable in this "
                "basis-free L² setting."
            )
        if xp.domain is not self:
            raise ValueError(
                "VolumeKernel3D was produced by a different Lebesgue3D "
                "instance and cannot be recovered through this space."
            )
        return xp.kernel

    def to_components(self, f: Function3D) -> np.ndarray:
        """Not implemented in Phase 2 (no basis).

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "Lebesgue3D has no finite basis in Phase 2.  "
            "to_components will be available once a basis provider is attached "
            "(Phase 3+)."
        )

    def from_components(self, c: np.ndarray) -> Function3D:
        """Not implemented in Phase 2 (no basis).

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "Lebesgue3D has no finite basis in Phase 2.  "
            "from_components will be available once a basis provider is "
            "attached (Phase 3+)."
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Lebesgue3D):
            return False
        return self._function_domain is other._function_domain

    def __hash__(self) -> int:
        return id(self._function_domain)

    # ------------------------------------------------------------------
    # Override HilbertSpace defaults that fall through to from_components
    # ------------------------------------------------------------------

    @property
    def zero(self) -> Function3D:
        """The zero function (additive identity) in this space."""
        return Function3D(
            self._function_domain,
            evaluate_callable=lambda x: (
                np.zeros(len(x)) if np.asarray(x).ndim == 2 else 0.0
            ),
        )

    def copy(self, f: Function3D) -> Function3D:
        """Return a copy of *f*."""
        return f.copy()

    def add(self, f: Function3D, g: Function3D) -> Function3D:
        """Compute *f* + *g*."""
        return f + g

    def subtract(self, f: Function3D, g: Function3D) -> Function3D:
        """Compute *f* − *g*."""
        return f - g

    def multiply(self, a: float, f: Function3D) -> Function3D:
        """Compute *a* · *f*."""
        return a * f

    def negative(self, f: Function3D) -> Function3D:
        """Compute −*f*."""
        return -f

    def inner_product(self, f: Function3D, g: Function3D) -> float:
        """Compute the L² inner product ⟨f, g⟩ = ∫_Ω f(x)g(x) dV.

        Delegates to :meth:`to_dual` so the pairing is authoritative.
        """
        return self.to_dual(f)(g)

    def norm(self, f: Function3D) -> float:
        """Compute the L² norm ``‖f‖ = √⟨f, f⟩``."""
        ip = self.inner_product(f, f)
        return float(np.sqrt(max(ip, 0.0)))

    def __repr__(self) -> str:
        return f"Lebesgue3D(domain={self._function_domain!r})"
