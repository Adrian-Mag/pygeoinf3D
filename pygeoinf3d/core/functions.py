"""Primal scalar functions on 3D domains.

This module provides :class:`Function3D`: a wrapper for scalar-valued
functions on a :class:`~pygeoinf3d.core.domain_base.Domain3D`.

Design principles
-----------------
- **Primal only.**  ``Function3D`` represents a function *f : Ω → ℝ*.  It is
  **not** a distribution container; point / path / surface distributions live
  in a separate dual hierarchy (Phase 6+).
- **Callable-backed.**  Phase 2 supports callable evaluation exclusively.
  Coefficient-backed representations require a basis provider (Phase 3+).
- **Two modes.**

  standalone
      Created with a :class:`~pygeoinf3d.core.domain_base.Domain3D` directly.
      ``is_attached == False``, ``space is None``.
  space-attached
      Created (or promoted via ``attach_to_space``) with a
      :class:`~pygeoinf.hilbert_space.HilbertSpace`.  ``is_attached == True``.

- **Domain identity for arithmetic.**  Addition and subtraction between two
  ``Function3D`` objects require the same domain *object* (``is`` check).
  This is conservative but safe for Phase 2.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_domain3d(obj) -> bool:
    """Duck-type check for Domain3D without a circular import."""
    # Domain3D has .dim == 3, .region, .contains, .integrate_volume
    return (
        hasattr(obj, "dim")
        and getattr(obj, "dim", None) == 3
        and hasattr(obj, "region")
        and hasattr(obj, "integrate_volume")
        and hasattr(obj, "contains")
    )


def _extract_domain(space_or_domain) -> Any:
    """Return the Domain3D from either a domain or a space."""
    if _is_domain3d(space_or_domain):
        return space_or_domain
    # Space attached: expect .function_domain attribute
    if hasattr(space_or_domain, "function_domain"):
        return space_or_domain.function_domain
    raise AttributeError(
        f"{type(space_or_domain).__name__!r} "
        "has no 'function_domain' attribute "
        "and does not look like a Domain3D."
    )


# ---------------------------------------------------------------------------
# Function3D
# ---------------------------------------------------------------------------

class Function3D:
    """A primal scalar function on a 3D domain.

    Parameters
    ----------
    space_or_domain:
        Either a :class:`~pygeoinf3d.core.domain_base.Domain3D` (standalone
        mode) or a :class:`~pygeoinf.hilbert_space.HilbertSpace` whose
        ``function_domain`` is a ``Domain3D`` (space-attached mode).
    evaluate_callable:
        Mandatory callable ``f(x)`` where *x* has shape ``(3,)``
        (single point) or ``(n, 3)`` (batch).  For a single point the return
        value should be a Python scalar or shape-``()`` array; for a batch it
        should be shape ``(n,)``.
    name:
        Optional human-readable label.

    Examples
    --------
    Standalone construction::

        dom = SomeDomain3D(...)
        f = Function3D(dom, evaluate_callable=lambda x: float(x[0]))

    Space-attached construction via ``attach_to_space``::

        space = Lebesgue3D(dom)
        f_attached = f.attach_to_space(space)
        f_attached.is_attached  # True
    """

    def __init__(
        self,
        space_or_domain,
        *,
        evaluate_callable: Optional[Callable] = None,
        name: Optional[str] = None,
    ) -> None:
        if evaluate_callable is None:
            raise ValueError(
                "Function3D requires 'evaluate_callable'.  "
                "Coefficient-based construction is not supported in Phase 2."
            )

        if _is_domain3d(space_or_domain):
            self._domain: Any = space_or_domain
            self._space: Optional[Any] = None
        else:
            # Space mode: validate it exposes a compatible function_domain
            if not hasattr(space_or_domain, "function_domain"):
                raise TypeError(
                    f"{type(space_or_domain).__name__!r} "
                    "is neither a Domain3D "
                    "nor a space with a 'function_domain' attribute."
                )
            self._space = space_or_domain
            self._domain = space_or_domain.function_domain

        self._callable: Callable = evaluate_callable
        self.name: Optional[str] = name

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def domain(self):
        """The :class:`~pygeoinf3d.core.domain_base.Domain3D`
        this function lives on."""
        return self._domain

    @property
    def space(self):
        """The attached function space, or ``None`` if standalone."""
        return self._space

    @property
    def is_attached(self) -> bool:
        """``True`` if this function is attached to a function space."""
        return self._space is not None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def __call__(self, x: np.ndarray) -> Union[float, np.ndarray]:
        """Evaluate the function at one or more points.

        Parameters
        ----------
        x:
            Shape ``(3,)`` for a single point, or ``(n, 3)`` for a batch.

        Returns
        -------
        float or ndarray
            Scalar for a single point; shape ``(n,)`` array for a batch.
        """
        x = np.asarray(x, dtype=float)
        return self._callable(x)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def _check_compatible(self, other: "Function3D") -> None:
        if not isinstance(other, Function3D):
            raise TypeError(
                f"Arithmetic with Function3D requires another Function3D, "
                f"got {type(other).__name__!r}."
            )
        if other.domain is not self.domain:
            raise ValueError(
                "Arithmetic requires Function3D objects on the same domain "
                "(identity check failed). Use functions defined on the same "
                "Domain3D instance."
            )

    def __add__(self, other: "Function3D") -> "Function3D":
        self._check_compatible(other)
        dom = self._domain
        f, g = self._callable, other._callable
        return Function3D(dom, evaluate_callable=lambda x: f(x) + g(x))

    def __sub__(self, other: "Function3D") -> "Function3D":
        self._check_compatible(other)
        dom = self._domain
        f, g = self._callable, other._callable
        return Function3D(dom, evaluate_callable=lambda x: f(x) - g(x))

    def __neg__(self) -> "Function3D":
        dom = self._domain
        f = self._callable
        return Function3D(dom, evaluate_callable=lambda x: -f(x))

    def __mul__(self, scalar) -> "Function3D":
        if isinstance(scalar, Function3D):
            # Pointwise product — prefer .pointwise_mul for clarity
            return self.pointwise_mul(scalar)
        dom = self._domain
        f = self._callable
        a = float(scalar)
        return Function3D(dom, evaluate_callable=lambda x: a * f(x))

    def __rmul__(self, scalar) -> "Function3D":
        return self.__mul__(scalar)

    def __truediv__(self, scalar) -> "Function3D":
        return self.__mul__(1.0 / float(scalar))

    def pointwise_mul(self, other: "Function3D") -> "Function3D":
        """Pointwise product ``(f ⊙ g)(x) = f(x) · g(x)``.

        Used for integrand formation, e.g. computing ``∫ f·g`` numerically.

        Parameters
        ----------
        other:
            Another ``Function3D`` on the **same** domain.
        """
        self._check_compatible(other)
        dom = self._domain
        f, g = self._callable, other._callable
        return Function3D(dom, evaluate_callable=lambda x: f(x) * g(x))

    # ------------------------------------------------------------------
    # copy / attach / detach
    # ------------------------------------------------------------------

    def copy(self) -> "Function3D":
        """Return a new ``Function3D`` with the same callable and domain.

        The callable is shared (not deep-copied), which is safe because
        ``Function3D`` callables are treated as immutable.
        """
        new = Function3D(
            self._domain,
            evaluate_callable=self._callable,
            name=self.name,
        )
        if self._space is not None:
            # Re-attach to the same space
            new._space = self._space
        return new

    def attach_to_space(self, space) -> "Function3D":
        """Return a new ``Function3D`` attached to *space*.

        Parameters
        ----------
        space:
            A function space whose ``function_domain`` is the same domain
            as this function (identity check).

        Returns
        -------
        Function3D
            A copy of this function marked as belonging to *space*.

        Raises
        ------
        ValueError
            If *space* is associated with a different domain instance.
        TypeError
            If *space* has no ``function_domain`` attribute.
        """
        if not hasattr(space, "function_domain"):
            raise TypeError(
                f"{type(space).__name__!r} has no 'function_domain' attribute."
            )
        if space.function_domain is not self._domain:
            raise ValueError(
                "Cannot attach to a space whose function_domain "
                "is a different Domain3D instance from this "
                "function's domain."
            )
        new = Function3D(
            self._domain,
            evaluate_callable=self._callable,
            name=self.name,
        )
        new._space = space
        return new

    def detach(self) -> "Function3D":
        """Return a standalone copy of this function
        (``is_attached == False``).

        If the function is already standalone, returns a copy of itself.

        For callable-backed functions, detachment is always safe because the
        evaluation rule is self-contained.  Raising on detach would only be
        appropriate for basis-coefficient-only functions (Phase 3+).
        """
        return Function3D(
            self._domain,
            evaluate_callable=self._callable,
            name=self.name,
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        mode = "attached" if self.is_attached else "standalone"
        name_part = f", name={self.name!r}" if self.name else ""
        return f"Function3D({mode!r}{name_part})"
