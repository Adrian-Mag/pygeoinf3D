"""Abstract base classes for coefficient representations.

A *coefficient representation* is a finite-dimensional analysis/synthesis
pair attached to a continuous function space.  It is deliberately separate
from the space itself so that:

1. The space retains pairing-first semantics.
2. Quadrature policy, Gram matrix assembly, and basis ordering remain
   confined to the representation/backend layer.
3. Multiple representations can exist for the same underlying space without
   polluting the space's identity.

Design follows the pygeoinf3D architecture plan
(docs/agent-docs/active-plans/pygeoinf3d-lebesgue-coefficient-architecture-plan.md).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CoefficientRepresentation3D(ABC):
    """Abstract finite-dimensional coefficient representation for a 3D space.

    A concrete subclass binds a continuous function space to a finite basis,
    providing analysis (``to_components``) and synthesis (``from_components``)
    maps without polluting the space itself.

    Subclass contract
    -----------------
    Must implement the four abstract attributes/methods:

    * ``space``  — the attached ``Lebesgue3D`` (or similar) space.
    * ``dim``    — number of basis coefficients (must be a positive ``int``).
    * ``to_components(f)``   — analysis map: ``Function3D → np.ndarray`` of
      shape ``(dim,)``.
    * ``from_components(c)`` — synthesis map: ``np.ndarray`` of shape
      ``(dim,)`` → ``Function3D``.

    The ``coefficient_space`` property is provided by the base class and
    always returns ``EuclideanSpace(dim)``.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def space(self):
        """The continuous function space that this representation is built on."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Coefficient-space dimension (number of basis elements)."""

    @abstractmethod
    def to_components(self, f) -> np.ndarray:
        """Map a primal function to its coefficient vector.

        Parameters
        ----------
        f:
            A ``Function3D`` in ``self.space``.

        Returns
        -------
        np.ndarray
            Coefficient vector of shape ``(self.dim,)``.
        """

    @abstractmethod
    def from_components(self, c: np.ndarray):
        """Synthesise a ``Function3D`` from a coefficient vector.

        Parameters
        ----------
        c:
            Coefficient vector of shape ``(self.dim,)``.

        Returns
        -------
        Function3D
            The synthesised primal function.
        """

    # ------------------------------------------------------------------
    # Derived / concrete helpers
    # ------------------------------------------------------------------

    @property
    def coefficient_space(self):
        """Return ``EuclideanSpace(dim)`` — the finite coefficient domain.

        The instance is not cached here; callers may cache if needed.
        """
        from pygeoinf import EuclideanSpace
        return EuclideanSpace(self.dim)
