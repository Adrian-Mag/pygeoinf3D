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
        """Return ``EuclideanSpace(dim)`` — the algebraic coefficient carrier.

        This is always plain :math:`\\mathbb{R}^n` (standard Euclidean space).
        The *induced Hilbert structure* on coefficients is in general different
        from the standard Euclidean inner product; use :meth:`coefficient_metric`
        to obtain it.

        The instance is not cached here; callers may cache if needed.
        """
        from pygeoinf import EuclideanSpace
        return EuclideanSpace(self.dim)

    def coefficient_metric(self):
        """Return the induced Hilbert structure on the coefficient carrier.

        The algebraic carrier is :meth:`coefficient_space` = ``EuclideanSpace(n)``.
        However, the physically correct inner product on coefficient vectors
        depends on the chosen basis family: for a non-orthonormal family it is
        the Gram-weighted inner product :math:`\\langle c_1, c_2 \\rangle_G = c_1^\\top G c_2`.

        Concrete implementations should override this method to return a
        :class:`~pygeoinf.hilbert_space.MassWeightedHilbertSpace` (or a
        compatible Hilbert space object) that encodes this structure.

        Raises
        ------
        NotImplementedError
            If the subclass or its backend does not implement coefficient metric
            computation.  Attach a backend that supports
            :py:meth:`~pygeoinf3d.representations.base.LebesgueCoefficientBackend.coefficient_metric`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement coefficient_metric(). "
            "Attach a backend that supports metric computation, such as "
            "OrthonormalProjectionBackend or GramProjectionBackend."
        )


# ---------------------------------------------------------------------------
# LebesgueCoefficientBackend — abstract numerical strategy
# ---------------------------------------------------------------------------


class LebesgueCoefficientBackend(ABC):
    r"""Abstract numerical backend for Lebesgue coefficient analysis and synthesis.

    A backend implements the full numerical workflow for a given basis strategy:
    moment assembly, Gram matrix construction, coefficient solves, and synthesis.
    It explicitly exposes the *induced coefficient-space metric* so that the
    algebraic carrier (:class:`~pygeoinf.hilbert_space.EuclideanSpace`\(n\)) can
    be distinguished from the Hilbert inner product structure that depends on the
    chosen basis family.

    Backends are stateless numerical strategies that accept ``(space, provider)``
    arguments on each call, allowing a single backend instance to be reused across
    multiple representations.  Concrete subclasses may add per-``(space, provider)``
    caching.

    Provider protocol
    -----------------
    All methods that receive a ``provider`` argument require:

    * ``provider.dim`` — number of basis functions.
    * ``provider.get_basis_function(i)`` — the ``i``-th basis function as a
      ``Function3D`` on the same domain as ``space``.
    """

    @abstractmethod
    def moments(self, space, provider, f) -> np.ndarray:
        r"""Compute raw moments :math:`b_i = \langle \phi_i, f \rangle`.

        Raw moments are NOT necessarily equal to the true basis coefficients
        unless the family is orthonormal.  Use :meth:`to_components` for
        coefficients.

        Returns
        -------
        b : np.ndarray, shape ``(n,)``
        """

    @abstractmethod
    def gram_matrix(self, space, provider) -> np.ndarray:
        r"""Return the :math:`n \times n` Gram matrix :math:`G_{ij} = \langle \phi_i, \phi_j \rangle`.

        Returns
        -------
        G : np.ndarray, shape ``(n, n)``
        """

    @abstractmethod
    def coefficient_metric(self, space, provider):
        r"""Return the induced coefficient-space Hilbert structure.

        The coefficient carrier is :class:`~pygeoinf.hilbert_space.EuclideanSpace`\(n\).
        For non-orthonormal families the physically correct inner product on
        coefficients is :math:`\langle c_1, c_2 \rangle_G = c_1^\top G c_2`, not
        the standard Euclidean one.

        Returns
        -------
        MassWeightedHilbertSpace
            Hilbert space on :math:`\mathbb{R}^n` whose inner product encodes
            the induced metric.
        """

    @abstractmethod
    def solve_coefficients(self, G: np.ndarray, b: np.ndarray) -> np.ndarray:
        r"""Solve :math:`G c = b` for the coefficient vector *c*.

        Parameters
        ----------
        G : np.ndarray, shape ``(n, n)``
        b : np.ndarray, shape ``(n,)``

        Returns
        -------
        c : np.ndarray, shape ``(n,)``
        """

    @abstractmethod
    def to_components(self, space, provider, f) -> np.ndarray:
        """Compute the true basis coefficient vector.

        This is the primary analysis map and must return correct coefficients
        even for non-orthonormal families (i.e., it should account for the
        Gram matrix).

        Returns
        -------
        c : np.ndarray, shape ``(n,)``
        """

    @abstractmethod
    def from_components(self, space, provider, c: np.ndarray):
        r"""Synthesise :math:`f = \sum_i c_i \phi_i`.

        Parameters
        ----------
        c : np.ndarray, shape ``(n,)``

        Returns
        -------
        Function3D
        """
