"""Concrete Lebesgue coefficient backends.

Implements two backends for the analysis/synthesis workflow:

* :class:`OrthonormalProjectionBackend` — assumes the provider family is
  orthonormal w.r.t. the space inner product; raw moments are coefficients.
* :class:`GramProjectionBackend` — general case: assembles the Gram matrix
  and solves :math:`G c = b` for the true basis coefficients.

Both backends implement the
:class:`~pygeoinf3d.representations.base.LebesgueCoefficientBackend` abstract
interface and expose a ``coefficient_metric()`` method that returns a
:class:`~pygeoinf.hilbert_space.MassWeightedHilbertSpace` encoding the induced
Hilbert structure on the algebraic coefficient carrier
``EuclideanSpace(n)``.

Design note on the coefficient carrier
---------------------------------------
The algebraic carrier of coefficient vectors is always
``EuclideanSpace(n)`` (plain :math:`\\mathbb{R}^n`).  The *induced* Hilbert
structure on that carrier is in general different from the standard Euclidean
inner product: it is determined by the Gram or mass operator associated with
the chosen basis family. ``coefficient_metric()`` makes this structure explicit
rather than hiding it behind an implicit Euclidean assumption.

Provider protocol
-----------------
Providers passed to backend methods must expose:

* ``.dim`` — ``int`` number of basis functions.
* ``.get_basis_function(i)`` — returns the ``i``-th basis function as a
  ``Function3D`` on the same domain as the space.
"""

from __future__ import annotations

import numpy as np

from pygeoinf3d.representations.base import LebesgueCoefficientBackend


# ---------------------------------------------------------------------------
# OrthonormalProjectionBackend
# ---------------------------------------------------------------------------


class OrthonormalProjectionBackend(LebesgueCoefficientBackend):
    r"""Backend for bases that are orthonormal w.r.t. the space inner product.

    Assumes :math:`\\langle \\phi_i, \\phi_j \\rangle = \\delta_{ij}`, so the
    Gram matrix is the identity and raw moments equal the true coefficients.
    This is the correct specialisation for tensor-product box eigenfunctions
    when the quadrature resolution is chosen consistently (Phase 3).

    The induced coefficient metric is the standard Euclidean inner product,
    exposing this explicitly via :meth:`coefficient_metric`.
    """

    def moments(self, space, provider, f) -> np.ndarray:
        r"""Compute raw moments :math:`b_i = \\langle \\phi_i, f \\rangle`.

        Parameters
        ----------
        space:
            A ``Lebesgue3D`` instance.
        provider:
            Basis provider exposing ``dim`` and ``get_basis_function(i)``.
        f:
            A ``Function3D`` in ``space``.

        Returns
        -------
        b : np.ndarray, shape ``(n,)``
        """
        n = provider.dim
        b = np.empty(n, dtype=float)
        for i in range(n):
            phi = provider.get_basis_function(i)
            b[i] = space.inner_product(phi, f)
        return b

    def gram_matrix(self, space, provider) -> np.ndarray:
        """Return the :math:`n \\times n` identity matrix.

        For an orthonormal family the Gram matrix is exactly :math:`I_n`.
        No integration is performed.
        """
        return np.eye(provider.dim, dtype=float)

    def coefficient_metric(self, space, provider):
        r"""Return the induced coefficient-space Hilbert structure.

        For an orthonormal basis the induced metric is the standard Euclidean
        inner product, represented as a
        :class:`~pygeoinf.hilbert_space.MassWeightedHilbertSpace` with
        identity mass operator so the API is uniform across backends.

        Returns
        -------
        MassWeightedHilbertSpace
            ``underlying_space = EuclideanSpace(n)``, mass = :math:`I_n`.
        """
        from pygeoinf import EuclideanSpace, MassWeightedHilbertSpace
        from pygeoinf.linear_operators import LinearOperator

        n = provider.dim
        euc = EuclideanSpace(n)
        identity = LinearOperator(
            euc, euc,
            lambda c: c.copy(),
            adjoint_mapping=lambda c: c.copy(),
        )
        return MassWeightedHilbertSpace(euc, identity, identity)

    def solve_coefficients(self, G: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return *b* unchanged (identity solve for orthonormal families)."""
        return b.copy()

    def to_components(self, space, provider, f) -> np.ndarray:
        """Return raw moments equal to coefficients for orthonormal bases."""
        return self.moments(space, provider, f)

    def from_components(self, space, provider, c: np.ndarray):
        r"""Synthesise :math:`f = \\sum_i c_i \\phi_i`.

        Parameters
        ----------
        c : np.ndarray, shape ``(n,)``
            Coefficient vector.

        Returns
        -------
        Function3D
            The linear combination of basis functions.
        """
        n = provider.dim
        if n == 0:
            return space.zero
        coeff = np.asarray(c, dtype=float)
        # Build linear combination via Function3D arithmetic.
        result = coeff[0] * provider.get_basis_function(0)
        for i in range(1, n):
            result = result + coeff[i] * provider.get_basis_function(i)
        return result


# ---------------------------------------------------------------------------
# GramProjectionBackend
# ---------------------------------------------------------------------------


class GramProjectionBackend(LebesgueCoefficientBackend):
    r"""Backend for general (non-orthogonal) basis families.

    Assembles the Gram matrix
    :math:`G_{ij} = \\langle \\phi_i, \\phi_j \\rangle`
    and solves :math:`G c = b` to recover the true basis coefficients from raw
    moments.  Gram matrices are cached per ``(id(space), id(provider))`` pair
    to avoid recomputation across multiple calls.

    The induced coefficient metric is
    :math:`\\langle c_1, c_2 \\rangle_G = c_1^\\top G c_2`,
    exposed explicitly via :meth:`coefficient_metric` as a
    :class:`~pygeoinf.hilbert_space.MassWeightedHilbertSpace`.
    """

    def __init__(self) -> None:
        # Cache: (id(space), id(provider)) → np.ndarray Gram matrix
        self._gram_cache: dict = {}

    def moments(self, space, provider, f) -> np.ndarray:
        r"""Compute raw moments :math:`b_i = \\langle \\phi_i, f \\rangle`.

        Parameters
        ----------
        space:
            A ``Lebesgue3D`` instance.
        provider:
            Basis provider exposing ``dim`` and ``get_basis_function(i)``.
        f:
            A ``Function3D`` in ``space``.

        Returns
        -------
        b : np.ndarray, shape ``(n,)``
        """
        n = provider.dim
        b = np.empty(n, dtype=float)
        for i in range(n):
            phi = provider.get_basis_function(i)
            b[i] = space.inner_product(phi, f)
        return b

    def gram_matrix(self, space, provider) -> np.ndarray:
        r"""Assemble (or return cached) Gram matrix.

        .. math::
            G_{ij} = \\langle \\phi_i, \\phi_j \\rangle

        The result is cached per ``(id(space), id(provider))`` pair.

        Returns
        -------
        G : np.ndarray, shape ``(n, n)``
        """
        key = (id(space), id(provider))
        if key not in self._gram_cache:
            from pygeoinf3d.representations.gram import assemble_gram
            self._gram_cache[key] = assemble_gram(space, provider)
        return self._gram_cache[key].copy()

    def coefficient_metric(self, space, provider):
        r"""Return the Gram-induced coefficient-space Hilbert structure.

        The induced inner product on the coefficient carrier is:

        .. math::
            \\langle c_1, c_2 \\rangle_G = c_1^\\top G\\, c_2

        This is encoded as a
        :class:`~pygeoinf.hilbert_space.MassWeightedHilbertSpace` with mass
        operator :math:`G` and inverse :math:`G^{-1}`.

        Returns
        -------
        MassWeightedHilbertSpace
            ``underlying_space = EuclideanSpace(n)``, mass = :math:`G`.
        """
        from pygeoinf import EuclideanSpace, MassWeightedHilbertSpace
        from pygeoinf3d.representations.gram import gram_operators

        G = self.gram_matrix(space, provider)
        mass_op, inv_mass_op = gram_operators(G)
        euc = EuclideanSpace(provider.dim)
        return MassWeightedHilbertSpace(euc, mass_op, inv_mass_op)

    def solve_coefficients(self, G: np.ndarray, b: np.ndarray) -> np.ndarray:
        r"""Solve :math:`G c = b` for the coefficient vector *c*.

        Uses Cholesky factorisation for symmetric positive-definite Gram
        matrices (the typical case) and falls back to LU for borderline inputs.

        Parameters
        ----------
        G : np.ndarray, shape ``(n, n)``
            Gram matrix (symmetric positive-definite).
        b : np.ndarray, shape ``(n,)``
            Raw-moments vector.

        Returns
        -------
        c : np.ndarray, shape ``(n,)``
        """
        import scipy.linalg
        try:
            return scipy.linalg.cho_solve(scipy.linalg.cho_factor(G), b)
        except np.linalg.LinAlgError:
            return scipy.linalg.lu_solve(scipy.linalg.lu_factor(G), b)

    def to_components(self, space, provider, f) -> np.ndarray:
        r"""Compute true basis coefficients satisfying :math:`G c = b`.

        Parameters
        ----------
        f : Function3D

        Returns
        -------
        c : np.ndarray, shape ``(n,)``
        """
        b = self.moments(space, provider, f)
        G = self.gram_matrix(space, provider)
        return self.solve_coefficients(G, b)

    def from_components(self, space, provider, c: np.ndarray):
        r"""Synthesise :math:`f = \\sum_i c_i \\phi_i`.

        Parameters
        ----------
        c : np.ndarray, shape ``(n,)``

        Returns
        -------
        Function3D
        """
        n = provider.dim
        if n == 0:
            return space.zero
        coeff = np.asarray(c, dtype=float)
        result = coeff[0] * provider.get_basis_function(0)
        for i in range(1, n):
            result = result + coeff[i] * provider.get_basis_function(i)
        return result
