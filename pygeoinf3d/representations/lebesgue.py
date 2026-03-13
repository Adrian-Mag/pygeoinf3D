"""Lebesgue-specific coefficient representation.

:class:`LebesgueRepresentation3D` binds a
:class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` space to a basis provider
and a numerical backend, implementing the
:class:`~pygeoinf3d.representations.base.CoefficientRepresentation3D` contract.

Responsibilities
----------------
* Validates space/provider domain compatibility at construction time.
* Exposes ``dim`` and ``coefficient_space`` from the provider.
* Delegates ``to_components`` and ``from_components`` to the backend.

This module intentionally contains **no numerical algorithms** — those belong
in backend objects introduced in Phase 2 of the architecture plan.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pygeoinf3d.representations.base import CoefficientRepresentation3D


class LebesgueRepresentation3D(CoefficientRepresentation3D):
    """Concrete coefficient representation for
    :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D`.

    Binds a ``Lebesgue3D`` space to a basis provider and a numerical backend.
    The provider enumerates basis functions and metadata; the backend performs
    the actual numerical analysis and synthesis.

    Parameters
    ----------
    space:
        A :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` instance.
    provider:
        Any object exposing a ``.dim`` attribute and, optionally, a ``.domain``
        attribute that must match ``space.function_domain``.
    backend:
        Any object implementing ``to_components(space, provider, f)`` and
        ``from_components(space, provider, c)``.

    Raises
    ------
    TypeError
        If *space* is not a ``Lebesgue3D`` instance.
    ValueError
        If *provider* has a ``.domain`` attribute that does not refer to the
        same object as ``space.function_domain``.
    """

    def __init__(self, space: Any, provider: Any, backend: Any) -> None:
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D

        if not isinstance(space, Lebesgue3D):
            raise TypeError(
                f"LebesgueRepresentation3D requires a Lebesgue3D space, "
                f"got {type(space).__name__!r}."
            )

        # Domain compatibility: if the provider exposes .domain, it must match.
        provider_domain = None
        for attr in ("domain", "function_domain"):
            d = getattr(provider, attr, None)
            if d is not None:
                provider_domain = d
                break

        if (
            provider_domain is not None
            and provider_domain is not space.function_domain
        ):
            raise ValueError(
                "LebesgueRepresentation3D: provider's domain does not match "
                "space.function_domain (identity check).  "
                "Ensure provider and space are built on the same Domain3D "
                "instance."
            )

        self._space = space
        self._provider = provider
        self._backend = backend

    # ------------------------------------------------------------------
    # CoefficientRepresentation3D abstract interface
    # ------------------------------------------------------------------

    @property
    def space(self):
        """The attached :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D`."""
        return self._space

    @property
    def dim(self) -> int:
        """Coefficient dimension, delegated from the provider."""
        return int(self._provider.dim)

    def to_components(self, f) -> np.ndarray:
        """Delegate analysis to the backend.

        Parameters
        ----------
        f:
            A ``Function3D`` in ``self.space``.

        Returns
        -------
        np.ndarray
            Coefficient vector of shape ``(self.dim,)``.
        """
        return self._backend.to_components(self._space, self._provider, f)

    def from_components(self, c: np.ndarray):
        """Delegate synthesis to the backend.

        Parameters
        ----------
        c:
            Coefficient vector of shape ``(self.dim,)``.

        Returns
        -------
        Function3D
            The synthesised primal function.
        """
        return self._backend.from_components(self._space, self._provider, c)

    # ------------------------------------------------------------------
    # Provider / backend accessors (advanced API)
    # ------------------------------------------------------------------

    @property
    def provider(self):
        """The basis provider bound to this representation."""
        return self._provider

    @property
    def backend(self):
        """The numerical backend bound to this representation."""
        return self._backend

    # ------------------------------------------------------------------
    # Phase 2 — coefficient metric and moment accessors
    # ------------------------------------------------------------------

    def coefficient_metric(self):
        r"""Return the induced Hilbert structure on the coefficient carrier.

        Delegates to ``backend.coefficient_metric(space, provider)``.  The
        algebraic carrier is always ``EuclideanSpace(dim)`` (accessible via
        :attr:`coefficient_space`); this method returns the *metric* that
        encodes the physically correct inner product on coefficient vectors —
        in general a Gram-weighted inner product rather than the standard
        Euclidean one.

        Returns
        -------
        MassWeightedHilbertSpace
            Hilbert space on :math:`\\mathbb{R}^{\\text{dim}}` with mass
            operator determined by the backend strategy.

        Raises
        ------
        NotImplementedError
            If the attached backend does not implement
            ``coefficient_metric(space, provider)``.
        """
        if not hasattr(self._backend, "coefficient_metric"):
            raise NotImplementedError(
                f"Backend {type(self._backend).__name__!r} does not implement "
                "coefficient_metric(space, provider).  Use "
                "OrthonormalProjectionBackend or GramProjectionBackend."
            )
        return self._backend.coefficient_metric(self._space, self._provider)

    def gram_matrix(self) -> "np.ndarray":
        r"""Return or compute the Gram matrix
        :math:`G_{ij} = \\langle \\phi_i, \\phi_j \\rangle`.

        Delegates to ``backend.gram_matrix(space, provider)``.

        Returns
        -------
        G : np.ndarray, shape ``(dim, dim)``
        """
        if not hasattr(self._backend, "gram_matrix"):
            raise NotImplementedError(
                f"Backend {type(self._backend).__name__!r} does not implement "
                "gram_matrix(space, provider)."
            )
        return self._backend.gram_matrix(self._space, self._provider)

    def moments(self, f) -> "np.ndarray":
        r"""Compute raw moments :math:`b_i = \\langle \\phi_i, f \\rangle`.

        Raw moments are NOT equal to the true basis coefficients unless the
        basis family is orthonormal.  Use :meth:`to_components` for true
        coefficients.

        Parameters
        ----------
        f : Function3D

        Returns
        -------
        b : np.ndarray, shape ``(dim,)``
        """
        if not hasattr(self._backend, "moments"):
            raise NotImplementedError(
                f"Backend {type(self._backend).__name__!r} does not implement "
                "moments(space, provider, f)."
            )
        return self._backend.moments(self._space, self._provider, f)

    def __repr__(self) -> str:
        return (
            f"LebesgueRepresentation3D("
            f"space={self._space!r}, "
            f"dim={self.dim}, "
            f"provider={type(self._provider).__name__!r}, "
            f"backend={type(self._backend).__name__!r})"
        )
