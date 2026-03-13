"""Gram matrix assembly and operator construction utilities.

These helpers are used by coefficient backends to:

1. Assemble the symmetric Gram matrix
   :math:`G_{ij} = \\langle \\phi_i, \\phi_j \\rangle`
   using a space's ``inner_product``.
2. Wrap the Gram matrix and its inverse as
   :class:`~pygeoinf.linear_operators.LinearOperator` objects on
   ``EuclideanSpace(n)``, suitable for
   constructing a :class:`~pygeoinf.hilbert_space.MassWeightedHilbertSpace`.

Provider protocol
-----------------
The provider passed to :func:`assemble_gram` must expose:

* ``.dim`` — ``int`` number of basis functions.
* ``.get_basis_function(i)`` — returns the ``i``-th basis function as a
  ``Function3D`` living on the same domain as the space.
"""

from __future__ import annotations

import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Gram assembly
# ---------------------------------------------------------------------------


def assemble_gram(space, provider) -> np.ndarray:
    r"""Assemble the :math:`n \\times n` Gram matrix.

    .. math::
         G_{ij} = \\langle \\phi_i,\\, \\phi_j \\rangle
             = \\int_{\\Omega} \\phi_i(\\mathbf{x})
               \\phi_j(\\mathbf{x})\\, dV

    Parameters
    ----------
    space:
        A :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` instance whose
        ``inner_product`` evaluates the :math:`L^2(\\Omega)` pairing.
    provider:
        A basis provider exposing ``.dim`` and ``.get_basis_function(i)``.

    Returns
    -------
    G : np.ndarray, shape ``(n, n)``
        Symmetric Gram matrix.  Positive-definite when the basis is
        linearly independent.
    """
    n = provider.dim
    basis = [provider.get_basis_function(i) for i in range(n)]

    G = np.empty((n, n), dtype=float)
    for i in range(n):
        G[i, i] = space.inner_product(basis[i], basis[i])
        for j in range(i + 1, n):
            val = space.inner_product(basis[i], basis[j])
            G[i, j] = val
            G[j, i] = val

    return G


# ---------------------------------------------------------------------------
# Gram → LinearOperators
# ---------------------------------------------------------------------------


def gram_operators(G: np.ndarray):
    r"""Construct mass and inverse-mass
    :class:`~pygeoinf.linear_operators.LinearOperator` objects from a dense
    Gram matrix.

    The returned operators act on ``EuclideanSpace(n)``
    and can be passed directly to
    :class:`~pygeoinf.hilbert_space.MassWeightedHilbertSpace` to define the
    induced coefficient-space Hilbert structure:

    .. math::
        \\langle c_1, c_2 \\rangle_{\\text{metric}}
        = \\langle G\\, c_1,\\, c_2 \\rangle_{\\mathbb{R}^n}
        = c_1^{\\top} G\\, c_2

    Parameters
    ----------
    G : np.ndarray, shape ``(n, n)``
        A symmetric positive-definite Gram matrix.

    Returns
    -------
    mass_op : LinearOperator
        Maps :math:`c \\mapsto G\\, c` on ``EuclideanSpace(n)``.
    inverse_mass_op : LinearOperator
        Maps :math:`b \\mapsto G^{-1}\\, b` on ``EuclideanSpace(n)``.

    Warns
    -----
    RuntimeWarning
        Emitted if the condition number of *G* exceeds :math:`10^{12}`.
    """
    import scipy.linalg

    from pygeoinf import EuclideanSpace
    from pygeoinf.linear_operators import LinearOperator

    n = G.shape[0]
    cond = np.linalg.cond(G)
    if cond > 1e12:
        warnings.warn(
            f"Gram matrix is poorly conditioned (cond={cond:.2e}). "
            "Coefficient solve may be inaccurate.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Try Cholesky (SPD fast path); fall back to LU for borderline matrices.
    try:
        factor = scipy.linalg.cho_factor(G)

        def _solve(b: np.ndarray) -> np.ndarray:
            return scipy.linalg.cho_solve(factor, b)
    except np.linalg.LinAlgError:
        lu_piv = scipy.linalg.lu_factor(G)

        def _solve(b: np.ndarray) -> np.ndarray:
            return scipy.linalg.lu_solve(lu_piv, b)

    # Immutable copy so operators are independent of external mutations to G.
    G_frozen = G.copy()
    euc = EuclideanSpace(n)

    mass_op = LinearOperator(
        euc, euc,
        lambda c: G_frozen @ c,
        adjoint_mapping=lambda c: G_frozen @ c,
    )
    inverse_mass_op = LinearOperator(
        euc, euc,
        _solve,
        adjoint_mapping=_solve,
    )
    return mass_op, inverse_mass_op
