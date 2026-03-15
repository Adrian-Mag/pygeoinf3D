"""Integer-power generalisation of the box Laplacian: $(-\\alpha \\Delta)^s$.

Provides :class:`GeneralizedBoxLaplacian`, a
:class:`~pygeoinf3d.operators.base.SpectralOperator3D` representing the
spectral operator $(-\\alpha \\Delta)^s$ for integer $s$ (``power``) and
positive scale $\\alpha$ (``alpha``).

Mathematical definition
-----------------------
Given orthonormal eigenfunction-eigenvalue pairs $(\\Phi_k, \\lambda_k)$ with
$-\\Delta \\Phi_k = \\lambda_k \\Phi_k$, the generalised operator acts as

.. math::

    ((-\\alpha\\Delta)^s f)(\\mathbf{x})
        = \\sum_{k=0}^{N-1} (\\alpha\\lambda_k)^s
            \\langle f, \\Phi_k \\rangle_{L^2}\\, \\Phi_k(\\mathbf{x})

where $(\\alpha\\lambda_k)^s$ is defined as follows:

- ``s = 0``: operator eigenvalue is ``1`` for every mode (identity).
- ``s > 0``: operator eigenvalue is ``(alpha * lambda_k) ** s``.
- ``s < 0``: operator eigenvalue is ``(alpha * lambda_k) ** s``, but modes
  satisfying ``|lambda_k| < eigenvalue_tol`` are projected out (pseudoinverse
  / mean-zero convention).

Public convention
-----------------
Powers are taken of the *positive* operator $(-\\alpha\\Delta)$, not the
raw $\\Delta$.  This ensures the operator is positive-semidefinite for
$\\alpha > 0$ and all Neumann/Dirichlet eigenvalues $\\lambda_k \\ge 0$.

Special cases
-------------
- ``GeneralizedBoxLaplacian(space, provider, power=1, alpha=1.0)`` implements
  the same operator as :class:`~pygeoinf3d.operators.laplacian_box.BoxLaplacian`.
- ``GeneralizedBoxLaplacian(space, provider, power=-1, alpha=1.0)`` implements
  the same eigenvalue semantics as
  :class:`~pygeoinf3d.operators.inverse_laplacian_box.InverseBoxLaplacian`
  with ``null_space_handling='pseudoinverse'``.

Implementation status
---------------------
**Phase 1 stub**: eigenvalue and eigenfunction access are fully implemented.
The :meth:`_apply` spectral projection mapping raises
:exc:`NotImplementedError` and will be completed in Phase 2 of the
generalized-box-laplacian plan.
"""

from __future__ import annotations

import numpy as np

from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.operators.base import SpectralOperator3D
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.providers.tensor_product import TensorProductProvider


class GeneralizedBoxLaplacian(SpectralOperator3D):
    """Spectral operator $(-\\alpha \\Delta)^s$ on a box domain.

    Represents integer powers of the positive box Laplacian.  The operator
    eigenvalues are $(\\alpha \\lambda_k)^s$ where $\\lambda_k$ are the
    Laplacian eigenvalues provided by a
    :class:`~pygeoinf3d.providers.tensor_product.TensorProductProvider`.

    For negative powers, modes with ``|lambda_k| < eigenvalue_tol`` are
    suppressed to zero (pseudoinverse / mean-zero projection). This is the only
    zero-mode strategy supported in Phase 1; shifted variants are out of scope.

    Parameters
    ----------
    space:
        The :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` on the box domain.
        Both the operator's domain and codomain are this space.
    provider:
        A :class:`~pygeoinf3d.providers.tensor_product.TensorProductProvider`
        whose box domain matches ``space.function_domain``.
    power:
        Integer power $s$.  ``power=0`` gives the identity, ``power=1`` is
        the Laplacian, ``power=-1`` is the inverse Laplacian, etc.
    alpha:
        Positive scale factor for the operator $(-\\alpha \\Delta)$.
        Default ``1.0``.
    eigenvalue_tol:
        Threshold below which a Laplacian eigenvalue is treated as zero for
        pseudoinverse handling when ``power < 0``. Default ``1e-10``.

    Raises
    ------
    TypeError
        If ``power`` is not an integer (e.g. ``power=0.5``).
    ValueError
        If ``alpha <= 0``.
    ValueError
        If ``provider``'s box domain differs from ``space.function_domain``.

    Notes
    -----
    **Phase 1 stub**: the :meth:`_apply` mapping raises
    :exc:`NotImplementedError`. Full spectral projection is implemented in
    Phase 2 of ``pygeoinf3d-generalized-box-laplacian-plan.md``.

    Examples
    --------
    ::

        from pygeoinf3d.geometries.box.domain import BoxDomain
        from pygeoinf3d.providers.tensor_product import TensorProductProvider
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        from pygeoinf3d.operators.generalized_box_laplacian import GeneralizedBoxLaplacian
        import numpy as np

        box      = BoxDomain(np.zeros(3), np.ones(3))
        space    = Lebesgue3D(box)
        provider = TensorProductProvider(box, truncation=(4, 4, 4), bc_type='dirichlet')

        # Power=2 Laplacian: eigenvalues = lambda_k^2
        op = GeneralizedBoxLaplacian(space, provider, power=2)
        ev0 = op.get_eigenvalue(0)   # = lambda_0 ** 2

        # Identity operator
        ident = GeneralizedBoxLaplacian(space, provider, power=0)
        ev0 = ident.get_eigenvalue(0)  # = 1.0
    """

    def __init__(
        self,
        space: Lebesgue3D,
        provider: TensorProductProvider,
        *,
        power: int,
        alpha: float = 1.0,
        eigenvalue_tol: float = 1e-10,
    ) -> None:
        # --- Validate power: must be an integer type ---
        if not isinstance(power, (int, np.integer)):
            raise TypeError(
                f"power must be an integer; got {type(power).__name__!r} "
                f"with value {power!r}. Only integer powers are supported."
            )

        # --- Validate alpha ---
        if alpha <= 0:
            raise ValueError(
                f"alpha must be strictly positive; got alpha={alpha!r}. "
                "The operator convention is powers of the positive operator "
                "(-alpha * Delta), so alpha > 0 is required."
            )

        # --- Validate box identity ---
        if space.function_domain is not provider._box:
            raise ValueError(
                "The provider's box domain must be the same object as "
                "space.function_domain."
            )

        self._space = space
        self._provider = provider
        self._power = int(power)
        self._alpha = float(alpha)
        self._eigenvalue_tol = float(eigenvalue_tol)

        # --- Pre-compute operator eigenvalues ---
        # Convention: mu_k = (alpha * lambda_k)^power
        # Special cases:
        #   power == 0  : mu_k = 1.0 (identity, regardless of alpha or lambda_k)
        #   power < 0, |lambda_k| < tol : mu_k = 0.0 (pseudoinverse suppression)
        #   otherwise   : mu_k = (alpha * lambda_k) ** power
        self._operator_eigenvalues: list[float] = []
        for i in range(provider.dim):
            lam_k = float(provider.get_eigenvalue(i))
            if self._power == 0:
                op_ev = 1.0
            elif self._power < 0 and abs(lam_k) < self._eigenvalue_tol:
                op_ev = 0.0
            else:
                op_ev = float((self._alpha * lam_k) ** self._power)
            self._operator_eigenvalues.append(op_ev)

        # Pre-cache eigenfunctions to avoid repeated provider calls
        self._eigenfunctions: list[Function3D] = [
            provider.get_eigenfunction(i) for i in range(provider.dim)
        ]

        super().__init__(
            space,
            space,
            self._apply,
            adjoint_mapping=self._apply,
        )

    # ------------------------------------------------------------------
    # SpectralOperator3D interface
    # ------------------------------------------------------------------

    def get_eigenvalue(self, index: int) -> float:
        """Return the operator eigenvalue $(\\alpha \\lambda_k)^s$ for mode *index*.

        Parameters
        ----------
        index : int
            Zero-based mode index in :math:`[0, \\text{n\\_modes})`.

        Returns
        -------
        float
            $(\\alpha \\lambda_k)^s$, or ``1.0`` if ``power == 0``, or ``0.0``
            for suppressed near-zero modes when ``power < 0``.
        """
        return self._operator_eigenvalues[index]

    def get_eigenfunction(self, index: int) -> Function3D:
        """Return the $L^2$-normalised eigenfunction $\\Phi_k$ for mode *index*.

        Eigenfunctions are forwarded unchanged from the underlying provider.
        """
        return self._eigenfunctions[index]

    @property
    def n_modes(self) -> int:
        """Number of modes in the truncated spectral expansion."""
        return len(self._operator_eigenvalues)

    @property
    def power(self) -> int:
        """The integer power $s$ of the operator $(-\\alpha \\Delta)^s$."""
        return self._power

    @property
    def alpha(self) -> float:
        """The positive scale factor $\\alpha$ of the operator $(-\\alpha \\Delta)^s$."""
        return self._alpha

    @property
    def eigenvalue_tol(self) -> float:
        """Threshold for near-zero eigenvalue detection (pseudoinverse mode)."""
        return self._eigenvalue_tol

    # ------------------------------------------------------------------
    # Internal application (Phase 1 stub — raises NotImplementedError)
    # ------------------------------------------------------------------

    def _apply(self, f: Function3D) -> Function3D:
        """Apply the spectral operator to *f*.

        .. note::
            **Phase 1 stub**: raises :exc:`NotImplementedError`.
            Full spectral projection will be implemented in Phase 2 of the
            generalized-box-laplacian plan.

        Raises
        ------
        NotImplementedError
            Always in Phase 1.
        """
        raise NotImplementedError(
            "GeneralizedBoxLaplacian._apply is not yet implemented. "
            "See Phase 2 of pygeoinf3d-generalized-box-laplacian-plan.md."
        )

    def __repr__(self) -> str:
        return (
            f"GeneralizedBoxLaplacian("
            f"power={self._power}, "
            f"alpha={self._alpha!r}, "
            f"bc_type={self._provider.bc_type!r}, "
            f"truncation={self._provider.truncation}"
            f")"
        )
