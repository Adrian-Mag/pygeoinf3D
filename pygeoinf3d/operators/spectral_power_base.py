"""Non-box-specific spectral-power base class for 3D operators.

Provides :class:`SpectralPowerBase3D`, a reusable abstract base for operators
whose spectrum is derived from a set of pre-computed operator eigenvalues and
eigenfunctions.  This class is *not* tied to tensor-product box geometry or
any particular quadrature scheme.  Concrete subclasses (e.g.
:class:`~pygeoinf3d.operators.generalized_box_laplacian.GeneralizedBoxLaplacian`)
implement :meth:`_apply` using domain-specific integration.

Eigenvalue transform convention
--------------------------------
The canonical rule for mapping Laplacian eigenvalues $\\lambda_k$ to operator
eigenvalues $\\mu_k$ for $(-\\alpha\\Delta)^s$ is encapsulated in the static
helper :meth:`SpectralPowerBase3D.compute_power_eigenvalues`:

.. math::

    \\mu_k = \\begin{cases}
        1 & s = 0 \\\\
        0 & s < 0,\\; |\\lambda_k| < \\varepsilon \\\\
        (\\alpha \\lambda_k)^s & \\text{otherwise}
    \\end{cases}

This ensures the operator is positive-semidefinite for $\\alpha > 0$ and
handles the Neumann zero-mode via pseudoinverse suppression when $s < 0$.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pygeoinf3d.operators.base import SpectralOperator3D
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.core.functions import Function3D


class SpectralPowerBase3D(SpectralOperator3D, ABC):
    """Abstract base for spectral operators with pre-computed eigenvalue lists.

    Stores a list of operator eigenvalues $\\mu_k$ and eigenfunctions
    $\\Phi_k$, and satisfies the
    :class:`~pygeoinf3d.operators.base.SpectralOperator3D` interface.
    Concrete subclasses implement :meth:`_apply` using domain-specific
    quadrature.

    Parameters
    ----------
    space:
        The :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` that acts as
        both domain and codomain of the operator.
    op_eigenvalues:
        Pre-computed operator eigenvalues $\\mu_k$, one per mode.
    eigenfunctions:
        Pre-fetched eigenfunctions $\\Phi_k$, one per mode.

    Notes
    -----
    The :meth:`compute_power_eigenvalues` static helper implements the
    canonical convention for mapping Laplacian eigenvalues $\\lambda_k$ to
    operator eigenvalues for $(-\\alpha\\Delta)^s$.
    """

    def __init__(
        self,
        space: Lebesgue3D,
        op_eigenvalues: list[float],
        eigenfunctions: list[Function3D],
    ) -> None:
        self._op_eigenvalues: list[float] = list(op_eigenvalues)
        self._eigenfunctions: list[Function3D] = list(eigenfunctions)
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
        """Return the operator eigenvalue $\\mu_k$ for mode *index*.

        Parameters
        ----------
        index : int
            Zero-based mode index in :math:`[0, \\text{n\\_modes})`.

        Returns
        -------
        float
            The pre-computed operator eigenvalue for mode *index*.
        """
        return self._op_eigenvalues[index]

    def get_eigenfunction(self, index: int) -> Function3D:
        """Return the $L^2$-normalised eigenfunction $\\Phi_k$ for mode *index*.

        Parameters
        ----------
        index : int
            Zero-based mode index in :math:`[0, \\text{n\\_modes})`.

        Returns
        -------
        :class:`~pygeoinf3d.core.functions.Function3D`
            The eigenfunction $\\Phi_k$.
        """
        return self._eigenfunctions[index]

    @property
    def n_modes(self) -> int:
        """Number of modes in the truncated spectral expansion."""
        return len(self._op_eigenvalues)

    # ------------------------------------------------------------------
    # Abstract apply — subclasses provide domain-specific quadrature
    # ------------------------------------------------------------------

    @abstractmethod
    def _apply(self, f: Function3D) -> Function3D:
        """Apply the spectral operator to *f*.

        Implementations should compute $L^2$ projection coefficients
        $c_k = \\langle f, \\Phi_k \\rangle_{L^2}$ with domain-appropriate
        quadrature, then synthesise the output as

        .. math::

            (Af)(\\mathbf{x}) = \\sum_{k=0}^{N-1} \\mu_k\\, c_k\\, \\Phi_k(\\mathbf{x}).

        Parameters
        ----------
        f : :class:`~pygeoinf3d.core.functions.Function3D`
            The input function.

        Returns
        -------
        :class:`~pygeoinf3d.core.functions.Function3D`
            The spectral operator applied to *f*.
        """

    # ------------------------------------------------------------------
    # Static helper for eigenvalue transformation convention
    # ------------------------------------------------------------------

    @staticmethod
    def compute_power_eigenvalues(
        laplacian_eigs: list[float],
        power: int,
        alpha: float = 1.0,
        eigenvalue_tol: float = 1e-10,
    ) -> list[float]:
        """Transform Laplacian eigenvalues to operator eigenvalues $\\mu_k$.

        Implements the canonical convention for integer-power spectral
        operators $(-\\alpha \\Delta)^s$:

        .. math::

            \\mu_k = \\begin{cases}
                1 & s = 0 \\\\
                0 & s < 0,\\; |\\lambda_k| < \\varepsilon \\\\
                (\\alpha \\lambda_k)^s & \\text{otherwise}
            \\end{cases}

        Parameters
        ----------
        laplacian_eigs:
            Laplacian eigenvalues $\\lambda_k$ (raw values, not negated).
        power:
            Integer power $s$.
        alpha:
            Positive scale factor.  Default ``1.0``.
        eigenvalue_tol:
            Near-zero threshold for pseudoinverse suppression when
            ``power < 0``.  Default ``1e-10``.

        Returns
        -------
        list[float]
            Operator eigenvalues $\\mu_k$, one per input eigenvalue.

        Examples
        --------
        ::

            # power=2, alpha=1: mu_k = lambda_k^2
            mus = SpectralPowerBase3D.compute_power_eigenvalues(
                [1.0, 4.0, 9.0], power=2
            )
            # -> [1.0, 16.0, 81.0]

            # power=-1, Neumann: zero mode suppressed
            mus = SpectralPowerBase3D.compute_power_eigenvalues(
                [0.0, 5.0, 10.0], power=-1
            )
            # -> [0.0, 0.2, 0.1]
        """
        power = int(power)
        result: list[float] = []
        for lam_k in laplacian_eigs:
            lam_k = float(lam_k)
            if power == 0:
                op_ev = 1.0
            elif power < 0 and abs(lam_k) < eigenvalue_tol:
                op_ev = 0.0
            else:
                op_ev = float((alpha * lam_k) ** power)
            result.append(op_ev)
        return result
