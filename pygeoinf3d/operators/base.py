"""Abstract base class for spectrally-implemented operators on 3D box domains.

Provides :class:`SpectralOperator3D`, an ABC that extends
:class:`~pygeoinf.linear_operators.LinearOperator` with the spectral
interface required by :class:`~pygeoinf3d.sampling.kl_sampler.KLSampler3D`
and any other consumer that needs direct access to eigenvalue/eigenfunction
pairs.

All operators implemented via a truncated spectral projection should
inherit from this class so their spectral structure is visible to generic
utilities.
"""

from __future__ import annotations

from abc import abstractmethod

from pygeoinf.linear_operators import LinearOperator

from pygeoinf3d.core.functions import Function3D


class SpectralOperator3D(LinearOperator):
    """Abstract base for spectrally-implemented 3D operators.

    A spectral operator is a :class:`~pygeoinf.linear_operators.LinearOperator`
    that can be diagonalised in terms of an orthonormal eigenfunction basis
    $(\\Phi_k,\\lambda_k)$ of the underlying function space:

    .. math::

        (Af)(\\mathbf{x}) = \\sum_{k=0}^{N-1} \\mu_k \\,
            \\langle f, \\Phi_k \\rangle_{L^2} \\, \\Phi_k(\\mathbf{x})

    where :math:`\\mu_k` is operator-specific (e.g. :math:`\\lambda_k` for
    :class:`~pygeoinf3d.operators.laplacian_box.BoxLaplacian`,
    :math:`1/\\lambda_k` for
    :class:`~pygeoinf3d.operators.inverse_laplacian_box.InverseBoxLaplacian`).

    Subclasses must implement:

    - :meth:`get_eigenvalue` — the operator's eigenvalue for mode *k*
    - :meth:`get_eigenfunction` — the corresponding eigenfunction
    - :attr:`n_modes` — the truncation dimension *N*

    In addition, as a :class:`~pygeoinf.linear_operators.LinearOperator`, they
    must supply a ``mapping`` callable (the ``_apply`` pattern) to the
    :class:`~pygeoinf.linear_operators.LinearOperator` constructor.
    """

    @abstractmethod
    def get_eigenvalue(self, index: int) -> float:
        """Return the operator eigenvalue for mode *index*.

        For :class:`~pygeoinf3d.operators.laplacian_box.BoxLaplacian` this is
        the raw Laplacian eigenvalue :math:`\\lambda_k`; for
        :class:`~pygeoinf3d.operators.inverse_laplacian_box.InverseBoxLaplacian`
        it is the (possibly modified) inverted eigenvalue
        :math:`\\tilde{\\lambda}_k^{-1}`.

        Parameters
        ----------
        index : int
            Zero-based mode index in :math:`[0, \\text{n\\_modes})`.

        Returns
        -------
        float
            The operator eigenvalue for mode *index*.
        """

    @abstractmethod
    def get_eigenfunction(self, index: int) -> Function3D:
        """Return the eigenfunction for mode *index*.

        Parameters
        ----------
        index : int
            Zero-based mode index in :math:`[0, \\text{n\\_modes})`.

        Returns
        -------
        :class:`~pygeoinf3d.core.functions.Function3D`
            The :math:`L^2`-normalised eigenfunction :math:`\\Phi_k`.
        """

    @property
    @abstractmethod
    def n_modes(self) -> int:
        """Number of modes in the truncated spectral expansion."""
