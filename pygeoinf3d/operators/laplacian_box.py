"""Spectral box Laplacian for axis-aligned box domains.

Provides :class:`BoxLaplacian`, a
:class:`~pygeoinf.linear_operators.LinearOperator` implementing the
negative Laplacian $-\\Delta$ on a
:class:`~pygeoinf3d.geometries.box.domain.BoxDomain` via truncated
spectral projection through a
:class:`~pygeoinf3d.providers.tensor_product.BoxLaplacianSpectrumProvider`.

Mathematical definition
-----------------------
Given eigenfunction-eigenvalue pairs $(\\Phi_k, \\lambda_k)$ with
$-\\Delta \\Phi_k = \\lambda_k \\Phi_k$ and $\\|\\Phi_k\\|_{L^2} = 1$, the
spectral projection operator acts as

.. math::

    (-\\Delta_N f)(\\mathbf{x})
        = \\sum_{k=0}^{N-1} \\lambda_k
            \\langle f, \\Phi_k \\rangle_{L^2}\\, \\Phi_k(\\mathbf{x})

where the $L^2$ inner product is computed via the box's tensor-product
quadrature.  The result is a :class:`~pygeoinf3d.core.functions.Function3D`
whose pointwise values are the spectral approximation of $-\\Delta f$.

Self-adjointness
----------------
The operator is formally self-adjoint with respect to the exact $L^2$
inner product because the eigenfunctions are orthonormal and the
eigenvalues are real. In this implementation, projection coefficients
are computed with finite quadrature, so the numerical operator is
symmetric only up to quadrature error.

Usage
-----
::

    from pygeoinf3d.geometries.box.domain import BoxDomain
    from pygeoinf3d.providers.tensor_product import BoxLaplacianSpectrumProvider
    from pygeoinf3d.spaces.lebesgue import Lebesgue3D
    from pygeoinf3d.operators.laplacian_box import BoxLaplacian
    import numpy as np

    box = BoxDomain(np.zeros(3), np.ones(3))
    space = Lebesgue3D(box)
    provider = BoxLaplacianSpectrumProvider(
        box,
        truncation=(4, 4, 4),
        bc_type='dirichlet',
    )
    lap = BoxLaplacian(space, provider)

    phi0 = provider.get_eigenfunction(0)
    lam0 = provider.get_eigenvalue(0)
    result = lap(phi0)   # ≈ lam0 * phi0 pointwise

Quadrature
----------
Projection coefficients $c_k = \\langle f, \\Phi_k \\rangle_{L^2}$ are computed
using the box's
:meth:`~pygeoinf3d.geometries.box.domain.BoxDomain.integrate_volume`
with resolution ``quad_resolution`` (default 20 per axis). Higher
values give more accurate projections at the cost of speed.
"""

from __future__ import annotations

from pygeoinf3d.operators.generalized_box_laplacian import GeneralizedBoxLaplacian
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.providers.tensor_product import BoxLaplacianSpectrumProvider


class BoxLaplacian(GeneralizedBoxLaplacian):
    """Spectral negative Laplacian on a box domain.

    Applies the truncated spectral projection

    .. math::

        (-\\Delta_N f)(\\mathbf{x})
        = \\sum_{k=0}^{N-1} \\lambda_k
          \\langle f, \\Phi_k \\rangle_{L^2}\\, \\Phi_k(\\mathbf{x})

    via the eigenfunctions provided by a
    :class:`~pygeoinf3d.providers.tensor_product.BoxLaplacianSpectrumProvider`.

    Parameters
    ----------
    space:
        The :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` on the box domain.
        Both the operator's domain and codomain are this space.
    provider:
        A :class:`~pygeoinf3d.providers.tensor_product.BoxLaplacianSpectrumProvider`
        whose box domain matches ``space.function_domain``.
    quad_resolution:
        Resolution passed to
        :meth:`~pygeoinf3d.geometries.box.domain.BoxDomain.structured_mesh`
        when computing the $L^2$ projections. Higher values give more
        accurate coefficients. Default is ``20``.

    Raises
    ------
    ValueError
        If ``provider``'s box domain differs from ``space.function_domain``.

    Examples
    --------
    ::

        lap = BoxLaplacian(space, provider, quad_resolution=30)
        phi0 = provider.get_eigenfunction(0)
        lam0 = provider.get_eigenvalue(0)
        Lphi0 = lap(phi0)
        # Lphi0(x) ≈ lam0 * phi0(x) pointwise
    """

    def __init__(
        self,
        space: Lebesgue3D,
        provider: BoxLaplacianSpectrumProvider,
        quad_resolution: int = 20,
    ) -> None:
        super().__init__(
            space, provider,
            power=1, alpha=1.0,
            quad_resolution=quad_resolution,
        )

    def __repr__(self) -> str:
        return (
            f"BoxLaplacian("
            f"bc_type={self._provider.bc_type!r}, "
            f"truncation={self._provider.truncation}, "
            f"quad_resolution={self._quad_resolution}"
            f")"
        )
