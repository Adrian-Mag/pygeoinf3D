"""Spectral box Laplacian for axis-aligned box domains.

Provides :class:`BoxLaplacian`, a
:class:`~pygeoinf.linear_operators.LinearOperator` implementing the
negative Laplacian $-\\Delta$ on a
:class:`~pygeoinf3d.geometries.box.domain.BoxDomain` via truncated
spectral projection through a
:class:`~pygeoinf3d.providers.tensor_product.TensorProductProvider`.

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
    from pygeoinf3d.providers.tensor_product import TensorProductProvider
    from pygeoinf3d.spaces.lebesgue import Lebesgue3D
    from pygeoinf3d.operators.laplacian_box import BoxLaplacian
    import numpy as np

    box = BoxDomain(np.zeros(3), np.ones(3))
    space = Lebesgue3D(box)
    provider = TensorProductProvider(
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

from typing import Union

import numpy as np

from pygeoinf.linear_operators import LinearOperator

from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.providers.tensor_product import TensorProductProvider


class BoxLaplacian(LinearOperator):
    """Spectral negative Laplacian on a box domain.

    Applies the truncated spectral projection

    .. math::

        (-\\Delta_N f)(\\mathbf{x})
        = \\sum_{k=0}^{N-1} \\lambda_k
          \\langle f, \\Phi_k \\rangle_{L^2}\\, \\Phi_k(\\mathbf{x})

    via the eigenfunctions provided by a
    :class:`~pygeoinf3d.providers.tensor_product.TensorProductProvider`.

    Parameters
    ----------
    space:
        The :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` on the box domain.
        Both the operator's domain and codomain are this space.
    provider:
        A :class:`~pygeoinf3d.providers.tensor_product.TensorProductProvider`
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
        provider: TensorProductProvider,
        quad_resolution: int = 20,
    ) -> None:
        if space.function_domain is not provider._box:
            raise ValueError(
                "The provider's box domain must be the same object as "
                "space.function_domain."
            )

        self._space = space
        self._provider = provider
        self._quad_resolution = quad_resolution

        # Pre-fetch modes for efficiency (avoids repeated list construction)
        self._modes: list = [
            (provider.get_eigenvalue(i), provider.get_eigenfunction(i))
            for i in range(provider.dim)
        ]

        # Pre-compute quadrature mesh and weights for projection integrals
        box = provider._box
        self._pts = box.structured_mesh(quad_resolution)
        self._weights = box.volume_weights(quad_resolution)

        super().__init__(
            space,
            space,
            self._apply,
            adjoint_mapping=self._apply,
        )

    # ------------------------------------------------------------------
    # Internal application
    # ------------------------------------------------------------------

    def _apply(self, f: Function3D) -> Function3D:
        """Apply the spectral negative Laplacian to *f*.

        Computes projection coefficients $c_k = \\langle f, \\Phi_k \\rangle$
        using pre-computed quadrature points and weights, then builds the
        output function as a weighted sum.

        Args:
            f: A :class:`~pygeoinf3d.core.functions.Function3D` to apply the
               operator to.

        Returns:
            :class:`~pygeoinf3d.core.functions.Function3D` approximating
            $(-\\Delta_N f)$.
        """
        # Evaluate f at all quadrature points once
        f_vals = np.asarray(f(self._pts), dtype=float)

        # Compute spectral coefficients c_k = <f, phi_k> = ∫ f * phi_k dV
        coeffs = []
        for lam_k, phi_k in self._modes:
            phi_vals = np.asarray(phi_k(self._pts), dtype=float)
            c_k = float(np.dot(f_vals * phi_vals, self._weights))
            coeffs.append(c_k)

        # Build the output function f_out(x) = sum_k lambda_k * c_k * phi_k(x)
        # We use a closure that captures the modes list
        modes_snapshot = list(zip(
            [lam for lam, _ in self._modes],
            [phi for _, phi in self._modes],
            coeffs,
        ))

        def evaluate(x: np.ndarray) -> Union[float, np.ndarray]:
            x = np.asarray(x, dtype=float)
            is_batch = x.ndim == 2
            if is_batch:
                total = np.zeros(len(x))
            else:
                total = 0.0
            for lam_k, phi_k, c_k in modes_snapshot:
                total = total + lam_k * c_k * phi_k(x)
            return total

        return Function3D(
            self._space.function_domain,
            evaluate_callable=evaluate,
        )

    def __repr__(self) -> str:
        return (
            f"BoxLaplacian("
            f"bc_type={self._provider.bc_type!r}, "
            f"truncation={self._provider.truncation}, "
            f"quad_resolution={self._quad_resolution}"
            f")"
        )
