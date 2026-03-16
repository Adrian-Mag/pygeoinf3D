"""Bessel-Sobolev spectral operator on a box domain.

Provides :class:`BesselSobolevBox`, a
:class:`~pygeoinf3d.operators.spectral_power_base.SpectralPowerBase3D`
representing the spectral operator whose eigenvalues are

.. math::

    \\mu_k = (\\mathrm{shift} + \\lambda_k)^{s/2}

where $\\lambda_k$ are the Laplacian eigenvalues supplied by a
:class:`~pygeoinf3d.providers.tensor_product.TensorProductProvider`,
``shift`` is a strictly positive real number, and $s$ (``order``) is any
real float.

Mathematical definition
-----------------------
Given orthonormal eigenfunction-eigenvalue pairs $(\\Phi_k, \\lambda_k)$ with
$-\\Delta \\Phi_k = \\lambda_k \\Phi_k$, this operator acts as

.. math::

    (Bf)(\\mathbf{x})
        = \\sum_{k=0}^{N-1} (\\mathrm{shift} + \\lambda_k)^{s/2}
            \\langle f, \\Phi_k \\rangle_{L^2}\\, \\Phi_k(\\mathbf{x}).

Special cases
-------------
- ``order=0``: operator eigenvalue is ``1.0`` for every mode (identity).
- ``order=2``: eigenvalue is ``shift + lambda_k`` — this equals
  ``(shift - Delta)`` applied spectrally.
- ``order=-2``: eigenvalue is ``1 / (shift + lambda_k)`` — the Green's
  function / resolvent of ``(shift - Delta)``.
- ``order=s`` composed with ``order=-s`` gives the identity (inverse pair).

Role in Sobolev ladder
----------------------
This operator is intended as an operator-level building block for later
Sobolev space embeddings: applying ``BesselSobolevBox(order=s)`` maps
$L^2$ functions to $H^s$-equivalent functions under the Bessel-Sobolev
norm $\\|(\\mathrm{shift} - \\Delta)^{s/2} \\cdot\\|_{L^2}$.

Why ``shift > 0`` is required
------------------------------
Requiring a strictly positive shift guarantees
$\\mathrm{shift} + \\lambda_k > 0$ for **all** modes — including the Neumann
zero mode $\\lambda_0 = 0$.  This avoids any singularity for negative orders,
eliminating the need for a special-case pseudoinverse projection.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.operators.spectral_power_base import SpectralPowerBase3D
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.providers.tensor_product import TensorProductProvider


class BesselSobolevBox(SpectralPowerBase3D):
    """Spectral Bessel-Sobolev operator on a box.

    Represents the spectral operator with eigenvalues
    $\\mu_k = (\\mathrm{shift} + \\lambda_k)^{s/2}$, where $\\lambda_k$ are the
    Laplacian eigenvalues provided by a
    :class:`~pygeoinf3d.providers.tensor_product.TensorProductProvider` and
    $s$ (``order``) is any real number.

    Because ``shift`` is required to be strictly positive, the operator is
    always well-defined for any real ``order`` — including negative values —
    without requiring zero-mode suppression.

    Parameters
    ----------
    space:
        The :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` on the box domain.
        Both the operator's domain and codomain are this space.
    provider:
        A :class:`~pygeoinf3d.providers.tensor_product.TensorProductProvider`
        whose box domain matches ``space.function_domain``.
    shift:
        Strictly positive real number added to each eigenvalue before
        exponentiation.  Must satisfy ``shift > 0``.
    order:
        Real exponent $s$ such that the operator eigenvalue for mode $k$ is
        $(\\mathrm{shift} + \\lambda_k)^{s/2}$.  May be any real float,
        including non-integer and negative values.
    quad_resolution:
        Resolution passed to
        :meth:`~pygeoinf3d.geometries.box.domain.BoxDomain.structured_mesh`
        when computing the $L^2$ projections.  Default ``20``.

    Raises
    ------
    ValueError
        If ``shift <= 0``.
    ValueError
        If ``provider``'s box domain differs from ``space.function_domain``.

    Examples
    --------
    ::

        import numpy as np
        from pygeoinf3d.geometries.box.domain import BoxDomain
        from pygeoinf3d.providers.tensor_product import TensorProductProvider
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        from pygeoinf3d.operators.bessel_sobolev_box import BesselSobolevBox

        box      = BoxDomain(np.zeros(3), np.ones(3))
        space    = Lebesgue3D(box)
        provider = TensorProductProvider(
            box,
            truncation=(4, 4, 4),
            bc_type='neumann',
        )

        # Sobolev-1 smoothing: (1 + lambda_k)^(1/2)
        op = BesselSobolevBox(space, provider, shift=1.0, order=1.0)
        phi0 = provider.get_eigenfunction(0)
        result = op(phi0)   # ≈ mu_0 * phi0 pointwise

        # Inverse: (1 + lambda_k)^(-1/2).
        # No singularity even for the Neumann zero mode.
        op_inv = BesselSobolevBox(space, provider, shift=1.0, order=-1.0)
    """

    def __init__(
        self,
        space: Lebesgue3D,
        provider: TensorProductProvider,
        *,
        shift: float,
        order: float,
        quad_resolution: int = 20,
    ) -> None:
        # --- Validate shift: must be strictly positive ---
        shift_value = float(shift)
        if not np.isfinite(shift_value):
            raise ValueError(
                f"shift must be finite; got shift={shift!r}."
            )
        if shift_value <= 0.0:
            raise ValueError(
                f"shift must be strictly positive; got shift={shift!r}. "
                "A positive shift ensures (shift + lambda_k) > 0 for all "
                "modes "
                "and avoids singularities for negative orders."
            )

        # --- Validate box identity ---
        if space.function_domain is not provider._box:
            raise ValueError(
                "The provider's box domain must be the same object as "
                "space.function_domain."
            )

        self._space = space
        self._provider = provider
        self._shift = shift_value
        self._order = float(order)
        self._quad_resolution = quad_resolution

        # Build operator eigenvalues: mu_k = (shift + lambda_k) ** (order / 2)
        half_order = self._order / 2.0
        op_eigenvalues: list[float] = [
            float(
                (self._shift + float(provider.get_eigenvalue(i)))
                ** half_order
            )
            for i in range(provider.dim)
        ]

        # Pre-fetch eigenfunctions to avoid repeated provider calls
        eigenfunctions: list[Function3D] = [
            provider.get_eigenfunction(i) for i in range(provider.dim)
        ]

        # Pre-compute quadrature mesh and weights for projection integrals
        box = provider._box
        self._pts = box.structured_mesh(quad_resolution)
        self._weights = box.volume_weights(quad_resolution)

        super().__init__(space, op_eigenvalues, eigenfunctions)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shift(self) -> float:
        """The positive shift in $(\\mathrm{shift}+\\lambda)^{s/2}$."""
        return self._shift

    @property
    def order(self) -> float:
        """The real exponent $s$ in $(\\mathrm{shift}+\\lambda)^{s/2}$."""
        return self._order

    # ------------------------------------------------------------------
    # Spectral projection apply — box quadrature
    # ------------------------------------------------------------------

    def _apply(self, f: Function3D) -> Function3D:
        """Apply the Bessel-Sobolev operator to *f*.

        Computes $L^2$ projection coefficients
        $c_k = \\langle f, \\Phi_k \\rangle_{L^2}$ using the pre-computed
        box quadrature, then synthesises the result as

        .. math::

            (Bf)(\\mathbf{x})
                = \\sum_{k=0}^{N-1} \\mu_k\\, c_k\\, \\Phi_k(\\mathbf{x})

        where $\\mu_k = (\\mathrm{shift} + \\lambda_k)^{s/2}$.

        Parameters
        ----------
        f : :class:`~pygeoinf3d.core.functions.Function3D`
            The input function.

        Returns
        -------
        :class:`~pygeoinf3d.core.functions.Function3D`
            The spectral operator applied to *f*.
        """
        # Evaluate f at all quadrature points once
        f_vals = np.asarray(f(self._pts), dtype=float)

        # Compute spectral coefficients c_k = <f, phi_k> = ∫ f * phi_k dV
        coeffs = []
        for phi_k in self._eigenfunctions:
            phi_vals = np.asarray(phi_k(self._pts), dtype=float)
            c_k = float(np.dot(f_vals * phi_vals, self._weights))
            coeffs.append(c_k)

        # Snapshot for closure: (mu_k, phi_k, c_k)
        modes_snapshot = list(
            zip(self._op_eigenvalues, self._eigenfunctions, coeffs)
        )

        def evaluate(x: np.ndarray) -> Union[float, np.ndarray]:
            x = np.asarray(x, dtype=float)
            is_batch = x.ndim == 2
            if is_batch:
                total = np.zeros(len(x))
            else:
                total = 0.0
            for mu_k, phi_k, c_k in modes_snapshot:
                total = total + mu_k * c_k * phi_k(x)
            return total

        return Function3D(
            self._space.function_domain,
            evaluate_callable=evaluate,
        )

    def __repr__(self) -> str:
        return (
            f"BesselSobolevBox("
            f"shift={self._shift!r}, "
            f"order={self._order!r}, "
            f"bc_type={self._provider.bc_type!r}, "
            f"truncation={self._provider.truncation}"
            f")"
        )
