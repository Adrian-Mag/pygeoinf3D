r"""Bessel-Sobolev spectral operator on a box domain.

Provides :class:`BesselSobolevBox`, a
:class:`~pygeoinf3d.operators.spectral_power_base.SpectralPowerBase3D`
representing the spectral operator whose eigenvalues are

.. math::

    \mu_k = (\mathrm{shift} + \lambda_k)^{s/2}

where $\lambda_k$ are the Laplacian eigenvalues supplied by a spectral-basis
provider, ``shift`` is a strictly positive real number, and $s$ (``order``)
is any real float.

Mathematical definition
-----------------------
Given orthonormal eigenfunction-eigenvalue pairs $(\Phi_k, \lambda_k)$ with
$-\Delta \Phi_k = \lambda_k \Phi_k$, this operator acts as

.. math::

    (Bf)(\mathbf{x})
        = \sum_{k=0}^{N-1} (\mathrm{shift} + \lambda_k)^{s/2}
            \langle f, \Phi_k \rangle_{L^2}\, \Phi_k(\mathbf{x}).

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
norm $\|(\mathrm{shift} - \Delta)^{s/2} \cdot\|_{L^2}$.

Why ``shift > 0`` is required
-----------------------------
Requiring a strictly positive shift guarantees
$\mathrm{shift} + \lambda_k > 0$ for **all** modes — including the Neumann
zero mode $\lambda_0 = 0$. This avoids any singularity for negative orders,
eliminating the need for a special-case pseudoinverse projection.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np

from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.operators.spectral_power_base import SpectralPowerBase3D
from pygeoinf3d.providers.tensor_product import TensorProductProvider
from pygeoinf3d.spaces.lebesgue import Lebesgue3D


def _parse_resolution3d(resolution) -> tuple[int, int, int]:
    """Normalise a quadrature resolution to a 3-tuple."""
    if isinstance(resolution, (int, np.integer)):
        n = int(resolution)
        return n, n, n
    if len(resolution) != 3:
        raise ValueError(
            "quad_resolution must be an int or a length-3 tuple."
        )
    return int(resolution[0]), int(resolution[1]), int(resolution[2])


def _get_provider_domain(provider: Any):
    """Infer the provider domain from either metadata or the first mode."""
    if hasattr(provider, "_box"):
        return provider._box

    if not hasattr(provider, "dim") or int(provider.dim) < 1:
        raise TypeError(
            "provider must expose either '_box' or at least one eigenfunction "
            "from which a domain can be inferred."
        )

    phi0 = provider.get_eigenfunction(0)
    if not hasattr(phi0, "domain"):
        raise TypeError(
            "provider.get_eigenfunction(i) must return Function3D-like "
            "objects with a 'domain' attribute."
        )
    return phi0.domain


def _eval_1d_basis(
    xs: np.ndarray,
    a: float,
    L: float,
    n_modes: int,
    bc_type: str,
) -> np.ndarray:
    """Return the (len(xs), n_modes) matrix of 1-D basis values."""
    xs = np.asarray(xs, dtype=float).ravel()
    n = len(xs)
    if bc_type == "neumann":
        result = np.empty((n, n_modes))
        result[:, 0] = 1.0 / np.sqrt(L)
        if n_modes > 1:
            k_arr = np.arange(1, n_modes, dtype=float)
            angles = np.pi * np.outer((xs - a) / L, k_arr)
            result[:, 1:] = np.sqrt(2.0 / L) * np.cos(angles)
    else:
        k_arr = np.arange(1, n_modes + 1, dtype=float)
        angles = np.pi * np.outer((xs - a) / L, k_arr)
        result = np.sqrt(2.0 / L) * np.sin(angles)
    return result


class BesselSobolevBox(SpectralPowerBase3D):
    r"""Spectral Bessel-Sobolev operator on a box.

    Represents the spectral operator with eigenvalues
    $\mu_k = (\mathrm{shift} + \lambda_k)^{s/2}$, where $\lambda_k$ are the
    Laplacian eigenvalues provided by a spectral-basis provider and
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
        Any provider exposing ``dim``, ``get_eigenvalue(i)``, and
        ``get_eigenfunction(i)``. The returned eigenfunctions must live on the
        same domain object as ``space.function_domain``. If the provider is a
        :class:`~pygeoinf3d.providers.tensor_product.TensorProductProvider`, a
        tensor-product fast path is enabled automatically when the quadrature is
        compatible.
    shift:
        Strictly positive real number added to each eigenvalue before
        exponentiation. Must satisfy ``shift > 0``.
    order:
        Real exponent $s$ such that the operator eigenvalue for mode $k$ is
        $(\mathrm{shift} + \lambda_k)^{s/2}$. May be any real float,
        including non-integer and negative values.
    quad_resolution:
        Resolution passed to ``space.function_domain.structured_mesh`` when
        computing the $L^2$ projections. Default ``20``.
    quad_location:
        Quadrature node placement forwarded to ``structured_mesh`` and
        ``volume_weights``. Default ``'cell_centers'``. The tensor-product fast
        path is only enabled for ``'cell_centers'``.

    Raises
    ------
    ValueError
        If ``shift <= 0``.
    ValueError
        If the provider domain differs from ``space.function_domain``.
    TypeError
        If the provider does not expose the required spectral interface.
    """

    def __init__(
        self,
        space: Lebesgue3D,
        provider: Any,
        *,
        shift: float,
        order: float,
        quad_resolution: Union[int, tuple[int, int, int]] = 20,
        quad_location: str = "cell_centers",
    ) -> None:
        shift_value = float(shift)
        if not np.isfinite(shift_value):
            raise ValueError(f"shift must be finite; got shift={shift!r}.")
        if shift_value <= 0.0:
            raise ValueError(
                f"shift must be strictly positive; got shift={shift!r}. "
                "A positive shift ensures (shift + lambda_k) > 0 for all "
                "modes and avoids singularities for negative orders."
            )

        missing = [
            name for name in ("dim", "get_eigenvalue", "get_eigenfunction")
            if not hasattr(provider, name)
        ]
        if missing:
            raise TypeError(
                "provider must expose dim, get_eigenvalue(i), and "
                f"get_eigenfunction(i); missing {missing!r}."
            )

        provider_domain = _get_provider_domain(provider)
        if space.function_domain is not provider_domain:
            raise ValueError(
                "The provider's domain must be the same object as "
                "space.function_domain."
            )

        self._space = space
        self._provider = provider
        self._shift = shift_value
        self._order = float(order)
        self._quad_resolution = quad_resolution
        self._quad_location = quad_location

        provider_dim = int(provider.dim)
        half_order = self._order / 2.0
        op_eigenvalues: list[float] = [
            float((self._shift + float(provider.get_eigenvalue(i))) ** half_order)
            for i in range(provider_dim)
        ]

        eigenfunctions: list[Function3D] = [
            provider.get_eigenfunction(i) for i in range(provider_dim)
        ]

        box = space.function_domain
        self._pts = box.structured_mesh(quad_resolution, location=quad_location)
        self._weights = box.volume_weights(
            quad_resolution,
            location=quad_location,
        )

        super().__init__(space, op_eigenvalues, eigenfunctions)

        self._fast_path_enabled = False
        if (
            isinstance(provider, TensorProductProvider)
            and quad_location == "cell_centers"
        ):
            Nx, Ny, Nz = _parse_resolution3d(quad_resolution)
            nx, ny, nz = provider.truncation
            bc = provider.bc_type
            low = provider._low
            lengths = provider._lengths

            xs_q = low[0] + lengths[0] * (np.arange(Nx) + 0.5) / Nx
            ys_q = low[1] + lengths[1] * (np.arange(Ny) + 0.5) / Ny
            zs_q = low[2] + lengths[2] * (np.arange(Nz) + 0.5) / Nz

            self._Phi_x_q = _eval_1d_basis(xs_q, low[0], lengths[0], nx, bc)
            self._Phi_y_q = _eval_1d_basis(ys_q, low[1], lengths[1], ny, bc)
            self._Phi_z_q = _eval_1d_basis(zs_q, low[2], lengths[2], nz, bc)

            self._wx_q = np.full(Nx, lengths[0] / Nx)
            self._wy_q = np.full(Ny, lengths[1] / Ny)
            self._wz_q = np.full(Nz, lengths[2] / Nz)

            kmin = 0 if bc == "neumann" else 1
            mu_3d = np.zeros((nx, ny, nz))
            for idx in range(provider_dim):
                kx, ky, kz = provider.get_multi_index(idx)
                mu_3d[kx - kmin, ky - kmin, kz - kmin] = op_eigenvalues[idx]
            self._mu_3d = mu_3d

            self._tp_low = low.copy()
            self._tp_lengths = lengths.copy()
            self._tp_bc = bc
            self._tp_n = (nx, ny, nz)
            self._tp_quad_shape = (Nx, Ny, Nz)
            self._fast_path_enabled = True

    @property
    def shift(self) -> float:
        r"""The positive shift in $(\mathrm{shift}+\lambda)^{s/2}$."""
        return self._shift

    @property
    def order(self) -> float:
        r"""The real exponent $s$ in $(\mathrm{shift}+\lambda)^{s/2}$."""
        return self._order

    def _apply(self, f: Function3D) -> Function3D:
        """Apply the Bessel-Sobolev operator to *f*.

        Uses the generic box quadrature configured at construction time. When
        the provider is a :class:`TensorProductProvider` and the quadrature is
        compatible, an internal tensor-product fast path is used; otherwise the
        method falls back to the provider-agnostic spectral projection loop.
        """
        if self._fast_path_enabled:
            Nx, Ny, Nz = self._tp_quad_shape
            nx, ny, nz = self._tp_n
            low, lengths, bc = self._tp_low, self._tp_lengths, self._tp_bc

            f_grid = np.asarray(f(self._pts), dtype=float).reshape(Nx, Ny, Nz)
            FW = f_grid * (
                self._wx_q[:, None, None]
                * self._wy_q[None, :, None]
                * self._wz_q[None, None, :]
            )
            T = self._Phi_x_q.T @ FW.reshape(Nx, Ny * Nz)
            T = np.einsum(
                "xjl,jy->xyl",
                T.reshape(nx, Ny, Nz),
                self._Phi_y_q,
                optimize=True,
            )
            c_3d = np.einsum("xyz,zw->xyw", T, self._Phi_z_q, optimize=True)
            d_3d = self._mu_3d * c_3d

            def evaluate(x: np.ndarray) -> Union[float, np.ndarray]:
                x = np.asarray(x, dtype=float)
                is_single = x.ndim == 1
                pts = x[None, :] if is_single else x
                Px = _eval_1d_basis(pts[:, 0], low[0], lengths[0], nx, bc)
                Py = _eval_1d_basis(pts[:, 1], low[1], lengths[1], ny, bc)
                Pz = _eval_1d_basis(pts[:, 2], low[2], lengths[2], nz, bc)
                result = np.einsum(
                    "xyz,ix,iy,iz->i",
                    d_3d,
                    Px,
                    Py,
                    Pz,
                    optimize=True,
                )
                return float(result[0]) if is_single else result

            return Function3D(
                self._space.function_domain,
                evaluate_callable=evaluate,
            )

        f_vals = np.asarray(f(self._pts), dtype=float)
        coeffs = []
        for phi_k in self._eigenfunctions:
            phi_vals = np.asarray(phi_k(self._pts), dtype=float)
            c_k = float(np.dot(f_vals * phi_vals, self._weights))
            coeffs.append(c_k)

        modes_snapshot = list(
            zip(self._op_eigenvalues, self._eigenfunctions, coeffs)
        )

        def evaluate(x: np.ndarray) -> Union[float, np.ndarray]:
            x = np.asarray(x, dtype=float)
            is_batch = x.ndim == 2
            total = np.zeros(len(x)) if is_batch else 0.0
            for mu_k, phi_k, c_k in modes_snapshot:
                total = total + mu_k * c_k * phi_k(x)
            return total

        return Function3D(
            self._space.function_domain,
            evaluate_callable=evaluate,
        )

    def __repr__(self) -> str:
        parts = [
            f"shift={self._shift!r}",
            f"order={self._order!r}",
        ]
        if hasattr(self._provider, "bc_type"):
            parts.append(f"bc_type={self._provider.bc_type!r}")
        if hasattr(self._provider, "truncation"):
            parts.append(f"truncation={self._provider.truncation}")
        if self._fast_path_enabled:
            parts.append("fast_path='tensor-product'")
        return f"BesselSobolevBox({', '.join(parts)})"
