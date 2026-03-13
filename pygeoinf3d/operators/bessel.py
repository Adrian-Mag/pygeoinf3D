"""Diagonal Bessel potential operators.

Public API (continuous-first)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`BesselOperator3D` -- maps ``Function3D -> Function3D`` via spectral
scaling, operating on a :class:`~pygeoinf3d.spaces.sobolev.Sobolev3D`.

Secondary spectral API
~~~~~~~~~~~~~~~~~~~~~~~
:class:`SpectralBesselOperator3D` -- maps coefficient arrays ``(N,) -> (N,)``
on a :class:`~pygeoinf3d.spaces.sobolev.SpectralSobolev3D`.

Mathematical definition
-----------------------
Both operators scale the k-th spectral coefficient by
(kappa^2 + lambda_k)^p for a given power p.

Key special cases
~~~~~~~~~~~~~~~~~
- p = s/2:  direct Bessel  (factory: :func:`bessel_direct`)
- p = -s/2: inverse Bessel (factory: :func:`bessel_inverse`)
- p = 0:    identity

Self-adjointness
----------------
Both operators are self-adjoint because the scaling is diagonal and
real-valued.
"""

from __future__ import annotations

import numpy as np

from pygeoinf.linear_operators import LinearOperator

from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.spaces.sobolev import Sobolev3D, SpectralSobolev3D


# ---------------------------------------------------------------------------
# SpectralBesselOperator3D -- secondary coefficient-vector version
# ---------------------------------------------------------------------------


class SpectralBesselOperator3D(LinearOperator):
    r"""Diagonal Bessel potential operator on a :class:`SpectralSobolev3D`.

    Scales the k-th spectral coefficient by
    ``(kappa^2 + lambda_k)^power``.

    Parameters
    ----------
    space : SpectralSobolev3D
        The coefficient-vector Hilbert space on which the operator acts
        (both domain and codomain).
    power : float
        The scaling exponent p.
    """

    def __init__(self, space: SpectralSobolev3D, *, power: float) -> None:
        self._space = space
        self._power = float(power)

        lams = space.eigenvalues
        base = space.kappa ** 2 + lams

        if self._power < 0 and np.any(base == 0.0):
            raise ValueError(
                "SpectralBesselOperator3D with "
                f"power={power!r} is not invertible: "
                "(kappa^2 + lambda_k) = 0 for at least one mode. "
                f"kappa={space.kappa!r}; use kappa > 0 or Dirichlet BC."
            )

        self._scale = base ** self._power

        super().__init__(
            space,
            space,
            self._apply,
            adjoint_mapping=self._apply,
        )

    def _apply(self, c: np.ndarray) -> np.ndarray:
        """Diagonal scaling: c_k -> (kappa^2 + lambda_k)^p * c_k."""
        return self._scale * np.asarray(c, dtype=float)

    @property
    def power(self) -> float:
        """The scaling exponent p."""
        return self._power

    def __repr__(self) -> str:
        return (
            f"SpectralBesselOperator3D("
            f"dim={self._space.dim}, "
            f"power={self._power:.4g}, "
            f"kappa={self._space.kappa:.4g}"
            f")"
        )


# ---------------------------------------------------------------------------
# BesselOperator3D -- continuous-first public API
# ---------------------------------------------------------------------------


class BesselOperator3D(LinearOperator):
    r"""Diagonal Bessel potential operator on a :class:`Sobolev3D`.

    Maps ``Function3D -> Function3D`` via spectral projection, scaling, and
    reconstruction::

        (B f)(x) = sum_k scale_k * f_hat_k * phi_k(x)

    where ``scale_k = (kappa^2 + lambda_k)^power`` and
    ``f_hat_k = <f, phi_k>_{L^2}`` are L^2 spectral coefficients.

    Parameters
    ----------
    space : Sobolev3D
        The continuous-first Hilbert space on which the operator acts
        (both domain and codomain).
    power : float
        The scaling exponent p.  Common choices:
        p = s/2 (:func:`bessel_direct`) and p = -s/2 (:func:`bessel_inverse`).

    Examples
    --------
    ::

        op = BesselOperator3D(sobolev, power=1.5)
        f  = sobolev.from_components(np.eye(sobolev.provider.dim)[0])
        gf = op(f)   # Function3D scaled by (kappa^2 + lambda_0)^1.5
    """

    def __init__(self, space: Sobolev3D, *, power: float) -> None:
        if not isinstance(space, Sobolev3D):
            raise TypeError(
                "BesselOperator3D requires a continuous-first Sobolev3D. "
                "Use SpectralBesselOperator3D for SpectralSobolev3D."
            )
        self._space = space
        self._power = float(power)

        lams = space.eigenvalues
        base = space.kappa ** 2 + lams

        if self._power < 0 and np.any(base == 0.0):
            raise ValueError(
                f"BesselOperator3D with power={power!r} is not invertible: "
                "(kappa^2 + lambda_k) = 0 for at least one mode. "
                f"kappa={space.kappa!r}; use kappa > 0 or Dirichlet BC."
            )

        self._scale = base ** self._power

        super().__init__(
            space,
            space,
            self._apply,
            adjoint_mapping=self._apply,
        )

    def _apply(self, f: Function3D) -> Function3D:
        """Project f, scale coefficients, reconstruct Function3D."""
        if not self._space.is_element(f):
            raise TypeError(
                "BesselOperator3D expects a Function3D on the same Domain3D "
                "instance as its Sobolev3D space."
            )
        c = self._space.to_components(f)
        return self._space.from_components(self._scale * c)

    @property
    def power(self) -> float:
        """The scaling exponent p."""
        return self._power

    def __repr__(self) -> str:
        return (
            f"BesselOperator3D("
            f"modes={self._space.provider.dim}, "
            f"power={self._power:.4g}, "
            f"kappa={self._space.kappa:.4g}"
            f")"
        )


# ---------------------------------------------------------------------------
# Convenience constructors (continuous-first API)
# ---------------------------------------------------------------------------


def bessel_direct(space: Sobolev3D) -> BesselOperator3D:
    r"""Bessel operator with exponent s/2 on a :class:`Sobolev3D`.

    Scales the k-th mode by ``(kappa^2 + lambda_k)^(s/2)``.

    Parameters
    ----------
    space : Sobolev3D

    Returns
    -------
    BesselOperator3D with ``power = space.s / 2``.
    """
    if not isinstance(space, Sobolev3D):
        raise TypeError(
            "bessel_direct expects a continuous-first Sobolev3D. "
            "Use SpectralBesselOperator3D for SpectralSobolev3D."
        )
    return BesselOperator3D(space, power=space.s / 2)


def bessel_inverse(space: Sobolev3D) -> BesselOperator3D:
    r"""Bessel operator with exponent -s/2 on a :class:`Sobolev3D`.

    Scales the k-th mode by ``(kappa^2 + lambda_k)^(-s/2)``.

    Parameters
    ----------
    space : Sobolev3D

    Returns
    -------
    BesselOperator3D with ``power = -space.s / 2``.

    Raises
    ------
    ValueError
        If kappa = 0 and a Neumann constant mode exists (lambda_0 = 0),
        since the scaling would divide by zero.
    """
    if not isinstance(space, Sobolev3D):
        raise TypeError(
            "bessel_inverse expects a continuous-first Sobolev3D. "
            "Use SpectralBesselOperator3D for SpectralSobolev3D."
        )
    return BesselOperator3D(space, power=-space.s / 2)
