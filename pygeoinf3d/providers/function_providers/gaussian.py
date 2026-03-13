"""Axis-aligned Gaussian function providers for volumetric kernels.

Each produced function evaluates the anisotropic Gaussian:

.. math::

    f_i(\\mathbf{x}) = A_i \\exp\\!\\left(
        -\\frac{1}{2}\\sum_{j=1}^{3}
        \\frac{(x_j - c_{ij})^2}{\\sigma_{ij}^2}
    \\right)

where
:math:`A_i = \\bigl((2\\pi)^{3/2}\\prod_{j=1}^{3}
\\sigma_{ij}\\bigr)^{-1}`
when ``normalized=True``, giving
:math:`\\int_{\\mathbb{R}^3} f_i\\,\\mathrm{d}V = 1`.
"""

from __future__ import annotations

import numpy as np

from pygeoinf3d.providers.base import FunctionProvider3D
from pygeoinf3d.core.functions import Function3D

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


class GaussianProvider3D(FunctionProvider3D):
    """Produces :class:`~pygeoinf3d.core.functions.Function3D` objects with
    axis-aligned Gaussian profiles.

    Parameters
    ----------
    domain : Domain3D
        The computational domain on which the functions live.
    centers : array_like, shape ``(n, 3)`` or ``(3,)``
        Gaussian centres.  A shape-``(3,)`` input is treated as a single
        centre (``dim == 1``).
    sigmas : float or array_like
        Per-axis standard deviations.

        * scalar → broadcast to ``(n, 3)``
        * shape ``(3,)`` → same per-axis sigma for every kernel
        * shape ``(n, 3)`` → independent per-kernel, per-axis sigmas

        All values must be strictly positive.
    normalized : bool, default ``True``
        If ``True``, divide each function by its analytic normalisation
        constant so that :math:`\\int f_i\\,\\mathrm{d}V = 1`.

    Raises
    ------
    ValueError
        If *centers* does not have shape ``(n, 3)``, if any sigma value is
        non-positive, or if *sigmas* has an incompatible shape.

    Examples
    --------
    ::

        domain = BoxDomain(np.zeros(3), np.ones(3))
        centers = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
        provider = GaussianProvider3D(domain, centers, sigmas=0.1)
        f = provider.get_function(0)   # Gaussian centred at [0.25, 0.25, 0.25]
    """

    def __init__(
        self,
        domain,
        centers,
        sigmas,
        *,
        normalized: bool = True,
    ) -> None:
        centers = np.asarray(centers, dtype=float)
        if centers.ndim == 1:
            centers = centers[np.newaxis, :]
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError(
                f"centers must have shape (n, 3) or (3,), got {centers.shape}"
            )
        n = len(centers)

        sigmas = np.asarray(sigmas, dtype=float)
        if sigmas.ndim == 0:
            sigmas = np.full((n, 3), float(sigmas))
        elif sigmas.ndim == 1 and sigmas.shape[0] == 3:
            sigmas = np.broadcast_to(sigmas[np.newaxis, :], (n, 3)).copy()
        elif sigmas.ndim == 2 and sigmas.shape == (n, 3):
            sigmas = sigmas.copy()
        else:
            raise ValueError(
                f"sigmas must be scalar, shape (3,), or shape ({n}, 3), "
                f"got {sigmas.shape}"
            )
        if np.any(sigmas <= 0):
            raise ValueError("All sigmas must be strictly positive.")

        self._domain = domain
        self._centers = centers  # (n, 3)
        self._sigmas = sigmas    # (n, 3)
        self._normalized = normalized

    # ------------------------------------------------------------------
    # FunctionProvider3D interface
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Number of Gaussian functions this provider can produce."""
        return len(self._centers)

    def get_function(self, i: int) -> Function3D:
        """Return the *i*-th Gaussian as a :class:`Function3D`.

        Parameters
        ----------
        i : int
            Index in ``range(dim)``.

        Returns
        -------
        Function3D
            Gaussian function centred at ``centers[i]`` with widths
            ``sigmas[i]``.

        Raises
        ------
        IndexError
            If *i* is outside ``[0, dim)``.
        """
        if not (0 <= i < self.dim):
            raise IndexError(
                f"Index {i} is out of range [0, {self.dim})"
            )

        c = self._centers[i]   # (3,)
        s = self._sigmas[i]    # (3,)

        if self._normalized:
            A = 1.0 / (_SQRT_2PI ** 3 * s[0] * s[1] * s[2])
        else:
            A = 1.0

        def _eval(x, _c=c, _s=s, _A=A):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                d = x - _c
                return float(_A * np.exp(-0.5 * np.dot(d ** 2, 1.0 / _s ** 2)))
            # batch (n, 3)
            d = x - _c          # broadcasts (n,3) - (3,)
            exponent = -0.5 * np.sum(d ** 2 / _s ** 2, axis=1)
            return _A * np.exp(exponent)

        return Function3D(
            self._domain,
            evaluate_callable=_eval,
            name=f"gaussian_{i}",
        )

    def __repr__(self) -> str:
        return (
            f"GaussianProvider3D(dim={self.dim}, "
            f"normalized={self._normalized})"
        )
