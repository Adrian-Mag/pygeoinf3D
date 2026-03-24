"""Axis-aligned box-car (indicator) function providers for volumetric kernels.

Each produced function is piecewise constant on an axis-aligned box:

.. math::

    f_i(\\mathbf{x}) =
    \\begin{cases}
        h_i & \\text{if } |x_j - c_{ij}| \\le w_{ij}/2
              \\text{ for all } j = 1,2,3, \\\\
        0   & \\text{otherwise,}
    \\end{cases}

where ``widths`` :math:`w_{ij}` are **full widths** (so the half-extent per
axis is :math:`w_{ij}/2`).  When ``normalized=True``,
:math:`h_i = 1/(w_{ix}\\,w_{iy}\\,w_{iz})` so that
:math:`\\int f_i\\,\\mathrm{d}V = 1`; otherwise :math:`h_i = 1`.
"""

from __future__ import annotations

import numpy as np

from pygeoinf3d.providers.base import FunctionProvider3D
from pygeoinf3d.core.functions import Function3D


class BoxcarProvider3D(FunctionProvider3D):
    """Axis-aligned box-car (indicator) function provider.

    Parameters
    ----------
    domain : Domain3D
        The computational domain on which the functions live.
    centers : array_like, shape ``(n, 3)`` or ``(3,)``
        Box centres.  A shape-``(3,)`` input is treated as a single
        centre (``dim == 1``).
    widths : float or array_like
        Full widths per axis.  Half-width per axis = ``widths / 2``.

        * scalar → broadcast to ``(n, 3)``
        * shape ``(3,)`` → same per-axis for every kernel
        * shape ``(n, 3)`` → independent per-kernel, per-axis widths

        All values must be strictly positive.
    normalized : bool, default ``True``
        If ``True``, set height = 1 / volume so that
        :math:`\\int f_i\\,\\mathrm{d}V = 1`
        (volume = :math:`\\prod_j w_{ij}`).
        If ``False``, height = 1.

    Raises
    ------
    ValueError
        If *centers* does not have shape ``(n, 3)``, if any width value is
        non-positive, or if *widths* has an incompatible shape.

    Examples
    --------
    ::

        domain = BoxDomain(np.zeros(3), np.ones(3))
        centers = np.array([[0.5, 0.5, 0.5]])
        provider = BoxcarProvider3D(domain, centers, widths=0.4)
        f = provider.get_function(0)
    """

    def __init__(
        self,
        domain,
        centers,
        widths,
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

        widths = np.asarray(widths, dtype=float)
        if widths.ndim == 0:
            widths = np.full((n, 3), float(widths))
        elif widths.ndim == 1 and widths.shape[0] == 3:
            widths = np.broadcast_to(widths[np.newaxis, :], (n, 3)).copy()
        elif widths.ndim == 2 and widths.shape == (n, 3):
            widths = widths.copy()
        else:
            raise ValueError(
                f"widths must be scalar, shape (3,), or shape ({n}, 3), "
                f"got {widths.shape}"
            )
        if np.any(widths <= 0):
            raise ValueError("All widths must be strictly positive.")

        self._domain = domain
        self._centers = centers   # (n, 3)
        self._widths = widths     # (n, 3) full widths
        self._normalized = normalized

    # ------------------------------------------------------------------
    # FunctionProvider3D interface
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Number of box-car functions this provider can produce."""
        return len(self._centers)

    def get_function(self, i: int) -> Function3D:
        """Return the *i*-th box-car function as a :class:`Function3D`.

        Parameters
        ----------
        i : int
            Index in ``range(dim)``.

        Returns
        -------
        Function3D
            Indicator function on the axis-aligned box centred at
            ``centers[i]`` with full-widths ``widths[i]``.

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
        w = self._widths[i]    # (3,) full widths
        hw = w * 0.5           # (3,) half-widths

        if self._normalized:
            h = 1.0 / (w[0] * w[1] * w[2])
        else:
            h = 1.0

        def _eval(x, _c=c, _hw=hw, _h=h):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                inside = bool(np.all(np.abs(x - _c) <= _hw))
                return float(_h) if inside else 0.0
            # batch (b, 3)
            inside = np.all(np.abs(x - _c) <= _hw, axis=1)  # (b,)
            result = np.zeros(x.shape[0], dtype=float)
            result[inside] = _h
            return result

        return Function3D(
            self._domain,
            evaluate_callable=_eval,
            name=f"boxcar_{i}",
        )

    def evaluate_batch(self, i_start: int, i_end: int, points: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of box-cars ``i_start:i_end`` at *points*.

        Parameters
        ----------
        i_start, i_end : int
            Half-open row range ``[i_start, i_end)``.
        points : ndarray of shape ``(n_pts, 3)``

        Returns
        -------
        ndarray of shape ``(i_end - i_start, n_pts)``
        """
        points = np.asarray(points, dtype=float)              # (n_pts, 3)
        c = self._centers[i_start:i_end]                      # (B, 3)
        hw = self._widths[i_start:i_end] * 0.5                # (B, 3) half-widths
        d = points[np.newaxis, :, :] - c[:, np.newaxis, :]   # (B, n_pts, 3)
        inside = np.all(np.abs(d) <= hw[:, np.newaxis, :], axis=2)  # (B, n_pts)
        if self._normalized:
            w = self._widths[i_start:i_end]
            h = 1.0 / (w[:, 0] * w[:, 1] * w[:, 2])          # (B,)
        else:
            h = np.ones(i_end - i_start)
        return h[:, np.newaxis] * inside.astype(float)        # (B, n_pts)

    def __repr__(self) -> str:
        return (
            f"BoxcarProvider3D(dim={self.dim}, "
            f"normalized={self._normalized})"
        )
