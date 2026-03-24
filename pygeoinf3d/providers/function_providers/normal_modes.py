"""Normal-modes function provider for volumetric 3D domains.

Each produced function is a linear combination of 3D sine modes (one per
spatial axis), modulated by an anisotropic Gaussian envelope:

.. math::

    f(\\mathbf{x}) = G(\\mathbf{x}) \\sum_{k=1}^{M} a_k
        \\prod_{j=1}^{3} \\sin\\!\\left(
            \\frac{2\\pi \\nu_{kj} (x_j - L_j^{\\rm low})}
                 {L_j^{\\rm high} - L_j^{\\rm low}}
        \\right)

where :math:`G` is a 3D Gaussian envelope:

.. math::

    G(\\mathbf{x}) = \\exp\\!\\left(
        -\\tfrac{1}{2} \\sum_{j=1}^{3}
        \\left(\\frac{x_j - c_j}{\\sigma_j}\\right)^2
    \\right)

This is the 3D analogue of :class:`intervalinf.providers.NormalModesProvider`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from pygeoinf3d.providers.base import FunctionProvider3D
from pygeoinf3d.core.functions import Function3D


class NormalModesProvider3D(FunctionProvider3D):
    """Indexed provider for random Gaussian-modulated trigonometric functions
    on a 3D box domain.

    Each function is a sum of :math:`M` tensor-product sine modes multiplied
    by an anisotropic Gaussian envelope.  The provider supports two usage
    modes:

    **Deterministic / indexed** (primary usage)
        ``get_function(i)`` always returns the same function for the same
        index, regardless of how many other functions have been drawn.  This
        is achieved by seeding a fresh RNG per index using a hash of the
        base seed and the index.

    **Sequential random**
        ``sample_function()`` draws the next function from the shared
        internal RNG (state advances with each call).

    Parameters
    ----------
    domain : Domain3D
        The 3D computational domain.  Must expose ``.bounding_box``
        with ``.low`` and ``.high`` attributes (i.e. a
        :class:`~pygeoinf3d.geometries.box.domain.BoxDomain`).
    n_modes_range : tuple of int, default ``(3, 8)``
        ``(min, max)`` number of sine modes :math:`M` to combine.
    coeff_range : tuple of float, default ``(-2.0, 2.0)``
        ``(min, max)`` range for mode amplitude coefficients :math:`a_k`.
    freq_range : tuple of float, default ``(0.5, 10.0)``
        ``(min, max)`` range for per-axis frequencies :math:`\\nu_{kj}`.
    gaussian_width_percent_range : tuple of float, default ``(10.0, 50.0)``
        ``(min, max)`` Gaussian width expressed as a percentage of the
        domain extent along each axis.
    random_state : int, optional
        Seed for the internal RNG.  If ``None``, a fixed default seed is
        used so that indexed access remains reproducible.
    dim : int, default ``None``
        If given, fixes the reported ``dim`` of this provider (useful when
        embedding the provider in a solver that expects a finite basis).
        If ``None``, ``dim`` raises ``ValueError`` (unbounded provider).

    Examples
    --------
    ::

        domain = BoxDomain(np.zeros(3), np.ones(3))
        provider = NormalModesProvider3D(domain, random_state=42, dim=100)
        f = provider.get_function(0)   # deterministic, always the same
        g = provider.get_function(7)   # deterministic, always the same

        values = f(np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]]))
    """

    has_tensor_product_structure: bool = False

    def __init__(
        self,
        domain,
        *,
        n_modes_range: Tuple[int, int] = (3, 8),
        coeff_range: Tuple[float, float] = (-2.0, 2.0),
        freq_range: Tuple[float, float] = (0.5, 10.0),
        gaussian_width_percent_range: Tuple[float, float] = (10.0, 50.0),
        random_state: Optional[int] = None,
        dim: Optional[int] = None,
    ) -> None:
        self._domain = domain

        # Extract box bounds
        bbox = domain.bounding_box()
        self._low = np.asarray(bbox.low, dtype=float)   # shape (3,)
        self._high = np.asarray(bbox.high, dtype=float) # shape (3,)
        self._lengths = self._high - self._low          # shape (3,)

        self.n_modes_range = n_modes_range
        self.coeff_range = coeff_range
        self.freq_range = freq_range
        self.gaussian_width_percent_range = gaussian_width_percent_range

        self._base_seed: int = random_state if random_state is not None else 12345
        self._rng = np.random.default_rng(random_state)
        self._dim = dim

    # ------------------------------------------------------------------
    # FunctionProvider3D interface
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Total number of functions this provider can produce.

        Raises
        ------
        ValueError
            If *dim* was not specified at construction (unbounded provider).
        """
        if self._dim is None:
            raise ValueError(
                "This NormalModesProvider3D was created without a fixed 'dim'. "
                "Pass dim=N at construction to enable evaluate_all / evaluate_batch."
            )
        return self._dim

    def get_function(self, i: int) -> Function3D:
        """Return the *i*-th deterministic function.

        The function is reproducibly derived from the base seed and *i*:
        calling ``get_function(i)`` twice always returns the same function.

        Parameters
        ----------
        i : int
            Non-negative index.

        Returns
        -------
        Function3D
        """
        if i < 0:
            raise ValueError(f"Index must be non-negative, got {i}.")
        params = self._params_for_index(i)
        return self._make_function(params, name=f"normal_mode_3d_{i}")

    def sample_function(self) -> Function3D:
        """Draw the next random function, advancing the shared RNG state.

        Returns
        -------
        Function3D
        """
        params = self._draw_params(self._rng)
        n = params["n_modes"]
        return self._make_function(params, name=f"normal_mode_3d_random_{n}modes")

    def evaluate_batch(self, i_start: int, i_end: int, points: np.ndarray) -> np.ndarray:
        """Evaluate functions ``i_start:i_end`` at *points* in one vectorized pass.

        Rather than building :class:`Function3D` objects per function and
        calling them sequentially, all parameters for the batch are assembled
        into padded arrays and the sine / Gaussian kernels are computed in a
        single set of numpy operations.

        Parameters
        ----------
        i_start, i_end : int
            Half-open range ``[i_start, i_end)`` of function indices.
        points : ndarray of shape ``(n_pts, 3)``
            Evaluation points.

        Returns
        -------
        ndarray of shape ``(i_end - i_start, n_pts)``
            Row *k* contains the values of function ``i_start + k`` at all
            *points*.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[np.newaxis, :]
        n_pts = points.shape[0]
        batch = i_end - i_start
        if batch <= 0:
            return np.empty((0, n_pts), dtype=float)

        # Collect per-function parameters, padding modes to max_modes
        all_params = [self._params_for_index(i) for i in range(i_start, i_end)]
        max_modes = max(p["n_modes"] for p in all_params)

        # Padded arrays: (batch, max_modes) and (batch, max_modes, 3)
        coeffs  = np.zeros((batch, max_modes), dtype=float)
        freqs   = np.zeros((batch, max_modes, 3), dtype=float)
        centers = np.empty((batch, 3), dtype=float)
        widths  = np.empty((batch, 3), dtype=float)

        for k, p in enumerate(all_params):
            m = p["n_modes"]
            coeffs[k, :m]   = p["coefficients"]
            freqs[k, :m, :] = p["frequencies"]
            centers[k]      = p["center"]
            widths[k]       = p["widths"]

        low     = self._low      # (3,)
        lengths = self._lengths  # (3,)

        # --- Gaussian envelopes -------------------------------------------
        # z: (batch, n_pts, 3)  =  (batch, 1, 3) vs (1, n_pts, 3)
        z = (points[np.newaxis, :, :] - centers[:, np.newaxis, :]) \
            / widths[:, np.newaxis, :]
        envelope = np.exp(-0.5 * np.sum(z**2, axis=2))  # (batch, n_pts)

        # --- Trigonometric combination -------------------------------------
        # normalised angular coordinate: u = 2π * (x - low) / lengths
        u = 2.0 * np.pi * (points - low) / lengths   # (n_pts, 3)

        # sines: (batch, max_modes, n_pts, 3)
        # freqs: (batch, max_modes, 1, 3)  *  u: (1, 1, n_pts, 3)
        sins = np.sin(
            freqs[:, :, np.newaxis, :]          # (batch, max_modes, 1, 3)
            * u[np.newaxis, np.newaxis, :, :]   # (1, 1, n_pts, 3)
        )                                        # → (batch, max_modes, n_pts, 3)

        # tensor product over axes → (batch, max_modes, n_pts)
        mode_vals = np.prod(sins, axis=3)

        # weighted sum over modes → (batch, n_pts)
        # coeffs: (batch, max_modes, 1) * mode_vals: (batch, max_modes, n_pts)
        trig_sum = np.sum(coeffs[:, :, np.newaxis] * mode_vals, axis=1)

        return trig_sum * envelope  # (batch, n_pts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _params_for_index(self, i: int) -> dict:
        """Generate reproducible parameters for index *i*."""
        seed = (self._base_seed + i * 1_000_003) % (2**31 - 1)
        rng = np.random.default_rng(seed)
        return self._draw_params(rng)

    def _draw_params(self, rng: np.random.Generator) -> dict:
        """Draw random function parameters from *rng*."""
        n_min, n_max = self.n_modes_range
        n_modes = int(rng.integers(n_min, n_max + 1))

        coefficients = rng.uniform(
            self.coeff_range[0], self.coeff_range[1], size=n_modes
        )
        # Per-mode, per-axis frequencies: shape (n_modes, 3)
        frequencies = rng.uniform(
            self.freq_range[0], self.freq_range[1], size=(n_modes, 3)
        )

        # Anisotropic Gaussian envelope: centre uniform in box
        center = rng.uniform(self._low, self._high)  # shape (3,)

        w_min, w_max = self.gaussian_width_percent_range
        width_pct = rng.uniform(w_min, w_max, size=3)  # shape (3,)
        widths = (width_pct / 100.0) * self._lengths   # shape (3,)

        return {
            "n_modes": n_modes,
            "coefficients": coefficients,  # (n_modes,)
            "frequencies": frequencies,    # (n_modes, 3)
            "center": center,              # (3,)
            "widths": widths,              # (3,)
        }

    def _make_function(self, params: dict, name: str) -> Function3D:
        """Build a :class:`Function3D` from pre-drawn parameters."""
        low = self._low
        lengths = self._lengths
        coefficients = params["coefficients"]   # (n_modes,)
        frequencies = params["frequencies"]     # (n_modes, 3)
        center = params["center"]               # (3,)
        widths = params["widths"]               # (3,)

        def evaluate(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            single = x.ndim == 1
            if single:
                x = x[np.newaxis, :]    # (1, 3)
            # x: (n_pts, 3)

            # --- Gaussian envelope -------------------------------------------
            # z: (n_pts, 3)
            z = (x - center) / widths
            envelope = np.exp(-0.5 * np.sum(z**2, axis=1))  # (n_pts,)

            # --- Trigonometric combination ------------------------------------
            # normalised coordinate: u[j] = 2π * (x_j - low_j) / length_j
            u = 2.0 * np.pi * (x - low) / lengths  # (n_pts, 3)
            # sine along each axis for each mode: (n_modes, n_pts, 3)
            # frequencies: (n_modes, 3) → (n_modes, 1, 3)
            freq_bcast = frequencies[:, np.newaxis, :]     # (n_modes, 1, 3)
            u_bcast = u[np.newaxis, :, :]                  # (1, n_pts, 3)
            sins = np.sin(freq_bcast * u_bcast)            # (n_modes, n_pts, 3)
            # tensor product over axes: prod along last dim → (n_modes, n_pts)
            mode_values = np.prod(sins, axis=2)            # (n_modes, n_pts)
            # weighted sum: (n_modes,) @ (n_modes, n_pts) → (n_pts,)
            trig_sum = coefficients @ mode_values           # (n_pts,)

            result = trig_sum * envelope                   # (n_pts,)
            return float(result[0]) if single else result

        return Function3D(self._domain, evaluate_callable=evaluate, name=name)
