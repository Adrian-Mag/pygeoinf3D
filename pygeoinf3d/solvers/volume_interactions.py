"""Volumetric kernel interaction backend for structured quadrature meshes.

Provides :class:`VolumeInteractionBackend`, which assembles and caches the
kernel-evaluation matrix

.. math::

    K_{ij} = f_i(\\mathbf{x}_j), \\quad i = 0, \\ldots, n_{\\text{kernels}}-1,
    \\quad j = 0, \\ldots, n_{\\text{pts}}-1,

and derives the Gram and cross-Gram matrices via weighted summation over the
quadrature points.

Design notes
------------
* Each instance owns its own quadrature points and weights — no shared state.
* ``K`` is computed lazily on the first call to :meth:`interaction_matrix`
  and cached thereafter.
* Kernel evaluations are dispatched to ``provider.evaluate_all(pts)``; the
  default loop-based implementation in :class:`FunctionProvider3D` is used
  when no vectorized override is present.
* :meth:`cross_gram` validates that both backends share the same quadrature
  mesh (identity check on the arrays) before computing the cross-product.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from pygeoinf3d.providers.base import FunctionProvider3D


class _NoiseCovModel:
    r"""Helper for applying :math:`C_D^{-1}` in push-through identity solves.

    Encapsulates the three cases for the data noise covariance:

    * **scalar** (case 1): :math:`C_D = \sigma^2 I`
    * **diagonal** (case 2): :math:`C_D = \mathrm{diag}(\sigma_i^2)`
    * **dense SPD** (case 3): full :math:`C_D` factorised via Cholesky

    Parameters
    ----------
    cd_inv_diag : ndarray or None
        Array broadcast for :math:`C_D^{-1} d = d \odot \text{cd\_inv\_diag}`
        (cases 1 and 2).  ``None`` for dense case.
    cd_cho : tuple or None
        Cholesky factorisation from :func:`scipy.linalg.cho_factor` (case 3).
        ``None`` for scalar/diagonal case.
    """

    def __init__(self, cd_inv_diag, cd_cho):
        self._cd_inv_diag = cd_inv_diag  # 1/σᵢ² per datum, or None
        self._cd_cho = cd_cho            # cho_factor(C_D), or None

    def apply_inv(self, d: np.ndarray) -> np.ndarray:
        r"""Compute :math:`C_D^{-1} d`."""
        if self._cd_inv_diag is not None:
            return d * self._cd_inv_diag
        from scipy.linalg import cho_solve
        return cho_solve(self._cd_cho, d)


class VolumeInteractionBackend:
    r"""Assembles and caches volumetric kernel evaluations on a structured
    quadrature mesh.

    Given a domain and a :class:`~pygeoinf3d.providers.base.FunctionProvider3D`
    with *n* functions, this backend assembles:

    * **interaction matrix** :math:`K \in \mathbb{R}^{n \times N}` where
      :math:`K_{ij} = f_i(\mathbf{x}_j)` and :math:`\mathbf{x}_j` are the
      quadrature points;
    * **Gram matrix** :math:`G \in \mathbb{R}^{n \times n}` where
      :math:`G_{ij} = \sum_k K_{ik}\,K_{jk}\,w_k \approx \int f_i f_j\,dV`;
    * **cross-Gram matrix** between two backends sharing the same quadrature
      mesh.

    The interaction matrix is computed once and cached; subsequent calls to
    :meth:`interaction_matrix`, :meth:`gram`, or :meth:`cross_gram` reuse the
    cached result at no extra cost.

    Parameters
    ----------
    domain:
        A :class:`~pygeoinf3d.geometries.box.domain.BoxDomain` (or any domain
        implementing :meth:`structured_mesh` and :meth:`volume_weights`).
    provider:
        A :class:`~pygeoinf3d.providers.base.FunctionProvider3D` whose
        functions are to be evaluated.
    resolution:
        Quadrature resolution — either a single integer *n* (giving an
        ``n×n×n`` grid) or a 3-tuple ``(nx, ny, nz)``.
    location:
        ``'cell_centers'`` (default) or ``'nodes'``.  Passed directly to
        :meth:`structured_mesh` and :meth:`volume_weights`.

    Examples
    --------
    ::

        import numpy as np
        from pygeoinf3d import BoxDomain
        from pygeoinf3d.providers.function_providers import GaussianProvider3D
        from pygeoinf3d.solvers import VolumeInteractionBackend

        domain   = BoxDomain(np.zeros(3), np.ones(3))
        centers  = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
        provider = GaussianProvider3D(domain, centers, sigmas=0.1)
        backend  = VolumeInteractionBackend(domain, provider, resolution=(16, 16, 16))

        K = backend.interaction_matrix()   # shape (2, 4096)
        G = backend.gram()                 # shape (2, 2)
    """

    def __init__(
        self,
        domain,
        provider: FunctionProvider3D,
        resolution: Union[int, Tuple[int, int, int]],
        location: str = "cell_centers",
        cache_k: bool = True,
    ) -> None:
        self._domain = domain
        self._provider = provider
        self._resolution = resolution
        self._location = location

        # Quadrature points and weights — owned by this instance; the domain's
        # internal _quad_cache means repeated calls here are essentially free.
        self._pts = domain.structured_mesh(resolution, location=location)
        self._weights = domain.volume_weights(resolution, location=location)

        # Interaction matrix cache — populated lazily on first access.
        # When cache_k=False the matrix is recomputed on every call (streaming mode).
        self._K_cache: Optional[np.ndarray] = None
        self._cache_k: bool = cache_k
        self._no_cache_warned: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interaction_matrix(self) -> np.ndarray:
        """Evaluate all kernels at every quadrature point.

        Returns
        -------
        ndarray of shape ``(n_kernels, n_pts)``
            :math:`K_{ij} = f_i(\\mathbf{x}_j)`.  Computed once and cached
            (unless *cache_k* was set to ``False`` at construction time).
        """
        if self._K_cache is not None:
            return self._K_cache
        K = self._provider.evaluate_all(self._pts)
        if self._cache_k:
            self._K_cache = K
        else:
            if not self._no_cache_warned:
                import warnings
                warnings.warn(
                    f"VolumeInteractionBackend: K matrix "
                    f"({self._provider.dim} × {len(self._weights)} float64) "
                    f"is not cached (memory budget exceeded). "
                    f"Each call to interaction_matrix() re-evaluates all kernels.",
                    UserWarning,
                    stacklevel=2,
                )
                self._no_cache_warned = True
        return K

    def gram(self) -> np.ndarray:
        r"""Compute the Gram matrix :math:`G = K\,\mathrm{diag}(w)\,K^\top`.

        .. math::

            G_{ij} = \sum_k K_{ik}\,w_k\,K_{jk}
                   \approx \int_\Omega f_i(\mathbf{x})\,f_j(\mathbf{x})\,\mathrm{d}V

        Returns
        -------
        ndarray of shape ``(n_kernels, n_kernels)``
        """
        K = self.interaction_matrix()
        w = self._weights
        # K @ diag(w) @ K.T  — written as matrix multiply to stay O(n²·N)
        return K @ (w[:, np.newaxis] * K.T)

    def cross_gram(self, other: "VolumeInteractionBackend") -> np.ndarray:
        r"""Compute the cross-Gram matrix between *self* and *other*.

        .. math::

            G_{ij} = \sum_k K^{(1)}_{ik}\,w_k\,K^{(2)}_{jk}
                   \approx \int_\Omega f_i(\mathbf{x})\,g_j(\mathbf{x})\,\mathrm{d}V

        Both backends must share the same quadrature mesh (same domain,
        resolution, and location).

        Parameters
        ----------
        other:
            Another :class:`VolumeInteractionBackend` built with the same
            domain, resolution, and location.

        Returns
        -------
        ndarray of shape ``(n_kernels_self, n_kernels_other)``

        Raises
        ------
        ValueError
            If the two backends do not share the same quadrature points.
        """
        if (
            self._domain is not other._domain
            or self._resolution != other._resolution
            or self._location != other._location
        ):
            raise ValueError(
                "cross_gram requires both backends to share the same "
                "domain, resolution, and location.  "
                f"self: resolution={self._resolution}, location={self._location!r}; "
                f"other: resolution={other._resolution}, location={other._location!r}."
            )
        K_self = self.interaction_matrix()
        K_other = other.interaction_matrix()
        w = self._weights
        return K_self @ (w[:, np.newaxis] * K_other.T)

    # ------------------------------------------------------------------
    # Chunked / memory-efficient operations for push-through identity
    # ------------------------------------------------------------------

    def _default_chunk_size(self, memory_budget_bytes: int = 256 * 1024 ** 2) -> int:
        """Rows of K to process at once, limited by memory budget."""
        N_q = len(self._weights)
        row_bytes = N_q * 8  # float64
        return max(1, memory_budget_bytes // row_bytes)

    def build_model_gram(
        self,
        noise_cov,
        chunk_size: Optional[int] = None,
    ) -> tuple:
        r"""Build the model-space Gram matrix :math:`M_{CD} = I + \tilde{K}^\top C_D^{-1} \tilde{K}`.

        Here :math:`\tilde{K} = K\,\mathrm{diag}(\sqrt{w})`.

        Parameters
        ----------
        noise_cov : float, or ndarray of shape ``(N_d,)``, or ndarray of shape ``(N_d, N_d)``
            Defines the data noise covariance :math:`C_D`:

            * **float** — :math:`C_D = \sigma^2 I` where :math:`\sigma =` ``noise_cov``
              (standard deviation).
            * **1-D array** — :math:`C_D = \mathrm{diag}(\sigma_i^2)` where
              :math:`\sigma_i =` ``noise_cov[i]`` (per-datum standard deviations).
            * **2-D array** — dense SPD matrix :math:`C_D =` ``noise_cov``.
              Requires the interaction matrix :math:`K` to be cached
              (``memory_budget_bytes`` must be sufficient).

        chunk_size : int or None
            Rows of :math:`K` to process per iteration (cases 1 and 2 only).
            Defaults to a memory-limited value from :meth:`_default_chunk_size`.

        Returns
        -------
        M_CD : ndarray of shape ``(N_q, N_q)``
            :math:`M_{CD} = I + \tilde{K}^\top C_D^{-1} \tilde{K}`.
        cho_M_CD : object
            Scipy Cholesky factorisation of :math:`M_{CD}`
            (from :func:`scipy.linalg.cho_factor`).
        noise_model : _NoiseCovModel
            Helper for applying :math:`C_D^{-1}`.

        Raises
        ------
        ValueError
            If ``noise_cov`` is a 1-D array with length ≠ ``N_d``.
        RuntimeError
            If ``noise_cov`` is a 2-D array but :math:`K` is not cached.
        """
        from scipy.linalg import cho_factor
        N_q = len(self._weights)
        sqrt_w = np.sqrt(self._weights)
        noise_cov_arr = np.asarray(noise_cov, dtype=float)
        N_d = self._provider.dim
        pts = self._pts

        M_CD = np.eye(N_q)
        B = chunk_size if chunk_size is not None else self._default_chunk_size()

        if noise_cov_arr.ndim == 0:
            # Case 1: scalar std dev σ → C_D = σ²I, C_D^{-1} = (1/σ²)I
            sigma = float(noise_cov_arr)
            inv_var = 1.0 / sigma ** 2
            cd_inv_diag = np.full(N_d, inv_var)
            for i in range(0, N_d, B):
                j = min(i + B, N_d)
                K_chunk = self._provider.evaluate_batch(i, j, pts)  # (b, N_q)
                Kw_chunk = K_chunk * sqrt_w[None, :]                # (b, N_q)
                M_CD += inv_var * (Kw_chunk.T @ Kw_chunk)
            return M_CD, cho_factor(M_CD), _NoiseCovModel(cd_inv_diag, None)

        elif noise_cov_arr.ndim == 1:
            # Case 2: per-datum std devs σ_i → C_D = diag(σ_i²)
            if len(noise_cov_arr) != N_d:
                raise ValueError(
                    f"noise_cov 1-D array must have length N_d={N_d}, "
                    f"got {len(noise_cov_arr)}."
                )
            inv_var = 1.0 / noise_cov_arr ** 2          # shape (N_d,)
            for i in range(0, N_d, B):
                j = min(i + B, N_d)
                K_chunk = self._provider.evaluate_batch(i, j, pts)  # (b, N_q)
                Kw_chunk = K_chunk * sqrt_w[None, :]                # (b, N_q)
                # K̃^T diag(inv_var[i:j]) K̃  chunk contribution
                M_CD += (Kw_chunk * inv_var[i:j, None]).T @ Kw_chunk
            return M_CD, cho_factor(M_CD), _NoiseCovModel(inv_var, None)

        elif noise_cov_arr.ndim == 2:
            # Case 3: dense SPD C_D — requires K to be cached
            if self._K_cache is None:
                raise RuntimeError(
                    "build_model_gram with a 2-D noise_cov (dense C_D) requires "
                    "the interaction matrix K to be cached. Set "
                    "memory_budget_bytes >= "
                    f"{N_d * N_q * 8} bytes."
                )
            from scipy.linalg import cho_factor as _cf, cho_solve as _cs
            cd_cho = _cf(noise_cov_arr)
            K = self._K_cache                             # (N_d, N_q)
            Kw = K * sqrt_w[None, :]                      # (N_d, N_q) = K̃
            CdinvKw = _cs(cd_cho, Kw)                     # C_D^{-1} K̃, (N_d, N_q)
            M_CD += Kw.T @ CdinvKw                        # (N_q, N_q)
            return M_CD, cho_factor(M_CD), _NoiseCovModel(None, cd_cho)

        else:
            raise ValueError(
                "noise_cov must be a scalar, 1-D array (per-datum std devs), "
                "or 2-D array (full C_D matrix)."
            )


    def normal_vector(
        self,
        d: np.ndarray,
        chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""Compute :math:`\tilde{K}^\top d = \sqrt{w} \odot (K^\top d)`, shape ``(N_q,)``.

        Uses the cached interaction matrix when available; otherwise streams
        in chunks to avoid materialising the full ``K``.

        Parameters
        ----------
        d : array_like of shape ``(N_d,)``
        chunk_size : int or None

        Returns
        -------
        ndarray of shape ``(N_q,)``
        """
        d = np.asarray(d, dtype=float).ravel()
        sqrt_w = np.sqrt(self._weights)
        if self._K_cache is not None:
            return sqrt_w * (self._K_cache.T @ d)
        N_q = len(self._weights)
        B = chunk_size if chunk_size is not None else self._default_chunk_size()
        result = np.zeros(N_q)
        N_d = self._provider.dim
        pts = self._pts
        for i in range(0, N_d, B):
            j = min(i + B, N_d)
            K_chunk = self._provider.evaluate_batch(i, j, pts)  # (b, N_q)
            result += K_chunk.T @ d[i:j]
        return sqrt_w * result

    def kernel_matvec(
        self,
        v: np.ndarray,
        chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""Compute :math:`K v`, shape ``(N_d,)``, where *v* has shape ``(N_q,)``.

        Uses the cached interaction matrix when available; otherwise streams
        in chunks.

        Parameters
        ----------
        v : array_like of shape ``(N_q,)``
        chunk_size : int or None

        Returns
        -------
        ndarray of shape ``(N_d,)``
        """
        v = np.asarray(v, dtype=float).ravel()
        if self._K_cache is not None:
            return self._K_cache @ v
        B = chunk_size if chunk_size is not None else self._default_chunk_size()
        N_d = self._provider.dim
        result = np.zeros(N_d)
        pts = self._pts
        for i in range(0, N_d, B):
            j = min(i + B, N_d)
            K_chunk = self._provider.evaluate_batch(i, j, pts)  # (b, N_q)
            result[i:j] = K_chunk @ v
        return result
