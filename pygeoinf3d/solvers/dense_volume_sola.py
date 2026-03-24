r"""Fast SOLA solver for homogeneous volumetric Gaussian stacks on BoxDomain.

Provides :class:`DenseVolumeSolaSolver`, which pre-computes the full
kernel-evaluation matrix :math:`K` and Gram matrix :math:`G` once at
construction time, then answers Gram queries, forward maps, and resolving
coefficient solves in :math:`O(m^2)` or better — no per-query quadrature.

Mathematical background
-----------------------
For *m* volume-kernel linear forms :math:`\ell_i(f) = \int k_i\,f\,\mathrm{d}V`
with kernel functions :math:`k_0, \ldots, k_{m-1}` from a
:class:`~pygeoinf3d.providers.base.FunctionProvider3D`:

.. math::

    \text{Gram}[i,j] &= \int_\Omega k_i(\mathbf{x})\,k_j(\mathbf{x})\,\mathrm{d}V
                       \approx \mathbf{K}\,\mathrm{diag}(\mathbf{w})\,\mathbf{K}^\top \\
    d_i              &= \ell_i(f) \approx \sum_k K_{ik}\,w_k\,f(\mathbf{x}_k) \\
    \mathbf{q}       &= G^{-1}\,\mathbf{c}, \quad
                       c_i = \int k_i\,k_\mathrm{target}\,\mathrm{d}V

where :math:`K_{ij} = k_i(\mathbf{x}_j)` is the *interaction matrix* and
:math:`w_j` are cell-volume quadrature weights from
:meth:`~pygeoinf3d.geometries.box.domain.BoxDomain.volume_weights`.

This is a standalone solver object, not a :class:`~pygeoinf.LinearOperator`
subclass.  Use :class:`~pygeoinf3d.operators.sola.SOLAOperator3D` when you
need a pygeoinf-compatible forward operator (e.g. for
:class:`~pygeoinf.LinearMinimumNormInversion`).

Design
------
*  :class:`VolumeInteractionBackend` owns all quadrature state and K cache.
*  Gram matrix is assembled once in ``__init__`` and cached as
   :attr:`_gram_matrix`.
*  :class:`scipy.linalg.cho_factor` provides an :math:`O(m^3)` Cholesky
   factorisation; :meth:`resolve` then costs only :math:`O(m^2)` per call.
*  ``regularization`` shifts the Gram matrix by :math:`\varepsilon I` before
   factorisation, stabilising nearly-singular configurations.

See Also
--------
:class:`~pygeoinf3d.operators.sola.SOLAOperator3D` : Generic forward operator.
:class:`~pygeoinf3d.solvers.spectral_gram.SpectralGramSolver` : Spectral fast path.
:func:`~pygeoinf3d.solvers.router.SolverRouter.recommend` : Automatic selection.
"""

from __future__ import annotations

import warnings
from typing import Union, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from pygeoinf3d.solvers.volume_interactions import VolumeInteractionBackend


def _noise_cov_key(noise_cov):
    """Return a hashable key for cache invalidation of noise_cov."""
    nc = np.asarray(noise_cov, dtype=float)
    if nc.ndim == 0:
        return ('scalar', float(nc))
    elif nc.ndim == 1:
        return ('diag', nc.tobytes())
    else:
        return ('dense', nc.tobytes())


class DenseVolumeSolaSolver:
    r"""Fast SOLA solver for homogeneous volumetric Gaussian stacks on BoxDomain.

    Pre-computes the kernel-evaluation matrix :math:`K` and Gram matrix
    :math:`G = K\,\mathrm{diag}(w)\,K^\top` at construction time using a
    :class:`~pygeoinf3d.solvers.volume_interactions.VolumeInteractionBackend`.
    A Cholesky factorisation of :math:`G` is also cached, making per-target
    resolving-coefficient solves run at :math:`O(m^2)` cost rather than
    :math:`O(m^3)`.

    This is the canonical fast path for repeated SOLA assembly on structured
    :class:`~pygeoinf3d.geometries.box.domain.BoxDomain` problems with
    :class:`~pygeoinf3d.providers.function_providers.gaussian.GaussianProvider3D`
    kernels.  The generic always-correct fallback is
    :class:`~pygeoinf3d.operators.sola.SOLAOperator3D`.

    Parameters
    ----------
    domain:
        A :class:`~pygeoinf3d.geometries.box.domain.BoxDomain` (or any domain
        implementing :meth:`structured_mesh` and :meth:`volume_weights`).
    provider:
        A :class:`~pygeoinf3d.providers.base.FunctionProvider3D` (e.g.
        :class:`~pygeoinf3d.providers.function_providers.gaussian.GaussianProvider3D`)
        whose functions form the kernel stack.
    resolution:
        Quadrature resolution — a single ``int`` for an isotropic grid
        or a 3-tuple ``(nx, ny, nz)`` for anisotropic grids.  Default ``20``
        matches :meth:`~pygeoinf3d.geometries.box.domain.BoxDomain.integrate_volume`'s
        default, ensuring exact agreement with
        :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` evaluations.
    location:
        ``'cell_centers'`` (default) or ``'nodes'``.
    regularization:
        Non-negative scalar :math:`\varepsilon` added to the diagonal of the
        Gram matrix before Cholesky factorisation.  Improves numerical stability
        when kernels are nearly collinear.  Default ``0.0``.

    Raises
    ------
    ValueError
        If ``provider.dim == 0``.
    ValueError
        If ``regularization < 0``.
    np.linalg.LinAlgError
        (from scipy) If the Gram matrix (after regularization) is not
        positive definite — consider increasing ``regularization``.

    Notes
    -----
    *Resolution matching.*  The default ``resolution=20`` aligns with
    :meth:`BoxDomain.integrate_volume`'s default, so
    ``DenseVolumeSolaSolver.gram()`` matches
    :class:`~pygeoinf3d.operators.sola.SOLAOperator3D`'s Gram matrix to
    machine precision.  Choose any resolution; the two are always equal when
    both use the same quadrature parameters.

    *Memory.*  The interaction matrix :math:`K` of shape
    ``(n_kernels, n_pts)`` is dense.  For ``resolution=20`` on a unit box,
    ``n_pts = 8000``; for 100 kernels this is ~6.4 MB of float64 data.
    """

    def __init__(
        self,
        domain,
        provider,
        resolution: Union[int, Tuple[int, int, int]] = 20,
        location: str = "cell_centers",
        regularization: float = 0.0,
        memory_budget_bytes: int = 512 * 1024 ** 2,
    ) -> None:
        if provider.dim == 0:
            raise ValueError(
                "DenseVolumeSolaSolver requires at least one kernel "
                "(provider.dim must be positive)."
            )
        if regularization < 0.0:
            raise ValueError(
                f"regularization must be non-negative; got {regularization!r}."
            )

        # Build backend — owns pts, weights, and K cache
        self._backend = VolumeInteractionBackend(domain, provider, resolution, location)

        # Push-through identity cache (populated lazily by _ensure_model_cho)
        self._model_M_cho = None
        self._model_noise_model = None   # _NoiseCovModel instance
        self._model_noise_key = None     # hashable cache key for invalidation

        # Memory budget check: does K fit within the budget?
        N_d = provider.dim
        N_q = len(self._backend._weights)
        k_bytes = N_d * N_q * 8  # float64
        self._k_fits_in_memory = k_bytes <= memory_budget_bytes

        if not self._k_fits_in_memory:
            # Streaming mode: K is never cached; gram/cho not available
            self._backend._cache_k = False
            self._gram_matrix = None
            self._cho = None
        else:
            # Assemble Gram matrix G = K @ diag(w) @ K^T
            G = self._backend.gram()  # (n_kernels, n_kernels) — also caches K
            n = G.shape[0]
            if regularization > 0.0:
                G = G + regularization * np.eye(n)
            self._gram_matrix = G
            # Cholesky factorisation for O(m²) repeated solves
            self._cho = cho_factor(G)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def gram(self) -> np.ndarray:
        r"""Return the Gram matrix :math:`G[i,j] = \int k_i\,k_j\,\mathrm{d}V`.

        Returns
        -------
        ndarray of shape ``(n_kernels, n_kernels)``
            Dense, symmetric positive (semi-)definite matrix.  A copy is
            returned; modifying it does not affect internal state.
        """
        return self._gram_matrix.copy()

    def forward(self, f) -> np.ndarray:
        r"""Apply the forward map :math:`d_i = \int k_i\,f\,\mathrm{d}V`.

        Evaluates *f* at the pre-computed quadrature points and contracts
        with the cached interaction matrix row-wise.

        Parameters
        ----------
        f:
            A :class:`~pygeoinf3d.core.functions.Function3D` to measure.
            Must accept a ``(n_pts, 3)`` array and return ``(n_pts,)`` values.

        Returns
        -------
        ndarray of shape ``(n_kernels,)``
            Data vector :math:`d_i = K_{i,:} \cdot (w \odot f(\mathbf{x}))`.
        """
        K = self._backend.interaction_matrix()  # (n_kernels, n_pts)
        w = self._backend._weights              # (n_pts,)
        pts = self._backend._pts               # (n_pts, 3)

        f_vals = np.asarray(f(pts), dtype=float).ravel()  # (n_pts,)
        return K @ (w * f_vals)                            # (n_kernels,)

    def resolve(self, target_K_row: np.ndarray, noise_cov=None, *, noise_std=None) -> np.ndarray:
        r"""Compute resolving coefficients :math:`q`.

        Two paths are available:

        * **Push-through path** (recommended, pass *noise_cov*): solves
          :math:`(C_D + GG^*)\,q = c_t` using the model-space
          push-through identity.  Works even when K is not cached.
        * **Legacy Cholesky path** (*noise_cov=None*, deprecated): solves
          :math:`G\,q = c_t` using the pre-cached data-space Cholesky
          factorisation.  Requires K to fit in the memory budget.

        Parameters
        ----------
        target_K_row:
            The target kernel evaluated at all quadrature points, shape ``(n_pts,)``.
        noise_cov : float, or ndarray, or None
            Data noise covariance :math:`C_D`.  Pass a non-None value to use
            the push-through solver (required when K exceeds the memory budget).
            ``None`` selects the deprecated Cholesky path.
        noise_std : float, keyword-only
            **Deprecated.** Use ``noise_cov`` instead.

        Returns
        -------
        ndarray of shape ``(n_kernels,)``
            Resolving coefficients.
        """
        if noise_std is not None:
            warnings.warn(
                "noise_std is deprecated; use noise_cov instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if noise_cov is None:
                noise_cov = noise_std
        target = np.asarray(target_K_row, dtype=float).ravel()  # (n_pts,)

        if noise_cov is not None:
            # Push-through path: solve (C_D + GG*) q = c_t
            w = self._backend._weights
            c_t = self._backend.kernel_matvec(w * target)  # K @ (w * target), shape (N_d,)
            return self.solve(c_t, noise_cov)
        else:
            # Legacy path: solve data-space Gram G q = c_t
            if self._cho is None:
                raise RuntimeError(
                    "resolve() called without noise_std but the data-space Gram "
                    "Cholesky is not available (K exceeds memory budget). "
                    "Pass noise_std to use the push-through solver."
                )
            warnings.warn(
                "resolve(target) without noise_std uses the data-space Cholesky "
                "solver. Pass noise_std to use the push-through solver "
                "(required for large N_d).",
                DeprecationWarning,
                stacklevel=2,
            )
            K = self._backend.interaction_matrix()  # (n_kernels, n_pts)
            w = self._backend._weights              # (n_pts,)
            cross = K @ (w * target)                # (n_kernels,)
            return cho_solve(self._cho, cross)

    # ------------------------------------------------------------------
    # Properties (read-only accessors)
    # ------------------------------------------------------------------

    @property
    def n_kernels(self) -> int:
        """Number of kernel functions (dimension of the data space)."""
        return self._backend._provider.dim

    @property
    def backend(self) -> VolumeInteractionBackend:
        """The underlying :class:`VolumeInteractionBackend` instance."""
        return self._backend

    def forward_operator(self, function_space):
        r"""Return the forward map as a :class:`~pygeoinf.LinearOperator`.

        Wraps :meth:`forward` as a :class:`~pygeoinf.LinearOperator` with
        domain *function_space* and codomain :class:`~pygeoinf.EuclideanSpace`
        ``(n_kernels)``.  The precomputed :math:`K` matrix is reused on every
        call — no quadrature is repeated.

        The exact adjoint in the :math:`L^2(\Omega)` sense is:

        .. math::

            A^*(d)(x) = \sum_{i=0}^{m-1} d_i\,k_i(x)

        where :math:`k_i` are the kernel functions from the provider.
        Because the forward map evaluates :math:`f` at the same quadrature
        nodes used to build :math:`K`, the composition
        :math:`A A^*` recovers the Gram matrix exactly.

        Parameters
        ----------
        function_space:
            A :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` (or compatible
            Hilbert space) that is the domain of the operator.

        Returns
        -------
        pygeoinf.LinearOperator
            Linear map from *function_space* to
            :class:`~pygeoinf.EuclideanSpace` ``(n_kernels)``.

        Notes
        -----
        Each call constructs a new :class:`~pygeoinf.LinearOperator` object,
        but all returned operators share the same cached :math:`K` matrix
        via :attr:`backend`.
        """
        import warnings
        if not getattr(self._backend, '_cache_k', True):
            warnings.warn(
                f"forward_operator: K matrix is not cached "
                f"({self._backend._provider.dim} kernels). "
                f"Each application of this operator will re-evaluate all kernel functions.",
                UserWarning,
                stacklevel=2,
            )

        from pygeoinf import EuclideanSpace
        from pygeoinf.linear_operators import LinearOperator
        from pygeoinf3d.spaces.forms import VolumeKernel3D
        from pygeoinf3d.core.functions import Function3D

        m = self.n_kernels
        codomain = EuclideanSpace(m)
        provider = self._backend._provider

        def mapping(f):
            return codomain.from_components(self.forward(f))

        def adjoint_mapping(d):
            # A*(d)(x) = ∑_i d_i k_i(x)  — exact L² adjoint
            c = codomain.to_components(d)  # shape (m,)

            def evaluate(x):
                x = np.asarray(x, dtype=float)
                single = x.ndim == 1
                if single:
                    x = x[np.newaxis, :]
                # K_x[i, j] = k_i(x_j), shape (m, n_pts_in)
                K_x = provider.evaluate_all(x)
                result = c @ K_x  # (n_pts_in,)
                return float(result[0]) if single else result

            return Function3D(
                function_space.function_domain, evaluate_callable=evaluate
            )

        def dual_mapping(yp):
            # yp ∈ EuclideanSpace(m).dual — returns a LinearForm on function_space
            c = codomain.dual.to_components(yp)  # shape (m,)
            forms = [
                VolumeKernel3D(function_space, provider.get_function(i))
                for i in range(m)
            ]
            result = float(c[0]) * forms[0]
            for i in range(1, m):
                result = result + float(c[i]) * forms[i]
            return result

        return LinearOperator(
            function_space,
            codomain,
            mapping,
            dual_mapping=dual_mapping,
            adjoint_mapping=adjoint_mapping,
        )

    # ------------------------------------------------------------------
    # Push-through identity methods (Phase 2)
    # ------------------------------------------------------------------

    def _ensure_model_cho(self, noise_cov, chunk_size=None) -> None:
        """Lazily build and cache the model-space Cholesky factorisation."""
        key = _noise_cov_key(noise_cov)
        if self._model_M_cho is None or self._model_noise_key != key:
            _, cho, noise_model = self._backend.build_model_gram(noise_cov, chunk_size)
            self._model_M_cho = cho
            self._model_noise_model = noise_model
            self._model_noise_key = key

    def solve(
        self,
        d: np.ndarray,
        noise_cov=None,
        chunk_size=None,
        *,
        noise_std=None,
    ) -> np.ndarray:
        r"""Solve :math:`(C_D + GG^*)\,q = d` via the push-through identity.

        Uses the model-space factorisation
        :math:`M_{CD} = I + \tilde{K}^\top C_D^{-1} \tilde{K}` to avoid
        building the :math:`(N_d \times N_d)` data-space system.

        Parameters
        ----------
        d : ndarray of shape (N_d,)
        noise_cov : float, or ndarray of shape (N_d,), or ndarray of shape (N_d, N_d)
            Data noise covariance :math:`C_D`.  For scalar/1-D inputs the values
            are treated as **standard deviations** :math:`\sigma` so that
            :math:`C_D = \sigma^2 I` or :math:`C_D = \mathrm{diag}(\sigma_i^2)`.
            A 2-D array is taken as the full SPD matrix :math:`C_D` directly.
        chunk_size : int or None
        noise_std : float, keyword-only
            **Deprecated.** Pass ``noise_cov`` instead. When supplied, the value
            is forwarded to ``noise_cov`` with a :class:`DeprecationWarning`.

        Returns
        -------
        ndarray of shape (N_d,)
            SOLA coefficients *q*.
        """
        if noise_std is not None:
            warnings.warn(
                "noise_std is deprecated; use noise_cov instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if noise_cov is None:
                noise_cov = noise_std  # same semantics: float = std dev
        self._ensure_model_cho(noise_cov, chunk_size)
        d = np.asarray(d, dtype=float).ravel()
        nm = self._model_noise_model
        sqrt_w = np.sqrt(self._backend._weights)
        f  = nm.apply_inv(d)                                          # C_D^{-1} d
        v  = self._backend.normal_vector(f, chunk_size)               # K̃^T f
        z  = cho_solve(self._model_M_cho, v)                          # M_CD^{-1} v
        Kz = self._backend.kernel_matvec(sqrt_w * z, chunk_size)      # K(sqrt_w * z)
        return f - nm.apply_inv(Kz)                                   # C_D^{-1}(d - K̃ z)

    def reconstruct(
        self,
        d: np.ndarray,
        noise_cov=None,
        chunk_size=None,
        *,
        noise_std=None,
    ):
        r"""Reconstruct the model function from data *d*.

        Returns a :class:`~pygeoinf3d.core.functions.Function3D` that
        evaluates :math:`\hat{m}(x) = \sum_i q_i\,k_i(x)` where
        :math:`q = (C_D + GG^*)^{-1}\,d`.

        Parameters
        ----------
        d : ndarray of shape (N_d,)
        noise_cov : float, or ndarray of shape (N_d,), or ndarray of shape (N_d, N_d)
            See :meth:`solve`.
        chunk_size : int or None
        noise_std : float, keyword-only
            **Deprecated.** Use ``noise_cov``.

        Returns
        -------
        Function3D
        """
        if noise_std is not None:
            warnings.warn(
                "noise_std is deprecated; use noise_cov instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if noise_cov is None:
                noise_cov = noise_std
        from pygeoinf3d.core.functions import Function3D

        q = self.solve(d, noise_cov, chunk_size)
        provider = self._backend._provider
        N_d = provider.dim
        if N_d > 5000 and self._backend._K_cache is None:
            warnings.warn(
                f"reconstruct: evaluating a sum of {N_d} kernels at each query point "
                f"(K not cached). Consider reducing N_d or increasing memory_budget_bytes.",
                UserWarning, stacklevel=2,
            )

        def _evaluate(x):
            x = np.atleast_2d(np.asarray(x, dtype=float))
            return provider.evaluate_all(x).T @ q

        return Function3D(self._backend._domain, evaluate_callable=_evaluate)

    def compute_sola_matrix(
        self,
        target_solver: "DenseVolumeSolaSolver",
        noise_cov=None,
        chunk_size=None,
        *,
        noise_std=None,
    ) -> np.ndarray:
        r"""Compute the SOLA operator matrix :math:`X = \tilde{K}_T M_{CD}^{-1} \tilde{K}_G^\top C_D^{-1}`.

        Parameters
        ----------
        target_solver : DenseVolumeSolaSolver
            Solver whose kernels define the target properties.
        noise_cov : float, or ndarray of shape (N_d,), or ndarray of shape (N_d, N_d)
            See :meth:`solve`.
        chunk_size : int or None
        noise_std : float, keyword-only
            **Deprecated.** Use ``noise_cov``.

        Returns
        -------
        ndarray of shape (N_p, N_d)
            ``X[k, j]`` is the SOLA estimate of property *k* from unit data
            vector :math:`e_j`.
        """
        if noise_std is not None:
            warnings.warn(
                "noise_std is deprecated; use noise_cov instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if noise_cov is None:
                noise_cov = noise_std
        self._ensure_model_cho(noise_cov, chunk_size)
        nm = self._model_noise_model
        sqrt_w = np.sqrt(self._backend._weights)
        N_d    = self._backend._provider.dim
        N_p    = target_solver._backend._provider.dim

        # Evaluate target kernels at the forward solver's quadrature pts
        K_T  = target_solver._backend._provider.evaluate_all(self._backend._pts)  # (N_p, N_q)
        KT_w = K_T * sqrt_w[None, :]                          # K̃_T, (N_p, N_q)

        # Z_T = M_CD^{-1} K̃_T^T,  shape (N_q, N_p)
        Z_T  = cho_solve(self._model_M_cho, KT_w.T)
        # H = K̃_T M_CD^{-1} diag(sqrt_w),  shape (N_p, N_q)
        H    = Z_T.T * sqrt_w[None, :]

        # X = H @ (C_D^{-1} K_G)^T  (chunked over forward kernels)
        X    = np.zeros((N_p, N_d), dtype=float)
        pts  = self._backend._pts
        cs   = chunk_size if chunk_size is not None else self._backend._default_chunk_size()

        if nm._cd_inv_diag is not None:
            # Cases 1 + 2: C_D^{-1} K_G = K_G * cd_inv_diag[:, None]  (row-scaled)
            for i in range(0, N_d, cs):
                j = min(i + cs, N_d)
                K_chunk = self._backend._provider.evaluate_batch(i, j, pts)  # (B, N_q)
                X[:, i:j] = H @ (K_chunk * nm._cd_inv_diag[i:j, None]).T    # (N_p, B)
        else:
            # Case 3: precompute C_D^{-1} K (K must be cached — ensured by RuntimeError)
            K_full  = self._backend.interaction_matrix()                     # (N_d, N_q)
            CdinvK  = cho_solve(nm._cd_cho, K_full)                          # (N_d, N_q)
            for i in range(0, N_d, cs):
                j = min(i + cs, N_d)
                X[:, i:j] = H @ CdinvK[i:j, :].T                            # (N_p, B)
        return X
