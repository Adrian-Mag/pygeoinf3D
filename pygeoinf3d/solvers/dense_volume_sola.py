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

from typing import Union, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from pygeoinf3d.solvers.volume_interactions import VolumeInteractionBackend


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

        # Assemble Gram matrix G = K @ diag(w) @ K^T
        G = self._backend.gram()  # (n_kernels, n_kernels)
        n = G.shape[0]
        if regularization > 0.0:
            G = G + regularization * np.eye(n)
        self._gram_matrix = G

        # Cholesky factorisation for O(m²) repeated solves
        # cho_factor raises scipy.linalg.LinAlgError if G is not positive definite.
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

    def resolve(self, target_K_row: np.ndarray) -> np.ndarray:
        r"""Compute resolving coefficients :math:`q` by solving :math:`G\,q = c_t`.

        The cross-gram vector is :math:`c_t[i] = \int k_i\,k_\mathrm{target}\,\mathrm{d}V
        \approx K_{i,:} \cdot (w \odot k_\mathrm{target}(\mathbf{x}))`.

        Uses the cached Cholesky factorisation for :math:`O(m^2)` cost.

        Parameters
        ----------
        target_K_row:
            The target kernel evaluated at all quadrature points, shape ``(n_pts,)``.
            Typically ``K[target_idx, :]`` from :attr:`~VolumeInteractionBackend.interaction_matrix`
            or ``target_fn(backend._pts)`` for an arbitrary target function.

        Returns
        -------
        ndarray of shape ``(n_kernels,)``
            Resolving coefficients :math:`q = G^{-1} c_t`.
        """
        K = self._backend.interaction_matrix()                    # (n_kernels, n_pts)
        w = self._backend._weights                                # (n_pts,)
        target = np.asarray(target_K_row, dtype=float).ravel()   # (n_pts,)

        # Cross-gram: c_t[i] = ∫ k_i · k_target dV ≈ K[i,:] · (w * target)
        cross = K @ (w * target)                                   # (n_kernels,)

        return cho_solve(self._cho, cross)                         # (n_kernels,)

    # ------------------------------------------------------------------
    # Properties (read-only accessors)
    # ------------------------------------------------------------------

    @property
    def n_kernels(self) -> int:
        """Number of kernel functions (dimension of the data space)."""
        return self._gram_matrix.shape[0]

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
