"""Karhunen-Loève sampler for Gaussian measures on 3D function spaces.

Provides :class:`KLSampler3D`, which turns any
:class:`~pygeoinf3d.operators.base.SpectralOperator3D` (treated as a
covariance operator) into a sampler for the induced Gaussian measure via the
truncated KL expansion

.. math::

    u(\\mathbf{x}) = m(\\mathbf{x}) +
        \\sum_{k=0}^{K-1} \\sqrt{\\lambda_k}\\,\\xi_k\\,\\Phi_k(\\mathbf{x}),
        \\quad \\xi_k \\sim N(0, 1),

where $(\\lambda_k, \\Phi_k)$ are the operator's eigenvalue/eigenfunction pairs
and *K* = ``n_modes``.

Typical usage (prior for a 3D PLI notebook)::

    from pygeoinf3d.geometries.box.domain import BoxDomain
    from pygeoinf3d.providers.tensor_product import BoxLaplacianSpectrumProvider
    from pygeoinf3d.spaces.lebesgue import Lebesgue3D
    from pygeoinf3d.operators.inverse_laplacian_box import InverseBoxLaplacian
    from pygeoinf3d.sampling import KLSampler3D
    from pygeoinf import GaussianMeasure
    import numpy as np

    box      = BoxDomain(np.zeros(3), np.ones(3))
    space    = Lebesgue3D(box)
    provider = BoxLaplacianSpectrumProvider(box, truncation=(5, 5, 5), bc_type='dirichlet')
    inv_lap  = InverseBoxLaplacian(space, provider)

    kl = KLSampler3D(inv_lap, n_modes=20)

    # Option 1: use sample() directly
    prior = GaussianMeasure(covariance=inv_lap, expectation=space.zero,
                            sample=kl.sample)

    # Option 2: use covariance_factor (L : R^K → Lebesgue3D, C ≈ L L*)
    prior = GaussianMeasure(covariance=inv_lap, expectation=space.zero,
                            sample=kl.sample)
    # (pygeoinf will use sample() for drawing; covariance_factor is available
    #  for constructing the measure via factor-based paths if needed)
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from pygeoinf import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator

from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.operators.base import SpectralOperator3D
from pygeoinf3d.spaces.lebesgue import Lebesgue3D


def _attach_kl_fast_paths(
    sampler: "KLSampler3D",
    xis: np.ndarray,
    fn: Function3D,
) -> None:
    """Tag a KL sample Function3D with fast-path evaluator callables.

    Attaches two private callables to *fn* so that
    :func:`~pygeoinf3d.plot.evaluate_function3d_on_regular_grid` and
    :func:`~pygeoinf3d.plot.evaluate_function3d_on_axis_slice` can
    dispatch to the vectorized tensor-product fast path instead of the
    point-by-point generic :class:`Function3D` evaluation path.

    Parameters
    ----------
    sampler:
        The :class:`KLSampler3D` that produced *fn*.
    xis:
        The standard-normal coefficients used to construct this sample.
    fn:
        The :class:`~pygeoinf3d.core.functions.Function3D` to tag.
    """
    xis_snap = xis.copy()
    fn._kl_fast_grid_evaluator = (  # type: ignore[attr-defined]
        lambda resolution, plot_n_modes=None, _s=sampler, _x=xis_snap:
            _s.evaluate_on_regular_grid(_x, resolution, plot_n_modes=plot_n_modes)
    )
    fn._kl_fast_slice_evaluator = (  # type: ignore[attr-defined]
        lambda axis, coord, resolution, plot_n_modes=None, _s=sampler, _x=xis_snap:
            _s.evaluate_on_axis_slice(_x, axis=axis, coord=coord, resolution=resolution, plot_n_modes=plot_n_modes)
    )


def _parse_tp_axis(axis) -> int:
    """Normalise axis argument to int index 0/1/2."""
    if isinstance(axis, (int, np.integer)):
        idx = int(axis)
    elif isinstance(axis, str):
        mapping = {'x': 0, 'y': 1, 'z': 2}
        if axis not in mapping:
            raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}.")
        idx = mapping[axis]
    else:
        raise TypeError(f"axis must be str or int; got {type(axis)!r}.")
    if idx not in (0, 1, 2):
        raise ValueError(f"axis index must be 0, 1, or 2; got {idx}.")
    return idx


class KLSampler3D:
    """Karhunen-Loève sampler for a Gaussian measure on :class:`Lebesgue3D`.

    Constructs samples from the Gaussian measure whose covariance operator is
    the supplied :class:`~pygeoinf3d.operators.base.SpectralOperator3D` via a
    truncated KL expansion:

    .. math::

        u = m + \\sum_{k=0}^{K-1} \\sqrt{\\lambda_k}\\,\\xi_k\\,\\Phi_k,
        \\quad \\xi_k \\sim N(0, 1),

    where :math:`(\\lambda_k, \\Phi_k)` come from ``operator.get_eigenvalue(k)``
    and ``operator.get_eigenfunction(k)``.  Eigenvalues that are zero or
    negative (e.g. the suppressed Neumann null-space mode of
    :class:`~pygeoinf3d.operators.inverse_laplacian_box.InverseBoxLaplacian`
    in pseudoinverse mode) contribute zero to the expansion because
    :math:`\\sqrt{\\max(\\lambda_k, 0)} = 0`.

    Requires
    --------
    - **Operator:** Any :class:`~pygeoinf3d.operators.base.SpectralOperator3D`
      (used as covariance).
    - **Fast path:** Operator must have a provider with ``has_tensor_product_structure = True``
      for vectorized einsum evaluation (applies to
      :class:`~pygeoinf3d.providers.tensor_product.BoxLaplacianSpectrumProvider`).

    Performance
    -----------
    - Fast path: O(N · p) per sample using vectorized einsum (no Python loop over modes).
    - Loop fallback: O(N · p) per mode call for operators without tensor-product structure.
    - Typical use: sample prior fields before or after inversion.

    Parameters
    ----------
    operator:
        A :class:`~pygeoinf3d.operators.base.SpectralOperator3D` whose
        ``domain`` is a :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D`.
        Typically an :class:`~pygeoinf3d.operators.inverse_laplacian_box.InverseBoxLaplacian`
        acting as the prior covariance.
    n_modes:
        Number of KL modes to retain, :math:`K`.  Must satisfy
        :math:`1 \\le K \\le` ``operator.n_modes``.
    mean:
        Mean function :math:`m`.  If *None* the zero function is used.
    rng:
        NumPy random generator for reproducibility.  If *None* a fresh
        default generator is created.

    Raises
    ------
    TypeError
        If ``operator`` is not a :class:`~pygeoinf3d.operators.base.SpectralOperator3D`.
    ValueError
        If ``n_modes < 1`` or ``n_modes > operator.n_modes``.

    See Also
    --------
    :class:`SpectralGramSolver` : For minimum-norm inversion rather than sampling.
    :func:`~pygeoinf3d.solvers.router.SolverRouter.recommend_sampler` : Automatic sampler selection.

    Examples
    --------
    ::

        kl = KLSampler3D(inv_lap, n_modes=20)
        sample_fn = kl.sample()               # Function3D
        vals = sample_fn(pts)                 # shape (n,) for pts shape (n, 3)

        L = kl.covariance_factor              # LinearOperator R^K → Lebesgue3D
        reconstruction = L(np.eye(K)[0])     # sqrt(lam_0) * phi_0
    """

    def __init__(
        self,
        operator: SpectralOperator3D,
        *,
        n_modes: int,
        mean: Optional[Function3D] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if not isinstance(operator, SpectralOperator3D):
            raise TypeError(
                f"operator must be a SpectralOperator3D; got {type(operator)!r}."
            )
        if n_modes < 1:
            raise ValueError(
                f"n_modes must be >= 1; got {n_modes!r}."
            )
        if n_modes > operator.n_modes:
            raise ValueError(
                f"n_modes={n_modes!r} exceeds operator.n_modes={operator.n_modes!r}."
            )

        self._operator = operator
        self._n_modes = n_modes
        self._space: Lebesgue3D = operator.domain  # type: ignore[assignment]
        self._mean: Function3D = (
            mean if mean is not None else self._space.zero
        )
        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

        # Cache eigenpairs at construction time (n_modes is required/known)
        self._eigenvalues: np.ndarray = np.array(
            [operator.get_eigenvalue(i) for i in range(n_modes)],
            dtype=float,
        )
        self._eigenfunctions: List[Function3D] = [
            operator.get_eigenfunction(i) for i in range(n_modes)
        ]
        # sqrt(max(lam, 0)) — handles zero/suppressed modes gracefully
        self._sqrt_eigenvalues: np.ndarray = np.sqrt(
            np.maximum(self._eigenvalues, 0.0)
        )

        # Detect tensor-product structure for fast-path evaluators
        from pygeoinf3d.providers.tensor_product import BoxLaplacianSpectrumProvider as _TPA
        _tp = getattr(operator, '_provider', None)
        self._tp_provider = _tp if isinstance(_tp, _TPA) else None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_modes(self) -> int:
        """Number of KL modes retained."""
        return self._n_modes

    @property
    def space(self) -> Lebesgue3D:
        """The Lebesgue3D space on which samples live."""
        return self._space

    @property
    def eigenvalues(self) -> np.ndarray:
        """Array of shape *(n_modes,)* with the operator eigenvalues."""
        return self._eigenvalues.copy()

    @property
    def has_tensor_product_structure(self) -> bool:
        """True if the operator is backed by a :class:`BoxLaplacianSpectrumProvider`.

        When True, the fast vectorized evaluators
        :meth:`evaluate_on_regular_grid` and :meth:`evaluate_on_axis_slice`
        are available.
        """
        return self._tp_provider is not None

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self) -> Function3D:
        """Draw a single sample from the Gaussian measure.

        Returns
        -------
        :class:`~pygeoinf3d.core.functions.Function3D`
            The sample :math:`u = m + \\sum_k \\sqrt{\\lambda_k}\\,\\xi_k\\,\\Phi_k`.
        """
        z = self._rng.standard_normal(self._n_modes)
        # weights[k] = sqrt(lam_k) * xi_k
        weights = self._sqrt_eigenvalues * z

        if self.has_tensor_product_structure:
            evaluate = self._build_vectorized_evaluate(weights)
        else:
            mean_fn = self._mean
            snapshot = list(zip(weights.tolist(), self._eigenfunctions))

            def evaluate(x: np.ndarray) -> Union[float, np.ndarray]:  # type: ignore[misc]
                x = np.asarray(x, dtype=float)
                total = mean_fn(x)
                for w_k, phi_k in snapshot:
                    if w_k != 0.0:
                        total = total + w_k * phi_k(x)
                return total

        fn = Function3D(
            self._space.function_domain,
            evaluate_callable=evaluate,
        )
        if self.has_tensor_product_structure:
            _attach_kl_fast_paths(self, z, fn)
        return fn

    def samples(self, n: int) -> List[Function3D]:
        """Draw *n* independent samples.

        Parameters
        ----------
        n:
            Number of samples.

        Returns
        -------
        list of :class:`~pygeoinf3d.core.functions.Function3D`
        """
        return [self.sample() for _ in range(n)]

    def sample_from_xis(self, xis: np.ndarray) -> Function3D:
        """Build a deterministic sample :class:`Function3D` from given xi values.

        Constructs

        .. math::

            u = m + \\sum_{k=0}^{K-1} \\sqrt{\\lambda_k}\\,\\xi_k\\,\\Phi_k

        without drawing new random numbers. Useful for testing and reproducible
        evaluation pipelines.

        Parameters
        ----------
        xis:
            Array of shape ``(n_modes,)`` — the standard-normal coefficients.

        Returns
        -------
        :class:`~pygeoinf3d.core.functions.Function3D`
            The reconstructed sample function.

        Raises
        ------
        ValueError
            If ``xis.shape != (n_modes,)``.
        """
        xis = np.asarray(xis, dtype=float)
        if xis.shape != (self._n_modes,):
            raise ValueError(
                f"xis must have shape ({self._n_modes},); got {xis.shape}."
            )
        weights = self._sqrt_eigenvalues * xis

        if self.has_tensor_product_structure:
            evaluate = self._build_vectorized_evaluate(weights)
        else:
            mean_fn = self._mean
            snapshot = list(zip(weights.tolist(), self._eigenfunctions))

            def evaluate(x: np.ndarray) -> Union[float, np.ndarray]:  # type: ignore[misc]
                x = np.asarray(x, dtype=float)
                total = mean_fn(x)
                for w_k, phi_k in snapshot:
                    if w_k != 0.0:
                        total = total + w_k * phi_k(x)
                return total

        fn = Function3D(self._space.function_domain, evaluate_callable=evaluate)
        if self.has_tensor_product_structure:
            _attach_kl_fast_paths(self, xis, fn)
        return fn

    # ------------------------------------------------------------------
    # Vectorized evaluate builder
    # ------------------------------------------------------------------

    def _build_vectorized_evaluate(self, weights: np.ndarray):
        """Build a fully-vectorized evaluate closure for the tensor-product case.

        Instead of iterating over modes in Python, this closure:

        1. Evaluates all required 1-D basis matrices in one vectorized call per
           axis (via :func:`~pygeoinf3d.providers.tensor_product._eval_1d_basis`).
        2. Uses ``np.einsum`` to compute the weighted mode sum without any
           Python loop, giving a :math:`O(K \\cdot N)` operation where *K* is
           the number of modes and *N* is the number of evaluation points.

        The expansion evaluated is

        .. math::

            u(\\mathbf{x}) = m(\\mathbf{x}) +
                \\sum_{k=0}^{K-1} w_k\\,\\phi_{k_x}(x)\\,\\phi_{k_y}(y)\\,\\phi_{k_z}(z)

        using the factored form of the tensor-product eigenfunctions.

        Only valid when :attr:`has_tensor_product_structure` is True.

        Parameters
        ----------
        weights:
            Array of shape ``(n_modes,)`` containing
            :math:`w_k = \\sqrt{\\lambda_k}\\,\\xi_k`.

        Returns
        -------
        callable
            An evaluate function matching the
            :class:`~pygeoinf3d.core.functions.Function3D` signature.
        """
        from pygeoinf3d.providers.tensor_product import _eval_1d_basis

        provider = self._tp_provider
        n_modes = self._n_modes
        mean_fn = self._mean
        _w = weights.copy()  # snapshot

        # Pre-compute 0-based column indices into the 1-D basis matrices
        # (one-time cost at closure construction, not repeated per eval call)
        bc = provider.bc_type
        kmin = 0 if bc == "neumann" else 1
        ax, ay, az = provider._low
        Lx, Ly, Lz = provider._lengths
        nx, ny, nz = provider._truncation

        multi_indices = [provider.get_multi_index(k) for k in range(n_modes)]
        ix = np.array([kx - kmin for kx, _, __ in multi_indices])
        iy = np.array([ky - kmin for _, ky, __ in multi_indices])
        iz = np.array([kz - kmin for _, __, kz in multi_indices])

        def evaluate(x: np.ndarray) -> Union[float, np.ndarray]:
            x = np.asarray(x, dtype=float)
            is_single = x.ndim == 1
            pts = x[None, :] if is_single else x
            # 1-D basis matrices: shape (n_pts, n_axis_modes)
            Bx = _eval_1d_basis(pts[:, 0], ax, Lx, nx, bc)
            By = _eval_1d_basis(pts[:, 1], ay, Ly, ny, bc)
            Bz = _eval_1d_basis(pts[:, 2], az, Lz, nz, bc)
            # result[p] = sum_k w[k] * Bx[p,ix[k]] * By[p,iy[k]] * Bz[p,iz[k]]
            result = np.einsum("k,pk,pk,pk->p", _w, Bx[:, ix], By[:, iy], Bz[:, iz])
            result += mean_fn(pts)
            return float(result[0]) if is_single else result

        return evaluate

    # ------------------------------------------------------------------
    # Fast tensor-product evaluators
    # ------------------------------------------------------------------

    def evaluate_on_regular_grid(
        self,
        xis: np.ndarray,
        resolution: Union[int, Tuple[int, int, int]] = 25,
        *,
        plot_n_modes: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized KL evaluation on a regular 3-D axis-aligned grid.

        Requires the operator to be backed by a :class:`BoxLaplacianSpectrumProvider`
        (i.e. :attr:`has_tensor_product_structure` is True).  Uses an
        ``einsum``-based tensor-product decomposition that avoids the Python
        loop over grid points used by the generic :class:`Function3D` path.

        The evaluation is numerically identical to
        :meth:`sample_from_xis`\\ ``(xis)(grid_points)`` up to floating-point
        rounding.

        Parameters
        ----------
        xis:
            Standard-normal coefficients, shape ``(n_modes,)``.
        resolution:
            Grid resolution. A positive ``int`` → ``(n, n, n)``; a 3-tuple of
            positive ints ``(Nx, Ny, Nz)`` → asymmetric grid.
        plot_n_modes:
            If given, only the first *plot_n_modes* modes are used (must be in
            ``[1, n_modes]``). The remaining modes in *xis* are ignored.
            Useful for fast approximate visualisation without constructing a
            separate :class:`KLSampler3D`.

        Returns
        -------
        X, Y, Z, values : np.ndarray, each of shape ``(Nx, Ny, Nz)``
            Meshgrid coordinates (``indexing='ij'``) and the evaluated field.

        Raises
        ------
        TypeError
            If the operator does not have tensor-product structure.
        ValueError
            If *plot_n_modes* is out of range.
        """
        if not self.has_tensor_product_structure:
            raise TypeError(
                "evaluate_on_regular_grid requires tensor-product structure. "
                "The operator backing this KLSampler3D does not expose a "
                "BoxLaplacianSpectrumProvider."
            )
        xis = np.asarray(xis, dtype=float)
        n_eff = self._resolve_plot_n_modes(plot_n_modes)

        provider = self._tp_provider
        bbox = provider._box.bounding_box()
        low, high = bbox.low, bbox.high

        if isinstance(resolution, (int, np.integer)):
            Nx = Ny = Nz = int(resolution)
        else:
            Nx, Ny, Nz = int(resolution[0]), int(resolution[1]), int(resolution[2])

        xs = np.linspace(low[0], high[0], Nx)
        ys = np.linspace(low[1], high[1], Ny)
        zs = np.linspace(low[2], high[2], Nz)

        Phi_x, Phi_y, Phi_z = self._build_basis_matrices(xs, ys, zs, n_eff)
        weights = self._sqrt_eigenvalues[:n_eff] * xis[:n_eff]

        # values[i,j,l] = sum_k w[k] * Phi_x[i,k] * Phi_y[j,k] * Phi_z[l,k]
        values = np.einsum('ik,jk,lk,k->ijl', Phi_x, Phi_y, Phi_z, weights)

        # Add mean (evaluated generically on the full grid)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        grid_pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        mean_vals = np.asarray(self._mean(grid_pts), dtype=float).reshape(Nx, Ny, Nz)
        values += mean_vals

        return X, Y, Z, values

    def evaluate_on_axis_slice(
        self,
        xis: np.ndarray,
        axis: Union[int, str] = 2,
        coord: Optional[float] = None,
        resolution: Union[int, Tuple[int, int]] = 100,
        *,
        plot_n_modes: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized KL evaluation on an axis-aligned 2-D slice.

        Requires :attr:`has_tensor_product_structure`.  For a slice fixing
        axis *a* at coordinate *c*, the evaluation collapses the *a*-direction
        1-D basis into a scalar per mode and then evaluates the 2-D sum via
        an ``einsum`` over the two free axes, giving it the same numerical
        result as the generic path but computed in fully vectorized form.

        Parameters
        ----------
        xis:
            Standard-normal coefficients, shape ``(n_modes,)``.
        axis:
            The axis to fix: ``'x'``/``'y'``/``'z'`` or ``0``/``1``/``2``.
        coord:
            Coordinate value along *axis* for the slice.  Defaults to the
            midpoint of the domain's bounding box in that direction.
        resolution:
            Positive ``int`` → ``(n, n)``; 2-tuple ``(Nu, Nv)`` for
            asymmetric grids.
        plot_n_modes:
            Visualization-only mode cap; same semantics as in
            :meth:`evaluate_on_regular_grid`.

        Returns
        -------
        grid_u, grid_v, values : np.ndarray, each of shape ``(Nu, Nv)``
            Meshgrid coordinates (``indexing='ij'``) on the two free axes
            and the evaluated field.

        Raises
        ------
        TypeError
            If the operator does not have tensor-product structure.
        ValueError
            If *plot_n_modes* is out of range.
        """
        if not self.has_tensor_product_structure:
            raise TypeError(
                "evaluate_on_axis_slice requires tensor-product structure. "
                "The operator backing this KLSampler3D does not expose a "
                "BoxLaplacianSpectrumProvider."
            )
        from pygeoinf3d.providers.tensor_product import _make_1d_phi

        xis = np.asarray(xis, dtype=float)
        n_eff = self._resolve_plot_n_modes(plot_n_modes)
        axis_idx = _parse_tp_axis(axis)

        provider = self._tp_provider
        bbox = provider._box.bounding_box()
        low, high = bbox.low, bbox.high

        if coord is None:
            coord = float(0.5 * (low[axis_idx] + high[axis_idx]))

        # The two free axes (in canonical order: the two axes that are not axis_idx)
        u_idx, v_idx = [i for i in range(3) if i != axis_idx]

        if isinstance(resolution, (int, np.integer)):
            Nu = Nv = int(resolution)
        else:
            Nu, Nv = int(resolution[0]), int(resolution[1])

        us = np.linspace(low[u_idx], high[u_idx], Nu)
        vs = np.linspace(low[v_idx], high[v_idx], Nv)

        bc = provider.bc_type
        ax = provider._low
        lengths = provider._lengths

        # Build 1D basis matrices for the two free axes and evaluate fixed axis
        Phi_u = np.empty((Nu, n_eff))
        Phi_v = np.empty((Nv, n_eff))
        phi_fixed = np.empty(n_eff)

        for k in range(n_eff):
            ki = provider.get_multi_index(k)  # (kx, ky, kz)
            Phi_u[:, k] = _make_1d_phi(ki[u_idx], ax[u_idx], lengths[u_idx], bc)(us)
            Phi_v[:, k] = _make_1d_phi(ki[v_idx], ax[v_idx], lengths[v_idx], bc)(vs)
            phi_fixed[k] = float(
                _make_1d_phi(ki[axis_idx], ax[axis_idx], lengths[axis_idx], bc)(coord)
            )

        # Effective weights absorb the fixed-axis contribution
        effective_weights = self._sqrt_eigenvalues[:n_eff] * xis[:n_eff] * phi_fixed

        # values[i,j] = sum_k eff_w[k] * Phi_u[i,k] * Phi_v[j,k]
        values = np.einsum('ik,jk,k->ij', Phi_u, Phi_v, effective_weights)

        # Add mean (evaluated generically on the slice grid)
        grid_u, grid_v = np.meshgrid(us, vs, indexing='ij')
        pts = np.empty((Nu * Nv, 3))
        pts[:, axis_idx] = float(coord)
        pts[:, u_idx] = grid_u.ravel()
        pts[:, v_idx] = grid_v.ravel()
        mean_vals = np.asarray(self._mean(pts), dtype=float).reshape(Nu, Nv)
        values += mean_vals

        return grid_u, grid_v, values

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_plot_n_modes(self, plot_n_modes: Optional[int]) -> int:
        """Validate and return the effective number of modes to use."""
        if plot_n_modes is None:
            return self._n_modes
        n = int(plot_n_modes)
        if n < 1 or n > self._n_modes:
            raise ValueError(
                f"plot_n_modes must be in [1, {self._n_modes}]; got {plot_n_modes!r}."
            )
        return n

    def _build_basis_matrices(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        n_modes: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute 1-D basis matrices for the tensor-product fast path.

        Returns ``Phi_x`` of shape ``(Nx, n_modes)``, ``Phi_y`` of shape
        ``(Ny, n_modes)``, and ``Phi_z`` of shape ``(Nz, n_modes)``, where
        ``Phi_x[i, k] = phi_{kx_k}(xs[i])``, etc.
        """
        from pygeoinf3d.providers.tensor_product import _make_1d_phi

        provider = self._tp_provider
        bc = provider.bc_type
        ax = provider._low
        lengths = provider._lengths

        Phi_x = np.empty((len(xs), n_modes))
        Phi_y = np.empty((len(ys), n_modes))
        Phi_z = np.empty((len(zs), n_modes))

        for k in range(n_modes):
            kx, ky, kz = provider.get_multi_index(k)
            Phi_x[:, k] = _make_1d_phi(kx, ax[0], lengths[0], bc)(xs)
            Phi_y[:, k] = _make_1d_phi(ky, ax[1], lengths[1], bc)(ys)
            Phi_z[:, k] = _make_1d_phi(kz, ax[2], lengths[2], bc)(zs)

        return Phi_x, Phi_y, Phi_z

    # ------------------------------------------------------------------
    # Covariance factor
    # ------------------------------------------------------------------

    @property
    def covariance_factor(self) -> LinearOperator:
        """Covariance factor :math:`L : \\mathbb{R}^K \\to \\text{Lebesgue3D}`.

        The operator satisfies :math:`C \\approx L L^*` (up to KL truncation),
        where :math:`C` is the covariance of the associated Gaussian measure.

        Applying :math:`L` to the *k*-th standard basis vector :math:`e_k`
        yields :math:`\\sqrt{\\lambda_k}\\,\\Phi_k`:

        .. math::

            L(w) = \\sum_{k=0}^{K-1} \\sqrt{\\lambda_k}\\, w_k\\, \\Phi_k

        The adjoint is :math:`L^*(f) = [\\sqrt{\\lambda_k}\\,\\langle f, \\Phi_k
        \\rangle_{L^2}]_k`.

        Returns
        -------
        :class:`~pygeoinf.linear_operators.LinearOperator`
            Factor operator :math:`L`.
        """
        K = self._n_modes
        domain_euc = EuclideanSpace(K)
        codomain = self._space
        sqrt_lam = self._sqrt_eigenvalues.copy()
        phis = list(self._eigenfunctions)
        box = self._space.function_domain

        def mapping(w: np.ndarray) -> Function3D:
            w = np.asarray(w, dtype=float)
            # Snapshot weights for this call: sqrt_lam[k] * w[k]
            mode_weights: List[float] = (sqrt_lam * w).tolist()
            snapshot = list(zip(mode_weights, phis))

            def evaluate(x: np.ndarray) -> Union[float, np.ndarray]:
                x = np.asarray(x, dtype=float)
                is_batch = x.ndim == 2
                total: Union[float, np.ndarray] = (
                    np.zeros(len(x)) if is_batch else 0.0
                )
                for sl_k_w_k, phi_k in snapshot:
                    if sl_k_w_k != 0.0:
                        total = total + sl_k_w_k * phi_k(x)
                return total

            return Function3D(box, evaluate_callable=evaluate)

        def adjoint_mapping(f: Function3D) -> np.ndarray:
            # L*(f)[k] = sqrt(lam_k) * <phi_k, f>_L2
            result = np.zeros(K)
            for k, (sl_k, phi_k) in enumerate(zip(sqrt_lam, phis)):
                if sl_k != 0.0:
                    result[k] = sl_k * codomain.inner_product(phi_k, f)
            return result

        return LinearOperator(
            domain_euc,
            codomain,
            mapping,
            adjoint_mapping=adjoint_mapping,
        )

    def __repr__(self) -> str:
        return (
            f"KLSampler3D("
            f"operator={self._operator!r}, "
            f"n_modes={self._n_modes}"
            f")"
        )
