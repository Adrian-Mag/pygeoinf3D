"""SOLA-style measurement operator for 3D function spaces.

Provides :class:`SOLAOperator3D`, a :class:`~pygeoinf.LinearOperator`
subclass that stacks general linear forms into a finite-dimensional output.
All three mappings (forward, dual, adjoint) are implemented
basis-free on the domain side.

Mathematical definition
-----------------------
Given $m$ linear forms $\\ell_1, \\ldots, \\ell_m$ on the function space
$V$ and an $m$-dimensional codomain $W$, the operator is

.. math::

    A : V \\to W, \\quad (Af)_i = \\ell_i(f), \\quad i = 1, \\ldots, m

Supported form types are
:class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` (volume integrals) and
:class:`~pygeoinf3d.spaces.forms.PathIntegralFunctional` (path integrals).
Mixed stacks (different concrete types in the same operator) are accepted.

Dual mapping
------------
The dual $A^* : W^* \\to V^*$ is given by

.. math::

    A^*(y') = \\sum_{i=1}^m c_i\\, \\ell_i,
    \\qquad c_i = (y')_i,

where $c = $ ``codomain.dual.to_components(y')``.
The result is a :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` whose
kernel is $\\sum_i c_i k_i$.  No component machinery on the domain side is
used.

Adjoint mapping
---------------
The adjoint $A^\\dagger : W \\to V$ is computed as

.. math::

    A^\\dagger(y) = R_V^{-1}\\bigl(A^*(R_W(y))\\bigr),

where $R_W$ is the Riesz map of the codomain (``codomain.to_dual``) and
$R_V^{-1}$ is its domain inverse (``domain.from_dual``).  This is correct
for any finite-dimensional Hilbert codomain, including
:class:`~pygeoinf.MassWeightedHilbertSpace`.

Design constraints
------------------
* The forward map does **not** call
  :meth:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D.to_components` or
  :meth:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D.from_components` on the
  domain side.  Each measurement is a pairing-first dual element evaluated
  directly.
* :meth:`~pygeoinf.LinearOperator.from_linear_forms` is **not** used.
* ``codomain.from_components`` controls how the raw scalar kernel
    evaluations are
  assembled into the output element.
* Kernels may alternatively be provided via a
  :class:`~pygeoinf3d.providers.base.KernelProvider3D` for lazy,
  cached, indexed access (Phase 3).

Usage
-----
Sequence-backed (eager)::

    from pygeoinf3d.operators.sola import SOLAOperator3D
    from pygeoinf3d.spaces.forms import VolumeKernel3D
    from pygeoinf import EuclideanSpace

    space = Lebesgue3D(some_domain)
    codomain = EuclideanSpace(n)
    kernels = [VolumeKernel3D(space, k_i) for k_i in kernel_functions]
    A = SOLAOperator3D(space, codomain, kernels)
    y = A(f)   # element of codomain; np.ndarray for EuclideanSpace

Provider-backed (lazy + cached)::

    from pygeoinf3d.providers.base import FunctionToKernelAdapter
    from pygeoinf3d.providers.tensor_product import BoxLaplacianSpectrumProvider

    provider = BoxLaplacianSpectrumProvider(box, truncation=4, bc_type='dirichlet')
    adapter = FunctionToKernelAdapter(provider, space)
    codomain = EuclideanSpace(adapter.dim)
    A = SOLAOperator3D(space, codomain, adapter)
    y = A(f)
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from pygeoinf import EuclideanSpace
from pygeoinf.linear_forms import LinearForm
from pygeoinf.linear_operators import LinearOperator, MatrixLinearOperator

from pygeoinf3d.providers.base import KernelProvider3D
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.spaces.sobolev import Sobolev3D


class SOLAOperator3D(LinearOperator):
    """SOLA-style linear operator stacking general linear forms into a finite-dimensional output.

    Evaluates each linear form on the input function and assembles the scalar
    outputs into a codomain element via ``codomain.from_components``.

    The *kernels* argument accepts either:

    * A **sequence** of :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D`
      or :class:`~pygeoinf3d.spaces.forms.PathIntegralFunctional` objects
      (eager, sequence-backed mode).  Mixed lists are accepted; or
    * A :class:`~pygeoinf3d.providers.base.KernelProvider3D` instance
      (lazy, provider-backed mode).  Kernels are fetched on demand via
      ``provider.get_kernel(i)`` and cached for stability.

    Requires
    --------
    - **Space:** :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` or
      :class:`~pygeoinf3d.spaces.sobolev.Sobolev3D`.
    - **Forms:** :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D`,
      :class:`~pygeoinf3d.spaces.forms.PathIntegralFunctional`,
      :class:`~pygeoinf3d.spaces.forms.SumLinearForm`, or mixed stacks.
      All forms must satisfy ``form.domain is domain``.
    - **Adjoint:** Requires ``domain.from_dual`` to be implemented;
      :class:`~pygeoinf3d.spaces.sobolev.Sobolev3D` with a
      :class:`~pygeoinf3d.operators.base.SpectralOperator3D` weight provides this.

    Performance
    -----------
    - O(m · N) per forward evaluation (m measurements, N quadrature/mode cost per form).
    - No pre-computation; slower than :class:`~pygeoinf3d.solvers.spectral_gram.SpectralGramSolver`
      for repeated solves on the same problem.  Suitable as a forward operator in
      :class:`~pygeoinf.LinearMinimumNormInversion`.

    Parameters
    ----------
    domain:
        The :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` **or**
        :class:`~pygeoinf3d.spaces.sobolev.Sobolev3D` on which the kernels
        are defined.  All kernel forms in *kernels* must have
        ``kernel.domain is domain`` (identity check).
    codomain:
        A finite-dimensional :class:`~pygeoinf.HilbertSpace` with
        ``codomain.dim == len(kernels)`` (or ``codomain.dim == provider.dim``
        in provider-backed mode).  Only ``from_components`` is exercised by
        the forward mapping.
    kernels:
        Either a sequence of :class:`~pygeoinf.LinearForm` subclass objects
        (e.g. :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` or
        :class:`~pygeoinf3d.spaces.forms.PathIntegralFunctional`, mixed
        stacks accepted), or a
        :class:`~pygeoinf3d.providers.base.KernelProvider3D`.

    Raises
    ------
    TypeError
        If *domain* is not a
        :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` or
        :class:`~pygeoinf3d.spaces.sobolev.Sobolev3D`.
    ValueError
        If kernel count does not match ``codomain.dim``.
    TypeError
        If *kernels* is a sequence and any element is not a
        :class:`~pygeoinf.LinearForm` (checked eagerly for sequences only).
    ValueError
        If *kernels* is a sequence and any kernel has a different
        ``domain`` identity than *domain* (sequence mode only).

    Notes
    -----
    The forward, dual, and adjoint mappings are all basis-free on the
    domain side.  No call to
    :meth:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D.to_components` or
    :meth:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D.from_components` is
    made.  The adjoint is correct for any finite-dimensional Hilbert
    codomain including :class:`~pygeoinf.MassWeightedHilbertSpace`.

    In provider-backed mode, per-kernel validation is deferred to first
    access: ``VolumeKernel3D`` will raise if the provider produces a kernel
    incompatible with *domain*.

    See Also
    --------
    :class:`SpectralGramSolver` : Fast batch solver for PathIntegralFunctional on Sobolev3D.
    :func:`~pygeoinf3d.solvers.router.SolverRouter.recommend` : Automatic solver selection.
    """

    def __init__(
        self,
        domain,
        codomain,
        kernels: Union[Sequence[LinearForm], KernelProvider3D],
    ) -> None:
        if not isinstance(domain, (Lebesgue3D, Sobolev3D)):
            raise TypeError(
                f"SOLAOperator3D requires a Lebesgue3D or Sobolev3D domain, "
                f"got {type(domain).__name__!r}."
            )

        if isinstance(kernels, KernelProvider3D):
            # ---- Provider-backed path (lazy + cached) ----
            if kernels.dim == 0:
                raise ValueError(
                    "SOLAOperator3D requires at least one kernel "
                    "(codomain.dim must be positive)."
                )
            if kernels.dim != codomain.dim:
                raise ValueError(
                    f"Number of kernels ({kernels.dim}) must equal "
                    f"codomain.dim ({codomain.dim})."
                )
            self._provider: KernelProvider3D | None = kernels
            self._kernels: list[LinearForm] | None = None
            self._n_kernels: int = kernels.dim
        else:
            # ---- Sequence-backed path (eager validation) ----
            kernels_list = list(kernels)

            if not kernels_list:
                raise ValueError(
                    "SOLAOperator3D requires at least one kernel "
                    "(codomain.dim must be positive)."
                )
            if len(kernels_list) != codomain.dim:
                raise ValueError(
                    f"Number of kernels ({len(kernels_list)}) must equal "
                    f"codomain.dim ({codomain.dim})."
                )
            for i, kernel in enumerate(kernels_list):
                if not isinstance(kernel, LinearForm):
                    raise TypeError(
                        f"kernels[{i}] is not a VolumeKernel3D or "
                        f"PathIntegralFunctional (LinearForm subclass); "
                        f"got {type(kernel).__name__!r}."
                    )
                if kernel.domain is not domain:
                    raise ValueError(
                        f"kernels[{i}].domain is not the same space instance "
                        "as the operator domain (identity check).  "
                        "Build each kernel form with the exact same "
                        "Lebesgue3D or Sobolev3D object passed as 'domain' here."
                    )
            self._provider = None
            self._kernels = kernels_list
            self._n_kernels = len(kernels_list)

        # Cache for provider-backed mode; unused in sequence-backed mode.
        self._kernel_cache: dict[int, LinearForm] = {}

        # Volumetric fast-path cache (built lazily via build_vol_cache).
        self._vol_cache_built: bool = False
        self._vol_K_w: np.ndarray | None = None
        self._vol_pts: np.ndarray | None = None

        # Path integral fast-path cache (built lazily via build_path_cache).
        self._path_cache_built: bool = False
        self._path_W: np.ndarray | None = None      # (N, n_quad) weight rows
        self._path_pts: np.ndarray | None = None    # (N * n_quad, 3) stacked pts
        self._path_n_quad: int = 0

        # Mixed fast-path cache (built lazily via build_mixed_cache).
        self._mixed_cache_built: bool = False
        self._mixed_vol_idx: np.ndarray | None = None   # (N_v,) original indices
        self._mixed_vol_K_w: np.ndarray | None = None   # (N_v, n_vol_pts)
        self._mixed_vol_pts: np.ndarray | None = None   # (n_vol_pts, 3)
        self._mixed_path_idx: np.ndarray | None = None  # (N_p,) original indices
        self._mixed_path_W: np.ndarray | None = None    # (N_p, n_quad)
        self._mixed_path_pts: np.ndarray | None = None  # (N_p * n_quad, 3)
        self._mixed_path_n_quad: int = 0

        # Cached fast path for G @ G.adjoint in Euclidean codomain.
        self._gram_adjoint_operator: MatrixLinearOperator | None = None

        # Cache for cross-Gram operators: T @ G.adjoint where T != G.
        # Key: id(other_sola), value: MatrixLinearOperator(G.codomain, T.codomain, ...).
        self._cross_gram_cache: dict[int, MatrixLinearOperator] = {}

        super().__init__(
            domain,
            codomain,
            self._forward,
            dual_mapping=self._dual_mapping,
            adjoint_mapping=self._adjoint_mapping,
        )

    # ------------------------------------------------------------------
    # Kernel access (with caching for provider-backed mode)
    # ------------------------------------------------------------------

    def _get_kernel(self, i: int) -> LinearForm:
        """Return the *i*-th kernel form, using the cache in provider-backed mode.

        In sequence-backed mode returns ``self._kernels[i]`` directly.
        In provider-backed mode, checks ``self._kernel_cache`` first,
        calls ``self._provider.get_kernel(i)`` on a miss, then caches
        and returns the result.

        Args:
            i: Index in ``range(self._n_kernels)``.

        Returns:
            The :class:`~pygeoinf.LinearForm` at index *i*.
        """
        if self._provider is None:
            # Sequence-backed: direct access, no caching needed.
            return self._kernels[i]  # type: ignore[index]
        if i not in self._kernel_cache:
            self._kernel_cache[i] = self._provider.get_kernel(i)
        return self._kernel_cache[i]

    def _all_kernels(self) -> list[LinearForm]:
        """Return all realized kernel forms in operator order."""
        return [self._get_kernel(i) for i in range(self._n_kernels)]

    def _try_build_fast_gram_matrix(self) -> np.ndarray | None:
        """Return a dense Gram matrix for ``G @ G.adjoint`` when safe.

        Fast path is enabled only in two cases:

          1. ``Sobolev3D`` with spectral ``inverse_weight`` and forms supported
              by :class:`~pygeoinf3d.solvers.spectral_gram.SpectralGramSolver`
              (including mixed ``VolumeKernel3D`` +
              ``PathIntegralFunctional`` stacks).
          2. ``Lebesgue3D`` with an all-``VolumeKernel3D`` stack on a
              ``BoxDomain``.

        Returns ``None`` when the operator does not meet these strict criteria,
        so the generic operator-composition route is used unchanged.
        """
        forms = self._all_kernels()
        if not forms:
            return None

        from pygeoinf3d.spaces.forms import (
            PathIntegralFunctional,
            VolumeKernel3D,
        )

        # Spectral Sobolev fast path — supports mixed volume/path stacks.
        if isinstance(self.domain, Sobolev3D):
            if not all(
                isinstance(form, (VolumeKernel3D, PathIntegralFunctional))
                for form in forms
            ):
                return None
            try:
                from pygeoinf3d.solvers.spectral_gram import SpectralGramSolver

                return SpectralGramSolver(self.domain, forms).gram_matrix
            except (TypeError, ValueError, AttributeError):
                return None

        # Structured Lebesgue all-volume fast path.
        if isinstance(self.domain, Lebesgue3D):
            if not all(isinstance(form, VolumeKernel3D) for form in forms):
                return None
            volume_forms = [
                form for form in forms if isinstance(form, VolumeKernel3D)
            ]

            from pygeoinf3d.geometries.box.domain import BoxDomain

            func_domain = self.domain.function_domain
            if not isinstance(func_domain, BoxDomain):
                return None

            pts = func_domain.structured_mesh(20, location="cell_centers")
            weights = func_domain.volume_weights(20, location="cell_centers")
            K = np.vstack(
                [
                    np.asarray(form.kernel(pts), dtype=float).ravel()
                    for form in volume_forms
                ]
            )
            return (K * weights[None, :]) @ K.T

        return None

    def _try_build_fast_cross_gram_operator(
        self, other_sola: "SOLAOperator3D"
    ) -> MatrixLinearOperator:
        """Return (cached) cross-Gram operator for ``self @ other_sola.adjoint``.

        Computes :math:`C_{TG} = L_T \\operatorname{diag}(\\nu) L_G^\\top` where
        :math:`L_T` and :math:`L_G` are the spectral interaction matrices for
        *self* and *other_sola* respectively, and :math:`\\nu` are the shared
        inverse-weight eigenvalues.

        The result is cached by ``id(other_sola)``.

        Args:
            other_sola: The right-hand :class:`SOLAOperator3D` (the "G" in
                ``T @ G.adjoint``).

        Returns:
            A :class:`~pygeoinf.linear_operators.MatrixLinearOperator` mapping
            ``other_sola.codomain → self.codomain`` with ``galerkin=True``.

        Raises:
            TypeError: If the domain's ``inverse_weight`` is not a
                :class:`~pygeoinf3d.operators.base.SpectralOperator3D`.
        """
        key = id(other_sola)
        if key in self._cross_gram_cache:
            return self._cross_gram_cache[key]

        from pygeoinf3d.solvers.spectral_gram import SpectralGramSolver

        forms_self = [self._get_kernel(i) for i in range(self._n_kernels)]
        forms_other = [other_sola._get_kernel(i) for i in range(other_sola._n_kernels)]

        L_T, nu = SpectralGramSolver._build_L_matrix(self.domain, forms_self)
        L_G, _ = SpectralGramSolver._build_L_matrix(other_sola.domain, forms_other)

        # C_TG[k, i] = Σ_j L_T[k,j] · ν_j · L_G[i,j]  →  shape (N_p, N_d)
        C_TG = (L_T * nu[None, :]) @ L_G.T

        op = MatrixLinearOperator(
            other_sola.codomain,
            self.codomain,
            C_TG,
            galerkin=True,
        )
        self._cross_gram_cache[key] = op
        return op

    def _try_build_fast_gram_operator(self) -> MatrixLinearOperator | None:
        """Return cached matrix-backed operator for ``G @ G.adjoint``.

        Returns ``None`` when the strict fast-path conditions are not met.
        """
        if self._gram_adjoint_operator is not None:
            return self._gram_adjoint_operator

        gram = self._try_build_fast_gram_matrix()
        if gram is None:
            return None

        # For EuclideanSpace, the standard and Galerkin matrices coincide.
        op = MatrixLinearOperator(
            self.codomain,
            self.codomain,
            gram,
            galerkin=True,
        )
        self._gram_adjoint_operator = op
        return op

    def __matmul__(self, other):
        """Compose operators with a specialized ``G @ G.adjoint`` fast path.

        The fast path is activated only when:

        - ``other`` is exactly the adjoint of ``self`` (checked via
          ``other._adjoint_base is self``),
        - the codomain is :class:`~pygeoinf.EuclideanSpace`, and
        - one of the strict Gram-matrix construction paths succeeds.

        Otherwise this method delegates to the generic
        :class:`~pygeoinf.LinearOperator` composition implementation.
        """
        if isinstance(other, LinearOperator) and isinstance(self.codomain, EuclideanSpace):
            adj_base = getattr(other, "_adjoint_base", None)

            # --- Self-Gram fast path: T @ T.adjoint ---
            if (
                adj_base is self
                and other.domain == self.codomain
                and other.codomain == self.domain
            ):
                fast_op = self._try_build_fast_gram_operator()
                if fast_op is not None:
                    return fast_op

            # --- Cross-Gram fast path: T @ G.adjoint (G != T, Sobolev3D domain) ---
            elif (
                isinstance(adj_base, SOLAOperator3D)
                and adj_base is not self
                and isinstance(self.domain, Sobolev3D)
                and adj_base.domain is self.domain
                and isinstance(adj_base.codomain, EuclideanSpace)
            ):
                return self._try_build_fast_cross_gram_operator(adj_base)

        return super().__matmul__(other)

    # ------------------------------------------------------------------
    # Volumetric fast-path cache
    # ------------------------------------------------------------------

    def build_vol_cache(
        self,
        resolution: int = 20,
        *,
        location: str = "cell_centers",
    ) -> None:
        """Pre-compute the kernel matrix for fast repeated forward evaluations.

        When all forms are :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D`,
        this method evaluates every kernel on a shared structured quadrature
        mesh **once** and stores the result as a pre-weighted matrix

        .. math::

            K_w[i, j] = k_i(\\mathbf{x}_j) \\cdot w_j

        of shape ``(n_forms, n_pts)``.  After calling this method every
        subsequent ``G(f)`` call reduces to:

        .. code-block:: text

            f_vals = f(pts)          # one batch evaluation
            raw    = K_w @ f_vals   # one matrix-vector product

        instead of ``n_forms`` separate ``∫ k_i · f dV`` integrations each
        of which would re-evaluate ``k_i`` on the mesh.

        Parameters
        ----------
        resolution:
            Quadrature grid resolution in each dimension (default ``20``,
            giving a ``20³ = 8000``-point mesh).
        location:
            ``'cell_centers'`` (default) or ``'nodes'``.  Passed to
            :meth:`~pygeoinf3d.geometries.box.domain.BoxDomain.structured_mesh`.

        Raises
        ------
        TypeError
            If any form is not a
            :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D`.
        TypeError
            If the domain's ``function_domain`` is not a
            :class:`~pygeoinf3d.geometries.box.domain.BoxDomain`.

        Notes
        -----
        Quadrature points and weights are fetched from
        :class:`~pygeoinf3d.geometries.box.domain.BoxDomain`'s own cache
        (``structured_mesh`` / ``volume_weights``) so there is no extra mesh
        allocation if the domain was already queried at the same resolution.

        Call ``build_vol_cache`` again with a different *resolution* to
        replace the existing cache.
        """
        from pygeoinf3d.spaces.forms import VolumeKernel3D
        # Validate that all forms are VolumeKernel3D, then delegate
        # kernel evaluation to each form's own build_cache().
        # VolumeKernel3D.build_cache raises TypeError if the domain does
        # not provide structured_mesh / volume_weights.
        for i in range(self._n_kernels):
            k = self._get_kernel(i)
            if not isinstance(k, VolumeKernel3D):
                raise TypeError(
                    f"build_vol_cache: forms[{i}] is a "
                    f"{type(k).__name__!r}, not a VolumeKernel3D.  "
                    "The fast-path cache is only available when all forms "
                    "are VolumeKernel3D.  Use the standard evaluation path "
                    "for mixed or PathIntegralFunctional stacks."
                )
            k.build_cache(resolution, location=location)

        # Assemble the operator-level matrix from per-form cached state.
        self._vol_K_w = np.vstack(
            [self._get_kernel(i)._cache_weighted_kernel
             for i in range(self._n_kernels)]
        )
        self._vol_pts = self._get_kernel(0)._cache_pts
        self._vol_cache_built = True

    def build_path_cache(self) -> None:
        """Pre-compute stacked quadrature data for fast repeated forward evaluations.

        When all forms are :class:`~pygeoinf3d.spaces.forms.PathIntegralFunctional`
        with the **same** ``n_quad``, this method stacks every path's cached
        quadrature points into one array and precomputes the trapezoid weight
        row for each path.  After calling this method every subsequent ``G(f)``
        call reduces to:

        .. code-block:: text

            # single vectorised batch eval over all N * n_quad points
            f_vals_2d = f(all_pts).reshape(N, n_quad)
            # one element-wise multiply + row sum  (no Python loop, no trapz call)
            raw = (W * f_vals_2d).sum(axis=1)

        instead of ``N`` separate ``f(pts_i)`` calls each followed by
        ``np.trapezoid``.

        **Why this is faster:**  The N Python-level function-call dispatches to
        ``f.__call__`` become one, and the N individual trapezoid contractions
        become a single NumPy einsum.  The total number of floating-point
        operations is unchanged; the saving is in Python overhead and cache
        locality.

        Raises
        ------
        TypeError
            If any form is not a
            :class:`~pygeoinf3d.spaces.forms.PathIntegralFunctional`.
        ValueError
            If the forms do not all share the same ``n_quad``.  Rebuild each
            form with a uniform ``n_quad`` value to use this cache.

        Notes
        -----
        The quadrature weight rows are read directly from each form's
        ``_quad_weights`` attribute (pre-computed at
        :class:`~pygeoinf3d.spaces.forms.PathIntegralFunctional` construction
        time), so ``build_path_cache`` itself performs no arithmetic beyond
        stacking two matrices.

        Call ``build_path_cache`` again to refresh the cache (e.g. if you
        replace forms after construction).
        """
        from pygeoinf3d.spaces.forms import PathIntegralFunctional

        # Validate: all forms must be PathIntegralFunctional.
        for i in range(self._n_kernels):
            k = self._get_kernel(i)
            if not isinstance(k, PathIntegralFunctional):
                raise TypeError(
                    f"build_path_cache: forms[{i}] is a "
                    f"{type(k).__name__!r}, not a PathIntegralFunctional.  "
                    "The path fast-path cache requires all forms to be "
                    "PathIntegralFunctional.  Use build_vol_cache for "
                    "VolumeKernel3D stacks, or leave the cache unbuilt for "
                    "mixed stacks."
                )

        # Validate: uniform n_quad.
        n_quad_vals = [self._get_kernel(i).n_quad for i in range(self._n_kernels)]
        if len(set(n_quad_vals)) != 1:
            raise ValueError(
                "build_path_cache requires all PathIntegralFunctional forms "
                f"to share the same n_quad; got distinct values: "
                f"{sorted(set(n_quad_vals))}.  "
                "Reconstruct the forms with a uniform n_quad= argument."
            )
        n_quad = n_quad_vals[0]

        # Stack all quadrature points and weight rows directly from each form's
        # pre-computed attributes (set at PathIntegralFunctional construction).
        all_pts = np.vstack(
            [self._get_kernel(i)._pts for i in range(self._n_kernels)]
        )
        W = np.vstack(
            [self._get_kernel(i)._quad_weights for i in range(self._n_kernels)]
        )

        self._path_W = W
        self._path_pts = all_pts
        self._path_n_quad = n_quad
        self._path_cache_built = True

    def build_mixed_cache(self, resolution: int = 20) -> None:
        """Pre-compute fast-path caches for any mix of volume and path forms.

        Handles stacks that contain **any combination** of
        :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` and
        :class:`~pygeoinf3d.spaces.forms.PathIntegralFunctional` forms —
        including all-volume and all-path stacks.  This is the recommended
        single-call alternative to calling :meth:`build_vol_cache` or
        :meth:`build_path_cache` when the form types are not known in advance.

        **What it precomputes:**

        * **Volume forms** — same as :meth:`build_vol_cache`: a pre-weighted
          kernel matrix ``K_w[i,j] = k_i(x_j) * w_j`` on a structured mesh.
        * **Path forms** — same as :meth:`build_path_cache`: stacked
          quadrature points and trapezoid weight rows.

        **Forward pass after calling this method:**

        .. code-block:: text

            # At most two batch function evaluations:
            f(vol_pts)           → K_w_vol  @ f_vals_vol   for volume rows
            f(all_path_pts)      → einsum W for path rows
            # Results are assembled back in original form order.

        Parameters
        ----------
        resolution:
            Quadrature grid resolution for volume forms (default ``20``,
            giving ``20³ = 8000`` points).  Ignored if there are no volume
            forms.

        Raises
        ------
        TypeError
            If any form is neither a ``VolumeKernel3D`` nor a
            ``PathIntegralFunctional``.
        TypeError
            If there are volume forms but the domain's ``function_domain``
            is not a :class:`~pygeoinf3d.geometries.box.domain.BoxDomain`.
        ValueError
            If the path forms do not all share the same ``n_quad``.  Rebuild
            each path form with a uniform ``n_quad=`` argument.

        Notes
        -----
        Call ``build_mixed_cache`` again with a different *resolution* to
        replace the existing cache.
        """
        from pygeoinf3d.spaces.forms import VolumeKernel3D, PathIntegralFunctional

        # Classify forms into vol / path groups.
        vol_idx_list: list[int] = []
        path_idx_list: list[int] = []
        for i in range(self._n_kernels):
            k = self._get_kernel(i)
            if isinstance(k, VolumeKernel3D):
                vol_idx_list.append(i)
            elif isinstance(k, PathIntegralFunctional):
                path_idx_list.append(i)
            else:
                raise TypeError(
                    f"build_mixed_cache: forms[{i}] is a "
                    f"{type(k).__name__!r}; only VolumeKernel3D and "
                    "PathIntegralFunctional are supported by this cache."
                )

        vol_idx = np.array(vol_idx_list, dtype=int)
        path_idx = np.array(path_idx_list, dtype=int)

        # --- Build volume sub-cache ---
        if vol_idx.size > 0:
            # Delegate to each form's own build_cache().
            # Raises TypeError if the domain is unstructured.
            for i in vol_idx:
                self._get_kernel(i).build_cache(resolution)
            K_w = np.vstack(
                [self._get_kernel(i)._cache_weighted_kernel for i in vol_idx]
            )
            pts_vol = self._get_kernel(vol_idx[0])._cache_pts
        else:
            pts_vol = None
            K_w = None

        # --- Build path sub-cache ---
        if path_idx.size > 0:
            n_quad_vals = [
                self._get_kernel(i).n_quad for i in path_idx
            ]
            if len(set(n_quad_vals)) != 1:
                raise ValueError(
                    "build_mixed_cache: path forms do not all share the same "
                    f"n_quad; got distinct values: {sorted(set(n_quad_vals))}. "
                    "Reconstruct the path forms with a uniform n_quad= argument."
                )
            n_quad = n_quad_vals[0]
            all_path_pts = np.vstack(
                [self._get_kernel(i)._pts for i in path_idx]
            )
            # Weight rows come directly from each form's pre-computed attribute.
            W_path = np.vstack(
                [self._get_kernel(i)._quad_weights for i in path_idx]
            )
        else:
            all_path_pts = None
            W_path = None
            n_quad = 0

        # Store.
        self._mixed_vol_idx = vol_idx
        self._mixed_vol_K_w = K_w
        self._mixed_vol_pts = pts_vol
        self._mixed_path_idx = path_idx
        self._mixed_path_W = W_path
        self._mixed_path_pts = all_path_pts
        self._mixed_path_n_quad = n_quad
        self._mixed_cache_built = True

    # ------------------------------------------------------------------
    # Internal mappings
    # ------------------------------------------------------------------

    def _forward(self, f) -> object:
        """Evaluate each kernel on *f*.

        Uses the pre-computed volumetric cache (``K_w @ f(pts)``) when
        :meth:`build_vol_cache` has been called, otherwise falls back to the
        standard per-form evaluation loop.

        Args:
            f: A :class:`~pygeoinf3d.core.functions.Function3D` from the
               domain space.

        Returns:
            An element of the codomain, constructed via
            ``codomain.from_components(raw_components)`` where
            ``raw_components[i] = kernels[i](f)``.
        """
        if self._vol_cache_built:
            # Volumetric fast path: single batch eval + matmul.
            f_vals = np.asarray(f(self._vol_pts), dtype=float).ravel()
            raw = self._vol_K_w @ f_vals
        elif self._path_cache_built:
            # Path fast path: one batch eval over all N*n_quad pts,
            # then a single einsum contraction (no Python loop, no trapz calls).
            f_vals_2d = np.asarray(
                f(self._path_pts), dtype=float
            ).reshape(self._n_kernels, self._path_n_quad)
            raw = np.einsum("ij,ij->i", self._path_W, f_vals_2d)
        elif self._mixed_cache_built:
            # Mixed fast path: at most two batch function evaluations,
            # one for volume rows and one for path rows.
            raw = np.empty(self._n_kernels, dtype=float)
            if self._mixed_vol_idx.size > 0:
                f_vals_vol = np.asarray(
                    f(self._mixed_vol_pts), dtype=float
                ).ravel()
                raw[self._mixed_vol_idx] = self._mixed_vol_K_w @ f_vals_vol
            if self._mixed_path_idx.size > 0:
                n_path = self._mixed_path_idx.size
                f_vals_path_2d = np.asarray(
                    f(self._mixed_path_pts), dtype=float
                ).reshape(n_path, self._mixed_path_n_quad)
                raw[self._mixed_path_idx] = np.einsum(
                    "ij,ij->i", self._mixed_path_W, f_vals_path_2d
                )
        else:
            raw = np.array(
                [self._get_kernel(i)(f) for i in range(self._n_kernels)],
                dtype=float,
            )
        return self.codomain.from_components(raw)

    def _dual_mapping(self, yp) -> LinearForm:
        """Dual map $A^*: W^* \\to V^*$.

        Extracts codomain-dual coordinates and returns the corresponding
        weighted sum of the kernel-induced forms.

        Return type depends on the form stack:

        * **Single form:** ``float(c[0]) * \\ell_0`` via the form's own
          ``__mul__``.
        * **All VolumeKernel3D:** a single
          :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` built by direct
          addition (backward-compatible; enables
          :meth:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D.from_dual`).
        * **Otherwise (mixed types or path stacks with n>1):** a
          :class:`~pygeoinf3d.spaces.forms.SumLinearForm` wrapping the
          weighted terms, consumed by
          :meth:`~pygeoinf3d.spaces.sobolev.Sobolev3D.from_dual`.

        Args:
            yp: An element of the codomain dual space (a
                :class:`~pygeoinf.LinearForm` on the codomain).

        Returns:
            A :class:`~pygeoinf.LinearForm` equal to
            $\\sum_i c_i \\ell_i$ where $c_i$ are the dual coordinates of *yp*.
        """
        c = self.codomain.dual.to_components(yp)
        if self._n_kernels == 1:
            return float(c[0]) * self._get_kernel(0)
        # For all-VolumeKernel3D stacks, use direct addition
        # (returns a single VolumeKernel3D — backward compatible)
        from pygeoinf3d.spaces.forms import VolumeKernel3D
        if all(isinstance(self._get_kernel(i), VolumeKernel3D)
               for i in range(self._n_kernels)):
            result = float(c[0]) * self._get_kernel(0)
            for i in range(1, self._n_kernels):
                result = result + float(c[i]) * self._get_kernel(i)
            return result
        # General case: wrap in SumLinearForm on the operator domain
        from pygeoinf3d.spaces.forms import SumLinearForm
        terms = [(float(c[i]), self._get_kernel(i))
                 for i in range(self._n_kernels)]
        return SumLinearForm(self.domain, terms)

    def _adjoint_mapping(self, y):
        """Adjoint map $A^\\dagger: W \\to V$.

        Applies the codomain Riesz map, the dual mapping, then the domain
        inverse-Riesz map.  This succeeds only when the dual result is a
        :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` (i.e. when the
        measurement forms are all ``VolumeKernel3D``).

        When path-integral forms are used, the domain must be a
        :class:`~pygeoinf3d.spaces.sobolev.Sobolev3D` whose *inverse_weight*
        is a :class:`~pygeoinf3d.operators.base.SpectralOperator3D` (e.g.
        ``BesselSobolevBox``).  If the ``Sobolev3D`` uses a non-spectral
        weight, ``from_dual`` raises :exc:`TypeError` with guidance.

        Args:
            y: An element of the codomain.

        Returns:
            A :class:`~pygeoinf3d.core.functions.Function3D` in the domain.

        Raises:
            TypeError:
                If the dual result is not a
                :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` (e.g. when
                path-integral forms are stacked against a plain
                :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` domain).
        """
        yp = self.codomain.to_dual(y)
        xp = self._dual_mapping(yp)
        return self.domain.from_dual(xp)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def kernels(self) -> list:
        """The list of measurement forms.

        In sequence-backed mode returns a copy of the stored list.
        In provider-backed mode, realizes all kernels on demand (with caching)
        and returns them as a list.  Repeated calls return identity-equal
        kernel objects because the cache is shared.

        Elements are :class:`~pygeoinf.LinearForm` subclass instances
        (:class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` in sequence or
        provider mode, or
        :class:`~pygeoinf3d.spaces.forms.PathIntegralFunctional` in
        sequence mode).

        Returns a new list object each call, so external mutation is safe.
        """
        return [self._get_kernel(i) for i in range(self._n_kernels)]
