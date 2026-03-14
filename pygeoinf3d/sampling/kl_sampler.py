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
    from pygeoinf3d.providers.tensor_product import TensorProductProvider
    from pygeoinf3d.spaces.lebesgue import Lebesgue3D
    from pygeoinf3d.operators.inverse_laplacian_box import InverseBoxLaplacian
    from pygeoinf3d.sampling import KLSampler3D
    from pygeoinf import GaussianMeasure
    import numpy as np

    box      = BoxDomain(np.zeros(3), np.ones(3))
    space    = Lebesgue3D(box)
    provider = TensorProductProvider(box, truncation=(5, 5, 5), bc_type='dirichlet')
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

from typing import List, Optional, Union

import numpy as np

from pygeoinf import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator

from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.operators.base import SpectralOperator3D
from pygeoinf3d.spaces.lebesgue import Lebesgue3D


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
        weights: List[float] = (self._sqrt_eigenvalues * z).tolist()

        mean_fn = self._mean
        snapshot = list(zip(weights, self._eigenfunctions))

        def evaluate(x: np.ndarray) -> Union[float, np.ndarray]:
            x = np.asarray(x, dtype=float)
            total = mean_fn(x)
            for w_k, phi_k in snapshot:
                if w_k != 0.0:
                    total = total + w_k * phi_k(x)
            return total

        return Function3D(
            self._space.function_domain,
            evaluate_callable=evaluate,
        )

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
