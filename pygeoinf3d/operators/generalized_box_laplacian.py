"""Integer-power generalisation of the box Laplacian: $(-\\alpha \\Delta)^s$.

Provides :class:`GeneralizedBoxLaplacian`, a
:class:`~pygeoinf3d.operators.spectral_power_base.SpectralPowerBase3D`
representing the spectral operator $(-\\alpha \\Delta)^s$ for integer $s$
(``power``) and positive scale $\\alpha$ (``alpha``).

Mathematical definition
-----------------------
Given orthonormal eigenfunction-eigenvalue pairs $(\\Phi_k, \\lambda_k)$ with
$-\\Delta \\Phi_k = \\lambda_k \\Phi_k$, the generalised operator acts as

.. math::

    ((-\\alpha\\Delta)^s f)(\\mathbf{x})
        = \\sum_{k=0}^{N-1} (\\alpha\\lambda_k)^s
            \\langle f, \\Phi_k \\rangle_{L^2}\\, \\Phi_k(\\mathbf{x})

where $(\\alpha\\lambda_k)^s$ is defined as follows:

- ``s = 0``: operator eigenvalue is ``1`` for every mode (identity).
- ``s > 0``: operator eigenvalue is ``(alpha * lambda_k) ** s``.
- ``s < 0``: operator eigenvalue is ``(alpha * lambda_k) ** s``, but modes
  satisfying ``|lambda_k| < eigenvalue_tol`` are projected out (pseudoinverse
  / mean-zero convention).

Public convention
-----------------
Powers are taken of the *positive* operator $(-\\alpha\\Delta)$, not the
raw $\\Delta$.  This ensures the operator is positive-semidefinite for
$\\alpha > 0$ and all Neumann/Dirichlet eigenvalues $\\lambda_k \\ge 0$.

Special cases
-------------
- ``GeneralizedBoxLaplacian(space, provider, power=1, alpha=1.0)`` implements
  the same operator as :class:`~pygeoinf3d.operators.laplacian_box.BoxLaplacian`.
- ``GeneralizedBoxLaplacian(space, provider, power=-1, alpha=1.0)`` implements
  the same eigenvalue semantics as
  :class:`~pygeoinf3d.operators.inverse_laplacian_box.InverseBoxLaplacian`
  with ``null_space_handling='pseudoinverse'``.

Implementation status
---------------------
Full implementation: eigenvalue access, eigenfunction access, and the
full spectral projection :meth:`_apply` are all implemented.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.operators.spectral_power_base import SpectralPowerBase3D
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.providers.tensor_product import BoxLaplacianSpectrumProvider


class GeneralizedBoxLaplacian(SpectralPowerBase3D):
    """Spectral operator $(-\\alpha \\Delta)^s$ on a box domain.

    Represents integer powers of the positive box Laplacian.  The operator
    eigenvalues are $(\\alpha \\lambda_k)^s$ where $\\lambda_k$ are the
    Laplacian eigenvalues provided by a
    :class:`~pygeoinf3d.providers.tensor_product.BoxLaplacianSpectrumProvider`.

    For negative powers, modes with ``|lambda_k| < eigenvalue_tol`` are
    suppressed to zero (pseudoinverse / mean-zero projection).

    Parameters
    ----------
    space:
        The :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` on the box domain.
        Both the operator's domain and codomain are this space.
    provider:
        A :class:`~pygeoinf3d.providers.tensor_product.BoxLaplacianSpectrumProvider`
        whose box domain matches ``space.function_domain``.
    power:
        Integer power $s$.  ``power=0`` gives the identity, ``power=1`` is
        the Laplacian, ``power=-1`` is the inverse Laplacian, etc.
    alpha:
        Positive scale factor for the operator $(-\\alpha \\Delta)$.
        Default ``1.0``.
    eigenvalue_tol:
        Threshold below which a Laplacian eigenvalue is treated as zero for
        pseudoinverse handling when ``power < 0``. Default ``1e-10``.
    quad_resolution:
        Resolution passed to
        :meth:`~pygeoinf3d.geometries.box.domain.BoxDomain.structured_mesh`
        when computing the $L^2$ projections.  Higher values give more
        accurate coefficients.  Default ``20``.

    Raises
    ------
    TypeError
        If ``power`` is not an integer (e.g. ``power=0.5``).
    ValueError
        If ``alpha <= 0``.
    ValueError
        If ``provider``'s box domain differs from ``space.function_domain``.

    Examples
    --------
    ::

        from pygeoinf3d.geometries.box.domain import BoxDomain
        from pygeoinf3d.providers.tensor_product import BoxLaplacianSpectrumProvider
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        from pygeoinf3d.operators.generalized_box_laplacian import GeneralizedBoxLaplacian
        import numpy as np

        box      = BoxDomain(np.zeros(3), np.ones(3))
        space    = Lebesgue3D(box)
        provider = BoxLaplacianSpectrumProvider(box, truncation=(4, 4, 4), bc_type='dirichlet')

        # Power=2 Laplacian: eigenvalues = lambda_k^2, full apply supported
        op = GeneralizedBoxLaplacian(space, provider, power=2)
        phi0 = provider.get_eigenfunction(0)
        result = op(phi0)   # ≈ lambda_0^2 * phi0 pointwise

        # Identity operator
        ident = GeneralizedBoxLaplacian(space, provider, power=0)
        ev0 = ident.get_eigenvalue(0)  # = 1.0
    """

    def __init__(
        self,
        space: Lebesgue3D,
        provider: BoxLaplacianSpectrumProvider,
        *,
        power: int,
        alpha: float = 1.0,
        eigenvalue_tol: float = 1e-10,
        quad_resolution: int = 20,
    ) -> None:
        # --- Validate power: must be an integer type ---
        if not isinstance(power, (int, np.integer)):
            raise TypeError(
                f"power must be an integer; got {type(power).__name__!r} "
                f"with value {power!r}. Only integer powers are supported."
            )

        # --- Validate alpha ---
        if alpha <= 0:
            raise ValueError(
                f"alpha must be strictly positive; got alpha={alpha!r}. "
                "The operator convention is powers of the positive operator "
                "(-alpha * Delta), so alpha > 0 is required."
            )

        # --- Validate box identity ---
        if space.function_domain is not provider._box:
            raise ValueError(
                "The provider's box domain must be the same object as "
                "space.function_domain."
            )

        self._space = space
        self._provider = provider
        self._power = int(power)
        self._alpha = float(alpha)
        self._eigenvalue_tol = float(eigenvalue_tol)
        self._quad_resolution = quad_resolution

        # Build operator eigenvalues using the canonical convention
        laplacian_eigs = [
            float(provider.get_eigenvalue(i)) for i in range(provider.dim)
        ]
        op_eigenvalues = SpectralPowerBase3D.compute_power_eigenvalues(
            laplacian_eigs,
            power=self._power,
            alpha=self._alpha,
            eigenvalue_tol=self._eigenvalue_tol,
        )

        # Pre-fetch eigenfunctions to avoid repeated provider calls
        eigenfunctions: list[Function3D] = [
            provider.get_eigenfunction(i) for i in range(provider.dim)
        ]

        # Pre-compute quadrature mesh and weights for projection integrals
        box = provider._box
        self._pts = box.structured_mesh(quad_resolution)
        self._weights = box.volume_weights(quad_resolution)

        super().__init__(space, op_eigenvalues, eigenfunctions)

    # ------------------------------------------------------------------
    # Properties (power / alpha / eigenvalue_tol are GeneralizedBoxLaplacian
    # specifics; get_eigenvalue/get_eigenfunction/n_modes are inherited)
    # ------------------------------------------------------------------

    @property
    def power(self) -> int:
        """The integer power $s$ of the operator $(-\\alpha \\Delta)^s$."""
        return self._power

    @property
    def alpha(self) -> float:
        """The positive scale factor $\\alpha$ of the operator $(-\\alpha \\Delta)^s$."""
        return self._alpha

    @property
    def eigenvalue_tol(self) -> float:
        """Threshold for near-zero eigenvalue detection (pseudoinverse mode)."""
        return self._eigenvalue_tol

    # ------------------------------------------------------------------
    # Spectral projection apply — box quadrature
    # ------------------------------------------------------------------

    def _apply(self, f: Function3D) -> Function3D:
        """Apply the spectral operator $(-\\alpha\\Delta)^s$ to *f*.

        Computes $L^2$ projection coefficients
        $c_k = \\langle f, \\Phi_k \\rangle_{L^2}$ using the pre-computed
        box quadrature, then synthesises the result as

        .. math::

            (Af)(\\mathbf{x})
                = \\sum_{k=0}^{N-1} \\mu_k\\, c_k\\, \\Phi_k(\\mathbf{x})

        where $\\mu_k = (\\alpha\\lambda_k)^s$ (or ``0`` for suppressed modes).

        Parameters
        ----------
        f : :class:`~pygeoinf3d.core.functions.Function3D`
            The input function.

        Returns
        -------
        :class:`~pygeoinf3d.core.functions.Function3D`
            The spectral operator applied to *f*.
        """
        # Evaluate f at all quadrature points once
        f_vals = np.asarray(f(self._pts), dtype=float)

        # Compute spectral coefficients c_k = <f, phi_k> = ∫ f * phi_k dV
        coeffs = []
        for phi_k in self._eigenfunctions:
            phi_vals = np.asarray(phi_k(self._pts), dtype=float)
            c_k = float(np.dot(f_vals * phi_vals, self._weights))
            coeffs.append(c_k)

        # Snapshot for closure: (mu_k, phi_k, c_k)
        modes_snapshot = list(zip(self._op_eigenvalues, self._eigenfunctions, coeffs))

        def evaluate(x: np.ndarray) -> Union[float, np.ndarray]:
            x = np.asarray(x, dtype=float)
            is_batch = x.ndim == 2
            if is_batch:
                total = np.zeros(len(x))
            else:
                total = 0.0
            for mu_k, phi_k, c_k in modes_snapshot:
                total = total + mu_k * c_k * phi_k(x)
            return total

        return Function3D(
            self._space.function_domain,
            evaluate_callable=evaluate,
        )

    def __repr__(self) -> str:
        return (
            f"GeneralizedBoxLaplacian("
            f"power={self._power}, "
            f"alpha={self._alpha!r}, "
            f"bc_type={self._provider.bc_type!r}, "
            f"truncation={self._provider.truncation}"
            f")"
        )
