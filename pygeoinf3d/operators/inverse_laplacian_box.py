"""Spectral pseudo-inverse of the box Laplacian for axis-aligned box domains.

Provides :class:`InverseBoxLaplacian`, a
:class:`~pygeoinf.linear_operators.LinearOperator` implementing the
spectral pseudo-inverse of the negative Laplacian $-\\Delta$ on a
:class:`~pygeoinf3d.geometries.box.domain.BoxDomain` via truncated
spectral projection through a
:class:`~pygeoinf3d.providers.tensor_product.BoxLaplacianSpectrumProvider`.

Mathematical definition
-----------------------
Given orthonormal eigenfunction-eigenvalue pairs $(\\Phi_k, \\lambda_k)$ with
$-\\Delta \\Phi_k = \\lambda_k \\Phi_k$, the spectral pseudo-inverse acts as

.. math::

    ((-\\Delta)^{-1}_N f)(\\mathbf{x})
        = \\sum_{k=0}^{N-1} \\tilde{\\lambda}_k^{-1}
            \\langle f, \\Phi_k \\rangle_{L^2}\\, \\Phi_k(\\mathbf{x})

where $\\tilde{\\lambda}_k^{-1}$ depends on the ``null_space_handling`` mode:

- **pseudoinverse**: $\\tilde{\\lambda}_k^{-1} = 1/\\lambda_k$ if
  $|\\lambda_k| \\ge$ ``eigenvalue_tol``, else $0$.  Near-zero modes
  (e.g. the constant Neumann mode) are projected out.
- **shift**: $\\tilde{\\lambda}_k^{-1} = 1/(\\lambda_k + \\delta)$ for all
  $k$, where $\\delta =$ ``shift`` $> 0$.
- **strict**: same as pseudoinverse, but raises :exc:`ValueError` at
  construction if any eigenvalue satisfies $|\\lambda_k| <$ ``eigenvalue_tol``.

Self-adjointness
----------------
The operator is self-adjoint with respect to the $L^2$ inner product because
the spectral coefficients $\\tilde{\\lambda}_k^{-1}$ are real and the
eigenfunctions are orthonormal. Numerically this holds up to quadrature error.

Usage
-----
::

    from pygeoinf3d.geometries.box.domain import BoxDomain
    from pygeoinf3d.providers.tensor_product import BoxLaplacianSpectrumProvider
    from pygeoinf3d.spaces.lebesgue import Lebesgue3D
    from pygeoinf3d.operators.inverse_laplacian_box import InverseBoxLaplacian
    import numpy as np

    box = BoxDomain(np.zeros(3), np.ones(3))
    space = Lebesgue3D(box)
    provider = BoxLaplacianSpectrumProvider(
        box,
        truncation=(4, 4, 4),
        bc_type='dirichlet',
    )
    inv_lap = InverseBoxLaplacian(space, provider)

    phi0 = provider.get_eigenfunction(0)
    lam0 = provider.get_eigenvalue(0)
    result = inv_lap(phi0)   # ≈ (1/lam0) * phi0 pointwise
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np

from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.operators.base import SpectralOperator3D
from pygeoinf3d.operators.generalized_box_laplacian import GeneralizedBoxLaplacian
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.providers.tensor_product import BoxLaplacianSpectrumProvider


class InverseBoxLaplacian(SpectralOperator3D):
    """Spectral pseudo-inverse of the negative Laplacian on a box domain.

    Applies the truncated spectral projection

    .. math::

        ((-\\Delta)^{-1}_N f)(\\mathbf{x})
        = \\sum_{k=0}^{N-1} \\tilde{\\lambda}_k^{-1}
          \\langle f, \\Phi_k \\rangle_{L^2}\\, \\Phi_k(\\mathbf{x})

    via the eigenfunctions provided by a
    :class:`~pygeoinf3d.providers.tensor_product.BoxLaplacianSpectrumProvider`.

    Parameters
    ----------
    space:
        The :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` on the box domain.
        Both the operator's domain and codomain are this space.
    provider:
        A :class:`~pygeoinf3d.providers.tensor_product.BoxLaplacianSpectrumProvider`
        whose box domain matches ``space.function_domain``.
    null_space_handling:
        How to handle eigenvalues near zero:

        - ``'pseudoinverse'`` (default): modes with
          $|\\lambda_k| <$ ``eigenvalue_tol`` are mapped to 0 (natural
          pseudoinverse / mean-free projection).
        - ``'shift'``: uses $1/(\\lambda_k + \\delta)$ for all modes, where
          $\\delta =$ ``shift`` must be positive.
        - ``'strict'``: raises :exc:`ValueError` at construction if any
          eigenvalue satisfies $|\\lambda_k| <$ ``eigenvalue_tol``.
    shift:
        Shift value $\\delta$ used when ``null_space_handling='shift'``.
        Must be strictly positive. Ignored for other modes.
    eigenvalue_tol:
        Threshold below which an eigenvalue is considered near-zero.
        Default ``1e-10``.
    quad_resolution:
        Resolution passed to
        :meth:`~pygeoinf3d.geometries.box.domain.BoxDomain.structured_mesh`
        when computing the $L^2$ projections. Default ``20``.

    Raises
    ------
    ValueError
        If ``provider``'s box domain differs from ``space.function_domain``.
    ValueError
        If ``null_space_handling`` is not one of the three allowed literals.
    ValueError
        If ``null_space_handling='shift'`` and ``shift <= 0``.
    ValueError
        If ``null_space_handling='strict'`` and any eigenvalue satisfies
        $|\\lambda_k| <$ ``eigenvalue_tol``.

    Examples
    --------
    ::

        inv_lap = InverseBoxLaplacian(space, provider, quad_resolution=30)
        phi0 = provider.get_eigenfunction(0)
        lam0 = provider.get_eigenvalue(0)
        result = inv_lap(phi0)
        # result(x) ≈ (1/lam0) * phi0(x) pointwise
    """

    def __init__(
        self,
        space: Lebesgue3D,
        provider: BoxLaplacianSpectrumProvider,
        null_space_handling: Literal['pseudoinverse', 'shift', 'strict'] = 'pseudoinverse',
        shift: float = 1.0,
        eigenvalue_tol: float = 1e-10,
        quad_resolution: int = 20,
    ) -> None:
        # --- Validate box identity ---
        if space.function_domain is not provider._box:
            raise ValueError(
                "The provider's box domain must be the same object as "
                "space.function_domain."
            )

        # --- Validate null_space_handling ---
        _valid_modes = ('pseudoinverse', 'shift', 'strict')
        if null_space_handling not in _valid_modes:
            raise ValueError(
                f"null_space_handling must be one of {_valid_modes}; "
                f"got {null_space_handling!r}."
            )

        # --- Validate shift ---
        if null_space_handling == 'shift' and shift <= 0:
            raise ValueError(
                f"shift must be strictly positive when "
                f"null_space_handling='shift'; got {shift!r}."
            )

        self._space = space
        self._provider = provider
        self._null_space_handling = null_space_handling
        self._shift = float(shift)
        self._eigenvalue_tol = float(eigenvalue_tol)
        self._quad_resolution = quad_resolution

        if null_space_handling == 'pseudoinverse':
            # Delegate to GeneralizedBoxLaplacian(power=-1) for spectral internals.
            self._delegate: GeneralizedBoxLaplacian | None = GeneralizedBoxLaplacian(
                space, provider,
                power=-1, alpha=1.0,
                eigenvalue_tol=eigenvalue_tol,
                quad_resolution=quad_resolution,
            )
            self._modes = None
            self._pts = None
            self._weights = None
        else:
            self._delegate = None
            # --- Build inverted-eigenvalue mode list for shift/strict ---
            # Each entry: (scaled_inv_eigenvalue, eigenfunction)
            self._modes: list = []
            for i in range(provider.dim):
                lam_k = provider.get_eigenvalue(i)
                phi_k = provider.get_eigenfunction(i)

                if null_space_handling == 'shift':
                    inv_lam_k = 1.0 / (lam_k + self._shift)
                elif abs(lam_k) < eigenvalue_tol:
                    # strict mode: raise on near-zero eigenvalues
                    raise ValueError(
                        f"null_space_handling='strict' but eigenvalue "
                        f"lambda_{i} = {lam_k!r} satisfies "
                        f"|lambda| < eigenvalue_tol={eigenvalue_tol!r}. "
                        f"Use 'pseudoinverse' or 'shift' to handle near-zero "
                        f"eigenvalues."
                    )
                else:
                    inv_lam_k = 1.0 / lam_k

                self._modes.append((inv_lam_k, phi_k))

            # --- Pre-compute quadrature mesh and weights ---
            box = provider._box
            self._pts = box.structured_mesh(quad_resolution)
            self._weights = box.volume_weights(quad_resolution)

        super().__init__(
            space,
            space,
            self._apply,
            adjoint_mapping=self._apply,
        )

    # ------------------------------------------------------------------
    # SpectralOperator3D interface
    # ------------------------------------------------------------------

    def get_eigenvalue(self, index: int) -> float:
        """Return the (inverted/modified) operator eigenvalue for mode *index*.

        This is the value $\\tilde{\\lambda}_k^{-1}$ actually used by the
        operator: ``1/lambda_k`` in pseudoinverse/strict modes (or ``0`` for
        suppressed null-space modes), and ``1/(lambda_k + shift)`` in shift
        mode.
        """
        if self._delegate is not None:
            return self._delegate.get_eigenvalue(index)
        return self._modes[index][0]

    def get_eigenfunction(self, index: int) -> Function3D:
        """Return the $L^2$-normalised eigenfunction $\\Phi_k$ for mode *index*."""
        if self._delegate is not None:
            return self._delegate.get_eigenfunction(index)
        return self._modes[index][1]

    @property
    def n_modes(self) -> int:
        """Number of modes in the truncated spectral expansion."""
        if self._delegate is not None:
            return self._delegate.n_modes
        return len(self._modes)

    @property
    def null_space_handling(self) -> str:
        """The null-space handling mode: 'pseudoinverse', 'shift', or 'strict'."""
        return self._null_space_handling

    @property
    def shift(self) -> float:
        """The shift value $\\delta$ (only meaningful for 'shift' mode)."""
        return self._shift

    @property
    def eigenvalue_tol(self) -> float:
        """Threshold below which an eigenvalue is considered near-zero."""
        return self._eigenvalue_tol

    # ------------------------------------------------------------------
    # Internal application
    # ------------------------------------------------------------------

    def _apply(self, f: Function3D) -> Function3D:
        """Apply the spectral pseudo-inverse Laplacian to *f*.

        For pseudoinverse mode, delegates to the GeneralizedBoxLaplacian engine.
        For shift/strict modes, computes projection using pre-computed
        quadrature points and weights with the pre-computed inverted eigenvalues.

        Args:
            f: A :class:`~pygeoinf3d.core.functions.Function3D` to apply the
               operator to.

        Returns:
            :class:`~pygeoinf3d.core.functions.Function3D` approximating
            $((-\\Delta)^{-1}_N f)$.
        """
        if self._delegate is not None:
            return self._delegate._apply(f)

        # Evaluate f at all quadrature points once
        f_vals = np.asarray(f(self._pts), dtype=float)

        # Compute spectral coefficients c_k = <f, phi_k> = ∫ f * phi_k dV
        coeffs = []
        for inv_lam_k, phi_k in self._modes:
            phi_vals = np.asarray(phi_k(self._pts), dtype=float)
            c_k = float(np.dot(f_vals * phi_vals, self._weights))
            coeffs.append(c_k)

        # Build output: f_out(x) = sum_k inv_lam_k * c_k * phi_k(x)
        modes_snapshot = list(zip(
            [inv_lam for inv_lam, _ in self._modes],
            [phi for _, phi in self._modes],
            coeffs,
        ))

        def evaluate(x: np.ndarray) -> Union[float, np.ndarray]:
            x = np.asarray(x, dtype=float)
            is_batch = x.ndim == 2
            if is_batch:
                total = np.zeros(len(x))
            else:
                total = 0.0
            for inv_lam_k, phi_k, c_k in modes_snapshot:
                total = total + inv_lam_k * c_k * phi_k(x)
            return total

        return Function3D(
            self._space.function_domain,
            evaluate_callable=evaluate,
        )

    def __repr__(self) -> str:
        return (
            f"InverseBoxLaplacian("
            f"bc_type={self._provider.bc_type!r}, "
            f"truncation={self._provider.truncation}, "
            f"null_space_handling={self._null_space_handling!r}, "
            f"quad_resolution={self._quad_resolution}"
            f")"
        )
