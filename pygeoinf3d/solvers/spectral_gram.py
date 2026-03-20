r"""Fast spectral Gram solver for minimum-norm inversion with path integrals.

Given *m* path-integral linear forms $\ell_1, \dots, \ell_m$ on a
:class:`~pygeoinf3d.spaces.sobolev.Sobolev3D` space backed by a
:class:`~pygeoinf3d.operators.bessel_sobolev_box.BesselSobolevBox` inverse
weight, this module computes the minimum-norm solution

.. math::

    u = G^\top (G G^\top)^{-1} d

entirely via pre-computed spectral interaction matrices, avoiding the
generic framework's Python-level loops over modes and forms.

Algorithm
---------
1. Pre-compute quadrature points $P_i$ for each path and stack them.
2. Batch-evaluate all eigenfunctions at all quadrature points:
   $\Phi(P_\text{flat})$ of shape $(N, m \cdot q)$.
3. Contract with trapezoid weights to form the **interaction matrix**
   $L_{i,k} = \ell_i(\Phi_k)$ of shape $(m, N)$.
4. Read the inverse-weight eigenvalues $\nu_k$ of shape $(N,)$.
5. Compute the **Gram matrix** as:
   $GG^\top = L \operatorname{diag}(\nu) L^\top$, shape $(m, m)$.
6. Solve $GG^\top c = d$ via eigendecomposition.
7. Reconstruct the representer $u(x) = \sum_k \alpha_k \Phi_k(x)$
   where $\alpha_k = \nu_k \sum_i c_i L_{i,k}$, using vectorized
   ``einsum`` evaluation.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.spaces.forms import PathIntegralFunctional


class SpectralGramSolver:
    r"""Fast minimum-norm solver for path-integral measurements on a Sobolev space.

    Pre-computes the spectral interaction matrix and Gram matrix at construction
    time, then solves for arbitrary data vectors in microseconds.

    Parameters
    ----------
    sobolev_space:
        A :class:`~pygeoinf3d.spaces.sobolev.Sobolev3D` whose
        ``inverse_weight`` is a
        :class:`~pygeoinf3d.operators.bessel_sobolev_box.BesselSobolevBox`.
    forms:
        List of :class:`~pygeoinf3d.spaces.forms.PathIntegralFunctional`
        instances defining the measurement operator.
    rcond:
        Relative condition number threshold for eigenvalue truncation in
        the Gram matrix pseudoinverse.  Eigenvalues below
        ``rcond * max(eigenvalue)`` are discarded.  Default ``1e-14``.
    """

    def __init__(
        self,
        sobolev_space,
        forms: List[PathIntegralFunctional],
        *,
        rcond: float = 1e-14,
    ) -> None:
        from pygeoinf3d.spaces.sobolev import Sobolev3D
        from pygeoinf3d.operators.base import SpectralOperator3D

        if not isinstance(sobolev_space, Sobolev3D):
            raise TypeError(
                f"Expected Sobolev3D; got {type(sobolev_space).__name__!r}."
            )
        inv_w = sobolev_space.inverse_weight
        if not isinstance(inv_w, SpectralOperator3D):
            raise TypeError(
                "inverse_weight must be a SpectralOperator3D "
                f"(e.g. BesselSobolevBox); got {type(inv_w).__name__!r}."
            )
        if not forms:
            raise ValueError("At least one form is required.")

        self._sobolev = sobolev_space
        self._forms = list(forms)
        self._m = len(forms)

        # ── Spectral data ───────────────────────────────────────────────
        provider = inv_w._provider
        n_modes = provider.dim
        self._n_modes = n_modes

        # Inverse-weight eigenvalues ν_k = (κ² + λ_k)^{-s/2}
        self._nu = np.array(
            [float(inv_w.get_eigenvalue(k)) for k in range(n_modes)]
        )

        # ── Quadrature nodes per path ───────────────────────────────────
        # All forms must share the same n_quad for stacking.
        n_quad = forms[0].n_quad
        t_arr = np.linspace(0.0, 1.0, n_quad)  # identical for all paths

        # Trapezoid weights: Δt * [0.5, 1, 1, ..., 1, 0.5]
        trap_w = np.ones(n_quad)
        trap_w[0] = 0.5
        trap_w[-1] = 0.5
        dt = 1.0 / (n_quad - 1)

        # Stack all quadrature points: (m, q, 3)
        all_pts = np.empty((self._m, n_quad, 3))
        scales = np.empty(self._m)
        for i, form in enumerate(forms):
            _, pts_i = form.path.arc_length_nodes(n_quad)
            all_pts[i] = pts_i
            scales[i] = form._scale * form.path.arclength * dt

        # ── Batch eigenfunction evaluation ──────────────────────────────
        # Flatten to (m*q, 3), evaluate all modes at once: (N, m*q)
        pts_flat = all_pts.reshape(-1, 3)
        Phi_flat = provider.evaluate_all_eigenfunctions(pts_flat)
        # Reshape to (N, m, q)
        Phi = Phi_flat.reshape(n_modes, self._m, n_quad)

        # ── Interaction matrix L[i, k] = ℓ_i(Φ_k) ─────────────────────
        # L[i,k] = scales[i] * (trap_w^T @ Phi[k, i, :])
        # Use einsum: contract over quadrature points
        self._L = np.einsum("kiq,q,i->ik", Phi, trap_w, scales)

        # ── Gram matrix ─────────────────────────────────────────────────
        # GG^T = L @ diag(ν) @ L^T
        L_nu = self._L * self._nu[None, :]  # (m, N)
        self._gram = L_nu @ self._L.T  # (m, m)

        # ── Eigendecomposition of Gram ──────────────────────────────────
        eigvals, eigvecs = np.linalg.eigh(self._gram)
        # Truncate near-zero eigenvalues
        max_ev = np.max(np.abs(eigvals))
        mask = np.abs(eigvals) > rcond * max_ev
        self._gram_eigvals = eigvals[mask]
        self._gram_eigvecs = eigvecs[:, mask]
        self._gram_inv_eigvals = 1.0 / self._gram_eigvals

        # Store provider for representer evaluation
        self._provider = provider
        self._box = sobolev_space.function_domain

    @property
    def gram_matrix(self) -> np.ndarray:
        """The Gram matrix $GG^\\top$ of shape ``(m, m)``."""
        return self._gram.copy()

    @property
    def interaction_matrix(self) -> np.ndarray:
        r"""The interaction matrix $L_{i,k} = \ell_i(\Phi_k)$, shape ``(m, N)``."""
        return self._L.copy()

    @property
    def eigenvalues(self) -> np.ndarray:
        r"""The inverse-weight eigenvalues $\nu_k$, shape ``(N,)``."""
        return self._nu.copy()

    def forward(self, f: Function3D) -> np.ndarray:
        r"""Evaluate the forward operator $G f$ using quadrature.

        Parameters
        ----------
        f:
            Function3D to evaluate.

        Returns
        -------
        np.ndarray
            Data vector of shape ``(m,)``.
        """
        return np.array([form(f) for form in self._forms], dtype=float)

    def solve(self, data: np.ndarray, *, return_coeffs: bool = False):
        r"""Compute the minimum-norm solution $u = G^\top (GG^\top)^{-1} d$.

        Parameters
        ----------
        data:
            Data vector of shape ``(m,)``.
        return_coeffs:
            If True, also return the spectral coefficients $\alpha_k$.

        Returns
        -------
        Function3D
            The minimum-norm reconstruction.
        np.ndarray (optional)
            Spectral coefficients, if ``return_coeffs=True``.
        """
        data = np.asarray(data, dtype=float)
        if data.shape != (self._m,):
            raise ValueError(
                f"data must have shape ({self._m},); got {data.shape}."
            )

        # Solve (GG^T) c = d via eigendecomposition
        # c = V diag(1/λ) V^T d
        proj = self._gram_eigvecs.T @ data  # (r,)
        c = self._gram_eigvecs @ (self._gram_inv_eigvals * proj)  # (m,)

        # Spectral coefficients: α_k = ν_k * Σ_i c_i L[i,k]
        alpha = self._nu * (self._L.T @ c)  # (N,)

        # Build Function3D with vectorized evaluation
        provider = self._provider
        box = self._box
        alpha_snapshot = alpha.copy()

        def evaluate(x, _alpha=alpha_snapshot, _prov=provider):
            x = np.asarray(x, dtype=float)
            is_single = x.ndim == 1
            pts = x[None, :] if is_single else x
            # (N, n_pts)
            Phi = _prov.evaluate_all_eigenfunctions(pts)
            result = _alpha @ Phi  # (n_pts,)
            return float(result[0]) if is_single else result

        u = Function3D(box, evaluate_callable=evaluate, name="min-norm solution")

        if return_coeffs:
            return u, alpha
        return u
