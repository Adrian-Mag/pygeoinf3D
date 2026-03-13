"""Sobolev spaces on box domains.

Public API (continuous-first): Sobolev3D -- H^s space with Function3D elements.
Secondary spectral API: SpectralSobolev3D -- coefficient-vector Hilbert space.
"""

from __future__ import annotations

import numpy as np

from pygeoinf import HilbertSpace
from pygeoinf.linear_forms import LinearForm

from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.providers.tensor_product import TensorProductProvider


# ---------------------------------------------------------------------------
# SpectralSobolev3D -- secondary coefficient-vector implementation
# ---------------------------------------------------------------------------


class SpectralSobolev3D(HilbertSpace):
    r"""Finite-dimensional spectral Sobolev space H^s on a box domain.

    .. note::  **Secondary API.**  Elements are numpy arrays of spectral
        coefficients.  For the continuous-first API use :class:`Sobolev3D`.

    Parameters
    ----------
    provider : TensorProductProvider
    s : float
        Sobolev order, s >= 0.
    kappa : float
        Mass-regularisation parameter, kappa >= 0.  Default 1.0.
    """

    def __init__(
        self,
        provider: TensorProductProvider,
        *,
        s: float,
        kappa: float = 1.0,
    ) -> None:
        if s < 0:
            raise ValueError(f"Sobolev order s must be >= 0; got {s}.")
        if kappa < 0:
            raise ValueError(f"kappa must be >= 0; got {kappa}.")

        self._provider = provider
        self._s = float(s)
        self._kappa = float(kappa)

        lams = np.array(
            [provider.get_eigenvalue(i) for i in range(provider.dim)],
            dtype=float,
        )
        self._eigenvalues = lams
        self._weights = (kappa ** 2 + lams) ** s

        if s > 0 and np.any(self._weights == 0.0):
            bad = lams[self._weights == 0.0]
            raise ValueError(
                f"SpectralSobolev3D with s={s!r} and kappa={kappa!r} is "
                "degenerate: (kappa^2 + lambda)^s = 0 for eigenvalue(s) "
                f"{bad}.  Use kappa > 0 or Dirichlet BC."
            )

    @property
    def dim(self) -> int:
        return self._provider.dim

    def to_components(self, c: np.ndarray) -> np.ndarray:
        return np.asarray(c, dtype=float)

    def from_components(self, c: np.ndarray) -> np.ndarray:
        return np.asarray(c, dtype=float)

    def to_dual(self, c: np.ndarray) -> LinearForm:
        return LinearForm(
            self,
            components=self._weights * np.asarray(c, dtype=float),
        )

    def from_dual(self, xp: LinearForm) -> np.ndarray:
        if not isinstance(xp, LinearForm) or xp.domain is not self:
            raise TypeError(
                "SpectralSobolev3D.from_dual requires a LinearForm defined "
                "on this exact SpectralSobolev3D instance."
            )
        if xp.components is None:
            raise TypeError(
                f"from_dual expects a LinearForm with .components; "
                f"got {type(xp).__name__!r}."
            )
        return np.asarray(xp.components, dtype=float) / self._weights

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpectralSobolev3D):
            return NotImplemented
        return (
            self._provider is other._provider
            and self._s == other._s
            and self._kappa == other._kappa
        )

    def is_element(self, c) -> bool:
        try:
            arr = np.asarray(c, dtype=float)
        except (ValueError, TypeError):
            return False
        return arr.shape == (self.dim,)

    @property
    def function_domain(self):
        return self._provider.box

    @property
    def provider(self) -> TensorProductProvider:
        return self._provider

    @property
    def s(self) -> float:
        return self._s

    @property
    def kappa(self) -> float:
        return self._kappa

    @property
    def eigenvalues(self) -> np.ndarray:
        return self._eigenvalues.copy()

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    def add(self, x, y):
        return np.asarray(x, dtype=float) + np.asarray(y, dtype=float)

    def subtract(self, x, y):
        return np.asarray(x, dtype=float) - np.asarray(y, dtype=float)

    def multiply(self, a, x):
        return float(a) * np.asarray(x, dtype=float)

    def negative(self, x):
        return -np.asarray(x, dtype=float)

    def copy(self, x):
        return np.asarray(x, dtype=float).copy()

    def inner_product(self, c1: np.ndarray, c2: np.ndarray) -> float:
        return float(
            np.dot(
                self._weights * np.asarray(c1, dtype=float),
                np.asarray(c2, dtype=float),
            )
        )

    def project(
        self,
        f: Function3D,
        *,
        quad_resolution: int = 20,
    ) -> np.ndarray:
        """Project f onto the spectral basis (L2 coefficients)."""
        box = self.function_domain
        pts = box.structured_mesh(quad_resolution)
        quad_w = box.volume_weights(quad_resolution)
        f_vals = np.asarray(f(pts), dtype=float)
        N = self.dim
        coeffs = np.empty(N)
        for i in range(N):
            phi = self._provider.get_eigenfunction(i)
            phi_vals = np.asarray(phi(pts), dtype=float)
            coeffs[i] = float(np.dot(f_vals * phi_vals, quad_w))
        return coeffs

    def to_function(self, c: np.ndarray) -> Function3D:
        """Reconstruct a Function3D from spectral coefficients."""
        c = np.asarray(c, dtype=float)
        modes = [
            (c[i], self._provider.get_eigenfunction(i))
            for i in range(self.dim)
            if c[i] != 0.0
        ]

        def evaluate(x: np.ndarray):
            x = np.asarray(x, dtype=float)
            total = np.zeros(len(x)) if x.ndim == 2 else 0.0
            for ci, phi in modes:
                total = total + ci * phi(x)
            return total

        return Function3D(self.function_domain, evaluate_callable=evaluate)

    @property
    def supports_trace(self) -> bool:
        return self._s > 0.5

    def assert_trace_allowed(self) -> None:
        if not self.supports_trace:
            raise RuntimeError(
                f"Trace is not defined for SpectralSobolev3D with "
                f"s={self._s:.4g}: requires s > 0.5 in 3D."
            )

    def __repr__(self) -> str:
        return (
            f"SpectralSobolev3D("
            f"dim={self.dim}, "
            f"s={self._s:.4g}, "
            f"kappa={self._kappa:.4g}, "
            f"bc_type={self._provider.bc_type!r}"
            f")"
        )


# ---------------------------------------------------------------------------
# Sobolev3D -- continuous-first public API
# ---------------------------------------------------------------------------


class Sobolev3D(HilbertSpace):
    """Sobolev space H^s on a box domain with Function3D elements.

    The H^s inner product is computed spectrally::

        <f, g>_{H^s} = sum_k (kappa^2 + lambda_k)^s * f_hat_k * g_hat_k

    where f_hat_k = <f, phi_k>_{L^2} are L^2 spectral coefficients.
    Spectral coefficients are optional helpers via
    to_components/from_components.

    Parameters
    ----------
    provider : TensorProductProvider
    s : float
        Sobolev order, s >= 0.
    kappa : float
        Mass-regularisation parameter, kappa >= 0.  Default 1.0.
    quad_resolution : int
        Per-axis quadrature resolution.  Default 20.
    """

    def __init__(
        self,
        provider: TensorProductProvider,
        *,
        s: float,
        kappa: float = 1.0,
        quad_resolution: int = 20,
    ) -> None:
        if s < 0:
            raise ValueError(f"Sobolev order s must be >= 0; got {s}.")
        if kappa < 0:
            raise ValueError(f"kappa must be >= 0; got {kappa}.")

        self._provider = provider
        self._s = float(s)
        self._kappa = float(kappa)
        self._quad_resolution = int(quad_resolution)

        lams = np.array(
            [provider.get_eigenvalue(i) for i in range(provider.dim)],
            dtype=float,
        )
        self._eigenvalues = lams
        self._weights = (kappa ** 2 + lams) ** s

        if s > 0 and np.any(self._weights == 0.0):
            bad = lams[self._weights == 0.0]
            raise ValueError(
                f"Sobolev3D with s={s!r} and kappa={kappa!r} is degenerate: "
                f"(kappa^2 + lambda)^s = 0 for eigenvalue(s) {bad}.  "
                "Use kappa > 0 or Dirichlet BC."
            )

        # Pre-compute quadrature mesh and eigenfunction values once.
        box = provider.box
        self._quad_pts = box.structured_mesh(self._quad_resolution)
        self._quad_weights = box.volume_weights(self._quad_resolution)

        N = provider.dim
        Q = len(self._quad_pts)
        self._phi_vals = np.empty((N, Q), dtype=float)
        for i in range(N):
            phi = provider.get_eigenfunction(i)
            self._phi_vals[i] = np.asarray(phi(self._quad_pts), dtype=float)

    # ------------------------------------------------------------------
    # HilbertSpace abstract interface
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Return the optional spectral component dimension.

        Elements remain Function3D objects, but a finite spectral projection
        is available as an explicit numerical representation.
        """
        return self._provider.dim

    def is_element(self, x) -> bool:
        """True iff x is a Function3D on this space's domain."""
        return isinstance(x, Function3D) and x.domain is self.function_domain

    def to_dual(self, f: Function3D):
        """Riesz map f -> phi_f in (H^s)*."""
        if not isinstance(f, Function3D):
            raise TypeError(
                "Sobolev3D.to_dual requires a Function3D primal element."
            )
        if f.domain is not self._provider.box:
            raise ValueError(
                "Sobolev3D.to_dual: f.domain does not match function_domain."
            )
        from pygeoinf3d.spaces.forms import SobolevKernel3D
        return SobolevKernel3D(self, f)

    def from_dual(self, xp) -> Function3D:
        """Recover the primal Function3D from a SobolevKernel3D."""
        from pygeoinf3d.spaces.forms import SobolevKernel3D
        if not isinstance(xp, SobolevKernel3D):
            raise TypeError(
                f"Sobolev3D.from_dual requires a SobolevKernel3D, "
                f"got {type(xp).__name__!r}."
            )
        if xp.domain is not self:
            raise ValueError(
                "SobolevKernel3D was produced by a different "
                "Sobolev3D instance."
            )
        return xp.kernel

    def to_components(self, f: Function3D) -> np.ndarray:
        """Project f onto spectral basis (L^2 coefficients), shape (N,)."""
        return self._project_to_coeffs(f)

    def from_components(self, c: np.ndarray) -> Function3D:
        """Reconstruct Function3D from spectral coefficients."""
        return self._reconstruct(np.asarray(c, dtype=float))

    def project(
        self,
        f: Function3D,
        *,
        quad_resolution: int | None = None,
    ) -> np.ndarray:
        """Alias for to_components with optional quadrature override."""
        return self._project_to_coeffs(f, quad_resolution=quad_resolution)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sobolev3D):
            return NotImplemented
        return (
            self._provider is other._provider
            and self._s == other._s
            and self._kappa == other._kappa
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def function_domain(self):
        return self._provider.box

    @property
    def provider(self) -> TensorProductProvider:
        return self._provider

    @property
    def s(self) -> float:
        return self._s

    @property
    def kappa(self) -> float:
        return self._kappa

    @property
    def quad_resolution(self) -> int:
        return self._quad_resolution

    @property
    def eigenvalues(self) -> np.ndarray:
        return self._eigenvalues.copy()

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    # ------------------------------------------------------------------
    # Vector space operations (Function3D)
    # ------------------------------------------------------------------

    @property
    def zero(self) -> Function3D:
        return Function3D(
            self.function_domain,
            evaluate_callable=lambda x: (
                np.zeros(len(x)) if np.asarray(x).ndim == 2 else 0.0
            ),
        )

    def copy(self, f: Function3D) -> Function3D:
        return f.copy()

    def add(self, f: Function3D, g: Function3D) -> Function3D:
        return f + g

    def subtract(self, f: Function3D, g: Function3D) -> Function3D:
        return f - g

    def multiply(self, a: float, f: Function3D) -> Function3D:
        return a * f

    def negative(self, f: Function3D) -> Function3D:
        return -f

    def inner_product(self, f: Function3D, g: Function3D) -> float:
        """H^s inner product via spectral projection."""
        return self.to_dual(f)(g)

    def norm(self, f: Function3D) -> float:
        ip = self.inner_product(f, f)
        return float(np.sqrt(max(ip, 0.0)))

    # ------------------------------------------------------------------
    # Trace readiness
    # ------------------------------------------------------------------

    @property
    def supports_trace(self) -> bool:
        return self._s > 0.5

    def assert_trace_allowed(self) -> None:
        if not self.supports_trace:
            raise RuntimeError(
                f"Trace is not defined for Sobolev3D with s={self._s:.4g}: "
                "requires s > 0.5."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project_to_coeffs(
        self,
        f: Function3D,
        *,
        quad_resolution: int | None = None,
    ) -> np.ndarray:
        """Compute L^2 spectral coefficients via quadrature."""
        if not isinstance(f, Function3D):
            raise TypeError(
                "Sobolev3D projection requires a Function3D primal element."
            )
        if f.domain is not self.function_domain:
            raise ValueError(
                "Sobolev3D projection requires a Function3D built on the "
                "same Domain3D instance as this space."
            )
        if quad_resolution is None or quad_resolution == self._quad_resolution:
            f_vals = np.asarray(f(self._quad_pts), dtype=float)
            # (N, Q) @ (Q,) -> (N,)
            return self._phi_vals @ (f_vals * self._quad_weights)
        # Slow path: custom resolution
        box = self.function_domain
        pts = box.structured_mesh(quad_resolution)
        quad_w = box.volume_weights(quad_resolution)
        f_vals = np.asarray(f(pts), dtype=float)
        N = self._provider.dim
        coeffs = np.empty(N)
        for i in range(N):
            phi = self._provider.get_eigenfunction(i)
            phi_vals = np.asarray(phi(pts), dtype=float)
            coeffs[i] = float(np.dot(f_vals * phi_vals, quad_w))
        return coeffs

    def _reconstruct(self, c: np.ndarray) -> Function3D:
        """Return Function3D evaluating sum_k c_k phi_k."""
        modes = [
            (c[i], self._provider.get_eigenfunction(i))
            for i in range(self._provider.dim)
            if c[i] != 0.0
        ]

        def evaluate(x: np.ndarray):
            x = np.asarray(x, dtype=float)
            total = np.zeros(len(x)) if x.ndim == 2 else 0.0
            for ci, phi in modes:
                total = total + ci * phi(x)
            return total

        return Function3D(self.function_domain, evaluate_callable=evaluate)

    def __repr__(self) -> str:
        return (
            f"Sobolev3D("
            f"s={self._s:.4g}, "
            f"kappa={self._kappa:.4g}, "
            f"bc_type={self._provider.bc_type!r}, "
            f"modes={self._provider.dim}, "
            f"quad_res={self._quad_resolution}"
            f")"
        )
