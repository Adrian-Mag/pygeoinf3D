"""Tensor-product spectral provider for axis-aligned box domains.

Provides :class:`TensorProductProvider`, a minimal spectral basis
manager for the negative Laplacian $-\\Delta$ on an axis-aligned box
with uniform homogeneous Dirichlet or Neumann boundary conditions.

Physics
-------
On $\\Omega = [a_x, b_x] \\times [a_y, b_y] \\times [a_z, b_z]$ the 1-D
eigenproblem for each axis is

.. math::

    -\\frac{\\mathrm{d}^2 \\varphi}{\\mathrm{d}t^2} = \\mu \\varphi

Dirichlet ($\\varphi(a) = \\varphi(b) = 0$, mode index $k \\ge 1$):

.. math::

    \\varphi_k(t)
    = \\sqrt{\\tfrac{2}{L}}\\,\\sin\\!\\left(\\frac{k\\pi(t-a)}{L}\\right),
    \\qquad \\mu_k = \\left(\\frac{k\\pi}{L}\\right)^2

Neumann ($\\varphi'(a) = \\varphi'(b) = 0$, mode index $k \\ge 0$):

.. math::

    \\varphi_0(t) = \\frac{1}{\\sqrt{L}},
    \\qquad
    \\varphi_k(t)
    = \\sqrt{\\tfrac{2}{L}}\\,\\cos\\!\\left(\\frac{k\\pi(t-a)}{L}\\right),
    \\qquad \\mu_k = \\left(\\frac{k\\pi}{L}\\right)^2\\quad(k \\ge 1)

Three-dimensional eigenfunctions are tensor products,
$\\Phi_{k_x,k_y,k_z}(x,y,z) = \\varphi_{k_x}(x) \\cdot \\varphi_{k_y}(y)
\\cdot \\varphi_{k_z}(z)$, with eigenvalue
$\\lambda = \\mu_{k_x} + \\mu_{k_y} + \\mu_{k_z}$.

Modes are ordered by increasing eigenvalue with lexicographic tie-break.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Union

import numpy as np

from pygeoinf3d.core.functions import Function3D


# ---------------------------------------------------------------------------
# 1-D eigenvalue and eigenfunction helpers
# ---------------------------------------------------------------------------

def _1d_eigenvalue(k: int, L: float) -> float:
    """Return the 1-D eigenvalue $(k\\pi/L)^2$."""
    return float((k * np.pi / L) ** 2)


def _make_1d_phi(
    k: int,
    a: float,
    L: float,
    bc_type: str,
) -> Callable[[np.ndarray], Union[float, np.ndarray]]:
    """Return a callable for the $k$-th 1-D normalised eigenfunction.

    For Dirichlet:
        $\\sqrt{2/L}\\,\\sin(k\\pi(t-a)/L)$, $k\\ge 1$.
    For Neumann:
        $1/\\sqrt{L}$ for $k=0$;
        $\\sqrt{2/L}\\,\\cos(k\\pi(t-a)/L)$ for $k\\ge 1$.

    Args:
        k: Mode index.
        a: Left endpoint of interval.
        L: Length of interval.
        bc_type: ``'dirichlet'`` or ``'neumann'``.

    Returns:
        Callable accepting a scalar or 1-D array of coordinates and returning
        correspondingly scalar or 1-D array values.
    """
    if bc_type == "dirichlet":
        amp = np.sqrt(2.0 / L)
        freq = k * np.pi / L
        return lambda t: amp * np.sin(freq * (np.asarray(t) - a))

    if k == 0:
        norm = 1.0 / np.sqrt(L)

        def constant_mode(t: np.ndarray) -> Union[float, np.ndarray]:
            values = np.asarray(t)
            if values.ndim > 0:
                return np.full_like(values, norm, dtype=float)
            return norm

        return constant_mode

    amp = np.sqrt(2.0 / L)
    freq = k * np.pi / L
    return lambda t: amp * np.cos(freq * (np.asarray(t) - a))


# ---------------------------------------------------------------------------
# TensorProductProvider
# ---------------------------------------------------------------------------

class TensorProductProvider:
    """Tensor-product spectral provider for a box domain.

    Enumerates the first ``n_x × n_y × n_z`` modes of the negative Laplacian on
    an axis-aligned box with uniform Dirichlet or Neumann boundary conditions.
    Modes are sorted by increasing eigenvalue with lexicographic tie-break.

    Parameters
    ----------
    domain_or_space:
        A :class:`~pygeoinf3d.geometries.box.domain.BoxDomain` directly, or a
        :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` (or any object with a
        ``function_domain`` attribute pointing to a ``BoxDomain``).
    truncation:
        Number of modes per axis: a single integer *n* (same for all three
        axes) or a 3-tuple ``(n_x, n_y, n_z)``.
    bc_type:
        ``'dirichlet'`` — zero Dirichlet on all faces
        (mode $k$ starts at 1).
        ``'neumann'`` — zero Neumann on all faces
        (mode $k$ starts at 0, so the constant is included).

    Raises
    ------
    ValueError
        If *bc_type* is not ``'dirichlet'`` or ``'neumann'``.

    Examples
    --------
    ::

        box = BoxDomain(np.zeros(3), np.ones(3))
        provider = TensorProductProvider(
            box,
            truncation=(4, 4, 4),
            bc_type='dirichlet',
        )
        phi0 = provider.get_eigenfunction(0)   # lowest Dirichlet mode
        lam0 = provider.get_eigenvalue(0)      # 3*pi^2 on unit box
    """

    def __init__(
        self,
        domain_or_space,
        *,
        truncation: Union[int, Tuple[int, int, int]] = 4,
        bc_type: str = "dirichlet",
    ) -> None:
        if bc_type not in ("dirichlet", "neumann"):
            raise ValueError(
                f"bc_type must be 'dirichlet' or 'neumann'; got {bc_type!r}."
            )

        # Resolve the BoxDomain
        if hasattr(domain_or_space, "function_domain"):
            box = domain_or_space.function_domain
        else:
            box = domain_or_space

        self._box = box
        self._bc_type = bc_type

        # Parse truncation
        if isinstance(truncation, (int, np.integer)):
            n = int(truncation)
            nx, ny, nz = n, n, n
        else:
            nx = int(truncation[0])
            ny = int(truncation[1])
            nz = int(truncation[2])
        self._truncation = (nx, ny, nz)

        # Box geometry
        bbox = box.bounding_box()
        self._low = bbox.low
        self._lengths = bbox.extents  # (Lx, Ly, Lz)
        if np.any(self._lengths <= 0.0):
            raise ValueError(
                "TensorProductProvider requires positive box extents."
            )

        # Build and sort mode list
        self._sorted_modes: List[Tuple[int, int, int]]
        self._sorted_eigenvalues: List[float]
        self._mode_to_index: Dict[Tuple[int, int, int], int]
        self._build_modes(nx, ny, nz)

    # ------------------------------------------------------------------
    # Internal mode construction
    # ------------------------------------------------------------------

    def _build_modes(self, nx: int, ny: int, nz: int) -> None:
        """Enumerate, compute eigenvalues, and sort all modes."""
        Lx, Ly, Lz = self._lengths

        if self._bc_type == "dirichlet":
            kx_range = range(1, nx + 1)
            ky_range = range(1, ny + 1)
            kz_range = range(1, nz + 1)
        else:
            kx_range = range(0, nx)
            ky_range = range(0, ny)
            kz_range = range(0, nz)

        entries: List[Tuple[float, int, int, int]] = []
        for kx in kx_range:
            for ky in ky_range:
                for kz in kz_range:
                    ev = (
                        _1d_eigenvalue(kx, Lx)
                        + _1d_eigenvalue(ky, Ly)
                        + _1d_eigenvalue(kz, Lz)
                    )
                    entries.append((ev, kx, ky, kz))

        # Sort by eigenvalue, with lexicographic tie-breaking.
        entries.sort(key=lambda t: (t[0], t[1], t[2], t[3]))

        self._sorted_modes = [(e[1], e[2], e[3]) for e in entries]
        self._sorted_eigenvalues = [e[0] for e in entries]
        self._mode_to_index = {m: i for i, m in enumerate(self._sorted_modes)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Total number of modes: $n_x \\cdot n_y \\cdot n_z$."""
        return len(self._sorted_modes)

    @property
    def bc_type(self) -> str:
        """Boundary condition type: ``'dirichlet'`` or ``'neumann'``."""
        return self._bc_type

    @property
    def truncation(self) -> Tuple[int, int, int]:
        """Number of modes per axis ``(n_x, n_y, n_z)``."""
        return self._truncation

    def get_multi_index(self, index: int) -> Tuple[int, int, int]:
        """Return the 1-D mode triple ``(kx, ky, kz)`` for flat *index*.

        Args:
            index: Flat index in ``range(dim)``.

        Returns:
            ``(kx, ky, kz)`` tuple.

        Raises:
            IndexError: If *index* is out of range.
        """
        if not 0 <= index < self.dim:
            raise IndexError(
                f"Index {index} out of range for provider with dim={self.dim}."
            )
        return self._sorted_modes[index]

    def get_index(self, multi_index: Tuple[int, int, int]) -> int:
        """Return the flat index for multi-index ``(kx, ky, kz)``.

        Args:
            multi_index: A triple of 1-D mode indices.

        Returns:
            Flat index in ``range(dim)``.

        Raises:
            KeyError: If the multi-index is not in this provider's mode set.
        """
        key = (
            int(multi_index[0]),
            int(multi_index[1]),
            int(multi_index[2]),
        )
        if key not in self._mode_to_index:
            raise KeyError(
                f"Multi-index {multi_index} not found in this "
                "TensorProductProvider."
            )
        return self._mode_to_index[key]

    def get_eigenvalue(self, index: int) -> float:
        """Return the eigenvalue $\\lambda$ at flat *index*.

        The eigenvalue is $\\mu_{k_x} + \\mu_{k_y} + \\mu_{k_z}$ where each
        $\\mu_k = (k\\pi/L)^2$.

        Args:
            index: Flat index in ``range(dim)``.

        Returns:
            Non-negative float eigenvalue.
        """
        if not 0 <= index < self.dim:
            raise IndexError(
                f"Index {index} out of range for provider with dim={self.dim}."
            )
        return self._sorted_eigenvalues[index]

    def get_eigenfunction(self, index: int) -> "Function3D":
        """Return the *index*-th normalised 3-D eigenfunction.

        The eigenfunction is the tensor product of three 1-D eigenfunctions::

            Phi_{kx,ky,kz}(x, y, z) = phi_kx(x) * phi_ky(y) * phi_kz(z)

        The returned ``Function3D`` accepts both a single point ``(3,)`` and a
        batch ``(n, 3)`` and returns a scalar or ``(n,)`` array respectively.

        Args:
            index: Flat index in ``range(dim)``.

        Returns:
            :class:`~pygeoinf3d.core.functions.Function3D` evaluated on the
            provider's box domain.
        """
        kx, ky, kz = self.get_multi_index(index)
        ax, ay, az = self._low
        Lx, Ly, Lz = self._lengths

        phi_x = _make_1d_phi(kx, ax, Lx, self._bc_type)
        phi_y = _make_1d_phi(ky, ay, Ly, self._bc_type)
        phi_z = _make_1d_phi(kz, az, Lz, self._bc_type)

        def evaluate(x: np.ndarray) -> Union[float, np.ndarray]:
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                return float(phi_x(x[0]) * phi_y(x[1]) * phi_z(x[2]))
            return phi_x(x[:, 0]) * phi_y(x[:, 1]) * phi_z(x[:, 2])

        return Function3D(self._box, evaluate_callable=evaluate)

    def __iter__(self):
        """Iterate over ``(eigenvalue, Function3D)`` pairs in order."""
        for i in range(self.dim):
            yield self.get_eigenvalue(i), self.get_eigenfunction(i)

    def __repr__(self) -> str:
        return (
            f"TensorProductProvider("
            f"box={self._box!r}, "
            f"truncation={self._truncation}, "
            f"bc_type={self._bc_type!r}, "
            f"dim={self.dim})"
        )
