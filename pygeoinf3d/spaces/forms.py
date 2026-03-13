"""Integration-backed and spectral-projection-backed linear forms
for 3D spaces.

Provides two kernel-backed :class:`~pygeoinf.LinearForm` subclasses:

:class:`VolumeKernel3D`
    Evaluation via ``∫ kernel(x) · g(x) dV``.  Used by :class:`Lebesgue3D`.

:class:`SobolevKernel3D`
    Evaluation via ``Σ_k (κ²+λ_k)^s ĉ_k(f) ĉ_k(g)`` where spectral
    coefficients are computed by quadrature.  Used by the continuous-first
    :class:`Sobolev3D`.

Both subclasses override ``_mapping_impl`` to bypass the
component-enumeration assumption in :class:`~pygeoinf.LinearForm`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from pygeoinf import LinearForm

if TYPE_CHECKING:
    from pygeoinf3d.core.functions import Function3D
    from pygeoinf3d.spaces.lebesgue import Lebesgue3D
    from pygeoinf3d.spaces.sobolev import Sobolev3D


class VolumeKernel3D(LinearForm):
    """A linear form on :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D`
    represented by a volume-integration kernel.

    The pairing is:

    .. math::

        \\langle \\phi_f,\\, g \\rangle =
        \\int_\\Omega f(\\mathbf{x})\\,g(\\mathbf{x})\\,\\mathrm{d}V

    where *f* is the :class:`~pygeoinf3d.core.functions.Function3D` kernel.

    Parameters
    ----------
    space:
        The :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` on which this
        form is defined.  This is the form's ``domain``.
    kernel:
        The primal function *f* whose inner product with arbitrary *g* this
        form evaluates.

    Notes
    -----
    ``VolumeKernel3D`` passes ``components=np.zeros(0)`` to bypass
    ``LinearForm``'s component-enumeration logic and instead routes all
    evaluation through its own overridden ``_mapping_impl``.
    """

    def __init__(self, space: "Lebesgue3D", kernel: "Function3D") -> None:
        if kernel.domain is not space.function_domain:
            raise ValueError(
                "VolumeKernel3D: kernel.domain does not match "
                "space.function_domain (identity check).  "
                "Pass a Function3D built on the same Domain3D instance "
                "as the Lebesgue3D space."
            )
        self._kernel: "Function3D" = kernel
        # Fake empty components so LinearForm.__init__ does NOT call
        # _compute_components (which would try to enumerate basis vectors).
        # The actual evaluation is done by our overridden _mapping_impl.
        super().__init__(
            space,
            components=np.zeros(0),
        )

    # ------------------------------------------------------------------
    # Disable component access — this form is basis-free
    # ------------------------------------------------------------------

    @property
    def components(self) -> None:  # type: ignore[override]
        """Not available in basis-free mode.

        Raises
        ------
        TypeError
            Always.  ``VolumeKernel3D`` has no finite-component representation.
            Use kernel-backed arithmetic instead.
        """
        raise TypeError(
            "VolumeKernel3D has no finite-component representation in "
            "basis-free mode.  Use kernel-backed arithmetic (e.g. "
            "VolumeKernel3D + VolumeKernel3D) instead of component operations."
        )

    # ------------------------------------------------------------------
    # Override the evaluation
    # ------------------------------------------------------------------

    def _mapping_impl(self, g: "Function3D") -> float:
        """Evaluate the pairing ``⟨f, g⟩ = ∫ f(x)·g(x) dV``."""
        domain = self.domain.function_domain  # type: ignore[attr-defined]
        if g.domain is not domain:
            raise ValueError(
                "VolumeKernel3D can only be evaluated on Function3D objects "
                "defined on the same Domain3D instance."
            )
        return domain.integrate_volume(self._kernel.pointwise_mul(g))

    # ------------------------------------------------------------------
    # Safe linear-form algebra (override inherited component-based machinery)
    # ------------------------------------------------------------------

    def copy(self) -> "VolumeKernel3D":
        """Return a shallow-callable copy.  Callable is shared (immutable)."""
        return VolumeKernel3D(
            cast("Lebesgue3D", self.domain), self._kernel.copy()
        )

    def __neg__(self) -> "VolumeKernel3D":
        return VolumeKernel3D(
            cast("Lebesgue3D", self.domain), -self._kernel
        )

    def __mul__(self, a: float) -> "VolumeKernel3D":
        return VolumeKernel3D(
            cast("Lebesgue3D", self.domain), float(a) * self._kernel
        )

    def __rmul__(self, a: float) -> "VolumeKernel3D":
        return self.__mul__(a)

    def __truediv__(self, a: float) -> "VolumeKernel3D":
        return self.__mul__(1.0 / float(a))

    def __add__(self, other) -> "VolumeKernel3D":
        if not isinstance(other, VolumeKernel3D):
            raise TypeError(
                f"Cannot add VolumeKernel3D and {type(other).__name__!r}.  "
                "Only VolumeKernel3D objects on the same space can be added."
            )
        if other.domain is not self.domain:
            raise ValueError(
                "Cannot add VolumeKernel3D objects from different Lebesgue3D "
                "instances (domain identity mismatch)."
            )
        return VolumeKernel3D(
            cast("Lebesgue3D", self.domain), self._kernel + other._kernel
        )

    def __sub__(self, other) -> "VolumeKernel3D":
        if not isinstance(other, VolumeKernel3D):
            raise TypeError(
                f"Cannot subtract {type(other).__name__!r} "
                "from VolumeKernel3D.  "
                "Only VolumeKernel3D objects on the same "
                "space can be subtracted."
            )
        if other.domain is not self.domain:
            raise ValueError(
                "Cannot subtract VolumeKernel3D objects "
                "from different Lebesgue3D instances "
                "(domain identity mismatch)."
            )
        return VolumeKernel3D(
            cast("Lebesgue3D", self.domain), self._kernel - other._kernel
        )

    # ------------------------------------------------------------------
    # In-place arithmetic — must override LinearForm's component-based
    # __imul__ and __iadd__ which mutate the fake empty _components array
    # and silently leave the kernel unchanged.
    # ------------------------------------------------------------------

    def __imul__(self, a: float) -> "VolumeKernel3D":
        """In-place scalar multiplication: self *= a."""
        self._kernel = float(a) * self._kernel
        return self

    def __iadd__(self, other) -> "VolumeKernel3D":
        """In-place addition: self += other."""
        if not isinstance(other, VolumeKernel3D):
            raise TypeError(
                f"Cannot in-place add {type(other).__name__!r} "
                "to VolumeKernel3D.  Only VolumeKernel3D objects "
                "on the same space can be added."
            )
        if other.domain is not self.domain:
            raise ValueError(
                "Cannot in-place add VolumeKernel3D objects from different "
                "Lebesgue3D instances (domain identity mismatch)."
            )
        self._kernel = self._kernel + other._kernel
        return self

    def __isub__(self, other) -> "VolumeKernel3D":
        """In-place subtraction: self -= other."""
        if not isinstance(other, VolumeKernel3D):
            raise TypeError(
                f"Cannot in-place subtract {type(other).__name__!r} from "
                "VolumeKernel3D.  Only VolumeKernel3D objects on the same "
                "space can be subtracted."
            )
        if other.domain is not self.domain:
            raise ValueError(
                "Cannot in-place subtract VolumeKernel3D objects "
                "from different Lebesgue3D instances "
                "(domain identity mismatch)."
            )
        self._kernel = self._kernel - other._kernel
        return self

    # ------------------------------------------------------------------
    # Expose the kernel for from_dual recovery
    # ------------------------------------------------------------------

    @property
    def kernel(self) -> "Function3D":
        """The primal function whose inner-product this form represents."""
        return self._kernel


# ---------------------------------------------------------------------------
# SobolevKernel3D — spectral-projection-backed linear form for Sobolev3D
# ---------------------------------------------------------------------------


class SobolevKernel3D(LinearForm):
    r"""Linear form on :class:`~pygeoinf3d.spaces.sobolev.Sobolev3D`
    backed by spectral projection.

    The pairing is:

    .. math::

        \langle \phi_f,\, g \rangle_{H^s}
        = \sum_{k=0}^{N-1} (\kappa^2 + \lambda_k)^s\,
          \hat{f}_k\, \hat{g}_k

    where :math:`\hat{f}_k = \langle f, \phi_k \rangle_{L^2}` are the
    :math:`L^2` spectral coefficients of *f* in the eigenbasis of the
    box Laplacian.

    Parameters
    ----------
    space:
        The :class:`~pygeoinf3d.spaces.sobolev.Sobolev3D` (continuous-first)
        on which this form is defined.
    kernel:
        The primal :class:`~pygeoinf3d.core.functions.Function3D`
        representer *f*.

    Notes
    -----
    Weighted spectral coefficients of *kernel* are pre-computed at
    construction; each evaluation only re-projects the test function *g*.
    The *_weighted_coeffs* keyword is a private fast path for arithmetic
    — callers should not pass it directly.
    """

    def __init__(
        self,
        space: "Sobolev3D",
        kernel: "Function3D",
        *,
        _weighted_coeffs: "np.ndarray | None" = None,
    ) -> None:
        if kernel.domain is not space.function_domain:
            raise ValueError(
                "SobolevKernel3D: kernel.domain does not match "
                "space.function_domain (identity check).  "
                "Pass a Function3D built on the same Domain3D instance "
                "as the Sobolev3D space."
            )
        self._kernel: "Function3D" = kernel
        self._space = space
        if _weighted_coeffs is not None:
            self._weighted_coeffs: np.ndarray = _weighted_coeffs
        else:
            c_f = space._project_to_coeffs(kernel)   # L² coefficients
            self._weighted_coeffs = space.weights * c_f  # w_k * ĉ_k(f)
        super().__init__(space, components=np.zeros(0))

    # ------------------------------------------------------------------
    # Disable component access — basis-free
    # ------------------------------------------------------------------

    @property
    def components(self) -> None:  # type: ignore[override]
        """Not available; this form is spectral-projection-backed.

        Raises
        ------
        TypeError
            Always.
        """
        raise TypeError(
            "SobolevKernel3D has no finite-component representation.  "
            "Use kernel-backed arithmetic or to_components/from_components "
            "on the parent Sobolev3D."
        )

    # ------------------------------------------------------------------
    # Override the evaluation
    # ------------------------------------------------------------------

    def _mapping_impl(self, g: "Function3D") -> float:
        r"""Evaluate :math:`\phi_f(g) = \sum_k w_k \hat{f}_k \hat{g}_k`."""
        if g.domain is not self._space.function_domain:
            raise ValueError(
                "SobolevKernel3D can only be evaluated on Function3D objects "
                "defined on the same Domain3D instance."
            )
        c_g = self._space._project_to_coeffs(g)
        return float(np.dot(self._weighted_coeffs, c_g))

    # ------------------------------------------------------------------
    # Expose the kernel for from_dual recovery
    # ------------------------------------------------------------------

    @property
    def kernel(self) -> "Function3D":
        """The primal function whose :math:`H^s` inner product this form
        represents."""
        return self._kernel

    # ------------------------------------------------------------------
    # Safe linear-form algebra
    # ------------------------------------------------------------------

    def copy(self) -> "SobolevKernel3D":
        """Return a copy with a copied kernel and coefficient array."""
        return SobolevKernel3D(
            self._space,
            self._kernel.copy(),
            _weighted_coeffs=self._weighted_coeffs.copy(),
        )

    def __neg__(self) -> "SobolevKernel3D":
        return SobolevKernel3D(
            self._space,
            -self._kernel,
            _weighted_coeffs=-self._weighted_coeffs,
        )

    def __mul__(self, a: float) -> "SobolevKernel3D":
        return SobolevKernel3D(
            self._space,
            float(a) * self._kernel,
            _weighted_coeffs=float(a) * self._weighted_coeffs,
        )

    def __rmul__(self, a: float) -> "SobolevKernel3D":
        return self.__mul__(a)

    def __truediv__(self, a: float) -> "SobolevKernel3D":
        return self.__mul__(1.0 / float(a))

    def __add__(self, other) -> "SobolevKernel3D":
        if not isinstance(other, SobolevKernel3D):
            raise TypeError(
                f"Cannot add SobolevKernel3D and {type(other).__name__!r}.  "
                "Only SobolevKernel3D objects on the same space can be added."
            )
        if other.domain is not self.domain:
            raise ValueError(
                "Cannot add SobolevKernel3D objects from different "
                "Sobolev3D instances (domain identity mismatch)."
            )
        return SobolevKernel3D(
            self._space,
            self._kernel + other._kernel,
            _weighted_coeffs=self._weighted_coeffs + other._weighted_coeffs,
        )

    def __sub__(self, other) -> "SobolevKernel3D":
        if not isinstance(other, SobolevKernel3D):
            raise TypeError(
                f"Cannot subtract {type(other).__name__!r} "
                "from SobolevKernel3D.  "
                "Only SobolevKernel3D objects on the same "
                "space can be subtracted."
            )
        if other.domain is not self.domain:
            raise ValueError(
                "Cannot subtract SobolevKernel3D objects "
                "from different Sobolev3D instances "
                "(domain identity mismatch)."
            )
        return SobolevKernel3D(
            self._space,
            self._kernel - other._kernel,
            _weighted_coeffs=self._weighted_coeffs - other._weighted_coeffs,
        )

    def __imul__(self, a: float) -> "SobolevKernel3D":
        """In-place scalar multiplication: self *= a."""
        self._kernel = float(a) * self._kernel
        self._weighted_coeffs = float(a) * self._weighted_coeffs
        return self

    def __iadd__(self, other) -> "SobolevKernel3D":
        """In-place addition: self += other."""
        if not isinstance(other, SobolevKernel3D):
            raise TypeError(
                f"Cannot in-place add {type(other).__name__!r} "
                "to SobolevKernel3D."
            )
        if other.domain is not self.domain:
            raise ValueError(
                "Cannot in-place add SobolevKernel3D objects from different "
                "Sobolev3D instances (domain identity mismatch)."
            )
        self._kernel = self._kernel + other._kernel
        self._weighted_coeffs = self._weighted_coeffs + other._weighted_coeffs
        return self

    def __isub__(self, other) -> "SobolevKernel3D":
        """In-place subtraction: self -= other."""
        if not isinstance(other, SobolevKernel3D):
            raise TypeError(
                f"Cannot in-place subtract {type(other).__name__!r} "
                "from SobolevKernel3D."
            )
        if other.domain is not self.domain:
            raise ValueError(
                "Cannot in-place subtract SobolevKernel3D objects "
                "from different Sobolev3D instances "
                "(domain identity mismatch)."
            )
        self._kernel = self._kernel - other._kernel
        self._weighted_coeffs = self._weighted_coeffs - other._weighted_coeffs
        return self
