"""Integration-backed linear forms for 3D function spaces.

This module provides :class:`VolumeKernel3D`, a :class:`~pygeoinf.LinearForm`
subclass whose evaluation is defined by
``∫ kernel(x) · g(x) dV`` rather than by a finite-dimensional component vector.

Using a kernel form avoids the component-enumeration assumption baked into
:class:`~pygeoinf.LinearForm` (which enumerates basis vectors via
``domain.from_components``) while remaining a first-class ``LinearForm``
for the rest of the pygeoinf ecosystem.

The trick mirrors ``intervalinf.spaces.forms.LinearFormKernel``:
the subclass overrides ``_mapping_impl`` with integration-based evaluation.
Because Python resolves ``self._mapping_impl`` dynamically in
``LinearForm.__init__``, the overridden method is what
:class:`~pygeoinf.NonLinearForm` stores and calls from ``__call__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from pygeoinf import LinearForm

if TYPE_CHECKING:
    from pygeoinf3d.core.functions import Function3D
    from pygeoinf3d.core.paths import Path3D
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


# ===========================================================================
# PathIntegralFunctional
# ===========================================================================

class PathIntegralFunctional(LinearForm):
    """A linear form on :class:`~pygeoinf3d.spaces.sobolev.Sobolev3D`
    defined by integration along a curve.

    The evaluation is:

    .. math::

        \\ell_\\gamma(f) = \\int_\\gamma f \\, \\mathrm{d}s
        \\approx |\\gamma| \\cdot \\frac{1}{n-1} \\left[
        \\frac{f(\\mathbf{p}_0)}{2} + \\sum_{i=1}^{n-2} f(\\mathbf{p}_i)
        + \\frac{f(\\mathbf{p}_{n-1})}{2} \\right]

    where :math:`|\\gamma|` is the arc length and
    :math:`\\mathbf{p}_0, \\ldots, \\mathbf{p}_{n-1}` are equally
    arc-length-spaced nodes (including endpoints) returned by
    :meth:`~pygeoinf3d.core.paths.Path3D.arc_length_nodes`.

    Parameters
    ----------
    space:
        The :class:`~pygeoinf3d.spaces.sobolev.Sobolev3D` on which this
        form is defined.  A plain
        :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` is rejected because
        path-integral functionals require the Sobolev regularity to admit
        a spectral representer under :meth:`Sobolev3D.from_dual`.
    path:
        The :class:`~pygeoinf3d.core.paths.Path3D` along which to integrate.
        If ``path.domain`` is not ``None`` it must be the same object as
        ``space.function_domain`` (identity check).
    n_quad:
        Number of quadrature nodes for the composite trapezoidal rule.
        Must be ≥ 2.  Default ``200``.

    Notes
    -----
    Like :class:`VolumeKernel3D`, this class passes ``components=np.zeros(0)``
    to bypass :class:`~pygeoinf.LinearForm`'s component-enumeration logic and
    routes all evaluation through the overridden ``_mapping_impl``.

    An optional ``_scale`` multiplier (default ``1.0``) is stored to support
    scalar algebra without copying the path object.
    """

    def __init__(
        self,
        space: "Sobolev3D",
        path: "Path3D",
        *,
        n_quad: int = 200,
        _scale: float = 1.0,
    ) -> None:
        from pygeoinf3d.core.paths import Path3D as _Path3D
        from pygeoinf3d.spaces.sobolev import Sobolev3D as _Sobolev3D
        if not isinstance(space, _Sobolev3D):
            raise TypeError(
                f"'space' must be a Sobolev3D instance; "
                f"got {type(space).__name__!r}.  "
                "PathIntegralFunctional requires a Sobolev3D domain so that "
                "path-integral representers can be computed via "
                "Sobolev3D.from_dual.  Wrap your Lebesgue3D with a "
                "Sobolev3D (e.g. using BesselSobolevBox as weight)."
            )
        if not isinstance(path, _Path3D):
            raise TypeError(
                f"'path' must be a Path3D instance; "
                f"got {type(path).__name__!r}."
            )
        if int(n_quad) < 2:
            raise ValueError(
                f"n_quad must be ≥ 2 for the composite trapezoidal rule; "
                f"got {n_quad!r}."
            )
        if path.domain is not None and path.domain is not space.function_domain:
            raise ValueError(
                "PathIntegralFunctional: path.domain does not match "
                "space.function_domain (identity check).  "
                "Pass a Path3D built with domain=None or with the same "
                "Domain3D instance as the Lebesgue3D space."
            )
        self._path: "Path3D" = path
        self._n_quad: int = int(n_quad)
        self._space: "Sobolev3D" = space
        self._scale: float = float(_scale)
        # Bypass component enumeration.
        super().__init__(space, components=np.zeros(0))

    # ------------------------------------------------------------------
    # Disable component access
    # ------------------------------------------------------------------

    @property
    def components(self) -> None:  # type: ignore[override]
        """Not available in basis-free mode.

        Raises
        ------
        TypeError
            Always.  ``PathIntegralFunctional`` has no finite-component
            representation.
        """
        raise TypeError(
            "PathIntegralFunctional has no finite-component representation "
            "in basis-free mode.  Use kernel-backed arithmetic instead of "
            "component operations."
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _mapping_impl(self, g: "Function3D") -> float:
        """Evaluate ``ℓ_γ(g)`` via composite trapezoidal rule."""
        t_arr, pts = self._path.arc_length_nodes(self._n_quad)
        vals = g(pts)  # batch call: shape (n_quad,)
        return self._scale * self._path.arclength * float(np.trapezoid(vals, t_arr))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> "Path3D":
        """The :class:`~pygeoinf3d.core.paths.Path3D` along which this
        form integrates."""
        return self._path

    @property
    def n_quad(self) -> int:
        """Number of quadrature nodes used in ``_mapping_impl``."""
        return self._n_quad

    # ------------------------------------------------------------------
    # Linear-form algebra
    # ------------------------------------------------------------------

    def copy(self) -> "PathIntegralFunctional":
        """Return a copy sharing the same immutable path."""
        return PathIntegralFunctional(
            self._space, self._path, n_quad=self._n_quad, _scale=self._scale
        )

    def __neg__(self) -> "PathIntegralFunctional":
        return PathIntegralFunctional(
            self._space, self._path, n_quad=self._n_quad, _scale=-self._scale
        )

    def __mul__(self, a: float) -> "PathIntegralFunctional":
        return PathIntegralFunctional(
            self._space, self._path,
            n_quad=self._n_quad, _scale=self._scale * float(a)
        )

    def __rmul__(self, a: float) -> "PathIntegralFunctional":
        return self.__mul__(a)

    def __truediv__(self, a: float) -> "PathIntegralFunctional":
        return self.__mul__(1.0 / float(a))

    def __add__(self, other) -> "PathIntegralFunctional":
        if not isinstance(other, PathIntegralFunctional):
            raise TypeError(
                f"Cannot add PathIntegralFunctional and "
                f"{type(other).__name__!r}.  Only two PathIntegralFunctional "
                "objects on the same path and space can be added."
            )
        if other._path is not self._path:
            raise ValueError(
                "Cannot add PathIntegralFunctional objects defined on "
                "different Path3D instances."
            )
        if other._space is not self._space:
            raise ValueError(
                "Cannot add PathIntegralFunctional objects from different "
                "Sobolev3D instances."
            )
        n = max(self._n_quad, other._n_quad)
        return PathIntegralFunctional(
            self._space, self._path,
            n_quad=n, _scale=self._scale + other._scale
        )

    def __sub__(self, other) -> "PathIntegralFunctional":
        if not isinstance(other, PathIntegralFunctional):
            raise TypeError(
                f"Cannot subtract {type(other).__name__!r} from "
                "PathIntegralFunctional.  Only two PathIntegralFunctional "
                "objects on the same path and space can be subtracted."
            )
        if other._path is not self._path:
            raise ValueError(
                "Cannot subtract PathIntegralFunctional objects defined on "
                "different Path3D instances."
            )
        if other._space is not self._space:
            raise ValueError(
                "Cannot subtract PathIntegralFunctional objects from "
                "different Sobolev3D instances."
            )
        n = max(self._n_quad, other._n_quad)
        return PathIntegralFunctional(
            self._space, self._path,
            n_quad=n, _scale=self._scale - other._scale
        )

    def __imul__(self, a: float) -> "PathIntegralFunctional":
        """In-place scalar multiplication: self *= a."""
        self._scale *= float(a)
        return self

    def __iadd__(self, other) -> "PathIntegralFunctional":
        """In-place addition: self += other."""
        if not isinstance(other, PathIntegralFunctional):
            raise TypeError(
                f"Cannot in-place add {type(other).__name__!r} to "
                "PathIntegralFunctional."
            )
        if other._path is not self._path:
            raise ValueError(
                "Cannot in-place add PathIntegralFunctional objects on "
                "different paths."
            )
        if other._space is not self._space:
            raise ValueError(
                "Cannot in-place add PathIntegralFunctional objects from "
                "different Sobolev3D instances."
            )
        self._scale += other._scale
        return self

    def __isub__(self, other) -> "PathIntegralFunctional":
        """In-place subtraction: self -= other."""
        if not isinstance(other, PathIntegralFunctional):
            raise TypeError(
                f"Cannot in-place subtract {type(other).__name__!r} from "
                "PathIntegralFunctional."
            )
        if other._path is not self._path:
            raise ValueError(
                "Cannot in-place subtract PathIntegralFunctional objects on "
                "different paths."
            )
        if other._space is not self._space:
            raise ValueError(
                "Cannot in-place subtract PathIntegralFunctional objects "
                "from different Sobolev3D instances."
            )
        self._scale -= other._scale
        return self

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PathIntegralFunctional("
            f"path={self._path!r}, "
            f"n_quad={self._n_quad}, "
            f"scale={self._scale})"
        )


# ===========================================================================
# SumLinearForm
# ===========================================================================

class SumLinearForm(LinearForm):
    """A weighted sum of :class:`~pygeoinf.LinearForm` objects on the same space.

    Evaluation is defined as:

    .. math::

        \\ell(f) = \\sum_i c_i \\cdot \\ell_i(f)

    This class is produced by :class:`~pygeoinf3d.operators.sola.SOLAOperator3D`
    ``_dual_mapping`` when the component forms cannot be directly added
    (e.g. :class:`PathIntegralFunctional` instances on different paths, or
    mixed form types such as :class:`VolumeKernel3D` and
    :class:`PathIntegralFunctional`).

    Parameters
    ----------
    space:
        A Hilbert space on which all component forms are defined.  Accepts
        :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D`,
        :class:`~pygeoinf3d.spaces.sobolev.Sobolev3D`, or any
        :class:`~pygeoinf.HilbertSpace`.  Every form in *terms* must satisfy
        ``form.domain is space`` (identity check).
    terms:
        A list of ``(coefficient, form)`` pairs.  Each *coefficient* is a
        :class:`float` and each *form* is a :class:`~pygeoinf.LinearForm`
        whose ``domain`` is *space*.

    Notes
    -----
    Like :class:`VolumeKernel3D` and :class:`PathIntegralFunctional`, this
    class passes ``components=np.zeros(0)`` to bypass
    :class:`~pygeoinf.LinearForm`'s component-enumeration logic and routes
    all evaluation through the overridden ``_mapping_impl``.

    ``SumLinearForm`` does **not** implement ``__add__`` / ``__sub__`` /
    ``__mul__`` algebra — it is a terminal representation produced by the
    SOLA dual map and consumed by Phase 5 representer logic.
    """

    def __init__(
        self,
        space,
        terms: "Sequence[tuple[float, LinearForm]]",
    ) -> None:
        if not terms:
            raise ValueError(
                "SumLinearForm requires at least one term."
            )
        for i, (c, form) in enumerate(terms):
            if not isinstance(form, LinearForm):
                raise TypeError(
                    f"terms[{i}][1] must be a LinearForm; "
                    f"got {type(form).__name__!r}."
                )
            if form.domain is not space:
                raise ValueError(
                    f"terms[{i}][1].domain is not the same space "
                    "instance as 'space' (identity check).  "
                    "All component forms must be defined on the same space."
                )
        self._terms: list[tuple[float, LinearForm]] = [
            (float(c), form) for c, form in terms
        ]
        super().__init__(space, components=np.zeros(0))

    # ------------------------------------------------------------------
    # Disable component access — this form is basis-free
    # ------------------------------------------------------------------

    @property
    def components(self) -> None:  # type: ignore[override]
        """Not available in basis-free mode.

        Raises
        ------
        TypeError
            Always.  ``SumLinearForm`` has no finite-component representation.
        """
        raise TypeError(
            "SumLinearForm has no finite-component representation in "
            "basis-free mode.  Evaluate the form directly via __call__."
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _mapping_impl(self, g: "Function3D") -> float:
        """Evaluate ``ℓ(g) = Σ c_i · ℓ_i(g)``."""
        return sum(c * form(g) for c, form in self._terms)

    # ------------------------------------------------------------------
    # Read-only access to terms
    # ------------------------------------------------------------------

    @property
    def terms(self) -> "list[tuple[float, LinearForm]]":
        """The list of ``(coefficient, form)`` pairs (read-only copy)."""
        return list(self._terms)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        pairs = ", ".join(
            f"({c!r}, {type(f).__name__})" for c, f in self._terms
        )
        return f"SumLinearForm(space={self.domain!r}, terms=[{pairs}])"
