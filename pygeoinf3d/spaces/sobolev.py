r"""Sobolev function space on a 3D domain.

Provides :class:`Sobolev3D`, a concrete :class:`~pygeoinf.HilbertSpace`
whose inner product is weighted by a positive-definite self-adjoint operator
*W* relative to an underlying :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D`
space:

.. math::

    \langle f,\, g \rangle_{H}
    = \langle W f,\, g \rangle_{L^2(\Omega)}
    = \int_\Omega (Wf)(\mathbf{x})\, g(\mathbf{x})\, \mathrm{d}V

Design notes
------------
*Pairing-first.*
  :meth:`to_dual` is authoritative and returns a
  :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` encoding
  :math:`g \mapsto \langle Wf,\, g \rangle_{L^2}`.  The inner product
  is derived from it.

*Basis-free.*
  ``dim == 0`` always. ``to_components`` and ``from_components`` raise
  :exc:`NotImplementedError`.

*Round-trip identity.*
  ``from_dual(to_dual(f)) == f`` pointwise, provided *inverse_weight* is
  a true left-inverse of *weight*:
  :math:`W^{-1}(Wf) = f`.

*Designed for Sobolev ladder.*
  The first intended weight back-end is
  :class:`~pygeoinf3d.operators.bessel_sobolev_box.BesselSobolevBox` with
  ``order=s`` and its paired ``order=-s`` operator as the inverse.  Later
  phases may add PDE-based or FEM-based weight operators without requiring
  any changes to :class:`Sobolev3D` itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pygeoinf import HilbertSpace

from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.spaces.forms import VolumeKernel3D


class Sobolev3D(HilbertSpace):
    r"""Sobolev Hilbert space on a
    :class:`~pygeoinf3d.core.domain_base.Domain3D`.

    The inner product is

    .. math::

        \langle f,\, g \rangle_{H}
        = \langle W f,\, g \rangle_{L^2(\Omega)}
        = \int_\Omega (Wf)(\mathbf{x})\, g(\mathbf{x})\, \mathrm{d}V

    where *W* is a positive-definite self-adjoint linear operator on the
    underlying :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` and
    *W*:sup:`-1` is its left-inverse.

    Parameters
    ----------
    base_space:
        The :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` supplying the
        domain geometry and :math:`L^2` pairing machinery.
    weight:
        A :class:`~pygeoinf.linear_operators.LinearOperator` from
        *base_space* to *base_space* representing *W*.  Must satisfy
        ``weight.domain == base_space`` and
        ``weight.codomain == base_space``.
    inverse_weight:
        A :class:`~pygeoinf.linear_operators.LinearOperator` from
        *base_space* to *base_space* representing :math:`W^{-1}`.  Must
        satisfy ``inverse_weight.domain == base_space`` and
        ``inverse_weight.codomain == base_space``.

    Raises
    ------
    TypeError
        If *base_space* is not a
        :class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D`,
        or if *weight* or *inverse_weight* do not expose ``.domain`` and
        ``.codomain`` attributes (i.e. are not operator-like objects).
        Positivity and self-adjointness of *weight* are caller contracts and
        are not verified here.
    ValueError
        If *weight* or *inverse_weight* have incompatible domain or codomain.

    Examples
    --------
    Using a scalar-multiple as a trivial weight::

        from pygeoinf.linear_operators import LinearOperator
        from pygeoinf3d import BoxDomain, Lebesgue3D, Function3D
        from pygeoinf3d.spaces.sobolev import Sobolev3D

        box   = BoxDomain(np.zeros(3), np.ones(3))
        space = Lebesgue3D(box)

        alpha = 2.0
        W    = LinearOperator(space, space, lambda f: alpha * f,
                              adjoint_mapping=lambda f: alpha * f)
        Winv = LinearOperator(space, space, lambda f: (1.0/alpha) * f,
                              adjoint_mapping=lambda f: (1.0/alpha) * f)

        H = Sobolev3D(space, W, Winv)
        ip = H.inner_product(f, g)   # = 2 · ∫ f·g dV

    Using :class:`~pygeoinf3d.operators.bessel_sobolev_box.BesselSobolevBox`
    as the weight operator::

        from pygeoinf3d import BoxLaplacianSpectrumProvider, BesselSobolevBox

        provider = BoxLaplacianSpectrumProvider(box, truncation=(8, 8, 8),
                                         bc_type='neumann')
        W    = BesselSobolevBox(space, provider, shift=1.0, order=2.0)
        Winv = BesselSobolevBox(space, provider, shift=1.0, order=-2.0)

        H  = Sobolev3D(space, W, Winv)
        fp = H.to_dual(f)          # VolumeKernel3D(space, W·f)
        f2 = H.from_dual(fp)       # W⁻¹(W·f) = f
    """

    def __init__(
        self,
        base_space: Lebesgue3D,
        weight: Any,
        inverse_weight: Any,
    ) -> None:
        if not isinstance(base_space, Lebesgue3D):
            raise TypeError(
                f"Sobolev3D expects a Lebesgue3D base_space; "
                f"got {type(base_space).__name__!r}."
            )
        if not (hasattr(weight, "domain") and hasattr(weight, "codomain")):
            raise TypeError(
                "Sobolev3D: 'weight' must be a linear operator with "
                "'.domain' and '.codomain' attributes "
                "(e.g. a pygeoinf LinearOperator); "
                f"got {type(weight).__name__!r}."
            )
        if not (
            hasattr(inverse_weight, "domain")
            and hasattr(inverse_weight, "codomain")
        ):
            raise TypeError(
                "Sobolev3D: 'inverse_weight' must be a linear operator with "
                "'.domain' and '.codomain' attributes "
                "(e.g. a pygeoinf LinearOperator); "
                f"got {type(inverse_weight).__name__!r}."
            )
        if weight.domain != base_space:
            raise ValueError(
                "weight.domain must equal base_space "
                "(both must be Lebesgue3D instances on the same Domain3D).  "
                f"Got weight.domain={weight.domain!r}."
            )
        if weight.codomain != base_space:
            raise ValueError(
                "weight.codomain must equal base_space "
                "(both must be Lebesgue3D instances on the same Domain3D).  "
                f"Got weight.codomain={weight.codomain!r}."
            )
        if inverse_weight.domain != base_space:
            raise ValueError(
                "inverse_weight.domain must equal base_space "
                "(both must be Lebesgue3D instances on the same Domain3D).  "
                f"Got inverse_weight.domain={inverse_weight.domain!r}."
            )
        if inverse_weight.codomain != base_space:
            raise ValueError(
                "inverse_weight.codomain must equal base_space "
                "(both must be Lebesgue3D instances on the same Domain3D).  "
                f"Got inverse_weight.codomain={inverse_weight.codomain!r}."
            )

        self._base_space = base_space
        self._weight = weight
        self._inverse_weight = inverse_weight

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def base_space(self) -> Lebesgue3D:
        """The underlying :class:`Lebesgue3D` supplying geometry and
        pairing.
        """
        return self._base_space

    @property
    def function_domain(self):
        """The :class:`~pygeoinf3d.core.domain_base.Domain3D` for this
        space.
        """
        return self._base_space.function_domain

    @property
    def weight(self):
        """The weight operator *W* : *base_space* → *base_space*."""
        return self._weight

    @property
    def inverse_weight(self):
        """The inverse weight operator *W*:sup:`-1` : *base_space* →
        *base_space*.
        """
        return self._inverse_weight

    # ------------------------------------------------------------------
    # HilbertSpace abstract interface
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Always ``0`` — :class:`Sobolev3D` is basis-free by design."""
        return 0

    def to_dual(self, f: Function3D) -> VolumeKernel3D:
        r"""Map a primal function to its Riesz dual.

        Returns a :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` that
        evaluates ``g ↦ ⟨Wf, g⟩_{L²} = ∫_Ω (Wf)(x)·g(x) dV``.

        The dual element is a :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D`
        attached to *base_space* with kernel :math:`Wf`.  Phase 3 (path
        integrals) will introduce non-volumetric functionals that do not
        pass through :meth:`to_dual`; they will instead be constructed
        directly as dual objects and will use :meth:`from_dual` to recover
        their Sobolev representers.

        Parameters
        ----------
        f:
            A :class:`~pygeoinf3d.core.functions.Function3D` in this space.

        Returns
        -------
        VolumeKernel3D
            Dual element representing *f* in the Sobolev inner product.            ``.domain`` is ``self`` (this :class:`Sobolev3D` instance),
            not the underlying ``base_space``.        """
        Wf = self._weight(f)
        return VolumeKernel3D(self, Wf)

    def from_dual(self, fp: Any) -> Function3D:
        r"""Recover a primal function from a dual element.

        Three resolution paths are supported:

        **Branch 1 — VolumeKernel3D** (round-trip from :meth:`to_dual`):
            Returns :math:`W^{-1}(\text{fp.kernel})`.

        **Branch 2 — SumLinearForm** (recursive):
            Applies :meth:`from_dual` to each term and accumulates the
            weighted sum of representers.

        **Branch 3 — General LinearForm** (spectral representer formula):
            Requires *inverse_weight* to be a
            :class:`~pygeoinf3d.operators.base.SpectralOperator3D`.
            Computes the representer

            .. math::

                r_\ell(\mathbf{x}) = \sum_{k=0}^{N-1}
                    \ell(\Phi_k) \cdot \nu_k \cdot \Phi_k(\mathbf{x}),

            where :math:`\Phi_k` are the eigenfunctions of *inverse_weight*
            and :math:`\nu_k = W^{-1}_k` are the corresponding eigenvalues.

        Parameters
        ----------
        fp:
            A dual element with ``.domain is self``: one of
            :class:`~pygeoinf3d.spaces.forms.VolumeKernel3D` (produced by
            :meth:`to_dual` or as a Sobolev measurement), a
            :class:`~pygeoinf3d.spaces.forms.SumLinearForm`, or any
            :class:`~pygeoinf.LinearForm` (requires *inverse_weight* to be a
            :class:`~pygeoinf3d.operators.base.SpectralOperator3D`).

        Returns
        -------
        Function3D
            The primal representer.

        Raises
        ------
        TypeError
            If *fp* is not one of the supported form types, or if
            *inverse_weight* is not a ``SpectralOperator3D`` when needed.
        ValueError
            If *fp* was produced by a different space instance
            (i.e. ``fp.domain is not self``).
        """
        # ── Branch 1: VolumeKernel3D ────────────────────────────────────────
        if isinstance(fp, VolumeKernel3D):
            if fp.domain is not self:
                raise ValueError(
                    "Sobolev3D.from_dual: the VolumeKernel3D was produced by a "
                    "different space instance than this Sobolev3D.  "
                    "Dual elements cannot cross space boundaries."
                )
            return self._inverse_weight(fp.kernel)

        # ── Branch 2: SumLinearForm (recursive) ─────────────────────────────
        from pygeoinf3d.spaces.forms import SumLinearForm  # avoid top-level circular
        if isinstance(fp, SumLinearForm):
            if fp.domain is not self:
                raise ValueError(
                    "Sobolev3D.from_dual: the SumLinearForm was not constructed on "
                    "this Sobolev3D.  "
                    "Dual elements cannot cross space boundaries."
                )
            result = self._base_space.zero
            for c_i, form_i in fp.terms:
                r_i = self.from_dual(form_i)
                result = self._base_space.add(
                    result, self._base_space.multiply(float(c_i), r_i)
                )
            return result

        # ── Branch 3: General LinearForm (spectral representer) ─────────────
        from pygeoinf import LinearForm as _LinearForm
        if isinstance(fp, _LinearForm):
            if fp.domain is not self:
                raise ValueError(
                    "Sobolev3D.from_dual: the LinearForm was not constructed on "
                    "this Sobolev3D.  "
                    "Dual elements cannot cross space boundaries."
                )
            from pygeoinf3d.operators.base import SpectralOperator3D
            if not isinstance(self._inverse_weight, SpectralOperator3D):
                raise TypeError(
                    "Sobolev3D.from_dual for non-VolumeKernel3D forms requires "
                    "inverse_weight to be a SpectralOperator3D "
                    "(e.g. BesselSobolevBox); "
                    f"got {type(self._inverse_weight).__name__!r}."
                )
            inv_w = self._inverse_weight
            n = inv_w.n_modes

            # Check whether a BoxLaplacianSpectrumProvider fast path is available.
            from pygeoinf3d.providers.tensor_product import (
                BoxLaplacianSpectrumProvider as _BLSP,
            )
            provider = getattr(inv_w, "_provider", None)
            has_fast_path = isinstance(provider, _BLSP)

            from pygeoinf3d.spaces.forms import PathIntegralFunctional as _PIF

            if has_fast_path and isinstance(fp, _PIF):
                # ── ULTRA-FAST PATH: vectorize BOTH analysis and synthesis ──
                # Analysis: evaluate ℓ on ALL eigenfunctions at once using the
                # cached quadrature nodes stored in the PathIntegralFunctional.
                #   Phi_quad[k, q] = φ_k(p_q)   shape (N, n_quad)
                #   c_k = scale · ∫ φ_k(p(t)) dt  ≈ scale · trapz(Phi_quad[k,:], t)
                Phi_quad = provider.evaluate_all_eigenfunctions(fp._pts)
                coeffs = fp._eval_scale * np.trapezoid(Phi_quad, fp._t_arr, axis=1)
                # Eigenvalues of the inverse-weight operator (not Laplacian eigs)
                nus = np.array([float(inv_w.get_eigenvalue(k)) for k in range(n)])
                alpha = coeffs * nus  # shape (n,)

                _alpha = alpha
                _prov = provider

                def evaluate(x, _a=_alpha, _p=_prov):
                    x = np.asarray(x, dtype=float)
                    is_single = x.ndim == 1
                    pts = x[None, :] if is_single else x
                    Phi = _p.evaluate_all_eigenfunctions(pts)  # (n, n_pts)
                    result = _a @ Phi  # (n_pts,)
                    return float(result[0]) if is_single else result

            elif has_fast_path:
                # ── FAST SYNTHESIS PATH: loop analysis, vectorized synthesis ──
                # Analysis: loop is unavoidable for a generic LinearForm
                # (we cannot assume anything about its quadrature structure).
                coeffs = np.array(
                    [float(fp(inv_w.get_eigenfunction(k))) for k in range(n)]
                )
                nus = np.array([float(inv_w.get_eigenvalue(k)) for k in range(n)])
                alpha = coeffs * nus  # shape (n,)

                _alpha = alpha
                _prov = provider

                def evaluate(x, _a=_alpha, _p=_prov):
                    x = np.asarray(x, dtype=float)
                    is_single = x.ndim == 1
                    pts = x[None, :] if is_single else x
                    Phi = _p.evaluate_all_eigenfunctions(pts)  # (n, n_pts)
                    result = _a @ Phi  # (n_pts,)
                    return float(result[0]) if is_single else result

            else:
                # ── FALLBACK: original loop over eigenfunctions ──────────────
                # Used when inverse_weight has no BoxLaplacianSpectrumProvider
                # (e.g. a custom SpectralOperator3D).
                coeffs = [float(fp(inv_w.get_eigenfunction(k))) for k in range(n)]
                nus = [float(inv_w.get_eigenvalue(k)) for k in range(n)]
                phis = [inv_w.get_eigenfunction(k) for k in range(n)]

                def evaluate(x, _c=coeffs, _nu=nus, _phi=phis):
                    x = np.asarray(x, dtype=float)
                    is_single = x.ndim == 1
                    pts = x[None, :] if is_single else x
                    total = np.zeros(len(pts))
                    for ck, nuk, phik in zip(_c, _nu, _phi):
                        total = total + ck * nuk * np.asarray(phik(pts), dtype=float)
                    return float(total[0]) if is_single else total

            return Function3D(
                self._base_space.function_domain, evaluate_callable=evaluate
            )

        # ── Fallback: unsupported type ───────────────────────────────────────
        raise TypeError(
            f"Sobolev3D.from_dual requires a VolumeKernel3D, SumLinearForm, "
            f"or LinearForm (backed by a SpectralOperator3D inverse_weight); "
            f"got {type(fp).__name__!r}."
        )

    def to_components(self, f: Function3D) -> np.ndarray:
        """Not implemented — :class:`Sobolev3D` is basis-free."""
        raise NotImplementedError(
            "Sobolev3D has no component representation; "
            "to_components is not available in basis-free mode."
        )

    def from_components(self, c: np.ndarray) -> Function3D:
        """Not implemented — :class:`Sobolev3D` is basis-free."""
        raise NotImplementedError(
            "Sobolev3D has no component representation; "
            "from_components is not available in basis-free mode."
        )

    # ------------------------------------------------------------------
    # Inner product (override for efficiency; avoids extra VolumeKernel3D
    # allocation when inner_product is called directly)
    # ------------------------------------------------------------------

    def inner_product(self, f: Function3D, g: Function3D) -> float:
        r"""Compute the Sobolev inner product.

        .. math::

            \langle f,\, g \rangle_{H}
            = \langle W f,\, g \rangle_{L^2(\Omega)}
            = \int_\Omega (Wf)(x)\, g(x)\, \mathrm{d}V

        Parameters
        ----------
        f, g:
            Primal functions in this space.

        Returns
        -------
        float
            The weighted inner product value.
        """
        return self._base_space.inner_product(self._weight(f), g)

    def norm(self, f: Function3D) -> float:
        r"""Compute the Sobolev norm
        :math:`\|f\|_H = \sqrt{\langle f,f\rangle_H}`.
        """
        ip = self.inner_product(f, f)
        return float(np.sqrt(max(ip, 0.0)))

    # ------------------------------------------------------------------
    # Vector-space operations — delegate to base_space
    # ------------------------------------------------------------------

    @property
    def zero(self) -> Function3D:
        """The zero function (additive identity) in this space."""
        return self._base_space.zero

    def add(self, f: Function3D, g: Function3D) -> Function3D:
        return self._base_space.add(f, g)

    def subtract(self, f: Function3D, g: Function3D) -> Function3D:
        return self._base_space.subtract(f, g)

    def multiply(self, a: float, f: Function3D) -> Function3D:
        return self._base_space.multiply(a, f)

    def negative(self, f: Function3D) -> Function3D:
        return self._base_space.negative(f)

    def copy(self, f: Function3D) -> Function3D:
        return self._base_space.copy(f)

    def is_element(self, x: Any) -> bool:
        return self._base_space.is_element(x)

    # ------------------------------------------------------------------
    # Equality and identity
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Two :class:`Sobolev3D` spaces are equal iff they share the same
        *base_space*, *weight*, and *inverse_weight* instances."""
        if not isinstance(other, Sobolev3D):
            return NotImplemented
        return (
            self._base_space == other._base_space
            and self._weight is other._weight
            and self._inverse_weight is other._inverse_weight
        )

    def __hash__(self) -> int:
        return hash((id(self._base_space.function_domain),
                     id(self._weight),
                     id(self._inverse_weight)))

    def __repr__(self) -> str:
        return (
            f"Sobolev3D("
            f"base_space={self._base_space!r}, "
            f"weight={self._weight!r})"
        )
