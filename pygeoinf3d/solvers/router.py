"""Solver router for pygeoinf3d.

Provides :class:`SolverRecommendation` and :class:`SolverRouter`, which
inspect a live :class:`~pygeoinf3d.spaces.sobolev.Sobolev3D` or
:class:`~pygeoinf3d.spaces.lebesgue.Lebesgue3D` space together with a list
of linear forms and recommend the best solver from :data:`SOLVER_REGISTRY`.

All heavy imports are deferred inside methods to avoid circular-import issues
at module load time, following the same pattern as
:mod:`pygeoinf3d.solvers.registry`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SolverRecommendation:
    """Result object returned by :meth:`SolverRouter.recommend`.

    Attributes
    ----------
    solver_spec:
        The recommended :class:`~pygeoinf3d.solvers.registry.SolverSpec`,
        or ``None`` if no compatible solver was found.
    reason:
        Human-readable explanation of why this solver was chosen (or why no
        recommendation could be made).
    warnings:
        Performance or correctness warnings relevant to this recommendation.
    """

    solver_spec: Any  # SolverSpec | None
    reason: str
    warnings: list[str] = field(default_factory=list)


class SolverRouter:
    """Inspects a space and list of forms and recommends the best solver.

    Decision order:

    1. **SpectralGramSolver** (fast) — requires :class:`Sobolev3D` with a
       :class:`SpectralOperator3D` inverse-weight whose underlying provider
       has :attr:`has_tensor_product_structure` ``== True``, and all forms
       must be :class:`PathIntegralFunctional` with the same ``n_quad``.
    2. **SOLAOperator3D** (general) — requires :class:`Lebesgue3D` or
       :class:`Sobolev3D` with all forms being
       :class:`VolumeKernel3D`, :class:`PathIntegralFunctional`, or
       :class:`SumLinearForm`.
    3. **No recommendation** — otherwise.

    A separate :meth:`recommend_sampler` method handles
    :class:`~pygeoinf3d.sampling.kl_sampler.KLSampler3D` recommendations.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self, space: Any, forms: list) -> SolverRecommendation:
        """Return the best solver for *space* and *forms*.

        Parameters
        ----------
        space:
            A concrete function space (typically :class:`Sobolev3D` or
            :class:`Lebesgue3D`).
        forms:
            List of linear-form objects describing the measurement operator.

        Returns
        -------
        SolverRecommendation
            The recommendation, which may have ``solver_spec=None`` if no
            registered solver is compatible.
        """
        # Lazy imports to avoid circular dependencies at module load time.
        from pygeoinf3d.spaces.sobolev import Sobolev3D
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        from pygeoinf3d.operators.base import SpectralOperator3D
        from pygeoinf3d.spaces.forms import (
            PathIntegralFunctional,
            VolumeKernel3D,
            SumLinearForm,
        )
        from pygeoinf3d.solvers.registry import SOLVER_REGISTRY

        # ------------------------------------------------------------------
        # 1. SpectralGramSolver fast path
        # ------------------------------------------------------------------
        if isinstance(space, Sobolev3D):
            inv_w = space.inverse_weight
            if isinstance(inv_w, SpectralOperator3D):
                provider = getattr(inv_w, "_provider", None)
                if (
                    hasattr(provider, "has_tensor_product_structure")
                    and provider.has_tensor_product_structure
                ):
                    if forms and all(isinstance(f, PathIntegralFunctional) for f in forms):
                        n_quads = {f.n_quad for f in forms}
                        if len(n_quads) == 1:
                            spec = SOLVER_REGISTRY.find_by_name("SpectralGramSolver")
                            return SolverRecommendation(
                                solver_spec=spec,
                                reason=(
                                    "Sobolev3D with spectral weight and all "
                                    "PathIntegralFunctional — fast spectral Gram "
                                    "path applies"
                                ),
                                warnings=[],
                            )

        # ------------------------------------------------------------------
        # 2. SOLAOperator3D general path
        # ------------------------------------------------------------------
        _sola_form_types = (VolumeKernel3D, PathIntegralFunctional, SumLinearForm)
        if isinstance(space, (Lebesgue3D, Sobolev3D)):
            if forms and all(isinstance(f, _sola_form_types) for f in forms):
                spec = SOLVER_REGISTRY.find_by_name("SOLAOperator3D")
                warnings: list[str] = []
                if isinstance(space, Sobolev3D):
                    warnings.append(
                        "This path is slower than SpectralGramSolver for large "
                        "spectral bases. Consider BesselSobolevBox + "
                        "PathIntegralFunctional for faster inversion."
                    )
                return SolverRecommendation(
                    solver_spec=spec,
                    reason=(
                        "General LinearForm stack — use SOLAOperator3D as forward "
                        "operator with pygeoinf.LinearMinimumNormInversion"
                    ),
                    warnings=warnings,
                )

        # ------------------------------------------------------------------
        # 3. No recommendation
        # ------------------------------------------------------------------
        return SolverRecommendation(
            solver_spec=None,
            reason=(
                "No compatible solver found in registry for the given space "
                "and form types."
            ),
            warnings=[],
        )

    def recommend_sampler(self, covariance_operator: Any) -> SolverRecommendation:
        """Return a sampler recommendation for *covariance_operator*.

        Parameters
        ----------
        covariance_operator:
            The covariance operator to be used for sampling (e.g. an
            :class:`~pygeoinf3d.operators.inverse_laplacian_box.InverseBoxLaplacian`
            or :class:`~pygeoinf3d.operators.bessel_sobolev_box.BesselSobolevBox`).

        Returns
        -------
        SolverRecommendation
            Recommends :class:`~pygeoinf3d.sampling.kl_sampler.KLSampler3D`
            when the operator is a :class:`SpectralOperator3D`, otherwise
            ``solver_spec=None``.
        """
        from pygeoinf3d.operators.base import SpectralOperator3D
        from pygeoinf3d.solvers.registry import SOLVER_REGISTRY

        if isinstance(covariance_operator, SpectralOperator3D):
            spec = SOLVER_REGISTRY.find_by_name("KLSampler3D")
            return SolverRecommendation(
                solver_spec=spec,
                reason=(
                    "SpectralOperator3D covariance — KL expansion sampling via "
                    "KLSampler3D is available"
                ),
                warnings=[],
            )

        return SolverRecommendation(
            solver_spec=None,
            reason=(
                "No compatible sampler found: operator is not a SpectralOperator3D."
            ),
            warnings=[],
        )

    def list_available(self) -> list:
        """Return all :class:`~pygeoinf3d.solvers.registry.SolverSpec` entries.

        Delegates to :meth:`SOLVER_REGISTRY.list_all`.
        """
        from pygeoinf3d.solvers.registry import SOLVER_REGISTRY

        return SOLVER_REGISTRY.list_all()
