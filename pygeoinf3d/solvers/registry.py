"""Solver registry for pygeoinf3d.

Provides :class:`SolverSpec`, :class:`SolverRegistry`, and a module-level
:data:`SOLVER_REGISTRY` singleton populated with all built-in solvers.

Importing this module does **not** pull in heavy solver/operator classes at
module level; all solver imports are deferred into :func:`_register_defaults`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Keep empty — real imports are done lazily below.


@dataclass
class SolverSpec:
    """Metadata record describing a solver or operator class.

    Args:
        name: Unique identifier used as the registry key.
        solver_class: The actual class being described.
        geometry_types: Geometry classes the solver requires (e.g. ``BoxDomain``).
        space_types: Function-space classes the solver requires (e.g. ``Sobolev3D``).
        operator_types: Operator/weight classes the solver requires
            (e.g. ``SpectralOperator3D``).
        functional_types: Linear-form classes the solver accepts.
        purpose: One of ``"inversion"``, ``"measurement_operator"``, or
            ``"sampling"``.
        performance_tier: One of ``"fast"``, ``"general"``, or ``"sampling"``.
        preconditions: Human-readable list of conditions that must hold before
            using this solver.
        notes: One-paragraph description of when to use this solver.
    """

    name: str
    solver_class: type
    geometry_types: tuple[type, ...]
    space_types: tuple[type, ...]
    operator_types: tuple[type, ...]
    functional_types: tuple[type, ...]
    purpose: str
    performance_tier: str
    preconditions: list[str] = field(default_factory=list)
    notes: str = ""


class SolverRegistry:
    """Simple name-keyed registry of :class:`SolverSpec` entries."""

    def __init__(self) -> None:
        self._specs: dict[str, SolverSpec] = {}

    def register(self, spec: SolverSpec) -> None:
        """Add *spec* to the registry under ``spec.name``."""
        self._specs[spec.name] = spec

    def find_by_name(self, name: str) -> SolverSpec:
        """Return the :class:`SolverSpec` registered under *name*.

        Raises:
            KeyError: If *name* is not in the registry.
        """
        if name not in self._specs:
            raise KeyError(f"No solver registered under name {name!r}")
        return self._specs[name]

    def list_all(self) -> list[SolverSpec]:
        """Return all registered specs in insertion order."""
        return list(self._specs.values())

    def find_by_purpose(self, purpose: str) -> list[SolverSpec]:
        """Return all specs whose ``purpose`` matches *purpose*."""
        return [s for s in self._specs.values() if s.purpose == purpose]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

SOLVER_REGISTRY: SolverRegistry = SolverRegistry()


def _register_defaults(registry: SolverRegistry) -> None:
    """Populate *registry* with the three built-in solver specs.

    All imports are local to avoid circular-import issues at module load time.
    """
    from pygeoinf3d.solvers.spectral_gram import SpectralGramSolver
    from pygeoinf3d.operators.sola import SOLAOperator3D
    from pygeoinf3d.sampling.kl_sampler import KLSampler3D
    from pygeoinf3d.spaces.sobolev import Sobolev3D
    from pygeoinf3d.spaces.lebesgue import Lebesgue3D
    from pygeoinf3d.operators.base import SpectralOperator3D
    from pygeoinf3d.spaces.forms import PathIntegralFunctional, VolumeKernel3D, SumLinearForm
    from pygeoinf3d.geometries.box.domain import BoxDomain

    registry.register(SolverSpec(
        name="SpectralGramSolver",
        solver_class=SpectralGramSolver,
        geometry_types=(BoxDomain,),
        space_types=(Sobolev3D,),
        operator_types=(SpectralOperator3D,),
        functional_types=(PathIntegralFunctional,),
        purpose="inversion",
        performance_tier="fast",
        preconditions=[
            "Domain must be a BoxDomain",
            "Function space must be Sobolev3D whose inverse_weight is a SpectralOperator3D"
            " (specifically BesselSobolevBox)",
            "All linear forms must be PathIntegralFunctional",
            "Spectral backend must be BoxLaplacianSpectrumProvider",
        ],
        notes=(
            "SpectralGramSolver exploits the tensor-product spectral structure of the"
            " Sobolev norm on a box domain to solve minimum-norm inversion problems at"
            " O(m³) eigendecomposition cost plus O(N·m·q) batch setup, making it"
            " 1 000–100 000× faster than generic approaches for typical seismic-tomography"
            " problem sizes.  Use this solver whenever the prior is a BesselSobolevBox"
            " covariance and all data functionals are path integrals."
        ),
    ))

    registry.register(SolverSpec(
        name="SOLAOperator3D",
        solver_class=SOLAOperator3D,
        geometry_types=(BoxDomain,),
        space_types=(Lebesgue3D, Sobolev3D),
        operator_types=(),
        functional_types=(VolumeKernel3D, PathIntegralFunctional, SumLinearForm),
        purpose="measurement_operator",
        performance_tier="general",
        preconditions=[
            "Domain is Lebesgue3D or Sobolev3D",
            "All forms must support forward, dual, and adjoint evaluation"
            " (i.e. from_dual must be computable)",
            "Forms may be VolumeKernel3D, PathIntegralFunctional, SumLinearForm,"
            " or mixed stacks",
        ],
        notes=(
            "SOLAOperator3D is not a standalone solver but a measurement operator that"
            " wraps a stack of linear forms into a pygeoinf-compatible forward operator."
            " Use it with pygeoinf.LinearMinimumNormInversion when a generic (non-spectral)"
            " inversion pipeline is required, or when the function space is Lebesgue3D."
            " Cost is O(m·N) per evaluation with no pre-computation."
        ),
    ))

    registry.register(SolverSpec(
        name="KLSampler3D",
        solver_class=KLSampler3D,
        geometry_types=(BoxDomain,),
        space_types=(Sobolev3D,),
        operator_types=(SpectralOperator3D,),
        functional_types=(),
        purpose="sampling",
        performance_tier="sampling",
        preconditions=[
            "Requires a SpectralOperator3D used as the covariance operator",
            "Fastest when SpectralOperator3D.has_tensor_product_structure is True"
            " (enables vectorised einsum path)",
        ],
        notes=(
            "KLSampler3D draws samples from a Gaussian random field whose covariance"
            " operator is a SpectralOperator3D.  The KL expansion is truncated at the"
            " requested number of modes; when the operator has tensor-product structure"
            " the sampler uses a vectorised einsum path that is significantly faster"
            " than the dense fallback."
        ),
    ))


_register_defaults(SOLVER_REGISTRY)
