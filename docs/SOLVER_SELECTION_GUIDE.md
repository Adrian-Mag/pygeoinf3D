# pygeoinf3D Solver Selection Guide

pygeoinf3D provides multiple solver and measurement-operator classes that address
different problem types (minimum-norm inversion, general forward modelling,
Gaussian sampling). Compatibility between a solver and a given space/operator/form
combination is made explicit through the `SolverRegistry`: every built-in class is
registered with a `SolverSpec` that records which space types, operator types, and
functional types it requires. Use `SolverRouter` for automatic, runtime-checked
recommendations, or consult the compatibility matrix below for manual selection.

---

## Quick Start via SolverRouter

```python
from pygeoinf3d import SolverRouter

router = SolverRouter()
rec = router.recommend(sobolev_space, forms)

print(rec.solver_spec.name)  # e.g. "SpectralGramSolver"
print(rec.reason)
for w in rec.warnings:
    print("WARNING:", w)
```

For sampling, use `recommend_sampler`:

```python
rec = router.recommend_sampler(covariance_operator)
print(rec.solver_spec.name)  # e.g. "KLSampler3D"
```

A `SolverRecommendation` with `solver_spec=None` means no registered solver is
compatible with the given inputs.

---

## ASCII Decision Tree

```
Is the problem a MINIMUM-NORM INVERSION?
├── YES
│   Is the space Sobolev3D with a SpectralOperator3D inverse-weight
│   AND the provider has has_tensor_product_structure == True?
│   ├── YES
│   │   Are ALL measurements PathIntegralFunctional with the same n_quad?
│   │   ├── YES → SpectralGramSolver ✓ (FAST, O(m³) eigendecomposition)
│   │   └── NO  → SOLAOperator3D + pygeoinf.LinearMinimumNormInversion (general)
│   └── NO (Lebesgue3D or Sobolev3D without tensor-product spectral weight)
│       └── SOLAOperator3D + pygeoinf.LinearMinimumNormInversion (general)
└── NO: Are you SAMPLING from a Gaussian prior?
    └── Is the covariance operator a SpectralOperator3D?
        ├── YES → KLSampler3D ✓ (fast with tensor-product structure)
        └── NO  → Use pygeoinf.GaussianMeasure directly
```

---

## Compatibility Matrix

| Solver | Space | Operator / Weight | Forms | Purpose | Performance | Notes |
|---|---|---|---|---|---|---|
| `SpectralGramSolver` | `Sobolev3D` | `SpectralOperator3D` (e.g. `BesselSobolevBox`); provider must have `has_tensor_product_structure = True` (e.g. `BoxLaplacianSpectrumProvider`) | `PathIntegralFunctional` (uniform `n_quad`) | Inversion | **FAST** (1 000–100 000×) | Exploits batch spectral Gram; requires `BoxLaplacianSpectrumProvider` for vectorised analysis step |
| `SOLAOperator3D` | `Lebesgue3D` or `Sobolev3D` | Any (duck-typed via `space.function_domain`) | `VolumeKernel3D`, `PathIntegralFunctional`, `SumLinearForm`, or mixed stacks | Measurement operator | General (O(m·N) per eval; no pre-computation) | Not a standalone solver — combine with `pygeoinf.LinearMinimumNormInversion` |
| `KLSampler3D` | Any (via `SpectralOperator3D` covariance) | `SpectralOperator3D` (covariance operator) | N/A (sampling, not inversion) | Sampling | Fast when `has_tensor_product_structure = True` (vectorised einsum) | For drawing samples from a Gaussian measure; not for inversion |

---

## When NOT to Use Each Solver

**`SpectralGramSolver`**

This solver is tightly coupled to the spectral structure of `Sobolev3D` with a
`SpectralOperator3D` inverse-weight that has `has_tensor_product_structure = True`
(currently only `BoxLaplacianSpectrumProvider`-backed operators satisfy this).
Do not use it if: the function space is `Lebesgue3D`; the weight operator is
non-spectral or does not expose a `BoxLaplacianSpectrumProvider`; any measurement
form is a `VolumeKernel3D` rather than a `PathIntegralFunctional`; or the path
forms use different `n_quad` values (mixed quadrature resolution triggers a
`ValueError`).

**`SOLAOperator3D`**

`SOLAOperator3D` is a *measurement operator*, not a complete solver. It wraps a
stack of linear forms into a `pygeoinf`-compatible `LinearOperator` but does not
solve any system on its own. Always combine it with
`pygeoinf.LinearMinimumNormInversion` or another `pygeoinf` inversion class.
For large spectral bases with repeated solves, `SpectralGramSolver` is
substantially faster; prefer `SOLAOperator3D` only when the forms are mixed-type,
when the space is `Lebesgue3D`, or when the tensor-product condition is not met.

**`KLSampler3D`**

`KLSampler3D` is a *sampler*, not an inversion solver. Its output is a random
`Function3D` drawn from a Gaussian measure with the given covariance operator.
Do not use it to compute a least-norm or posterior-mean estimate. If you need
both a prior sample and an inversion, use `KLSampler3D` for prior sampling and
`SpectralGramSolver` (or `SOLAOperator3D` + `LinearMinimumNormInversion`) for
the inversion step.

---

## Programmatic Registry Inspection

```python
from pygeoinf3d import SOLVER_REGISTRY

# List all registered solvers with their purpose and performance tier
for spec in SOLVER_REGISTRY.list_all():
    print(spec.name, spec.purpose, spec.performance_tier)

# Find all inversion solvers
inversion_solvers = SOLVER_REGISTRY.find_by_purpose("inversion")

# Look up a specific solver by name
spec = SOLVER_REGISTRY.find_by_name("SpectralGramSolver")
print(spec.preconditions)
print(spec.notes)
```

---

## Adding a New Solver

Follow these five steps to register a new solver or operator in pygeoinf3D:

1. **Create the solver class** in `pygeoinf3d/solvers/<name>.py` (or the
   appropriate sub-module if it is an operator or sampler). Inherit from
   `pygeoinf.LinearOperator` (for operators) or define a standalone class with
   the expected interface.

2. **Create a `SolverSpec`** describing the solver's requirements:

   ```python
   from pygeoinf3d.solvers.registry import SolverSpec
   spec = SolverSpec(
       name="MySolver",
       solver_class=MySolver,
       geometry_types=(BoxDomain,),
       space_types=(Sobolev3D,),
       operator_types=(SpectralOperator3D,),
       functional_types=(PathIntegralFunctional,),
       purpose="inversion",          # one of: "inversion", "measurement_operator", "sampling"
       performance_tier="fast",      # one of: "fast", "general", "sampling"
       preconditions=["Condition A", "Condition B"],
       notes="One-paragraph description of when to use this solver.",
   )
   ```

3. **Register it** by calling `SOLVER_REGISTRY.register(spec)` inside
   `_register_defaults()` in `pygeoinf3d/solvers/registry.py` (or call
   `SOLVER_REGISTRY.register(spec)` directly in your module for extension
   packages).

4. **Add a logic branch** to `SolverRouter.recommend()` in
   `pygeoinf3d/solvers/router.py` if the new solver handles inversion (or update
   `recommend_sampler()` if it handles sampling). Follow the existing pattern:
   check type conditions on `space` and `forms`, call
   `SOLVER_REGISTRY.find_by_name("MySolver")`, and return a
   `SolverRecommendation` with an informative `reason` string.

5. **Write tests** for the new `SolverSpec` (check `name`, `purpose`,
   `performance_tier`), the router path (check `rec.solver_spec.name`), and
   any new solver logic. Update this guide and the living reference
   (`docs/agent-docs/references/living/pygeoinf3d-reference.md`) to reflect the
   new entry.
