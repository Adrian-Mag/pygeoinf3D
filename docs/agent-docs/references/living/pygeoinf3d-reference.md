# pygeoinf3D Living Reference

> **Status:** Foundation, SOLA operator, volumetric kernel providers, KL sampler / fast-path plot dispatch, least-norm demo, synced-slice comparison, `InverseBoxLaplacian`, `SpectralOperator3D`, and `GeneralizedBoxLaplacian` (all 5 phases) are **complete and archived**. `BoxLaplacian` and `InverseBoxLaplacian` are thin wrappers over `GeneralizedBoxLaplacian`; `SpectralPowerBase3D` provides the reusable spectral-power engine. The Lebesgue coefficient architecture (phases 1–2 done; phases 3–5 active) is the one remaining active track.
>
> **Active work:** `pygeoinf3d-lebesgue-coefficient-architecture-plan.md` (phases 3–5 — backend integration, operator wiring, end-to-end demos).
>
> **Last updated:** 2026-03-15

---

## Active Architecture Work

| Plan | Status |
|------|--------|
| `pygeoinf3d-lebesgue-coefficient-architecture-plan.md` | **Active** — phases 3–5 in progress |

All other plans (foundation, SOLA, volumetric-kernel-providers, least-norm demo, synced-slice comparison, KL fast-path plot, InverseBoxLaplacian, SpectralOperator3D KL sampler) have been moved to `completed-plans/`.

---

## Canonical Workflow

The recommended end-to-end path from geometry to measurements:

```python
import numpy as np
from pygeoinf import EuclideanSpace
from pygeoinf3d import (
    BoxDomain, Lebesgue3D, Function3D,
    FunctionToKernelAdapter, SOLAOperator3D,
)
from pygeoinf3d.providers.function_providers import GaussianProvider3D
from pygeoinf3d.plot import plot_function3d_axis_slice

# 1. Domain and L² space
box   = BoxDomain(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
space = Lebesgue3D(box)

# 2. Function providers  →  Function3D objects on demand
centers  = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
fprov    = GaussianProvider3D(box, centers=centers, sigmas=0.2)
# Other choices: BumpProvider3D, BoxcarProvider3D

# 3. Adapt to a KernelProvider3D  →  VolumeKernel3D objects on demand
kprov = FunctionToKernelAdapter(fprov, space)
# Alternatively build VolumeKernel3D objects explicitly and pass a list:
#   from pygeoinf3d import VolumeKernel3D
#   kernels = [VolumeKernel3D(space, fprov.get_function(i)) for i in range(fprov.dim)]

# 4. SOLA measurement operator  A: Lebesgue3D → R^m
op = SOLAOperator3D(space, EuclideanSpace(fprov.dim), kprov)

# 5. Apply to a function  →  measurement vector
#    Callable must handle (3,) single point → scalar and (n,3) batch → (n,)
def f_callable(x):
    x = np.asarray(x)
    return np.exp(-np.sum(x**2, axis=-1))

f    = Function3D(box, evaluate_callable=f_callable)
meas = op(f)   # shape (2,) — one volume integral per kernel

# 6. Optional: inspect a kernel on an axis slice
plot_function3d_axis_slice(kprov.get_kernel(0).kernel, axis='z', coord=0.0)
```

**Runnable demo notebook:**
- `demos/least_norm_inversion_demo.ipynb` shows the same workflow embedded in
    pygeoinf's `LinearForwardProblem` + `LinearMinimumNormInversion` interface,
    using a few dozen Gaussian volumetric kernels, visualizing both the
    recovered model and a representative kernel slice, and now including an
    optional Plotly slider for interactive axis-aligned slicing.  Phase 5 adds a
    synced three-panel comparison of the true model, least-norm solution, and
    absolute total kernel sensitivity via `compare_function3d_axis_sliders`.

**Volumetric-only boundary (current limitation):**
All `VolumeKernel3D` instances compute integrals over 3D volumes using
`Domain3D.integrate_volume`. The following kernel types are **not yet
implemented**:
- *Surface-integral kernels* — flux measurements or boundary data
- *Line-integral kernels* — path-averaged seismic travel times
- *Point-evaluation kernels* — seismometer point readings

These require `LinearForm` subclasses that pair against boundary traces or
point values rather than volumetric inner products. However, adding such
non-volumetric kernels is **not** a drop-in extension with the current pipeline:

- `KernelProvider3D.get_kernel` has return type `VolumeKernel3D`; a non-volumetric
  `LinearForm` subclass would require updating this contract.
- The sequence-backed `SOLAOperator3D` constructor enforces
  `isinstance(kernel, VolumeKernel3D)` and rejects any other `LinearForm` subclass.
- `Lebesgue3D.from_dual` only accepts `VolumeKernel3D`; the **adjoint mapping**
  will need a parallel extension when non-volumetric duals are introduced.

In summary, supporting non-volumetric kernels requires coordinated API and
implementation changes across `KernelProvider3D`, `SOLAOperator3D`, and
`Lebesgue3D.from_dual`. The design notes for this future extension are tracked by
the active Lebesgue coefficient architecture plan (phases 3–5); the earlier
standalone design note has been archived to `references/legacy/`.

---

## Package Overview

**pygeoinf3D** (`pygeoinf3d`) is an extension package for concrete 3D function spaces and operators built on top of `pygeoinf`. It uses `pygeoinf.EuclideanSpace(3)` and `pygeoinf.Subset` objects as the semantic geometry layer.

---

## Repository Layout

```
pygeoinf3D/
├── pyproject.toml
├── README.md
├── demos/
│   └── least_norm_inversion_demo.ipynb   # Noiseless minimum-norm inversion with SOLAOperator3D
├── pygeoinf3d/
│   ├── __init__.py          # Re-exports all public API
│   ├── plot.py              # Visualization (static slices + Plotly sliders + volume/isosurface helpers)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # AxisAlignedBoundingBox, CoordinateSystem
│   │   ├── region.py        # Region3D
│   │   ├── domain_base.py   # Domain3D, StructuredDomain3D
│   │   ├── boundary.py      # Boundary3D
│   │   └── functions.py     # Function3D  ← Phase 2
│   ├── spaces/
│   │   ├── __init__.py      # Re-exports Lebesgue3D, VolumeKernel3D
│   │   ├── forms.py         # VolumeKernel3D  ← Phase 2
│   │   └── lebesgue.py      # Lebesgue3D  ← Phase 2, updated Phase 4
│   ├── geometries/
│   │   ├── __init__.py
│   │   └── box/
│   │       ├── __init__.py
│   │       └── domain.py    # BoxDomain  ← Phase 3
│   ├── providers/
│   │   ├── __init__.py        # Re-exports FunctionProvider3D, KernelProvider3D, FunctionToKernelAdapter, TensorProductProvider; imports function_providers
│   │   ├── base.py            # FunctionProvider3D (ABC), KernelProvider3D (ABC), FunctionToKernelAdapter  ← SOLA Phase 3
│   │   ├── tensor_product.py  # TensorProductProvider  ← Phase 3
│   │   └── function_providers/
│   │       ├── __init__.py    # Re-exports FunctionProvider3D + 3 concrete providers ← kernel-providers Phase 1+2
│   │       ├── gaussian.py    # GaussianProvider3D  ← kernel-providers Phase 2
│   │       ├── bump.py        # BumpProvider3D       ← kernel-providers Phase 2
│   │       └── boxcar.py      # BoxcarProvider3D     ← kernel-providers Phase 2
│   ├── operators/
│   │   ├── __init__.py        # Re-exports SpectralOperator3D, SpectralPowerBase3D, BoxLaplacian, InverseBoxLaplacian, GeneralizedBoxLaplacian, SOLAOperator3D
│   │   ├── base.py            # SpectralOperator3D (ABC)  ← spectral-kl plan
│   │   ├── spectral_power_base.py  # SpectralPowerBase3D (abstract, ABC mixin)  ← generalized-box-laplacian Phase 2
│   │   ├── laplacian_box.py   # BoxLaplacian  ← thin wrapper over GeneralizedBoxLaplacian (power=1, alpha=1) — Phase 3
│   │   ├── inverse_laplacian_box.py  # InverseBoxLaplacian  ← pseudoinverse mode delegates to GeneralizedBoxLaplacian(power=-1); shift/strict use own apply — Phase 3
│   │   ├── generalized_box_laplacian.py  # GeneralizedBoxLaplacian (full impl, inherits SpectralPowerBase3D)  ← generalized-box-laplacian Phases 1–2
│   │   └── sola.py            # SOLAOperator3D  ← SOLA Phase 1+2+3
│   ├── sampling/              # KL expansion sampler  ← spectral-kl plan
│   │   ├── __init__.py        # Re-exports KLSampler3D
│   │   └── kl_sampler.py      # KLSampler3D
│   └── representations/       # Lebesgue coefficient architecture Phases 1-2
│       ├── __init__.py        # Re-exports all representation public API
│       ├── base.py            # CoefficientRepresentation3D (ABC), LebesgueCoefficientBackend (ABC)
│       ├── lebesgue.py        # LebesgueRepresentation3D (concrete)
│       ├── backends.py        # OrthonormalProjectionBackend, GramProjectionBackend  ← architecture Phase 2
│       └── gram.py            # assemble_gram, gram_operators helpers  ← architecture Phase 2
├── tests/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_region.py
│   │   ├── test_domain_base.py
│   │   ├── test_boundary.py
│   │   └── test_functions.py         ← Phase 2
│   ├── spaces/
│   │   ├── __init__.py
│   │   └── test_lebesgue.py          ← Phase 2, extended Phase 4
│   ├── geometries/
│   │   ├── __init__.py
│   │   └── test_box_domain.py        ← Phase 3
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── test_tensor_product.py             ← Phase 3
│   │   ├── test_function_providers.py         ← kernel-providers Phase 1 (package import + protocol contract)
│   │   └── test_concrete_function_providers.py ← kernel-providers Phase 2 (Gaussian / Bump / Boxcar)
│   ├── operators/
│   │   ├── __init__.py
│   │   ├── test_box_laplacian.py     ← Phase 3 + spectral interface tests + regression convention tests
│   │   ├── test_inverse_box_laplacian.py  ← InverseLaplacian plan + spectral interface tests + regression convention tests
    │   │   ├── test_generalized_box_laplacian.py  ← generalized-box-laplacian plan Phase 1 semantics and future-phase apply coverage
│   │   └── test_sola.py              ← SOLA Phase 1+2+3+4
│   ├── sampling/
│   │   ├── __init__.py
│   │   ├── test_kl_sampler.py        ← spectral-kl plan (23 tests)
│   │   └── test_kl_generalized_integration.py  ← generalized-box-laplacian plan Phase 4 (10 tests)
│   └── representations/              # Lebesgue coefficient architecture Phases 1-2
│       ├── __init__.py
│       ├── test_lebesgue_representation.py  ← architecture Phase 1
│       └── test_projection_backends.py     ← architecture Phase 2
│   └── test_plot.py                  ← Phase 3 (plot revision)
└── docs/
    └── agent-docs/
        ├── active-plans/
        ├── completed-plans/
        ├── references/
        │   └── living/
        │       └── pygeoinf3d-reference.md   ← this file
        └── theory/
```

---

## Dependency Policy

| Group       | Packages                            | Status         |
|-------------|-------------------------------------|----------------|
| Core        | numpy, scipy, matplotlib            | Required       |
| Geometry    | trimesh, meshzoo                    | Optional extra |
| Spherical   | pyshtools, Cartopy                  | Optional extra |
| Heavy FEM   | FEniCS, dolfinx, ngsolve, firedrake | Deferred v2+   |

---

## Architecture: Three-Layer Separation

```
pygeoinf.Subset / EuclideanSpace(3)    ← Pure geometry semantics (pygeoinf)
         │
         ▼
    Region3D                           ← Thin geometric wrapper (core/region.py)
         │                               Guarantees domain is EuclideanSpace(3)
         ▼
    Domain3D  (ABC)                    ← Computational domain (core/domain_base.py)
         │                               NOT a HilbertSpace
         ▼
    StructuredDomain3D  (ABC)          ← Optional: adds coord system + mesh + BCs
         │
         ▼
    BoxDomain / BallDomain (Phase 3+)  ← Concrete geometry implementations
```

Function spaces (`Lebesgue3D`, `Sobolev3D`, …) inherit from `pygeoinf.HilbertSpace`
and are defined **on** a `Domain3D` — they are not domains.

Planned dual elements such as point, path, surface, or volume distributions must
remain distinct from primal function-space elements. They should be modeled later
as separate dual objects that pair against sufficiently regular spaces rather than
being folded into `Function3D`.

In keeping with `pygeoinf`, dual pairings should be treated as more fundamental
than inner products. Future space APIs should therefore expose pairing-compatible
structure cleanly and treat inner products and Riesz maps as derived Hilbert-space
machinery rather than the only organizing principle.

---

## Core Classes (Phase 1)

### `AxisAlignedBoundingBox` — `pygeoinf3d.core.config`

A frozen-style dataclass for axis-aligned bounding boxes in ℝ³.

| Member | Type | Description |
|--------|------|-------------|
| `low` | `np.ndarray` (3,) | Lower corner |
| `high` | `np.ndarray` (3,) | Upper corner |
| `.center` | property → `ndarray` (3,) | Midpoint |
| `.extents` | property → `ndarray` (3,) | Side lengths |
| `.contains(point)` | method → `bool` | Closed membership test |

Raises `ValueError` if shapes are not `(3,)` or if any `low[i] > high[i]`.

---

### `CoordinateSystem` — `pygeoinf3d.core.config`

```python
class CoordinateSystem(Enum):
    CARTESIAN   # Standard (x, y, z)
    SPHERICAL   # (r, theta, phi)
    CYLINDRICAL # (r, phi, z)
```

Used by `StructuredDomain3D.coordinate_system`.

---

### `Region3D` — `pygeoinf3d.core.region`

Thin semantic wrapper around a `pygeoinf.Subset` in `EuclideanSpace(3)`.

```python
Region3D(subset: Subset, *, bounding_box: AxisAlignedBoundingBox | None = None)
```

| Member | Description |
|--------|-------------|
| `.subset` | The wrapped `Subset` |
| `.ambient_space` | Returns `EuclideanSpace(3)` |
| `.bounding_box()` | Returns the explicit or inferred `AxisAlignedBoundingBox` |
| `.contains(point)` | Delegates to `subset.is_element(point)` |
| `.boundary_subset` | Returns `subset.boundary` (e.g. `Sphere` for a `Ball`) |

If `bounding_box` is omitted, `Region3D` attempts to infer a finite
axis-aligned box for convex subsets via `support_function` or
`directional_bound` in the coordinate directions.

Raises `ValueError` if `subset.domain` is not `EuclideanSpace(3)` or if a
bounding box cannot be inferred.

---

### `Domain3D` — `pygeoinf3d.core.domain_base`

Abstract base for computational 3D domains. **Not** a `HilbertSpace`.

```python
class Domain3D(ABC):
    def __init__(self, region: Region3D): ...
```

| Member | Description |
|--------|-------------|
| `.dim` → `3` | Fixed physical dimension |
| `.ambient_space` → `EuclideanSpace(3)` | Shared cached instance |
| `.region` → `Region3D` | The underlying geometric region |
| `.contains(point)` | Delegates to `region.contains` |
| `.bounding_box()` | Delegates to `region.bounding_box()` (overridable) |
| `.sample_interior(n, *, rng)` | **Abstract** — returns `(n, 3)` array |
| `.integrate_volume(f, *, method, **kw)` | **Abstract** — returns `float` |
| `.boundary` | **Abstract property** — returns `Boundary3D` |

---

### `StructuredDomain3D` — `pygeoinf3d.core.domain_base`

Extends `Domain3D` for domains with natural coordinate/grid structure.

**Additional abstract members:**

| Member | Description |
|--------|-------------|
| `.coordinate_system` | **Abstract property** → `CoordinateSystem` |
| `.structured_mesh(resolution, *, location)` | **Abstract** — domain-specific mesh/grid |
| `.volume_weights(resolution, **kw)` | **Abstract** — quadrature weights array |
| `.boundary_components` | **Abstract property** → `dict[str, Boundary3D]` |

**Optional fast-path stubs** (raise `NotImplementedError` by default):

| Method | Description |
|--------|-------------|
| `.integrate_volume_on_mesh(values, weights)` | Fast integral on mesh values |
| `.sample_boundary(n, *, rng)` | Sample boundary surface points |
| `.outward_normal(points, *, component)` | Unit outward normals at boundary points |

---

### `Boundary3D` — `pygeoinf3d.core.boundary`

Lightweight first-class boundary object tied to a parent `Domain3D`.

```python
class Boundary3D(ABC):
    def __init__(self, parent_domain: Domain3D): ...
```

| Member | Description |
|--------|-------------|
| `.parent_domain` | The owning `Domain3D` |
| `.ambient_space` | Delegates to `parent_domain.ambient_space` |
| `.dim` → `2` | Topological dimension (surface in ℝ³) |
| `.contains(point)` | **Abstract** — boundary membership test |

---

## Phase 2 Classes

### `Function3D` — `pygeoinf3d.core.functions`

Primal scalar function on a `Domain3D`.  **Not** a distribution container.

```python
Function3D(space_or_domain, *, evaluate_callable: Callable, name: str | None = None)
```

Two construction modes:
- **Standalone** — pass a `Domain3D` directly.
- **Space-attached** — pass a space whose `.function_domain` is a `Domain3D`.

| Member | Description |
|--------|-------------|
| `.domain` | The `Domain3D` this function lives on |
| `.space` | Attached space or `None` |
| `.is_attached` | `True` if space is set |
| `.__call__(x)` | Evaluate at `(3,)` point or `(n,3)` batch |
| `.copy()` | Shallow copy (callable shared, safe due to immutability) |
| `+` / `-` / `-` (neg) | Arithmetic — requires same domain identity |
| `* scalar` / `scalar *` | Scalar multiplication |
| `.pointwise_mul(g)` | Pointwise product `(f⊙g)(x)=f(x)g(x)` for integrand formation |
| `.attach_to_space(space)` | Return attached copy; raises if wrong domain |
| `.detach()` | Return standalone copy |

Raises `ValueError` if `evaluate_callable` is omitted.  Raises `ValueError` on
arithmetic with functions on different domain instances.

---

### `Lebesgue3D` — `pygeoinf3d.spaces.lebesgue`

L² Hilbert space on a `Domain3D`.  Inherits from `pygeoinf.HilbertSpace`.

```python
Lebesgue3D(domain: Domain3D, *, representation: CoefficientRepresentation3D | None = None)
```

| Member | Description |
|--------|-------------|
| `.function_domain` | The underlying `Domain3D` |
| `.dim` | `0` when no representation is attached; `representation.dim` otherwise |
| `.to_dual(f)` | Maps `f` → `VolumeKernel3D` evaluating `g ↦ ∫f·g dV` — always available |
| `.from_dual(xp)` | Recovers `Function3D` from own `VolumeKernel3D`; rejects others — always available |
| `.to_components(f)` | Delegates to `representation.to_components(f)`; raises `NotImplementedError` if no rep attached |
| `.from_components(c)` | Delegates to `representation.from_components(c)`; raises `NotImplementedError` if no rep attached |
| `.coordinate_projection` | `LinearOperator(self → EuclideanSpace(dim))` — meaningful once representation is attached |
| `.coordinate_inclusion` | `LinearOperator(EuclideanSpace(dim) → self)` — meaningful once representation is attached |
| `.inner_product(f,g)` | `∫f·g dV` via `to_dual(f)(g)` — always available |
| `.norm(f)` | `√⟨f,f⟩` — always available |
| `.zero` | Constant zero `Function3D` |
| `.add/.subtract/.multiply/.negative` | Vector-space ops on `Function3D` |
| `==` | True iff `function_domain is other.function_domain` — ignores attached representation |

**Key invariant:** Space identity and all pairing-first semantics (`to_dual`, `from_dual`,
`inner_product`) are independent of any attached representation. Two `Lebesgue3D` objects
on the same `Domain3D` instance are equal even if one has a representation attached.

**Design boundary:** `Lebesgue3D` never owns quadrature policy, Gram matrices, or basis
ordering. Those belong in representation backends (Phase 2 of the architecture plan).

---

### `VolumeKernel3D` — `pygeoinf3d.spaces.forms`

Integration-backed `LinearForm` for `Lebesgue3D` pairings.

```python
VolumeKernel3D(space: Lebesgue3D, kernel: Function3D)
```

Inherits from `pygeoinf.LinearForm`.  Overrides `_mapping_impl` so
`__call__(g) = ∫_Ω kernel(x)·g(x) dV`.

| Member | Description |
|--------|-------------|
| `.kernel` | The primal `Function3D` representer |
| `.domain` | The `Lebesgue3D` this form is defined on |
| `.__call__(g)` | Numerical volume integration via `domain.function_domain.integrate_volume` |
| `copy()` | Returns a new `VolumeKernel3D` with a copied kernel |
| `__neg__` / `__mul__` / `__rmul__` / `__truediv__` | Kernel-backed out-of-place arithmetic |
| `__add__` / `__sub__` | Kernel-backed out-of-place add/sub; require same domain identity |
| `__imul__` | In-place scalar multiply — mutates stored kernel; overrides `LinearForm.__imul__` |
| `__iadd__` | In-place add — mutates stored kernel; rejects non-`VolumeKernel3D` and foreign domains |
| `__isub__` | In-place subtract — mutates stored kernel; same guard as `__iadd__` |

**Evaluation contract:**  `VolumeKernel3D(space, f)(g) ≈ ∫ f·g dV`.
Accuracy depends on the `Domain3D.integrate_volume` back-end.

**In-place safety note:**  `LinearForm.__imul__` and `LinearForm.__iadd__` mutate the
parent's `_components` array, which is a fake `np.zeros(0)` placeholder in
`VolumeKernel3D`.  These are overridden to mutate `self._kernel` instead.
`__isub__` is also overridden for consistency (Python's default fallback to
`__sub__` would silently rebind rather than mutate).

---

## Plotting Module — `pygeoinf3d.plot`

Slice-based and volumetric visualization utilities for `Function3D` objects.
Core Matplotlib slice plots are always available; optional Plotly support adds
interactive slice sliders, dropdown-controlled volume rendering, and 3D
isosurface views for notebook exploration.

### Public API

```python
from pygeoinf3d.plot import (
    evaluate_function3d_on_axis_slice,
    evaluate_function3d_on_regular_grid,
    plot_function3d_axis_slice,
    plot_function3d_axis_slider,
    plot_function3d_volume_dropdown,
    plot_function3d_isosurface,
    compare_function3d_axis_sliders,
)
```

#### `evaluate_function3d_on_axis_slice`

```python
evaluate_function3d_on_axis_slice(
    function: Function3D,
    axis: int | str,
    coord: float | None = None,
    resolution: int | tuple[int, int] = 100,
    *,
    plot_n_modes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Samples a `Function3D` on an axis-aligned planar slice of its domain.
Returns `(grid_u, grid_v, values)` all of shape `(nu, nv)`, where `u` and `v`
are the two axes orthogonal to the slice axis (in the order given by
`_get_other_axes`).

**Fast path (Phase 2):** If the function object carries a `_kl_fast_slice_evaluator`
attribute (set by `KLSampler3D` when the operator has tensor-product structure),
the vectorized einsum path is used instead of generic point-by-point evaluation.
Generic `Function3D` instances go through the original path unchanged.

**`plot_n_modes` (Phase 3):** When set to a positive integer, the fast path uses
only the first `plot_n_modes` KL modes instead of all `n_modes`. Silently ignored
for generic `Function3D` objects.

| Arg | Description |
|-----|-------------|
| `axis` | `'x'`/`'y'`/`'z'` or `0`/`1`/`2` |
| `coord` | slice position; defaults to domain bbox midpoint |
| `resolution` | positive `int` → `(n,n)`; 2-tuple of positive ints → `(nu,nv)` |
| `plot_n_modes` | KL mode cap for visualization; `None` = all modes |

Raises `ValueError` for: invalid axis, coord outside domain bounds,
malformed resolution, or `plot_n_modes ≤ 0`.

Batch-evaluates via `function(points)` where `points` has shape `(nu*nv, 3)`.
Coerces result with `np.asarray(...)` before reshape to handle scalar returns.

#### `evaluate_function3d_on_regular_grid`

```python
evaluate_function3d_on_regular_grid(
    function: Function3D,
    resolution: int = 25,
    *,
    plot_n_modes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

Samples a `Function3D` on a regular 3D axis-aligned grid covering the full
bounding box of the domain. Returns `(X, Y, Z, values)`, each with shape
`(resolution, resolution, resolution)`.

**Fast path (Phase 2):** If the function carries a `_kl_fast_grid_evaluator`
attribute (set by `KLSampler3D` for tensor-product operators), the vectorized
einsum path is used instead of generic batch evaluation.

**`plot_n_modes` (Phase 3):** When set, caps the number of KL modes used
during evaluation. Silently ignored for generic `Function3D` objects.

This is the shared sampling backend for the volumetric Plotly helpers.

#### `plot_function3d_axis_slice`

```python
plot_function3d_axis_slice(
    function: Function3D,
    axis: int | str,
    coord: float | None = None,
    resolution: int | tuple[int, int] = 100,
    show_plot: bool = True,
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
    plot_n_modes: int | None = None,
    **kwargs,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, dict[str, Any]]
```

Plots a `pcolormesh` heatmap of the slice.  Returns `(fig, ax, payload)` where
`payload` contains `{'grid_u', 'grid_v', 'values', 'axis_idx', 'coord'}`.

If `ax` is provided, `fig` is obtained via `cast(Figure, ax.figure)`.  The
`plot_n_modes` parameter is forwarded to the underlying evaluator.

#### `plot_function3d_axis_slider`

```python
plot_function3d_axis_slider(
    function: Function3D,
    axis: int | str = 'z',
    *,
    n_slices: int = 40,
    resolution: int | tuple[int, int] = 100,
    show_plot: bool = True,
    title: str | None = None,
    colorscale: str = 'Viridis',
    vmin: float | None = None,
    vmax: float | None = None,
    plot_n_modes: int | None = None,
) -> plotly.graph_objects.Figure
```

Creates a Plotly heatmap figure with slider-controlled frames spanning the
domain bounds along the chosen slice axis. The slider coordinate traverses
the box in `x`, `y`, or `z`, and the displayed heatmap uses the two orthogonal
axes on the plot plane.

| Arg | Description |
|-----|-------------|
| `axis` | `'x'`/`'y'`/`'z'` or `0`/`1`/`2` |
| `n_slices` | positive integer number of slider positions |
| `resolution` | positive `int` or positive 2-tuple for each 2D slice |
| `colorscale` | Plotly colorscale name |
| `vmin`, `vmax` | Optional fixed colour limits shared across all frames |
| `plot_n_modes` | KL mode cap for all slice evaluations (ignored for generic functions) |

Raises `ImportError` if Plotly is not installed. Install the optional extra:

```bash
pip install pygeoinf3d[interactive]
```

#### `plot_function3d_volume_dropdown`

```python
plot_function3d_volume_dropdown(
    functions: list[Function3D],
    *,
    resolution: int = 25,
    opacity: float = 0.15,
    surface_count: int = 15,
    colorscale: str = 'RdBu_r',
    robust_percentile: float = 95.0,
    labels: list[str] | None = None,
    show_plot: bool = True,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    plot_n_modes: int | None = None,
) -> plotly.graph_objects.Figure
```

Creates a Plotly `go.Volume` viewer with one trace per function and a dropdown
menu to switch between them. The functions are sampled internally on a shared
regular 3D grid, so notebook demos can stay at the level of `Function3D`
objects rather than manual point sampling.

If `vmin` and `vmax` are omitted, the colour range is chosen symmetrically
from the specified percentile of absolute sampled values.

#### `plot_function3d_isosurface`

```python
plot_function3d_isosurface(
    function: Function3D,
    *,
    resolution: int = 30,
    n_levels: int = 3,
    sigma: float = 1.0,
    opacity: float = 0.5,
    colorscale: str = 'RdBu_r',
    vmin: float | None = None,
    vmax: float | None = None,
    show_plot: bool = True,
    title: str | None = None,
    plot_n_modes: int | None = None,
) -> plotly.graph_objects.Figure
```

Creates a Plotly `go.Isosurface` figure for a single `Function3D`, sampling the
function internally on a regular 3D grid. If `vmin` and `vmax` are omitted,
the iso-range defaults to `± sigma * std(values)`.

#### `compare_function3d_axis_sliders`

```python
compare_function3d_axis_sliders(
    functions: list[Function3D],
    axis: int | str = 'z',
    *,
    n_slices: int = 40,
    resolution: int | tuple[int, int] = 100,
    show_plot: bool = True,
    titles: list[str] | None = None,
    colorscale: str = 'Viridis',
    vmin: float | None = None,
    vmax: float | None = None,
    plot_n_modes: int | None = None,
) -> plotly.graph_objects.Figure
```

Creates a Plotly figure with one subplot (heatmap) per function, all synced
to a single slider that moves the slice plane along the chosen axis.  Colour
limits are derived globally from all panels and all slice positions so that
panels are directly comparable.

| Arg | Description |
|-----|-------------|
| `functions` | Non-empty list of `Function3D` (all with compatible bounding boxes) |
| `axis` | `'x'`/`'y'`/`'z'` or `0`/`1`/`2` |
| `n_slices` | positive integer number of slider positions |
| `resolution` | positive `int` or positive 2-tuple for each 2D slice |
| `titles` | Per-panel titles; defaults to `function.name` or `'Function i'` |
| `colorscale` | Plotly colorscale name, shared across all panels |
| `vmin`, `vmax` | Optional fixed colour limits; inferred globally if omitted |

**Frame structure:** Each `go.Frame` carries `n_panels` `Heatmap` objects
and sets `traces=[0, 1, …, n_panels-1]`, so advancing the slider updates
all subplots simultaneously.

**Validation errors:**
- Empty `functions` → `ValueError`
- `titles` length ≠ `len(functions)` → `ValueError`
- Incompatible bounding boxes (not close to each other) → `ValueError`
- `n_slices ≤ 0` or malformed `resolution` → `ValueError`
- Plotly not installed → `ImportError`

### Private Helpers

| Helper | Description |
|--------|-------------|
| `_parse_axis(axis)` | Normalises `int`/`str` axis to `0`/`1`/`2` |
| `_get_other_axes(axis_idx)` | Returns `(u_idx, v_idx)` for the two free axes |
| `_validate_resolution(res)` | Validates and returns `(nu, nv)`; raises `ValueError` for non-positive or malformed input |
| `_validate_n_slices(n)` | Validates interactive slider length |
| `_import_plotly_graph_objects()` | Lazy optional Plotly import with install hint |
| `_import_plotly_make_subplots()` | Lazy import of `plotly.subplots.make_subplots` with same install hint |

### Type Notes

- `Axes` imported from `matplotlib.axes`, not `plt.Axes` (not exported).
- `Figure` imported from `matplotlib.figure`, not `plt.Figure` (not exported).
- `cast(Figure, ax.figure)` used when reusing existing axes (`ax.figure` returns
  `Figure | SubFigure`; cast narrows for the return type).
- Plotly is imported lazily inside `plot_function3d_axis_slider`, so the core
    plotting module remains usable without the `interactive` extra.

### Tests — `tests/test_plot.py`

| Test | Description |
|------|-------------|
| `test_evaluate_slice_z_center` | Slice at z=0.5; shape, bounds, and value at corner |
| `test_evaluate_slice_x_axis` | Slice at x=0 with tuple resolution (5,5); checks values=y+z |
| `test_plot_slice_smoke` | Runs with Agg backend; checks fig/ax/payload structure |
| `test_invalid_input` | Bad axis and out-of-bounds coord raise `ValueError` |
| `test_resolution_validation` | resolution=0, -5, (5,5,5), (0,5) all raise `ValueError` |
| `test_default_coord` | Omitting coord defaults to bbox midpoint |
| `test_plot_axis_slider_smoke` | Plotly slider figure has frames, labels, and a heatmap trace |
| `test_plot_axis_slider_x_axis_labels` | x-axis slider uses y/z plot axes and correct slider prefix |
| `test_plot_axis_slider_y_axis_labels` | y-axis slider uses x/z plot axes and correct slider prefix |
| `test_plot_axis_slider_numpy_integer_axis` | `np.integer` axis inputs are accepted |
| `test_plot_axis_slider_validation` | `n_slices <= 0` raises `ValueError` |
| `test_plot_axis_slider_requires_plotly` | Missing Plotly raises the optional-dependency `ImportError` |
| `test_compare_function3d_axis_sliders_smoke` | 2-panel figure has 2 data traces, 4 frames, each frame has 2 data entries |
| `test_compare_function3d_axis_sliders_titles` | Custom titles appear in `fig.layout.annotations` |
| `test_compare_function3d_axis_sliders_single_panel` | Single function works; 1 trace, correct slider prefix |
| `test_compare_function3d_axis_sliders_axis_labels` | z-axis slider → x/y plot axis labels |
| `test_compare_function3d_axis_sliders_validation_empty` | Empty functions list raises `ValueError` |
| `test_compare_function3d_axis_sliders_validation_titles_length` | Mismatched titles length raises `ValueError` |
| `test_compare_function3d_axis_sliders_validation_bbox_mismatch` | Incompatible bounding boxes raise `ValueError` |
| `test_compare_function3d_axis_sliders_requires_plotly` | Monkeypatched missing Plotly raises `ImportError` |

---

## Phase 3 Classes

### `BoxDomain` — `pygeoinf3d.geometries.box.domain`

Concrete `StructuredDomain3D` for axis-aligned boxes in ℝ³.

```python
BoxDomain(low_or_bbox: np.ndarray | AxisAlignedBoundingBox,
          high: np.ndarray | None = None)
```

Accepts either `BoxDomain(low, high)` or `BoxDomain(AxisAlignedBoundingBox(...))`.

| Member | Description |
|--------|-------------|
| `.bounding_box()` | Returns internal `AxisAlignedBoundingBox` |
| `.center`, `.extents`, `.volume` | Properties from AABB |
| `.coordinate_system` | `CoordinateSystem.CARTESIAN` |
| `.contains(point)` | Direct AABB bounds test (overrides Domain3D default) |
| `.sample_interior(n, rng)` | `rng.uniform(low, high, (n,3))` |
| `.structured_mesh(resolution, location)` | `(N,3)` ndarray. `location='cell_centers'` (default) produces midpoints; `location='nodes'` produces boundary-inclusive linspace grid |
| `.volume_weights(resolution, location)` | `(N,)` quadrature weights summing exactly to `volume` |
| `.integrate_volume(f, resolution, location)` | Tensor-product midpoint/trapezoidal quadrature |
| `.integrate_volume_on_mesh(values, weights)` | `np.dot(values, weights)` |
| `.boundary` | `_BoxBoundary` — full surface |
| `.boundary_components` | Dict `{'x_min','x_max','y_min','y_max','z_min','z_max'}` → `_BoxFaceBoundary` |

**Private helpers:**
- `_BoxSubset(Subset)` — wraps AABB; `is_element` uses coordinate bounds with `rtol` tolerance.
- `_BoxBoundary(Boundary3D)` — `contains(point)` checks any of 6 faces.
- `_BoxFaceBoundary(Boundary3D)` — single face. Args: `(parent_domain, axis: int, coord_value: float)`.

---

### `TensorProductProvider` — `pygeoinf3d.providers.tensor_product`

Spectral basis manager for tensor-product eigenfunctions of − ∇² on a box.

```python
TensorProductProvider(domain_or_space,
                      *,
                      truncation: int | tuple[int,int,int] = (4,4,4),
                      bc_type: str = 'dirichlet')
```

Accepts `BoxDomain` directly or any object with `.function_domain`.
`bc_type` must be `'dirichlet'` or `'neumann'`.

Eigenfunctions are sorted by ascending eigenvalue, with lexicographic `(kx,ky,kz)` tie-break.

**Dirichlet modes** (kx,ky,kz ≥ 1):
$$\phi_{k_x k_y k_z}(x,y,z) = \sqrt{\frac{8}{L_x L_y L_z}}\,\sin\!\left(\frac{k_x\pi(x-a_x)}{L_x}\right)\sin\!\left(\frac{k_y\pi(y-a_y)}{L_y}\right)\sin\!\left(\frac{k_z\pi(z-a_z)}{L_z}\right)$$

**Neumann modes** (kx,ky,kz ≥ 0, with cosines, k=0 gives constant):
same structure with cosines and normalisation factor $1/\sqrt{L}$ for k=0.

| Member | Description |
|--------|-------------|
| `.dim` | `nx * ny * nz` — total number of modes |
| `.get_multi_index(i)` | `(kx,ky,kz)` for flat index `i` |
| `.get_index(multi_index)` | Flat index; raises `KeyError` if not in truncation |
| `.get_eigenvalue(i)` | Eigenvalue λ = $(k_x\pi/L_x)^2 + (k_y\pi/L_y)^2 + (k_z\pi/L_z)^2$ |
| `.get_eigenfunction(i)` | `Function3D` computing φ_i; handles `(3,)` and `(n,3)` |
| `.get_function(i)` | Alias for `.get_eigenfunction(i)` to satisfy generic function-provider consumers |
| `__iter__` | Yields `(eigenvalue, Function3D)` pairs |

---

### `SOLAOperator3D` — `pygeoinf3d.operators.sola`

Measurement operator stacking volume kernels into a finite-dimensional
codomain.  Inherits from `pygeoinf.LinearOperator`.  **Phase 4 complete:**
forward, dual, and adjoint mappings all basis-free on the domain side;
provider-backed lazy + cached kernel access; public API locked with export
tests; extension boundary documented (see below).

```python
SOLAOperator3D(
    domain: Lebesgue3D,
    codomain,                         # any HilbertSpace with dim == n_kernels
    kernels: Sequence[VolumeKernel3D] | KernelProvider3D,
)
```

Accepts either a **sequence** of `VolumeKernel3D` objects (eager, sequence-backed)
or a **`KernelProvider3D`** instance (lazy, provider-backed with internal caching).

The operator evaluates:
$$Af = \bigl(\ell_1(f),\,\ldots,\,\ell_m(f)\bigr) \in W, \quad
    \ell_i(f) = \int_\Omega k_i(\mathbf{x})\,f(\mathbf{x})\,\mathrm{d}V.$$

Output is assembled as `codomain.from_components(raw_components)` where
`raw_components[i] = kernels[i](f)`.

**Dual mapping** ($A^*: W^* \to V^*$):
$$A^*(y') = \sum_i c_i \ell_i, \quad c = \texttt{codomain.dual.to\_components}(y').$$
Returns a `VolumeKernel3D` with kernel $\sum_i c_i k_i$.
Never calls `Lebesgue3D.to_components` or `from_components`.

**Adjoint mapping** ($A^\dagger: W \to V$):
```
A†(y) = domain.from_dual(A*(codomain.to_dual(y)))
```
Correct for `EuclideanSpace` and `MassWeightedHilbertSpace` codomains.

**Constructor validation (sequence-backed):**
- Raises `TypeError` if `domain` is not a `Lebesgue3D`.
- Raises `ValueError` if `len(kernels) != codomain.dim`.
- Raises `TypeError` if any element of `kernels` is not a `VolumeKernel3D`.
- Raises `ValueError` if any kernel's `.domain` is not the exact same
  `Lebesgue3D` instance as `domain` (identity check).

**Constructor validation (provider-backed):**
- Same `TypeError` / `ValueError` for domain/dim checks.
- Per-kernel compatibility is deferred to first access (lazy).

| Member | Description |
|--------|-------------|
| `.kernels` | Realized `list[VolumeKernel3D]`; for sequence mode returns copy; for provider mode, materializes lazily via cache |
| `._get_kernel(i)` | Internal: returns `_kernels[i]` (sequence) or cached `_provider.get_kernel(i)` (provider) |
| `.__call__(f)` | Forward mapping; returns codomain element via `from_components` |
| `.dual(yp)` | Dual mapping; returns weighted `VolumeKernel3D` sum of kernels |
| `.adjoint(y)` | Adjoint mapping; returns `Function3D` via Riesz round-trip |
| `.domain` | The `Lebesgue3D` passed at construction |
| `.codomain` | The finite-dimensional target space |

**Design invariants:**
- Does **not** call `Lebesgue3D.to_components` / `from_components` on the domain.
- Does **not** use `LinearOperator.from_linear_forms`.
- Dual uses `codomain.dual.to_components(yp)`, **not** `codomain.to_components(y)`.
- Provider-backed mode uses `self._kernel_cache: dict[int, VolumeKernel3D]` to ensure repeated `_get_kernel(i)` calls return identical objects.

**Extension boundary (Phase 4):**

The forward and dual APIs are **representation-independent**: they work for any `VolumeKernel3D` sequence or `KernelProvider3D` without accessing domain coefficients or any basis.

The adjoint (`A†`) is **narrower**: it calls `domain.from_dual(xp)` to realise the dual element as a primal `Function3D`. `Lebesgue3D.from_dual` currently only accepts `VolumeKernel3D` instances that belong to the **same space instance**. Because `SOLAOperator3D.dual()` always returns a `VolumeKernel3D` on the same domain, the round-trip holds for all current usage — but only due to this alignment.

Consequence for extensibility:
- **Non-trivial to generalise**: introducing new kernel families (e.g. surface integrals, point evaluations) requires coordinated API changes: (a) the `KernelProvider3D.get_kernel` return type is currently annotated as `VolumeKernel3D`; (b) the sequence-backed `SOLAOperator3D` constructor enforces `isinstance(kernel, VolumeKernel3D)` and rejects other `LinearForm` subclasses; and (c) the provider-backed constructor defers but would still fail at first access. Non-volumetric kernels therefore cannot be plugged in via the current pipeline without changing these type contracts.
- **Adjoint also requires extension**: if the dual-mapping return type ever changes to something other than a `VolumeKernel3D`, `domain.from_dual` will raise `TypeError`. To support adjoint on such spaces, either override `adjoint` in the subclass or extend `Lebesgue3D.from_dual` to handle the new dual type.

---

### `FunctionProvider3D` — `pygeoinf3d.providers.base`

Abstract base for objects that supply `Function3D` on indexed demand.

```python
class FunctionProvider3D(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def get_function(self, i: int) -> Function3D: ...
```

| Member | Description |
|--------|-------------|
| `.dim` | **Abstract** — total number of functions |
| `.get_function(i)` | **Abstract** — return the *i*-th `Function3D`; raises `IndexError` if out of range |

---

### `KernelProvider3D` — `pygeoinf3d.providers.base`

Abstract base for objects that supply `VolumeKernel3D` on indexed demand.

```python
class KernelProvider3D(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def get_kernel(self, i: int) -> VolumeKernel3D: ...
```

| Member | Description |
|--------|-------------|
| `.dim` | **Abstract** — total number of kernels |
| `.get_kernel(i)` | **Abstract** — return the *i*-th `VolumeKernel3D`; raises `IndexError` if out of range |

Designed for structural subtyping (ABC).  Custom providers inherit from it.
`FunctionToKernelAdapter` is the standard concrete implementation.

---

### `FunctionToKernelAdapter` — `pygeoinf3d.providers.base`

Adapts a `Function3D` provider (e.g. `TensorProductProvider`) into a `KernelProvider3D`.

```python
FunctionToKernelAdapter(function_provider, space: Lebesgue3D)
```

`function_provider` must expose `dim: int` and `get_function(i: int) -> Function3D`.

| Member | Description |
|--------|-------------|
| `.dim` | Delegates to `function_provider.dim` |
| `.get_kernel(i)` | Returns `VolumeKernel3D(space, function_provider.get_function(i))` |

Compatibility between `function_provider` and `space` is validated lazily:
`VolumeKernel3D` will raise `ValueError` on the first call if the provider's domain
does not match `space.function_domain` (identity check).

**Typical usage:**
```python
box = BoxDomain(np.zeros(3), np.ones(3))
space = Lebesgue3D(box)
provider = TensorProductProvider(box, truncation=4, bc_type='dirichlet')
adapter = FunctionToKernelAdapter(provider, space)
op = SOLAOperator3D(space, EuclideanSpace(adapter.dim), adapter)
```

---

## Kernel-Providers Phase 2 Classes

### `GaussianProvider3D` — `pygeoinf3d.providers.function_providers.gaussian`

Produces `Function3D` objects with anisotropic Gaussian profiles.

```python
GaussianProvider3D(domain, centers, sigmas, *, normalized=True)
```

| Argument | Type/Shape | Description |
|----------|-----------|-------------|
| `domain` | `Domain3D` | Computational domain |
| `centers` | `(n,3)` or `(3,)` | Gaussian centres; `(3,)` → `dim=1` |
| `sigmas` | scalar / `(3,)` / `(n,3)` | Per-axis std-devs; must be > 0 |
| `normalized` | `bool` (default `True`) | If True, ∫ f_i dV = 1 analytically |

$$f_i(\mathbf{x}) = A_i \exp\!\left(-\tfrac{1}{2}\sum_j \tfrac{(x_j-c_{ij})^2}{\sigma_{ij}^2}\right),
\quad A_i = \bigl((2\pi)^{3/2}\sigma_{ix}\sigma_{iy}\sigma_{iz}\bigr)^{-1} \text{ if normalised}$$

| Member | Description |
|--------|-------------|
| `.dim` | `len(centers)` |
| `.get_function(i)` | Returns Gaussian centred at `centers[i]`; raises `IndexError` if out of range |

**Raises:** `ValueError` for non-positive sigmas, wrong centers shape, or incompatible sigmas shape.

---

### `BumpProvider3D` — `pygeoinf3d.providers.function_providers.bump`

Axis-separable $C^\infty$ compactly-supported bump function provider.

```python
BumpProvider3D(domain, centers, widths, *, k=1.0, normalized=True)
```

| Argument | Type/Shape | Description |
|----------|-----------|-------------|
| `domain` | `Domain3D` | Computational domain |
| `centers` | `(n,3)` or `(3,)` | Bump centres |
| `widths` | scalar / `(3,)` / `(n,3)` | Full widths per axis; half-width = `widths/2` |
| `k` | `float` (default `1.0`) | Shape parameter; larger → more concentrated |
| `normalized` | `bool` (default `True`) | If True, ∫ f_i dV = 1 |

$$f_i(\mathbf{x}) = C_i \prod_{j=1}^{3} \phi_k\!\left(\tfrac{2(x_j-c_{ij})}{w_{ij}}\right), \quad \phi_k(t) = \begin{cases} \exp\!\left(\tfrac{-kt^2}{1-t^2}\right) & |t|<1 \\ 0 & |t|\ge 1 \end{cases}$$

Compact support: $|x_j - c_{ij}| \le w_{ij}/2$ for all $j$. Normalization constant $C_i$
computed via `scipy.integrate.quad` (cached per unique $k$).

| Member | Description |
|--------|-------------|
| `.dim` | `len(centers)` |
| `.get_function(i)` | Returns separable bump; support is axis-aligned box; **exactly zero** outside support |

**Raises:** `ValueError` for non-positive widths, $k\le 0$, wrong centers shape.

---

### `BoxcarProvider3D` — `pygeoinf3d.providers.function_providers.boxcar`

Axis-aligned box-car (indicator) function provider.

```python
BoxcarProvider3D(domain, centers, widths, *, normalized=True)
```

| Argument | Type/Shape | Description |
|----------|-----------|-------------|
| `domain` | `Domain3D` | Computational domain |
| `centers` | `(n,3)` or `(3,)` | Box centres |
| `widths` | scalar / `(3,)` / `(n,3)` | Full widths per axis; must be > 0 |
| `normalized` | `bool` (default `True`) | If True, h = 1/(w_x·w_y·w_z) so ∫ f_i dV = 1 |

$$f_i(\mathbf{x}) = \begin{cases} h_i & |x_j - c_{ij}| \le w_{ij}/2\ \forall j \\ 0 & \text{otherwise} \end{cases}$$

| Member | Description |
|--------|-------------|
| `.dim` | `len(centers)` |
| `.get_function(i)` | Returns box-car function; **exactly 1 (or h)** inside box, **exactly 0** outside |

**Raises:** `ValueError` for non-positive widths, wrong centers shape, incompatible widths shape.

---

### Public API — `pygeoinf3d.providers.function_providers`

```python
from pygeoinf3d.providers.function_providers import (
    FunctionProvider3D,   # re-export of abstract base
    GaussianProvider3D,
    BumpProvider3D,
    BoxcarProvider3D,
)
```

All three concrete providers:
- Subclass `FunctionProvider3D` directly
- Accept `Domain3D` (`BoxDomain` in practice) at construction
- Support `dim` = `len(centers)`
- Support `get_function(i)` with `IndexError` for out-of-range `i`
- Preserve domain identity in produced `Function3D` objects
- Are directly compatible with `FunctionToKernelAdapter`

---

### `SpectralOperator3D` — `pygeoinf3d.operators.base`

Abstract base class for operators that are spectrally implemented (i.e. diagonalised in an
eigenfunction basis). Extends `pygeoinf.LinearOperator`. Any operator that applies a
spectral projection should inherit from this class so that generic utilities
(`KLSampler3D`, future spectral solvers) can consume its eigenpairs.

**Required abstract interface:**
| Symbol | Description |
|---|---|
| `get_eigenvalue(index: int) → float` | Operator eigenvalue for mode *k* |
| `get_eigenfunction(index: int) → Function3D` | L²-normalised eigenfunction Φ_k |
| `n_modes: int` (property) | Truncation dimension *N* |

Note: the *operator* eigenvalue need not be the Laplacian eigenvalue — for
`InverseBoxLaplacian` it is the (modified) inverted eigenvalue 1/λ_k.

---

### `BoxLaplacian` — `pygeoinf3d.operators.laplacian_box`

Spectral negative Laplacian − ∇² on a box domain. Inherits from `SpectralOperator3D`.

**Future design note:**
See `docs/agent-docs/references/legacy/operator-specific-integration-future-note.md`
for the architectural argument that the current quadrature-based projection is
an acceptable generic fallback, but dedicated operator-specific numerical
backends (for example separable transform backends for box domains) should sit
behind the operator rather than being pushed into `BoxDomain`.

```python
BoxLaplacian(space: Lebesgue3D,
             provider: TensorProductProvider,
             quad_resolution: int = 20)
```

Validates `space.function_domain is provider._box`. Constructed as self-adjoint via
`LinearOperator.__init__(space, space, self._apply, adjoint_mapping=self._apply)`.

**`SpectralOperator3D` interface:** `get_eigenvalue(i)` and `get_eigenfunction(i)`
forward to `self._modes[i][0/1]` (cached at construction from provider). `n_modes`
returns `len(self._modes)` = `provider.dim`.

**`_apply(f) → Function3D`:**
1. Evaluate `f` at pre-computed quadrature mesh once.
2. For each mode k: `c_k = ⟨f, φ_k⟩ ≈ Σ f(x_i) φ_k(x_i) w_i`.
3. Return `Function3D` evaluating `Σ_k λ_k c_k φ_k(x)`.

Quadrature mesh/weights are pre-computed at construction; all mode callables are
pre-fetched and stored as a list for reuse across calls.

---

### `InverseBoxLaplacian` — `pygeoinf3d.operators.inverse_laplacian_box`

Spectral pseudo-inverse of the negative Laplacian on a box domain. Inherits from `SpectralOperator3D`. Self-adjoint.

```python
InverseBoxLaplacian(space: Lebesgue3D,
                    provider: TensorProductProvider,
                    null_space_handling: Literal['pseudoinverse', 'shift', 'strict'] = 'pseudoinverse',
                    shift: float = 1.0,
                    eigenvalue_tol: float = 1e-10,
                    quad_resolution: int = 20)
```

**`null_space_handling` options:**
- `'pseudoinverse'` (default): modes with |λ_k| < `eigenvalue_tol` map to 0 (natural pseudoinverse; Neumann constant mode silently projected out).
- `'shift'`: uses `1/(λ_k + shift)` for all modes; `shift` must be > 0.
- `'strict'`: raises `ValueError` at construction if any eigenvalue is near-zero (safe for Dirichlet BCs).

**Properties:** `null_space_handling`, `shift`, `eigenvalue_tol`

**`_apply(f) → Function3D`:** Identical pattern to `BoxLaplacian._apply` but scales by the pre-computed inverted eigenvalues stored in `self._modes` as `(inv_lam_k, phi_k)` tuples.

**`SpectralOperator3D` interface:** `get_eigenvalue(i)` returns `self._modes[i][0]` (the inverted/modified eigenvalue). `get_eigenfunction(i)` returns `self._modes[i][1]`. `n_modes` returns `len(self._modes)`.

**Usage in Bayesian priors:** Acts as a covariance operator — apply to white noise samples to produce samples from a Laplacian-based Gaussian prior. For mean-zero priors with Neumann BCs, use `null_space_handling='pseudoinverse'`.

---

### `SpectralPowerBase3D` — `pygeoinf3d.operators.spectral_power_base`

Non-box-specific reusable abstract base for spectral operators with pre-computed
eigenvalue lists. Inherits from `SpectralOperator3D` *and* `ABC` (enforcing
`@abstractmethod` even though `LinearOperator` does not use `ABCMeta`).

```python
SpectralPowerBase3D(space: Lebesgue3D,
                   op_eigenvalues: list[float],
                   eigenfunctions: list[Function3D])
```

Stores `self._op_eigenvalues` and `self._eigenfunctions`. Passes `self._apply` as
both forward and adjoint mapping to `SpectralOperator3D.__init__` (self-adjoint).

**Concrete `SpectralOperator3D` interface** (not abstract, provided by this base):
| Member | Description |
|--------|-------------|
| `get_eigenvalue(index) → float` | `self._op_eigenvalues[index]` |
| `get_eigenfunction(index) → Function3D` | `self._eigenfunctions[index]` |
| `n_modes: int` (property) | `len(self._op_eigenvalues)` |

**Abstract `_apply(f) → Function3D`** — subclasses must implement domain-specific
quadrature. Pattern: compute `c_k = ⟨f,Φ_k⟩` then synthesise `Σ μ_k c_k Φ_k(x)`.

**Static helper:**
```python
SpectralPowerBase3D.compute_power_eigenvalues(
    laplacian_eigs: list[float],
    power: int,
    alpha: float = 1.0,
    eigenvalue_tol: float = 1e-10,
) -> list[float]
```
Canonical convention for $(-\alpha\Delta)^s$:
$$\mu_k = \begin{cases} 1 & s=0 \\ 0 & s<0,|\lambda_k|<\varepsilon \\ (\alpha\lambda_k)^s & \text{otherwise} \end{cases}$$

**Does not know about:** box geometry, `TensorProductProvider`, or quadrature setup.

**Exported from:** `pygeoinf3d.operators`, `pygeoinf3d`.

---

### `GeneralizedBoxLaplacian` — `pygeoinf3d.operators.generalized_box_laplacian`

*(Phase 2 complete — full spectral projection implemented.)*

Integer-power generalisation of the box Laplacian: $(-\alpha\Delta)^s$ for
integer `power` $s$ and positive `alpha` $\alpha > 0$. Inherits from
`SpectralPowerBase3D`. Self-adjoint.

```python
GeneralizedBoxLaplacian(space: Lebesgue3D,
                        provider: TensorProductProvider,
                        *,
                        power: int,
                        alpha: float = 1.0,
                        eigenvalue_tol: float = 1e-10,
                        quad_resolution: int = 20)
```

**Eigenvalue formula:** delegates to `SpectralPowerBase3D.compute_power_eigenvalues`:
- `power == 0`: `mu_k = 1.0` for **all** modes (identity).
- `power < 0` and `|lambda_k| < eigenvalue_tol`: `mu_k = 0.0` (pseudoinverse suppression).
- Otherwise: `mu_k = (alpha * lambda_k) ** power`.

**`get_eigenvalue`, `get_eigenfunction`, `n_modes`** are inherited from `SpectralPowerBase3D`.

**Validation:**
- `power` must be a Python `int` or `np.integer`; `float` raises `TypeError`.
- `alpha` must be > 0; otherwise raises `ValueError`.
- mismatched `space.function_domain` and provider box domain raises `ValueError`.

**Properties:** `power: int`, `alpha: float`, `eigenvalue_tol: float`

**`_apply(f) → Function3D`:** quadrature-based spectral projection (same pattern as `BoxLaplacian`):
1. Evaluate `f` at pre-computed `box.structured_mesh(quad_resolution)` once.
2. `c_k = ⟨f, φ_k⟩ ≈ Σ f(x_i) φ_k(x_i) w_i`
3. Return `Function3D` evaluating `Σ_k μ_k c_k φ_k(x)` via closure.

**Special-case equivalences (apply level):**
- `GeneralizedBoxLaplacian(..., power=1, alpha=1.0)(f)` ≈ `BoxLaplacian(...)(f)` pointwise.
- `GeneralizedBoxLaplacian(..., power=-1, alpha=1.0)(f)` ≈ `InverseBoxLaplacian(..., null_space_handling='pseudoinverse')(f)` pointwise.
- `GeneralizedBoxLaplacian(..., power=0)(f)` ≈ `f` pointwise (identity).
- Neumann zero mode under `power=-1`: contribution is `0 * c_0 * φ_0(x) = 0` (correctly suppressed).

---

## Sampling

### `KLSampler3D` — `pygeoinf3d.sampling.kl_sampler`

Karhunen-Loève sampler for Gaussian measures on `Lebesgue3D`. Accepts any
`SpectralOperator3D` as the covariance operator and truncates the KL expansion
at `n_modes` modes.

```python
KLSampler3D(operator: SpectralOperator3D,
            *,
            n_modes: int,
            mean: Function3D | None = None,
            rng: np.random.Generator | None = None)
```

- `operator`: must be a `SpectralOperator3D`; typically an `InverseBoxLaplacian`.
- `n_modes`: required; must satisfy `1 ≤ n_modes ≤ operator.n_modes`.
- `mean`: mean function (defaults to `operator.domain.zero`).
- Eigenpairs are **cached at construction** (no lazy loading).
- `sqrt(max(lam_k, 0))` handles suppressed zero-modes (Neumann pseudoinverse) automatically.

| Member | Description |
|---|---|
| `n_modes: int` | Number of KL modes retained |
| `space: Lebesgue3D` | The function space |
| `eigenvalues: np.ndarray` | Shape `(n_modes,)` copy of operator eigenvalues |
| `has_tensor_product_structure: bool` | True if operator is backed by `TensorProductProvider`; enables fast evaluators |
| `sample() → Function3D` | Draw one sample: `m + Σ sqrt(λ_k) ξ_k Φ_k`. If `has_tensor_product_structure`, attaches `_kl_fast_grid_evaluator` and `_kl_fast_slice_evaluator` to the returned `Function3D`. |
| `samples(n) → list[Function3D]` | Draw *n* independent samples |
| `sample_from_xis(xis) → Function3D` | Deterministic sample from given xi array `(n_modes,)`; raises `ValueError` on bad shape. Also attaches fast-path callables if TP structure is present. |
| `covariance_factor` (property) | `LinearOperator: EuclideanSpace(K) → Lebesgue3D` with `L(w) = Σ sqrt(λ_k) w_k Φ_k` |
| `evaluate_on_regular_grid(xis, resolution, *, plot_n_modes)` | **Fast path.** Vectorized `einsum` evaluation on regular 3D grid. Returns `(X, Y, Z, values)` each shape `(Nx, Ny, Nz)`. Requires TP structure; raises `TypeError` otherwise. |
| `evaluate_on_axis_slice(xis, axis, coord, resolution, *, plot_n_modes)` | **Fast path.** Vectorized `einsum` evaluation on axis-aligned 2D slice. Returns `(grid_u, grid_v, values)` each shape `(Nu, Nv)`. Requires TP structure; raises `TypeError` otherwise. |

**Fast-path auto-dispatch (Phase 2 + 3):** `_attach_kl_fast_paths(sampler, xis, fn)` is a
module-level helper (private) that stores two lambda callables directly on the
`Function3D` instance:
- `fn._kl_fast_grid_evaluator(resolution, plot_n_modes=None)` → `(X, Y, Z, values)`
- `fn._kl_fast_slice_evaluator(axis, coord, resolution, plot_n_modes=None)` → `(grid_u, grid_v, values)`

These are checked by `evaluate_function3d_on_regular_grid` and
`evaluate_function3d_on_axis_slice` in `plot.py`, enabling automatic fast-path
dispatch in all volumetric and slice-based Plotly helpers. **Phase 3** threads
`plot_n_modes` from the public plot API through to the fast-path callables,
providing a visualization-only KL mode cap without modifying the underlying sample.

**Adjoint of `covariance_factor`:** `L*(f) = [sqrt(λ_k) ⟨f, Φ_k⟩]_k`, computed via `space.inner_product`.

**Wiring into GaussianMeasure (PLI workflow):**
```python
from pygeoinf import GaussianMeasure
kl = KLSampler3D(inv_lap, n_modes=20)
prior = GaussianMeasure(covariance=inv_lap, expectation=space.zero, sample=kl.sample)
```

---

## Phase 4 Classes (Representation Boundary)

### `CoefficientRepresentation3D` — `pygeoinf3d.representations.base`

Abstract base class for finite-dimensional coefficient representations.

```python
class CoefficientRepresentation3D(ABC):
    @property @abstractmethod  def space(self): ...       # attached Lebesgue3D
    @property @abstractmethod  def dim(self) -> int: ... # number of coefficients
    @abstractmethod def to_components(self, f) -> np.ndarray: ...
    @abstractmethod def from_components(self, c: np.ndarray): ...
    @property def coefficient_space(self): ...  # EuclideanSpace(dim) — provided by base
```

**Design boundary:** Does not own quadrature policy or numerical algorithms. Subclasses
delegate numerical work to a backend object.  The optional concrete method
`coefficient_metric()` raises `NotImplementedError` by default; subclasses that
attach a Phase-2 backend (see below) can expose it.

---

### `LebesgueCoefficientBackend` — `pygeoinf3d.representations.base`

Abstract numerical strategy object for Lebesgue coefficient workflows.

```python
class LebesgueCoefficientBackend(ABC):
    @abstractmethod  def moments(self, space, provider, f) -> np.ndarray
    @abstractmethod  def gram_matrix(self, space, provider) -> np.ndarray
    @abstractmethod  def coefficient_metric(self, space, provider) -> MassWeightedHilbertSpace
    @abstractmethod  def solve_coefficients(self, G, b) -> np.ndarray
    @abstractmethod  def to_components(self, space, provider, f) -> np.ndarray
    @abstractmethod  def from_components(self, space, provider, c) -> Function3D
```

**Provider protocol** expected by all backend methods:
- `provider.dim` — number of basis functions.
- `provider.get_basis_function(i)` — returns the `i`-th `Function3D` basis element.

**Design boundary:** Backends are stateless strategies (modulo caching). They do not
attach to a specific space or provider; those are passed on each call.

---

## Lebesgue Coefficient Architecture Phase 2 Classes

### `OrthonormalProjectionBackend` — `pygeoinf3d.representations.backends`

Backend for bases orthonormal w.r.t. the space inner product.

```python
OrthonormalProjectionBackend()   # no constructor arguments
```

| Method | Description |
|--------|-------------|
| `.moments(space, provider, f)` | `b_i = ⟨φ_i, f⟩` via `space.inner_product` |
| `.gram_matrix(space, provider)` | Returns `I_n` (no integration, orthonormal assumption) |
| `.coefficient_metric(space, provider)` | `MassWeightedHilbertSpace(EuclideanSpace(n), I, I)` — standard Euclidean metric |
| `.solve_coefficients(G, b)` | Returns `b.copy()` (identity solve) |
| `.to_components(space, provider, f)` | Delegates to `moments` |
| `.from_components(space, provider, c)` | Builds `Σ c_i φ_i` via `Function3D` arithmetic |

**When to use:** tensor-product box eigenfunctions with consistent quadrature (Phase 3+).

---

### `GramProjectionBackend` — `pygeoinf3d.representations.backends`

General backend for non-orthogonal basis families.

```python
GramProjectionBackend()   # no constructor arguments
```

| Method | Description |
|--------|-------------|
| `.moments(space, provider, f)` | `b_i = ⟨φ_i, f⟩` via `space.inner_product` |
| `.gram_matrix(space, provider)` | Assembles `G_{ij}=⟨φ_i,φ_j⟩`; **cached** per `(id(space), id(provider))` |
| `.coefficient_metric(space, provider)` | `MassWeightedHilbertSpace(EuclideanSpace(n), G, G⁻¹)` |
| `.solve_coefficients(G, b)` | Cholesky solve `G c = b` (LU fallback for borderline SPD) |
| `.to_components(space, provider, f)` | Assembles moments, then solves `G c = b` |
| `.from_components(space, provider, c)` | Builds `Σ c_i φ_i` via `Function3D` arithmetic |

**Key invariant:** `G @ to_components(f) ≈ moments(f)` for any function `f`.

---

### `assemble_gram` / `gram_operators` — `pygeoinf3d.representations.gram`

```python
from pygeoinf3d.representations.gram import assemble_gram, gram_operators

G = assemble_gram(space, provider)           # np.ndarray (n, n)
mass_op, inv_mass_op = gram_operators(G)   # LinearOperator pair on EuclideanSpace(n)
```

- `assemble_gram`: iterates over `provider.dim` basis pairs, calls `space.inner_product`.
- `gram_operators`: wraps `G` in a `LinearOperator` and builds the Cholesky/LU inverse.
  Returns a pair suitable for `MassWeightedHilbertSpace(euc, mass_op, inv_mass_op)`.
  Emits `RuntimeWarning` if `cond(G) > 1e12`.

---

### Updated: `LebesgueRepresentation3D` — Architecture Phase 2 additions

Concrete `CoefficientRepresentation3D` for `Lebesgue3D` spaces.

```python
LebesgueRepresentation3D(space: Lebesgue3D, provider, backend)
```

**Constructor validation:**
- Raises `TypeError` if `space` is not a `Lebesgue3D`.
- Raises `ValueError` if `provider.domain` exists and does not match `space.function_domain`.
- Accepts providers without a `.domain` attribute (permissive, for Phase 1 compatibility).

| Member | Description |
|--------|-------------|
| `.space` | The attached `Lebesgue3D` |
| `.dim` | `int(provider.dim)` |
| `.coefficient_space` | `EuclideanSpace(dim)` — algebraic carrier |
| `.provider` | The basis provider |
| `.backend` | The numerical backend |
| `.to_components(f)` | Delegates `backend.to_components(space, provider, f)` |
| `.from_components(c)` | Delegates `backend.from_components(space, provider, c)` |
| `.coefficient_metric()` | Delegates `backend.coefficient_metric(space, provider)` — **architecture Phase 2** |
| `.gram_matrix()` | Delegates `backend.gram_matrix(space, provider)` — **architecture Phase 2** |
| `.moments(f)` | Delegates `backend.moments(space, provider, f)` — **architecture Phase 2** |

**Usage:**
```python
from pygeoinf3d.representations.backends import GramProjectionBackend
rep = LebesgueRepresentation3D(space, provider, backend)
space_rep = Lebesgue3D(domain, representation=rep)
c = space_rep.to_components(f)           # delegates through backend
f_syn = space_rep.from_components(c)
metric = rep.coefficient_metric()        # MassWeightedHilbertSpace — architecture Phase 2
G = rep.gram_matrix()                    # np.ndarray (n,n) — architecture Phase 2
b = rep.moments(f)                       # raw moments — architecture Phase 2
proj = space_rep.coordinate_projection  # LinearOperator (dim=rep.dim)
incl = space_rep.coordinate_inclusion   # LinearOperator (dim=rep.dim)
```

**Backend compatibility:** `.coefficient_metric()`, `.gram_matrix()`, and `.moments()`
raise `NotImplementedError` if the attached backend does not implement the
corresponding method.  Use `OrthonormalProjectionBackend` or `GramProjectionBackend`
(both from `pygeoinf3d.representations.backends`) for full architecture Phase 2 functionality.

---

```python
from pygeoinf3d import (
    AxisAlignedBoundingBox,
    CoordinateSystem,
    Region3D,
    Domain3D,
    StructuredDomain3D,
    Boundary3D,
    Function3D,          # Phase 2
    Lebesgue3D,          # Phase 2
    VolumeKernel3D,      # Phase 2
    BoxDomain,           # Phase 3
    TensorProductProvider, # Phase 3
    FunctionProvider3D,  # SOLA Phase 3
    KernelProvider3D,    # SOLA Phase 3
    FunctionToKernelAdapter, # SOLA Phase 3
    BoxLaplacian,        # Phase 3
    SOLAOperator3D,      # SOLA Phase 1+2+3
    CoefficientRepresentation3D,  # Phase 4
    LebesgueRepresentation3D,     # Phase 4
    # Lebesgue coefficient architecture Phase 2 — Gram-aware backends
    LebesgueCoefficientBackend,
    OrthonormalProjectionBackend,
    GramProjectionBackend,
)
```

---

## Planned Additions (Future Phases)

| Phase | Addition |
|-------|----------|
| ~~SOLA 2~~ | ~~`SOLAOperator3D` dual/adjoint mappings via codomain Riesz map and `Lebesgue3D.from_dual`~~ **DONE** |
| ~~SOLA 3~~ | ~~Provider-backed lazy kernel access in `SOLAOperator3D` via `KernelProvider3D` and `FunctionToKernelAdapter`~~ **DONE** |
| ~~SOLA 4~~ | ~~Public API lock, export test, full reference update~~ **DONE** |
| ~~kernel-providers 2~~ | ~~`GaussianProvider3D`, `BumpProvider3D`, `BoxcarProvider3D` under `function_providers`~~ **DONE** |
| ~~Phase 4 docs~~ | ~~Canonical workflow, volumetric boundary statement, README refresh, test counts~~ **DONE** |
| next | `Sobolev3D`, Bessel-Sobolev operators, and trace-ready structure on BoxDomain |
| 5 | `BallDomain` (StructuredDomain3D, spherical coordinates) |
| 6 | Dual elements and geometric distributions (point/path/surface/volume) |
| 7 | End-to-end inversion workflow demo |

---

## Test Coverage

Run with:
```bash
conda run -n inferences3 python -m pytest tests/ -q
```

**430 tests passing** (foundation Phases 1-3 + coefficient architecture Phases 1-2 + SOLA Phases 1-4 + kernel-providers Phases 1-2 + plot Phase 3 + interactive slider extension):
- `tests/core/test_config.py` — 13 tests
- `tests/core/test_region.py` — 14 tests
- `tests/core/test_domain_base.py` — 21 tests
- `tests/core/test_boundary.py` — 8 tests
- `tests/core/test_functions.py` — 32 tests  ← Phase 2
- `tests/spaces/test_lebesgue.py` — 72 tests  ← Phase 2
- `tests/geometries/test_box_domain.py` — 45 tests  ← Phase 3
- `tests/providers/test_tensor_product.py` — 28 tests  ← Phase 3
- `tests/providers/test_function_providers.py` — 12 tests  ← kernel-providers Phase 1
- `tests/providers/test_concrete_function_providers.py` — 70 tests  ← kernel-providers Phase 2
- `tests/operators/test_box_laplacian.py` — 17 tests  ← Phase 3 + spectral interface
- `tests/representations/test_lebesgue_representation.py` — 22 tests  ← architecture Phase 1
- `tests/representations/test_projection_backends.py` — 40 tests  ← architecture Phase 2
- `tests/operators/test_sola.py` — 29 tests  ← SOLA Phases 1+2+3+4
- `tests/operators/test_inverse_box_laplacian.py` — 32 tests  ← InverseLaplacian plan + spectral interface
- `tests/sampling/test_kl_sampler.py` — 23 tests  ← spectral-kl plan
- `tests/sampling/test_kl_generalized_integration.py` — 10 tests  ← generalized-box-laplacian plan Phase 4
- `tests/test_plot.py` — 17 tests  ← plot Phase 3 + slider + volume/isosurface helpers

---

## Key Design Decisions

1. `Region3D` is intentionally **not** a `Subset` subclass — it's a wrapper. This avoids entangling pygeoinf3D's computational layer with pygeoinf's geometry hierarchy.
2. `Domain3D` is **not** a `HilbertSpace`. Function spaces are separate objects built on domains.
3. `StructuredDomain3D` is an **optional intermediate layer** — not every domain needs it. Simple unstructured domains can implement `Domain3D` directly.
4. `Boundary3D` is a **first-class object** from Phase 1, not a dict of arrays. This enables future BCs and integration on boundary surfaces without refactoring.
5. `EuclideanSpace(3)` is shared as a **module-level singleton** (`_R3`) in `domain_base.py` to avoid repeated instantiation.
6. `Function3D` is **primal only**: callable-backed, domain- or space-attached. Not a distribution container. Callable must handle `(3,)` and `(n, 3)` inputs.
7. `Lebesgue3D` has `dim == 0` (basis-free Phase 2). `to_components`/`from_components` raise `NotImplementedError`. The inner product is derived from `to_dual` (pairing-first).
8. `VolumeKernel3D` subclasses `pygeoinf.LinearForm` and overrides `_mapping_impl` to do numerical volume integration. Python's dynamic dispatch routes `__call__` through the override. This avoids the basis-enumeration assumption in `LinearForm._compute_components`. All arithmetic methods (`__add__`, `__sub__`, `__mul__`, `__neg__`, and in-place `__imul__`, `__iadd__`, `__isub__`) are explicitly overridden to keep the object kernel-backed; `LinearForm.__imul__` and `__iadd__` are especially dangerous because they mutate the fake `_components=np.zeros(0)` placeholder silently.
9. `from_dual` is **conservative**: only accepts `VolumeKernel3D` objects owned by *this* space instance. Generic `LinearForm` objects and foreign-space kernels are rejected.
10. `SOLAOperator3D` is **fully basis-free on the domain side**: it uses `VolumeKernel3D.__call__` directly for each kernel without enumerating any domain basis. `LinearOperator.from_linear_forms` is explicitly avoided. The **dual mapping** extracts codomain-dual coordinates via `codomain.dual.to_components(yp)` and returns a weighted `VolumeKernel3D` sum. The **adjoint mapping** chains `codomain.to_dual(y)` → dual mapping → `domain.from_dual(xp)`, making it correct for any finite-dimensional Hilbert codomain including `MassWeightedHilbertSpace`.
11. `SOLAOperator3D` supports two construction modes: **sequence-backed** (eager, validates all kernels at construction) and **provider-backed** (lazy, kernels fetched via `KernelProvider3D.get_kernel(i)` on demand and cached in `_kernel_cache`). The internal `_get_kernel(i)` method is the single dispatch point that both `_forward` and `_dual_mapping` use. The `.kernels` property materializes the full list via `_get_kernel` in both modes (returning cached objects for provider mode).
