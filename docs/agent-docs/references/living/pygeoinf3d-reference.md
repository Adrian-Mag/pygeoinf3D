# pygeoinf3D Living Reference

> **Status:** Phase 2 complete — Function3D and Lebesgue3D with pairing-first design.
> **Last updated:** 2026-03-12

---

## Package Overview

**pygeoinf3D** (`pygeoinf3d`) is an extension package for concrete 3D function spaces and operators built on top of `pygeoinf`. It uses `pygeoinf.EuclideanSpace(3)` and `pygeoinf.Subset` objects as the semantic geometry layer.

---

## Repository Layout

```
pygeoinf3D/
├── pyproject.toml
├── README.md
├── pygeoinf3d/
│   ├── __init__.py          # Re-exports all public API
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # AxisAlignedBoundingBox, CoordinateSystem
│   │   ├── region.py        # Region3D
│   │   ├── domain_base.py   # Domain3D, StructuredDomain3D
│   │   ├── boundary.py      # Boundary3D
│   │   └── functions.py     # Function3D  ← Phase 2
│   └── spaces/
│       ├── __init__.py      # Re-exports Lebesgue3D, VolumeKernel3D
│       ├── forms.py         # VolumeKernel3D  ← Phase 2
│       └── lebesgue.py      # Lebesgue3D  ← Phase 2
├── tests/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_region.py
│   │   ├── test_domain_base.py
│   │   ├── test_boundary.py
│   │   └── test_functions.py     ← Phase 2
│   └── spaces/
│       ├── __init__.py
│       └── test_lebesgue.py      ← Phase 2
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
Lebesgue3D(domain: Domain3D)
```

| Member | Description |
|--------|-------------|
| `.function_domain` | The underlying `Domain3D` |
| `.dim` → `0` | Basis-free (no component representation in Phase 2) |
| `.to_dual(f)` | Maps `f` → `VolumeKernel3D` evaluating `g ↦ ∫f·g dV` |
| `.from_dual(xp)` | Recovers `Function3D` from own `VolumeKernel3D`; rejects others |
| `.to_components(f)` | Raises `NotImplementedError` (no basis) |
| `.from_components(c)` | Raises `NotImplementedError` (no basis) |
| `.inner_product(f,g)` | `∫f·g dV` via `to_dual(f)(g)` |
| `.norm(f)` | `√⟨f,f⟩` |
| `.zero` | Constant zero `Function3D` |
| `.add/.subtract/.multiply/.negative` | Vector-space ops on `Function3D` |
| `==` | True iff `function_domain is other.function_domain` |

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

## Public API (`pygeoinf3d.__init__`)

```python
from pygeoinf3d import (
    AxisAlignedBoundingBox,
    CoordinateSystem,
    Region3D,
    Domain3D,
    StructuredDomain3D,
    Boundary3D,
    Function3D,     # Phase 2
    Lebesgue3D,     # Phase 2
    VolumeKernel3D, # Phase 2
)
```

---

## Planned Additions (Future Phases)

| Phase | Addition |
|-------|----------|
| 3 | `BoxDomain` (StructuredDomain3D, Cartesian tensor-product) |
| 4 | `Sobolev3D`, Bessel-Sobolev operators, and trace-ready structure on BoxDomain |
| 5 | `BallDomain` (StructuredDomain3D, spherical coordinates) |
| 6 | Dual elements and geometric distributions (point/path/surface/volume) |
| 7 | End-to-end inversion workflow demo |

---

## Test Coverage

Run with:
```bash
conda run -n inferences3 python -m pytest tests/ -q
```

**141 tests, all passing** (Phase 1 + Phase 2):
- `tests/core/test_config.py` — 13 tests
- `tests/core/test_region.py` — 14 tests
- `tests/core/test_domain_base.py` — 21 tests
- `tests/core/test_boundary.py` — 8 tests
- `tests/core/test_functions.py` — 37 tests  ← Phase 2
- `tests/spaces/test_lebesgue.py` — 48 tests  ← Phase 2

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
