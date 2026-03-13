# pygeoinf3D Living Reference

> **Status:** Phase 4 complete — continuous-first refactor applied.
> **Last updated:** 2026-03-13

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
│   ├── spaces/
│   │   ├── __init__.py      # Re-exports Lebesgue3D, VolumeKernel3D, Sobolev3D,
│   │   │                    #   SobolevKernel3D, SpectralSobolev3D
│   │   ├── forms.py         # VolumeKernel3D, SobolevKernel3D  ← Phase 2/4
│   │   ├── lebesgue.py      # Lebesgue3D  ← Phase 2
│   │   └── sobolev.py       # Sobolev3D, SpectralSobolev3D  ← Phase 4
│   ├── geometries/
│   │   ├── __init__.py
│   │   └── box/
│   │       ├── __init__.py
│   │       └── domain.py    # BoxDomain  ← Phase 3
│   ├── providers/
│   │   ├── __init__.py
│   │   └── tensor_product.py  # TensorProductProvider  ← Phase 3
│   └── operators/
│       ├── __init__.py
│       ├── laplacian_box.py   # BoxLaplacian  ← Phase 3
│       └── bessel.py          # BesselOperator3D, SpectralBesselOperator3D,
│                                #   bessel_direct, bessel_inverse  ← Phase 4
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
│   │   ├── test_lebesgue.py          ← Phase 2
│   │   └── test_sobolev.py           ← Phase 4
│   ├── geometries/
│   │   ├── __init__.py
│   │   └── test_box_domain.py        ← Phase 3
│   ├── providers/
│   │   ├── __init__.py
│   │   └── test_tensor_product.py    ← Phase 3
│   └── operators/
│       ├── __init__.py
│       ├── test_box_laplacian.py     ← Phase 3
│       └── test_bessel.py            ← Phase 4
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
| `.sample_boundary(n, *, rng)` | Sample surface-area-uniform boundary points from 6 faces |
| `.outward_normal(points, *, component)` | Outward unit normals; `component` pins a named face. **Semantically valid for boundary points only** — interior points silently receive nearest-face normal |
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
| `.box` | The underlying `BoxDomain` (public accessor) |
| `.get_multi_index(i)` | `(kx,ky,kz)` for flat index `i` |
| `.get_index(multi_index)` | Flat index; raises `KeyError` if not in truncation |
| `.get_eigenvalue(i)` | Eigenvalue λ = $(k_x\pi/L_x)^2 + (k_y\pi/L_y)^2 + (k_z\pi/L_z)^2$ |
| `.get_eigenfunction(i)` | `Function3D` computing φ_i; handles `(3,)` and `(n,3)` |
| `__iter__` | Yields `(eigenvalue, Function3D)` pairs |

---

### `BoxLaplacian` — `pygeoinf3d.operators.laplacian_box`

Spectral negative Laplacian − ∇² on a box domain. Inherits from `pygeoinf.LinearOperator`.

```python
BoxLaplacian(space: Lebesgue3D,
             provider: TensorProductProvider,
             quad_resolution: int = 20)
```

Validates `space.function_domain is provider._box`. Constructed as self-adjoint via
`LinearOperator.__init__(space, space, self._apply, adjoint_mapping=self._apply)`.

**`_apply(f) → Function3D`:**
1. Evaluate `f` at pre-computed quadrature mesh once.
2. For each mode k: `c_k = ⟨f, φ_k⟩ ≈ Σ f(x_i) φ_k(x_i) w_i`.
3. Return `Function3D` evaluating `Σ_k λ_k c_k φ_k(x)`.

Quadrature mesh/weights are pre-computed at construction; all mode callables are
pre-fetched and stored as a list for reuse across calls.

---

## Phase 4 Classes

### `Sobolev3D` — `pygeoinf3d.spaces.sobolev` (continuous-first public API)

H^s Hilbert space on a box domain with **`Function3D` elements** (continuous-first design).
Inherits from `pygeoinf.HilbertSpace`.  Dual elements are `SobolevKernel3D` forms.

```python
Sobolev3D(provider: TensorProductProvider, *, s: float, kappa: float = 1.0,
          quad_resolution: int = 20)
```

Inner product: $\langle f, g \rangle_{H^s} = \sum_k (\kappa^2 + \lambda_k)^s \hat f_k \hat g_k$
where $\hat f_k = \langle f, \phi_k \rangle_{L^2}$ is computed via pre-cached quadrature.

| Member | Description |
|--------|-------------|
| `.dim` | `N = provider.dim` — optional spectral component dimension |
| `.s` | Sobolev order |
| `.kappa` | Mass-regularisation parameter |
| `.quad_resolution` | Per-axis quadrature resolution (default 20) |
| `.function_domain` | The underlying `BoxDomain` |
| `.provider` | The `TensorProductProvider` |
| `.eigenvalues` | Copy of $\lambda_k$ array, shape `(N,)` |
| `.weights` | Copy of $(\kappa^2+\lambda_k)^s$ array, shape `(N,)` |
| `.to_dual(f)` | `SobolevKernel3D(self, f)` — Riesz map for `Function3D` |
| `.from_dual(xp)` | Recovers `xp.kernel` from own `SobolevKernel3D`; rejects others |
| `.to_components(f)` | $L^2$ spectral coefficients via pre-cached quadrature, shape `(N,)` |
| `.from_components(c)` | `Function3D` reconstructed from coefficient array |
| `.project(f, *, quad_resolution)` | Alias for `to_components` with optional quad override |
| `.inner_product(f, g)` | `to_dual(f)(g)` |
| `.norm(f)` | `sqrt(inner_product(f, f))` |
| `.is_element(x)` | `True` iff `x` is `Function3D` on this space's domain |
| `.zero` | Constant-zero `Function3D` |
| `.add/.subtract/.multiply/.negative` | Vector-space ops on `Function3D` |
| `.supports_trace` | `True` iff `s > 0.5` |
| `.assert_trace_allowed()` | Raises `RuntimeError` if `s <= 0.5` |

Raises `ValueError` if `s < 0`, `kappa < 0`, or degenerate weights.

**Internal pre-computation:**
- `_quad_pts` — structured mesh `(Q, 3)` computed once at construction
- `_quad_weights` — quadrature weights `(Q,)` computed once at construction
- `_phi_vals` — `(N, Q)` matrix of eigenfunction values for fast `matmul` projection

---

### `SpectralSobolev3D` — `pygeoinf3d.spaces.sobolev` (secondary coefficient-vector API)

Coefficient-vector Sobolev space $H^s$ on a box domain.  Elements are **numpy arrays of shape `(N,)`**.
This was previously called `Sobolev3D` before the continuous-first refactor.

```python
SpectralSobolev3D(provider: TensorProductProvider, *, s: float, kappa: float = 1.0)
```

| Member | Description |
|--------|-------------|
| `.dim` | `N` = `provider.dim` |
| `.to_components(c)` / `.from_components(c)` | Identity |
| `.to_dual(c)` | `LinearForm(self, components=weights * c)` |
| `.from_dual(xp)` | `xp.components / weights` |
| `.inner_product(c1, c2)` | `sum(weights * c1 * c2)` |
| `.is_element(c)` | `True` iff array-like shape `(N,)` |
| `.project(f, *, quad_resolution)` | $L^2$ spectral coefficients via quadrature |
| `.to_function(c)` | `Function3D` reconstructed from coefficients |
| `.supports_trace` | `True` iff `s > 0.5` |

Raises `ValueError` if degenerate (Neumann + kappa=0 + s>0).

---

### `SobolevKernel3D` — `pygeoinf3d.spaces.forms`

Spectral-projection-backed `LinearForm` for `Sobolev3D` duals.

```python
SobolevKernel3D(space: Sobolev3D, kernel: Function3D)
```

Inherits from `pygeoinf.LinearForm`. Implements $\phi_f(g) = \sum_k w_k \hat f_k \hat g_k$
where $w_k = (\kappa^2 + \lambda_k)^s$ and weights are pre-multiplied into `_weighted_coeffs`.

| Member | Description |
|--------|-------------|
| `.kernel` | The primal `Function3D` representer |
| `.domain` | The owning `Sobolev3D` instance |
| `.__call__(g)` | `dot(weighted_coeffs, space._project_to_coeffs(g))` |
| `copy()` | New `SobolevKernel3D` with same space and kernel |
| `__neg__` / `__mul__` / `__rmul__` / `__truediv__` | Kernel-backed out-of-place arithmetic |
| `__add__` / `__sub__` | Kernel-backed; require same space identity |
| `__imul__` / `__iadd__` / `__isub__` | In-place: mutate both `_kernel` and `_weighted_coeffs` |

---

### `BesselOperator3D` — `pygeoinf3d.operators.bessel` (continuous-first public API)

Diagonal Bessel potential operator on a `Sobolev3D`. Maps `Function3D → Function3D`.
Inherits from `pygeoinf.LinearOperator`.

```python
BesselOperator3D(space: Sobolev3D, *, power: float)
```

Self-adjoint. `_apply(f)`: project `f → c`, scale `c_k *= scale_k`, reconstruct `Function3D`.
Where `scale_k = (kappa^2 + lambda_k)^power`.

| Member | Description |
|--------|-------------|
| `.power` | The exponent $p$ |
| `domain`, `codomain` | Both are the same `Sobolev3D` instance |

**Convenience constructors (continuous-first):**

```python
bessel_direct(space)   # power = s/2  → BesselOperator3D
bessel_inverse(space)  # power = -s/2 → BesselOperator3D
```

Raises `ValueError` if `power < 0` and any `(kappa^2 + lambda_k) = 0`.

---

### `SpectralBesselOperator3D` — `pygeoinf3d.operators.bessel` (secondary coefficient-vector API)

Diagonal Bessel operator on a `SpectralSobolev3D`. Maps coefficient arrays `(N,) → (N,)`.
This was previously called `BesselOperator3D` before the continuous-first refactor.

```python
SpectralBesselOperator3D(space: SpectralSobolev3D, *, power: float)
```

`_apply(c) = scale * c` (pure numpy). Raises `ValueError` if `power < 0` and any base is zero.

---

```python
from pygeoinf3d import (
    AxisAlignedBoundingBox,
    CoordinateSystem,
    Region3D,
    Domain3D,
    StructuredDomain3D,
    Boundary3D,
    Function3D,              # Phase 2
    Lebesgue3D,              # Phase 2
    VolumeKernel3D,          # Phase 2
    BoxDomain,               # Phase 3
    TensorProductProvider,   # Phase 3
    BoxLaplacian,            # Phase 3
    Sobolev3D,               # Phase 4 — continuous-first (Function3D elements)
    SobolevKernel3D,         # Phase 4 — Riesz dual form for Sobolev3D
    SpectralSobolev3D,       # Phase 4 — secondary coefficient-vector API
    BesselOperator3D,        # Phase 4 — continuous-first (Function3D -> Function3D)
    SpectralBesselOperator3D,# Phase 4 — secondary coefficient-vector API
    bessel_direct,           # Phase 4
    bessel_inverse,          # Phase 4
)
```

---

## Planned Additions (Future Phases)

| Phase | Addition |
|-------|----------|
| 5 | `BallDomain` (StructuredDomain3D, spherical coordinates) |
| 6 | Dual elements and geometric distributions (point/path/surface/volume) |
| 7 | End-to-end inversion workflow demo |

---

## Test Coverage

Run with:
```bash
conda run -n inferences3 python -m pytest tests/ -q
```

**373 tests, all passing** (Phase 1 + Phase 2 + Phase 3 + Phase 4):
- `tests/core/test_config.py` — 13 tests
- `tests/core/test_region.py` — 14 tests
- `tests/core/test_domain_base.py` — 21 tests
- `tests/core/test_boundary.py` — 8 tests
- `tests/core/test_functions.py` — 37 tests  ← Phase 2
- `tests/spaces/test_lebesgue.py` — 48 tests  ← Phase 2
- `tests/spaces/test_sobolev.py` — 108 tests  ← Phase 4 (79 SpectralSobolev3D + 29 Sobolev3D continuous-first)
- `tests/geometries/test_box_domain.py` — 45 tests  ← Phase 3
- `tests/providers/test_tensor_product.py` — 28 tests  ← Phase 3
- `tests/operators/test_box_laplacian.py` — 12 tests  ← Phase 3
- `tests/operators/test_bessel.py` — 39 tests  ← Phase 4 (23 SpectralBesselOperator3D + 16 BesselOperator3D continuous-first)

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
10. `Sobolev3D` is continuous-first: elements are **`Function3D`** objects, not numpy arrays. It also exposes an optional spectral component map with `dim == provider.dim`, so `to_components(f)` returns L² spectral coefficients via fast matmul against a pre-cached `_phi_vals` matrix and `from_components(c)` reconstructs a `Function3D`. `to_dual(f)` returns `SobolevKernel3D`, and `from_dual(xp)` accepts only kernels owned by the same `Sobolev3D` instance. `SpectralSobolev3D` is the explicit secondary coefficient-based API (old name: `Sobolev3D`).
11. `BesselOperator3D` maps `Function3D → Function3D` via project → scale → reconstruct. Self-adjoint. `bessel_direct`/`bessel_inverse` use `power = ±s/2`. `SpectralBesselOperator3D` is the secondary coefficient-based API (old name: `BesselOperator3D`). **Invertibility guard** applies to both classes.
12. `BoxDomain.sample_boundary` distributes samples proportionally to face areas via `rng.multinomial`, then fills each face with uniform samples. `BoxDomain.outward_normal` assigns normals by finding the face with minimum unsigned distance per point; ties broken by face-index order (x_min first). **Precondition**: meaningful only for boundary points; interior points silently receive nearest-face normal.
