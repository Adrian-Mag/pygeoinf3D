# pygeoinf3D Living Reference

> **Status:** SOLA operator Phase 4 complete — API locked with export tests and extension boundary documented.
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
│   │   ├── __init__.py      # Re-exports Lebesgue3D, VolumeKernel3D
│   │   ├── forms.py         # VolumeKernel3D  ← Phase 2
│   │   └── lebesgue.py      # Lebesgue3D  ← Phase 2, updated Phase 4
│   ├── geometries/
│   │   ├── __init__.py
│   │   └── box/
│   │       ├── __init__.py
│   │       └── domain.py    # BoxDomain  ← Phase 3
│   ├── providers/
│   │   ├── __init__.py        # Re-exports FunctionProvider3D, KernelProvider3D, FunctionToKernelAdapter, TensorProductProvider
│   │   ├── base.py            # FunctionProvider3D (ABC), KernelProvider3D (ABC), FunctionToKernelAdapter  ← SOLA Phase 3
│   │   └── tensor_product.py  # TensorProductProvider  ← Phase 3
│   ├── operators/
│   │   ├── __init__.py
│   │   ├── laplacian_box.py   # BoxLaplacian  ← Phase 3
│   │   └── sola.py            # SOLAOperator3D  ← SOLA Phase 1+2+3
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
│   │   └── test_tensor_product.py    ← Phase 3
│   ├── operators/
│   │   ├── __init__.py
│   │   ├── test_box_laplacian.py     ← Phase 3
│   │   └── test_sola.py             ← SOLA Phase 1+2+3+4
│   └── representations/              # Lebesgue coefficient architecture Phases 1-2
│       ├── __init__.py
│       ├── test_lebesgue_representation.py  ← architecture Phase 1
│       └── test_projection_backends.py     ← architecture Phase 2
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
- **Safe to generalise**: new kernel families (e.g. surface integrals, point evaluations) can be plugged in via a custom `KernelProvider3D` and will work directly with the forward and dual mappings.
- **Requires care for adjoint**: if a future subclass changes the dual-mapping return type to something other than a `VolumeKernel3D`, `domain.from_dual` will raise `TypeError`. To support adjoint on such spaces, either override `adjoint` in the subclass or extend `Lebesgue3D.from_dual` to handle the new dual type.

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

**335 tests reported passing** (foundation Phases 1-3 + coefficient architecture Phases 1-2 + SOLA Phases 1-4):
- `tests/core/test_config.py` — 13 tests
- `tests/core/test_region.py` — 14 tests
- `tests/core/test_domain_base.py` — 21 tests
- `tests/core/test_boundary.py` — 8 tests
- `tests/core/test_functions.py` — 37 tests  ← Phase 2
- `tests/spaces/test_lebesgue.py` — 84 tests  ← Phase 2, extended Phase 4
- `tests/geometries/test_box_domain.py` — 45 tests  ← Phase 3
- `tests/providers/test_tensor_product.py` — 28 tests  ← Phase 3
- `tests/operators/test_box_laplacian.py` — 12 tests  ← Phase 3
- `tests/representations/test_lebesgue_representation.py` — 22 tests  ← architecture Phase 1
- `tests/representations/test_projection_backends.py` — 40 tests  ← architecture Phase 2
- `tests/operators/test_sola.py` — 28 tests  ← SOLA Phases 1+2+3+4

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
