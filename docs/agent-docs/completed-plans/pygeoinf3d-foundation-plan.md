## Plan: pygeoinf3D Foundation

Create a new extension package, pygeoinf3D, that mirrors the intervalinf architecture for 3D function spaces relevant to geophysics while improving its geometry design. The package should establish a shared 3D substrate first with an explicit split between pure region semantics, computational domains, and structured domains, using `EuclideanSpace(3)` and `Subset` objects from `pygeoinf` as the semantic geometry layer, then land one low-risk geometry (box/parallelepiped), then one high-value geometry (ball), with cylinder and ellipsoid deferred until the architecture is stable.

The plan must also preserve a clean distinction between primal elements (pointwise functions that belong to a function space) and dual elements (point, line, surface, or volume distributions acting on those functions). In keeping with `pygeoinf`, dual pairings should be treated as more fundamental than inner products: the inner product is one important induced structure, but APIs should not assume every relevant datum is most naturally represented through an $L^2$-style inner-product viewpoint. Early phases should implement the primal space hierarchy only; distribution support should arrive only after trace-capable Sobolev infrastructure exists.

**Dependency Policy**

| Group       | Packages                            | pyproject.toml key | Status     |
|-------------|-------------------------------------|--------------------|------------|
| Core        | numpy, scipy, matplotlib            | `dependencies`     | Required   |
| Geometry    | trimesh, meshzoo                    | `extras.geometry`  | Optional   |
| Spherical   | pyshtools, Cartopy                  | `extras.spherical` | Optional   |
| Heavy FEM   | FEniCS, dolfinx, ngsolve, firedrake | —                  | **Deferred v2+** |

Heavy FEM/mesh ecosystems introduce binary-distribution complexity and conflicting conda/pip environments that are disproportionate for v1 scope. They must not be added without a full architectural review.

**Decisions Already Made**
1. `Region3D` will be a thin wrapper over `pygeoinf.subsets.Subset`, not a separate heavyweight geometry hierarchy.
2. `Domain3D` will be a computational abstraction built on top of `Region3D` and will not inherit from `HilbertSpace`.
3. `StructuredDomain3D` will be an optional intermediate layer for geometries with natural grids, coordinate systems, tensor-product structure, or spectral structure.
4. `Function3D` and similar objects in early phases will represent primal space elements only, not generalized distributions.
5. Point, curve, surface, and volume distributions will be introduced later as dual-element objects that pair against sufficiently regular spaces; they must not be conflated with primal functions.

**Minimal API Sketch**
1. **`Region3D`**
    - **Purpose:** Thin semantic wrapper around a `Subset` in `EuclideanSpace(3)`.
    - **Required attributes and methods:**
        1. `ambient_space` -> returns `EuclideanSpace(3)`
        2. `subset` -> returns the wrapped `Subset`
        3. `contains(point)` -> membership test for a point in `R^3`
        4. `boundary_subset` -> returns the geometric boundary as a `Subset` where available
        5. `bounding_box()` -> returns an axis-aligned bounding box for sampling and plotting support

2. **`Domain3D`**
    - **Purpose:** Abstract computational domain built on top of a `Region3D`.
    - **Required attributes and methods:**
        1. `ambient_space` -> returns `EuclideanSpace(3)`
        2. `region` -> returns the underlying `Region3D`
        3. `contains(point)` -> delegates semantic membership to the region
        4. `bounding_box()` -> finite computational bounds
        5. `sample_interior(n, *, rng=None)` -> sample interior points for testing and stochastic routines
        6. `integrate_volume(f, *, method=None, **kwargs)` -> integrate a scalar callable over the domain volume
        7. `boundary` -> returns a boundary object or boundary-domain view suitable for boundary conditions
        8. `dim` -> returns the physical dimension, fixed at `3`

3. **`StructuredDomain3D`**
    - **Purpose:** Optional domain class for geometries with strong numerical structure.
    - **Required attributes and methods:**
        1. `coordinate_system` -> reports the native coordinates used by the domain
        2. `structured_mesh(resolution, *, location='cell_centers')` -> returns a geometry-appropriate mesh or grid
        3. `volume_weights(resolution, **kwargs)` -> weights compatible with the structured mesh
        4. `boundary_components` -> labeled boundary pieces for Dirichlet, Neumann, and Robin conditions
    - **Optional fast paths:**
        1. `integrate_volume_on_mesh(values, weights=None)`
        2. `sample_boundary(n, *, rng=None)`
        3. `outward_normal(points, *, component=None)`

4. **Concrete domain expectations**
    - `BoxDomain` should implement `StructuredDomain3D` with Cartesian tensor-product structure.
    - `BallDomain` should implement `StructuredDomain3D` with spherical coordinate support.

5. **Function-space and dual-space expectations**
    - `Lebesgue3D`, `Sobolev3D`, and related spaces should be modeled as `pygeoinf.HilbertSpace` subclasses defined on a `Domain3D`.
    - Dual elements should be modeled separately from primal elements, with pairings attached to the space layer rather than to `Region3D` or `Domain3D`.
    - Pairing operations should be first-class in the design; inner products and Riesz maps should be treated as derived structure where appropriate.
    - Boundary, curve, and point supported objects should be expressed through dedicated support or distribution abstractions, not by overloading `Function3D`.

**Phases 7**
1. **Phase 1: Package Skeleton and Core Geometry Contracts**
    - **Objective:** Establish the package layout, the explicit `Region3D` or `Subset` -> `Domain3D` -> `StructuredDomain3D` split, and the base representation conventions before any operators are implemented.
    - **Files/Functions to Modify/Create:** pygeoinf3D/pyproject.toml, pygeoinf3D/README.md, pygeoinf3D/pygeoinf3d/core/region.py, pygeoinf3D/pygeoinf3d/core/domain_base.py, pygeoinf3D/pygeoinf3d/core/boundary.py, pygeoinf3D/pygeoinf3d/core/config.py, pygeoinf3D/tests/core/*
    - **Tests to Write:** region-membership tests, subset-wrapping tests, domain contract tests, mesh or grid generation tests, boundary-condition validation tests, config defaults tests
    - **Steps:**
        1. Write failing tests for common 3D domain contracts and boundary specifications.
        2. Implement a thin `Region3D` layer or direct `Subset` adapter for pure geometry semantics in `EuclideanSpace(3)`.
        3. Implement a minimal `Domain3D` base that wraps a geometric region and provides computational hooks without inheriting from `HilbertSpace`.
        4. Add an optional `StructuredDomain3D` intermediate layer for geometries with natural grids, tensor-product structure, or coordinate charts.
        5. Add package metadata, dependency policy, and test scaffolding.

2. **Phase 2: Function Representation and 3D Lebesgue Space**
    - **Objective:** Recreate the intervalinf continuous-first, basis-optional pattern for 3D scalar functions while making it explicit that these are primal space elements, not generalized distributions, and that the resulting space API must remain compatible with pairing-first dual constructions.
    - **Files/Functions to Modify/Create:** pygeoinf3D/pygeoinf3d/core/functions.py, pygeoinf3D/pygeoinf3d/spaces/lebesgue.py, pygeoinf3D/tests/core/test_functions.py, pygeoinf3D/tests/spaces/test_lebesgue.py
    - **Tests to Write:** callable-backed function evaluation, attach-detach semantics, dual-pairing tests against simple representers, 3D inner-product tests, norm and dual-map tests, explicit rejection of treating distributions as primal functions
    - **Steps:**
        1. Write failing tests for a 3D `Function` object with domain-only and space-attached modes.
        2. Implement `Function3D` as the primal element wrapper for scalar fields on a `Domain3D`.
        3. Implement `Lebesgue3D` as a concrete `pygeoinf.HilbertSpace` defined on a `Domain3D`, exposing the dual pairing cleanly rather than burying it behind inner-product-only helpers.
        4. Add initial quadrature and integration paths sufficient for correctness on simple domains.

3. **Phase 3: Box Geometry MVP**
    - **Objective:** Deliver the first fully usable geometry using tensor-product structure, because it has the lowest technical risk.
    - **Files/Functions to Modify/Create:** pygeoinf3D/pygeoinf3d/geometries/box/domain.py, pygeoinf3D/pygeoinf3d/providers/tensor_product.py, pygeoinf3D/pygeoinf3d/operators/laplacian_box.py, pygeoinf3D/tests/providers/test_tensor_product.py, pygeoinf3D/tests/operators/test_box_laplacian.py
    - **Tests to Write:** separable basis enumeration tests, eigenvalue-ordering tests, box Laplacian spectral tests, Dirichlet and Neumann regression tests
    - **Steps:**
        1. Write failing tests for box-domain membership, tensor-product basis indexing, and low-order Laplacian modes.
        2. Implement `BoxDomain` and parallelepiped support as concrete `StructuredDomain3D` realizations on top of subset-based geometry and tensor-product provider machinery.
        3. Implement the first spectral Laplacian and verify integration with pygeoinf operators.

4. **Phase 4: Sobolev and Bessel-Type Operators on Box Domains**
    - **Objective:** Add the first inference-ready function spaces on top of the box geometry, including the regularity and trace structure needed before introducing geometric distributions.
    - **Files/Functions to Modify/Create:** pygeoinf3D/pygeoinf3d/spaces/sobolev.py, pygeoinf3D/pygeoinf3d/operators/bessel.py, pygeoinf3D/tests/spaces/test_sobolev.py, pygeoinf3D/tests/operators/test_bessel.py
    - **Tests to Write:** Sobolev dual-map tests, mass-operator tests, covariance-eigenvalue tests, compatibility tests with pygeoinf inversion components, trace and boundary-pairing readiness tests
    - **Steps:**
        1. Write failing tests for `H^s` structure on the box geometry.
        2. Implement Bessel-Sobolev operators and `MassWeightedHilbertSpace` integration.
        3. Add the minimal trace or boundary-evaluation contracts required for later surface-supported dual elements.
        4. Verify Gaussian-measure and inversion compatibility at the operator level.

5. **Phase 5: Ball Geometry MVP**
    - **Objective:** Deliver the second major geometry by combining radial and angular machinery into a volumetric ball space.
    - **Files/Functions to Modify/Create:** pygeoinf3D/pygeoinf3d/geometries/ball/domain.py, pygeoinf3D/pygeoinf3d/providers/ball.py, pygeoinf3D/pygeoinf3d/operators/laplacian_ball.py, pygeoinf3D/tests/providers/test_ball_providers.py, pygeoinf3D/tests/operators/test_ball_laplacian.py
    - **Tests to Write:** radial normalization tests, spherical-harmonic coupling tests, low-mode eigenpair tests, boundary-condition tests on the ball
    - **Steps:**
        1. Write failing tests around the lowest ball eigenmodes and normalization.
        2. Implement `BallDomain` as a concrete `StructuredDomain3D`, extend the current radial ideas beyond `ell = 0`, and define a consistent ball basis ordering.
        3. Implement the ball Laplacian and validate it against analytic low-mode expectations.

6. **Phase 6: Dual Elements and Geometric Distributions**
    - **Objective:** Introduce point-, curve-, surface-, and volume-supported dual elements as separate objects that act on sufficiently regular function spaces.
    - **Files/Functions to Modify/Create:** pygeoinf3D/pygeoinf3d/duals/base.py, pygeoinf3D/pygeoinf3d/duals/distributions.py, pygeoinf3D/pygeoinf3d/core/supports.py, pygeoinf3D/tests/duals/test_distributions.py, pygeoinf3D/tests/core/test_supports.py
    - **Tests to Write:** point-evaluation pairing tests, boundary surface functional tests, path-integral functional tests, regularity-restriction tests, dual pairing consistency tests
    - **Steps:**
        1. Write failing tests for point, path, and surface functionals acting on spaces with sufficient regularity.
        2. Introduce a dedicated dual-element abstraction rather than overloading `Function3D`.
        3. Add minimal support carriers for embedded points, curves, and surfaces where needed.
        4. Implement pairings against boundary traces and interior functions, with explicit errors when the chosen primal space is too weak.

7. **Phase 7: First End-to-End User Workflow**
    - **Objective:** Prove pygeoinf3D is a real extension package, not just a geometry library, by running a complete inversion or inference workflow.
    - **Files/Functions to Modify/Create:** pygeoinf3D/demos/*, pygeoinf3D/tests/integration/test_pygeoinf_integration.py, pygeoinf3D/docs/*, pygeoinf3D/docs/agent-docs/*
    - **Tests to Write:** end-to-end operator and inversion smoke tests, reproducible synthetic examples, basic performance sanity tests, one workflow using geometric dual data such as path or surface observations
    - **Steps:**
        1. Write failing integration tests using pygeoinf algorithms with pygeoinf3D spaces.
        2. Add one minimal synthetic inversion example on the box, then optionally one on the ball.
        3. Include at least one example driven by non-volumetric data such as line integrals or boundary observations.
        4. Document the architecture, supported geometries, and limitations for v1.

**Open Questions 5**
1. Should pygeoinf3D be basis-optional from day one, or should v1 be basis-backed first and add continuous-first workflows later?
2. For the ball, are optional dependencies like pyshtools acceptable, or do we want a pure NumPy and SciPy baseline first?
3. For box geometries, do we want full per-face boundary conditions in v1, or only uniform Dirichlet and Neumann initially?
4. Should v1 handle only scalar fields, or should the architecture reserve space for vector and tensor fields immediately?
5. Should `boundary` on `Domain3D` be a lightweight boundary-view object, or should boundary pieces themselves be first-class domain objects from the start?
6. When dual elements arrive, should embedded curves and surfaces be represented as dedicated support objects, or as parameterized measures attached directly to dual-element classes?
7. In Phase 2, how closely should `Function3D` mirror `pygeoinf`'s pairing and dual-map conventions from day one, versus starting minimal and tightening compatibility in Phase 4?
