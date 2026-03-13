## Plan: pygeoinf3D Lebesgue Coefficient Architecture

This plan defines a side-track architecture effort for pygeoinf3D: add coefficient-space analysis and synthesis for `Lebesgue3D` without polluting the primal space or kernel-based dual forms with quadrature policy, Gram assembly, basis enumeration, or backend-specific numerical logic. The goal is to stay close to the intervalinf blueprint where it is mathematically sound, while being explicit about one necessary refinement: coefficient extraction in the non-orthogonal case must be Gram-aware or biorthogonal-aware rather than hard-coded as simple per-basis normalization.

The core design rule is that `Lebesgue3D` remains a pairing-first Hilbert space on a `Domain3D`, `VolumeKernel3D` remains a basis-free integration-backed dual object, and all finite-dimensional representation choices are delegated to a separate representation layer that can be swapped, cached, and specialized by geometry.

**Why this tangent is being addressed now**
1. The current `Lebesgue3D` API is mathematically clean but incomplete for pygeoinf workflows that depend on `to_components`, `from_components`, `coordinate_projection`, and `coordinate_inclusion`.
2. The current `BoxLaplacian` already contains ad hoc analysis and synthesis logic, which is a sign that the coefficient-space boundary is real and should be made explicit before more operators are built.
3. If this boundary is not designed now, later Sobolev, Gaussian-measure, and inversion work will likely repeat the same numerical logic in inconsistent places.

**Architecture Principles**
1. **Pairing-first remains authoritative.**
   `Lebesgue3D.to_dual` and `Lebesgue3D.from_dual` define the Hilbert structure; coefficient maps are optional finite-dimensional representations, not the definition of the space.
2. **Representation is optional.**
   A `Lebesgue3D` without an attached representation remains valid and basis-free.
3. **Numerics are delegated.**
   Quadrature grids, weights, Gram matrices, Gram solvers, and biorthogonalization must not live in `Lebesgue3D` or `VolumeKernel3D`.
4. **Coefficient vectors inherit a metric.**
   The finite coefficient carrier may be represented algebraically as `R^n`, but its Hilbert-space inner product is generally induced by the chosen representation via a Gram or mass operator. It is only the standard Euclidean inner product when the chosen family is orthonormal for the pairing used.
4. **Domains provide integration capability, not projection policy.**
   `Domain3D` and `BoxDomain` may provide integration and structured-mesh primitives, but not the rules for which basis to use or how to solve for coefficients.
5. **Providers enumerate basis data; backends perform numerical projection.**
   Basis providers supply finite families of basis functions and related metadata. Representation backends use those providers plus domain integration to implement analysis and synthesis.
6. **Deviation from intervalinf must be explicit.**
   intervalinf places `to_components` and `from_components` directly on the space and in some cases computes coefficients by basiswise normalization. pygeoinf3D will keep the space-level API but route it through a dedicated backend so that general Gram-aware projection is possible.

**Relationship to Existing Packages**
1. **pygeoinf**
   - `HilbertSpace` expects `to_components` and `from_components` whenever a finite representation is exposed.
   - `coordinate_projection` and `coordinate_inclusion` are derived from those methods and become part of the operator algebra.
   - `MassWeightedHilbertSpace` shows that metric-like structure should be modeled explicitly rather than smuggled through ad hoc coefficient conventions.
2. **intervalinf**
   - intervalinf correctly treats `Lebesgue` as a continuous function space with kernel-based duals.
   - intervalinf also exposes optional basis-backed workflows on the space itself.
   - pygeoinf3D should follow that high-level shape, but the coefficient backend should be designed for general Gram solves and biorthogonal bases rather than assuming orthogonality.
3. **pygeoinf3D current state**
   - `Lebesgue3D` and `VolumeKernel3D` are clean and basis-free.
   - `TensorProductProvider` already acts like a basis enumerator.
   - `BoxLaplacian` currently performs private projection/reconstruction work that should move into a shared representation/backend layer.

**Target Package Structure After This Work**

The package should evolve toward the following structure:

1. `pygeoinf3d/core/`
   - unchanged as the geometric and primal-function substrate.
   - continues to define `Domain3D`, `StructuredDomain3D`, `Function3D`, and related geometry classes.
2. `pygeoinf3d/spaces/`
   - continues to hold mathematical Hilbert spaces and kernel-backed dual forms.
   - `lebesgue.py` remains thin and pairing-first.
   - `forms.py` remains basis-free and duality-focused.
3. `pygeoinf3d/providers/`
   - continues to hold basis/spectrum providers such as `TensorProductProvider`.
   - providers expose basis functions and indexing metadata but do not own projection numerics.
4. `pygeoinf3d/representations/` (new)
   - new home for finite-dimensional representation objects.
   - contains representation interfaces and concrete numerical backends.
   - becomes the canonical place for analysis maps, synthesis maps, Gram assembly, Gram solves, and caching.
5. `pygeoinf3d/operators/`
   - operators that need finite-dimensional projection consume representation objects rather than reimplementing projection logic.

**Proposed New Directory Layout**

1. `pygeoinf3d/representations/__init__.py`
   - re-exports public representation APIs.
2. `pygeoinf3d/representations/base.py`
   - abstract interfaces for coefficient representations and numerical backends.
3. `pygeoinf3d/representations/lebesgue.py`
   - Lebesgue-specific representation wrapper that binds a `Lebesgue3D`, a basis provider, and a backend.
4. `pygeoinf3d/representations/backends.py`
   - concrete backend implementations for orthonormal projection and general Gram-based projection.
5. `pygeoinf3d/representations/gram.py`
   - shared Gram assembly and factorization helpers.
6. `pygeoinf3d/tests/representations/`
   - tests for representation objects and backends.

If the backend layer remains small, `backends.py` and `gram.py` can initially be merged into `lebesgue.py` under `representations/`; the conceptual separation should still be preserved.

**Target Public API**

The intended public API should look like this after the work lands:

1. `Lebesgue3D(domain, representation=None)`
   - basis-free by default.
   - if `representation` is provided, component methods delegate to it.
2. `space.to_components(f)`
   - returns true expansion coefficients in the attached coefficient space.
   - raises a clear error if no representation is attached.
3. `space.from_components(c)`
   - returns the synthesis of coefficient vector `c` back into a `Function3D`.
4. `space.coordinate_projection`
   - automatically becomes meaningful when `to_components` is available.
5. `space.coordinate_inclusion`
   - automatically becomes meaningful when `from_components` is available.
6. `LebesgueRepresentation3D(space, provider, backend)`
   - first-class representation object for coefficient workflows.
7. `representation.analysis_operator`
   - optional explicit operator view from `Lebesgue3D` to `EuclideanSpace(n)`.
8. `representation.synthesis_operator`
   - optional explicit operator view from `EuclideanSpace(n)` to `Lebesgue3D`.
9. `representation.coefficient_metric()`
   - exposes the induced coefficient-space metric or mass operator.
10. `representation.weighted_coefficient_space()`
   - optional Hilbert-space view of the coefficient carrier using the induced metric.
11. `representation.gram_matrix()`
   - exposes the Gram matrix when meaningful.
12. `representation.biorthogonal_dual_basis()`
   - optional advanced API for backends that materialize the dual basis explicitly.

**Important distinction:**
1. `EuclideanSpace(n)` is the natural algebraic carrier for coefficient vectors.
2. The induced Hilbert structure on those coefficients is generally not the plain Euclidean one.
3. For non-orthonormal families, the correct coefficient-space inner product is determined by the Gram or mass operator associated with the representation.
4. Phase 2 should decide whether the public API exposes this as
   1. `EuclideanSpace(n)` plus a separate `coefficient_metric()` operator, or
   2. a derived `MassWeightedHilbertSpace(EuclideanSpace(n), M)` style object.

**Detailed Responsibility Split**

1. `pygeoinf3d.spaces.lebesgue.Lebesgue3D`
   - **Keeps:** domain identity, Hilbert-space equality, pairing-first `to_dual`, conservative `from_dual`, arithmetic helpers, optional delegation to an attached representation.
   - **Does not keep:** quadrature grids, Gram matrices, solver caches, provider-specific mode ordering, spectral projection loops.
2. `pygeoinf3d.spaces.forms.VolumeKernel3D`
   - **Keeps:** integration-backed dual evaluation and linear-form arithmetic.
   - **Does not keep:** component recovery, basis indexing, coefficient caches.
3. `pygeoinf3d.providers.tensor_product.TensorProductProvider`
   - **Keeps:** basis enumeration, eigenvalues, eigenfunctions, truncation metadata, ordering conventions.
   - **Does not keep:** projection coefficients, quadrature policies, Gram solves.
4. `pygeoinf3d.representations.base`
   - **Keeps:** abstract contracts for `dim`, `to_components`, `from_components`, optional operator views, and optional Gram/dual-basis accessors.
5. `pygeoinf3d.representations.backends`
   - **Keeps:** numerical algorithms for moments, Gram assembly, orthonormal shortcuts, linear solves, caching.
6. `pygeoinf3d.operators.laplacian_box.BoxLaplacian`
   - **Keeps:** operator semantics for `-Δ`.
   - **Changes:** consumes a box-compatible representation/backend instead of building its own private projection mechanism.

**Core Design Objects**

1. **`CoefficientRepresentation3D`**
   - **Purpose:** abstract finite-dimensional representation attached to a continuous space.
   - **Required attributes and methods:**
     1. `space` -> the `Lebesgue3D` being represented.
     2. `dim` -> coefficient-space dimension.
   3. `coefficient_space` -> algebraic carrier, typically `pygeoinf.EuclideanSpace(dim)`.
   4. `coefficient_metric()` -> induced metric or mass operator for the coefficient-space Hilbert structure.
     4. `to_components(f)` -> analysis map producing true coefficients.
     5. `from_components(c)` -> synthesis map producing a `Function3D`.
     6. `analysis_operator` -> explicit operator view, if exposed.
     7. `synthesis_operator` -> explicit operator view, if exposed.
2. **`LebesgueRepresentation3D`**
   - **Purpose:** concrete representation for `Lebesgue3D` using a basis provider and a numerical backend.
   - **Required constructor inputs:**
     1. `space`
     2. `provider`
     3. `backend`
   - **Expected behavior:**
     1. validates compatibility between space domain and provider domain.
     2. exposes `dim` through the provider.
   3. exposes the induced coefficient metric through the backend once Phase 2 lands.
   4. delegates numerical work to the backend.
3. **`LebesgueCoefficientBackend`**
   - **Purpose:** numerical strategy object for analysis and synthesis.
   - **Required methods:**
     1. `moments(space, provider, f)` -> raw vector `b_i = <phi_i, f>`.
     2. `gram_matrix(space, provider)` -> Gram matrix `G`.
   3. `coefficient_metric(space, provider)` -> metric or mass operator induced on the coefficient carrier.
   4. `solve_coefficients(gram, moments)` -> coefficient vector `c` satisfying `G c = b`.
   5. `to_components(space, provider, f)` -> true coefficients.
   6. `from_components(space, provider, c)` -> synthesized `Function3D`.
   - **Optional methods:**
     1. `dual_basis(space, provider)` -> explicit biorthogonal basis or representers.
     2. `supports_orthonormal_shortcut` -> indicates when `G` can be skipped.

**Backend Strategy for Phase 1 of This Work**

The initial backend set should be small and deliberate:

1. **`OrthonormalProjectionBackend`**
   - assumes the provider basis is orthonormal with respect to the domain integration rule being used.
   - analysis uses raw moments directly as coefficients.
   - appropriate for tensor-product Dirichlet or Neumann box eigenfunctions when quadrature resolution is chosen consistently.
2. **`GramProjectionBackend`**
   - computes raw moments, assembles the Gram matrix, and solves `G c = b`.
   - this is the general and mathematically correct path.
   - should be the long-term default for arbitrary non-orthogonal bases.

The orthonormal backend is a performance specialization, not the conceptual foundation.

**Expected Changes to Existing Files**

1. `pygeoinf3d/spaces/lebesgue.py`
   - add optional `representation` constructor argument.
   - add representation-aware `dim` behavior.
   - implement `to_components` and `from_components` by delegation.
   - keep `to_dual` and `from_dual` unchanged in spirit.
2. `pygeoinf3d/spaces/forms.py`
   - likely no public API change.
   - may gain clarifying docstrings stating that `VolumeKernel3D` intentionally stays basis-free.
3. `pygeoinf3d/providers/tensor_product.py`
   - may gain basis-oriented helpers such as `get_basis_function(i)` or `basis_functions()` aliases if needed by the representation layer.
   - should not gain projection routines.
4. `pygeoinf3d/operators/laplacian_box.py`
   - refactor to reuse a `LebesgueRepresentation3D` instead of owning projection internals.
5. `pygeoinf3d/__init__.py`, `pygeoinf3d/spaces/__init__.py`, `pygeoinf3d/operators/__init__.py`
   - export new public representation classes only once the architecture is stable.
6. `docs/agent-docs/references/living/pygeoinf3d-reference.md`
   - update architecture and file map once the work is implemented.

**Expected New Files and Their Contents**

1. `pygeoinf3d/representations/base.py`
   - abstract representation interface(s).
   - minimal shared validation and operator-construction helpers.
2. `pygeoinf3d/representations/lebesgue.py`
   - `LebesgueRepresentation3D`.
   - space-provider-backend compatibility checks.
   - thin API façade.
3. `pygeoinf3d/representations/backends.py`
   - `LebesgueCoefficientBackend` abstract base.
   - `OrthonormalProjectionBackend`.
   - `GramProjectionBackend`.
4. `pygeoinf3d/representations/gram.py`
   - helpers for dense Gram assembly, symmetry checks, conditioning warnings, and possibly cached factorizations.
5. `tests/representations/test_lebesgue_representation.py`
   - delegation, round-trip, and operator-view tests.
6. `tests/representations/test_projection_backends.py`
   - orthonormal and Gram-aware coefficient tests.

**API Behavior Decisions Proposed by This Plan**
1. `Lebesgue3D.dim` should be `0` when no representation is attached.
2. `Lebesgue3D.dim` should equal `representation.dim` when a representation is attached.
3. `to_components` should return true basis coefficients, not raw moments.
4. Raw moments should remain available as backend-level or representation-level advanced API, not as the default meaning of components.
5. If no representation is attached, `to_components` and `from_components` should continue to raise a clear `NotImplementedError` or `RuntimeError` explaining that a coefficient representation is required.
6. `VolumeKernel3D.components` should remain unavailable in basis-free mode unless a later explicit design introduces dual coefficient representations.
7. The representation should distinguish between the algebraic coefficient carrier and the induced coefficient-space Hilbert structure.
8. Phase 2 should likely expose a coefficient metric or mass operator explicitly, rather than silently treating coefficients as standard Euclidean vectors.

**How This Plan Follows or Deviates from intervalinf**
1. **Follows intervalinf:**
   - continuous-first `Lebesgue` space.
   - kernel-based dual forms.
   - optional basis-backed workflows.
   - basis providers separate from primal function objects.
2. **Explicit deviation:**
   - pygeoinf3D will not place Gram and quadrature policy directly inside `Lebesgue3D`.
   - pygeoinf3D will prefer backend-driven coefficient recovery so that non-orthogonal bases are handled correctly.
3. **Reason for deviation:**
   - pygeoinf3D needs a reusable architecture for 3D geometries where numerical realization is likely to vary by domain and by basis family.
   - the backend split prevents the space layer from becoming a hidden spectral implementation.

**Interaction with pygeoinf Operator Algebra**

This work should preserve full compatibility with pygeoinf’s operator machinery:

1. Once a representation is attached, `Lebesgue3D.coordinate_projection` should behave as the analysis operator induced by `to_components`.
2. Once a representation is attached, `Lebesgue3D.coordinate_inclusion` should behave as the synthesis operator induced by `from_components`.
3. Future operators can then be built compositionally as
   1. analysis in coefficient space,
   2. diagonal or sparse action in coefficient space,
   3. synthesis back to the primal space.
4. This gives a pygeoinf-native route to Galerkin or spectral operators without hard-coding projection internals into each operator.
5. When operator adjoints or Hilbert-space geometry matter in coefficient space, the induced coefficient metric must be used rather than assuming the plain Euclidean inner product.

**Phases 5**
1. **Phase 1: Representation Boundary and API Contracts**
   - **Objective:** Define the representation abstraction and wire `Lebesgue3D` to delegate coefficient operations without changing its pairing-first meaning.
   - **Files/Functions to Modify/Create:** `pygeoinf3d/spaces/lebesgue.py`, `pygeoinf3d/representations/base.py`, `pygeoinf3d/representations/lebesgue.py`, `tests/spaces/test_lebesgue.py`, `tests/representations/test_lebesgue_representation.py`
   - **Tests to Write:** basis-free default tests, representation-attached dimension tests, delegation tests for `to_components` and `from_components`, failure tests when no representation is attached.
   - **Steps:**
       1. Write tests that preserve current basis-free behavior.
       2. Introduce the representation interface and minimal concrete wrapper.
       3. Delegate `Lebesgue3D` component methods through the representation.
       4. Verify `coordinate_projection` and `coordinate_inclusion` become meaningful once a representation is attached.

2. **Phase 2: Gram-Aware Backend Infrastructure**
   - **Objective:** Implement reusable coefficient backends that separate raw moments, Gram assembly, and coefficient solves.
   - **Files/Functions to Modify/Create:** `pygeoinf3d/representations/backends.py`, `pygeoinf3d/representations/gram.py`, `tests/representations/test_projection_backends.py`
   - **Tests to Write:** orthonormal projection tests, Gram-solve tests for non-orthogonal mock bases, coefficient-metric tests, consistency tests between raw moments and solved coefficients.
   - **Steps:**
       1. Write failing tests for orthonormal and non-orthogonal cases.
       2. Implement moment assembly through domain integration.
      3. Implement Gram assembly, induced coefficient metric construction, and coefficient solves.
      4. Decide whether the public API exposes `EuclideanSpace(n) + metric` or a `MassWeightedHilbertSpace`-style coefficient Hilbert space.
      5. Add conservative caching where it belongs in the backend.

3. **Phase 3: Tensor-Product Box Integration**
   - **Objective:** Make `TensorProductProvider` usable as the first real basis provider for the new representation layer.
   - **Files/Functions to Modify/Create:** `pygeoinf3d/providers/tensor_product.py`, `pygeoinf3d/representations/lebesgue.py`, `tests/providers/test_tensor_product.py`, `tests/representations/test_lebesgue_representation.py`
   - **Tests to Write:** eigenfunction-to-coefficient tests, synthesis round-trip tests, provider-compatibility tests with `BoxDomain` and `Lebesgue3D`.
   - **Steps:**
       1. Add any missing basis accessors on the provider.
       2. Bind the provider cleanly to `LebesgueRepresentation3D`.
       3. Validate known box eigenfunctions map to expected canonical coefficients.

4. **Phase 4: Operator Refactor onto Shared Representation**
   - **Objective:** Refactor projection-using operators to reuse the shared backend rather than keeping private analysis logic.
   - **Files/Functions to Modify/Create:** `pygeoinf3d/operators/laplacian_box.py`, `tests/operators/test_box_laplacian.py`
   - **Tests to Write:** regression tests showing unchanged Laplacian action, tests confirming representation-backed projection path is used consistently.
   - **Steps:**
       1. Write regression tests against the current spectral behavior.
       2. Replace private projection code in `BoxLaplacian` with calls into the shared representation.
       3. Verify correctness and preserve the current public operator semantics.

5. **Phase 5: Documentation and Blueprint Alignment**
   - **Objective:** Document the final API, architecture boundary, and intervalinf alignment clearly enough that later phases do not regress into embedding numerics in spaces.
   - **Files/Functions to Modify/Create:** `README.md`, `docs/agent-docs/references/living/pygeoinf3d-reference.md`, representation module docstrings, optional demo snippets.
   - **Tests to Write:** none beyond doc examples if added.
   - **Steps:**
       1. Update the living reference with the new file map and API boundary.
       2. Document how representation-attached and basis-free `Lebesgue3D` differ.
       3. Explicitly note where pygeoinf3D follows intervalinf and where it intentionally refines it.

**Open Questions 6**
1. Should the first implementation expose only a single attached representation per `Lebesgue3D`, or should a space support multiple named representations simultaneously?
2. Should raw moments be exposed publicly on `LebesgueRepresentation3D`, or only through backend internals?
3. Should the first tensor-product backend default to the orthonormal shortcut, or should Gram projection be the default even for box eigenfunctions?
4. Do we want explicit condition-number checks and warnings for Gram systems in v1 of this architecture?
5. Should dual-basis objects be materialized explicitly as kernel-backed forms, or kept implicit through Gram solves until there is a real use case?
6. Should future Sobolev or trace-ready spaces reuse this same representation package, or should that be postponed until the Lebesgue design is proven stable?

**Recommended Answers to Open Questions**
1. Start with a single attached representation per `Lebesgue3D` to keep equality and operator semantics simple.
2. Expose raw moments as advanced API on the representation, not on the space.
3. Use the orthonormal shortcut only as an opt-in or clearly labeled specialized backend; keep the general design centered on Gram-aware projection.
4. Add at least basic conditioning diagnostics in the Gram helpers.
5. Keep dual bases implicit initially.
6. Reuse this package structure later for Sobolev-type spaces if and only if the Lebesgue workflow proves stable first.
7. Treat `EuclideanSpace(dim)` as the algebraic coefficient carrier only; expose the induced Hilbert structure through a metric or mass operator in Phase 2.