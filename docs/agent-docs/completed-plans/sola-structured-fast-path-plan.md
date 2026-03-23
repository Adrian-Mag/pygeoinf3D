## Plan: Structured SOLA Fast Paths

Build principled, structure-aware SOLA acceleration inside the libraries by keeping `SOLAOperator3D` as the generic fallback while adding reusable interaction backends and solver-level fast paths in `pygeoinf3D`. The first payoff comes from caching quadrature data and assembling repeated volumetric form interactions once; later phases extend the same pattern to spectral Sobolev and mixed form stacks, with only small generic hooks added to `pygeoinf` where matrix-aware propagation already belongs.

**Phases 5 phases**
1. **Phase 1: Quadrature Cache Primitives**
    - **Objective:** Introduce reusable structured-domain quadrature caching so repeated volumetric evaluations stop rebuilding the same mesh and weights.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/pygeoinf3d/geometries/box/domain.py`; new tests in `pygeoinf3D/tests/geometries/` or `pygeoinf3D/tests/core/`.
    - **Tests to Write:** `test_box_domain_structured_mesh_cache_reuses_same_array_shape_and_values`, `test_box_domain_volume_weights_cache_reuses_same_values`, `test_integrate_volume_matches_cached_mesh_dot_product`.
    - **Steps:**
        1. Write tests for repeated `structured_mesh` and `volume_weights` requests at the same resolution/location and for consistency with `integrate_volume_on_mesh`.
        2. Run those tests to confirm the current implementation recomputes without a reusable cache abstraction.
        3. Add internal cache plumbing in `BoxDomain` keyed by `(resolution, location)` without changing public semantics.
        4. Re-run the focused tests to confirm identical numerical output and stable cache reuse behavior.

2. **Phase 2: Volumetric Interaction Backend**
    - **Objective:** Add a reusable backend that assembles and caches volumetric kernel evaluations on a structured quadrature mesh for any batch-evaluable `Function3D` family, not just Gaussians.
    - **Files/Functions to Modify/Create:** new module `pygeoinf3D/pygeoinf3d/solvers/volume_interactions.py` or `pygeoinf3D/pygeoinf3d/representations/volume_interactions.py`; `pygeoinf3D/pygeoinf3d/providers/base.py`; optional optimization in `pygeoinf3D/pygeoinf3d/providers/function_providers/gaussian.py`; exports in `pygeoinf3D/pygeoinf3d/solvers/__init__.py`; tests under `pygeoinf3D/tests/solvers/` or `pygeoinf3D/tests/representations/`.
    - **Tests to Write:** `test_volume_interaction_backend_matches_kernel_by_kernel_quadrature`, `test_volume_interaction_backend_cross_gram_matches_direct_integration`, `test_volume_interaction_backend_falls_back_when_provider_has_no_batch_api`, `test_gaussian_provider_evaluate_all_matches_get_function_loop`.
    - **Steps:**
        1. Write failing tests comparing cached matrix assembly against direct `VolumeKernel3D` quadrature for small volumetric stacks.
        2. Define an interaction backend that owns `(pts, weights)` and materializes kernel evaluation matrices for a fixed provider or kernel list.
        3. Add an optional batched-evaluation protocol on providers and implement it for `GaussianProvider3D`, while preserving a loop fallback for arbitrary providers.
        4. Re-run the focused tests to confirm backend correctness for both batched and fallback paths.

3. **Phase 3: Dense Volume SOLA Solver + Router Integration**
    - **Objective:** Add a solver-level fast path for repeated volumetric SOLA assembly on structured domains, leaving `SOLAOperator3D` as the generic operator.
    - **Files/Functions to Modify/Create:** new module `pygeoinf3D/pygeoinf3d/solvers/dense_volume_sola.py`; `pygeoinf3D/pygeoinf3d/solvers/registry.py`; `pygeoinf3D/pygeoinf3d/solvers/router.py`; `pygeoinf3D/docs/SOLVER_SELECTION_GUIDE.md`; tests in `pygeoinf3D/tests/solvers/test_dense_volume_sola.py` and router tests.
    - **Tests to Write:** `test_dense_volume_sola_gram_matches_generic_sola_operator_matrix`, `test_dense_volume_sola_solution_matches_generic_minimum_norm_solution`, `test_dense_volume_sola_resolving_coefficients_match_generic_operator_action`, `test_solver_router_prefers_dense_volume_sola_for_box_volume_stacks`.
    - **Steps:**
        1. Write failing tests that compare a new dense volumetric solver against the existing generic `SOLAOperator3D` path on a small box-domain problem.
        2. Implement solver construction using the volumetric interaction backend to assemble `GG*`, target cross terms, and repeated data-space solves via BLAS/Cholesky.
        3. Register the solver and extend `SolverRouter` to recommend it when the space/domain/forms satisfy the structured volumetric preconditions.
        4. Re-run the focused tests to confirm numerical equivalence and correct solver selection.

4. **Phase 4: Spectral Mixed-Form Interaction Extension**
    - **Objective:** Generalize the existing spectral fast-path machinery so Sobolev problems with repeated path and volumetric forms can share one interaction matrix backend when a spectral inverse weight is available.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/pygeoinf3d/solvers/spectral_gram.py`; possibly new module `pygeoinf3D/pygeoinf3d/solvers/spectral_interactions.py`; `pygeoinf3D/pygeoinf3d/spaces/forms.py`; `pygeoinf3D/pygeoinf3d/solvers/router.py`; tests in `pygeoinf3D/tests/solvers/` and `pygeoinf3D/tests/spaces/`.
    - **Tests to Write:** `test_spectral_interaction_backend_matches_volume_form_coefficients`, `test_spectral_solver_supports_homogeneous_volume_stack_on_sobolev`, `test_spectral_solver_supports_mixed_path_and_volume_stack`, `test_router_falls_back_when_mixed_stack_is_not_supported`.
    - **Steps:**
        1. Write failing tests for spectral interaction coefficients `L[i, k] = ell_i(Phi_k)` for both path and volumetric forms against direct reference computations.
        2. Refactor the current spectral solver to separate interaction-matrix assembly from solve logic.
        3. Add batched coefficient assembly for `VolumeKernel3D` against `evaluate_all_eigenfunctions`, then support mixed stacks where the full interaction matrix can be assembled safely.
        4. Re-run the focused tests to confirm correctness and graceful fallback behavior.

5. **Phase 5: Generic pygeoinf Hooks and Documentation**
    - **Objective:** Add only the small generic hooks in `pygeoinf` needed to let specialized backends propagate dense operators and Gaussian covariances efficiently, then document the architecture.
    - **Files/Functions to Modify/Create:** `pygeoinf/pygeoinf/linear_operators.py`; `pygeoinf/pygeoinf/gaussian_measure.py`; `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`; `pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md`; new or updated tests in `pygeoinf/tests/` if present, otherwise package-local pygeoinf3D integration tests.
    - **Tests to Write:** `test_affine_mapping_uses_explicit_matrix_covariance_path`, `test_dense_volume_sola_covariance_matches_generic_affine_mapping`, `test_linear_operator_matrix_override_preserves_generic_behavior`.
    - **Steps:**
        1. Write failing tests for covariance propagation and dense operator reuse where an explicit matrix is already available from the specialized backend.
        2. Add minimal generic hooks for matrix-aware propagation without teaching `pygeoinf` about boxes, Gaussians, or paths.
        3. Update living references and solver-selection docs to describe generic fallback vs specialized backend selection.
        4. Re-run the focused tests to confirm the generic path still behaves identically while specialized solvers can avoid repeated matrix materialization.

**Open Questions 4 questions**
1. Should the volumetric interaction backend cache dense kernel matrices per provider instance, or should solver instances own all caches to avoid stale/memory-heavy shared state?
2. For mixed Sobolev stacks, should the first implementation support full path+volume interaction assembly, or should Phase 4 initially accept homogeneous stacks only and fall back otherwise?
3. Should provider batching be formalized as a new abstract method on `FunctionProvider3D`, or kept as an optional duck-typed capability to avoid breaking existing providers?
4. Should the dense volumetric solver return explicit `LinearOperator` objects for downstream pygeoinf workflows, or remain a solver/assembler object with dedicated methods for Gram, solve, covariance, and resolving-kernel assembly?
