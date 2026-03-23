## Phase 3 Complete: Dense Volume SOLA Solver + Router Integration

Added `DenseVolumeSolaSolver` — a fast SOLA solver for homogeneous volumetric Gaussian stacks on `BoxDomain`. The solver precomputes the Gram matrix and Cholesky factorization at construction time, then answers all forward, Gram, and resolve queries with BLAS operations. `SolverRouter` was extended with an opt-in `provider` kwarg to recommend `DenseVolumeSolaSolver` when appropriate; all existing routing remains unchanged.

**Files created/changed:**
- `pygeoinf3d/solvers/dense_volume_sola.py` — new `DenseVolumeSolaSolver` class
- `pygeoinf3d/solvers/registry.py` — registered `DenseVolumeSolaSolver` as 1st entry; removed unused import
- `pygeoinf3d/solvers/router.py` — added `provider` kwarg + routing rule for dense volume path
- `pygeoinf3d/solvers/__init__.py` — added `DenseVolumeSolaSolver` export + `__all__`
- `tests/solvers/test_dense_volume_sola.py` — 12 new tests
- `tests/solvers/test_solver_router.py` — updated count test
- `tests/solvers/test_solver_registry.py` — updated count test; removed unused import
- `docs/agent-docs/references/living/pygeoinf3d-reference.md` — updated for Phase 3

**Functions created/changed:**
- `DenseVolumeSolaSolver.__init__` — builds `VolumeInteractionBackend`, precomputes G and Cholesky
- `DenseVolumeSolaSolver.gram()` — returns copy of precomputed G
- `DenseVolumeSolaSolver.forward(f)` — `K @ (w * f(pts))`, shape `(n_kernels,)`
- `DenseVolumeSolaSolver.resolve(target_K_row)` — `cho_solve(cho, K @ (w * target))` — O(m²) per call
- `DenseVolumeSolaSolver.n_kernels`, `.backend` properties
- `SolverRouter.recommend(...)` — added `*, provider=None` kwarg, new routing branch 1b

**Tests created/changed:**
- `test_gram_matches_generic_sola_operator`
- `test_gram_is_symmetric`
- `test_gram_is_positive_definite`
- `test_resolving_coefficients_match_direct_solve`
- `test_cross_gram_matches_generic_form_evaluation`
- `test_resolving_satisfies_normal_equations`
- `test_forward_matches_sola_operator`
- `test_forward_shape`
- `test_forward_is_linear_in_f`
- `test_router_recommends_dense_volume_sola_for_box_gaussian_stack`
- `test_router_still_recommends_sola_without_provider`
- `test_router_dense_volume_sola_recommendation_has_reason`

**Review Status:** APPROVED (after removing unused imports)

**Git Commit Message:**
```
feat(solvers): add DenseVolumeSolaSolver with Cholesky-cached SOLA fast path

- Add DenseVolumeSolaSolver: precomputes Gram + Cholesky at construction
- gram(), forward(f), resolve(target) all via BLAS (no repeated K assembly)
- Optional diagonal regularization for ill-conditioned problems
- Register in SolverRegistry; add to SolverRouter via opt-in provider kwarg
- 12 new tests: Gram symmetry/PD, resolve normal equations, forward linearity
- Remove unused imports in registry.py and test_solver_registry.py

Plan: pygeoinf3D/docs/agent-docs/active-plans/sola-structured-fast-path-plan.md
Phase: 3 of 5
Related: pygeoinf3D/docs/agent-docs/active-plans/sola-structured-fast-path-phase-3-complete.md
```
