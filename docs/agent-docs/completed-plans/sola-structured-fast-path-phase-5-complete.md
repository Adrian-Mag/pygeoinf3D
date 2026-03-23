## Phase 5 Complete: Generic pygeoinf Hooks and Documentation

Added `DenseVolumeSolaSolver.forward_operator(function_space)` — a method that exposes the cached-K forward map as a `pygeoinf.LinearOperator` with an exact L² adjoint, enabling `GaussianMeasure.affine_mapping()` integration without modifying pygeoinf. The adjoint representer is vectorized via `provider.evaluate_all`, preserving the no-Python-loop invariant. Living reference updated with the complete fast-path architecture section.

**Files created/changed:**
- `pygeoinf3d/solvers/dense_volume_sola.py` — added `forward_operator(function_space)` method
- `tests/solvers/test_dense_volume_sola_integration.py` — 10 new integration tests
- `docs/agent-docs/references/living/pygeoinf3d-reference.md` — updated with fast-path architecture section

**Functions created/changed:**
- `DenseVolumeSolaSolver.forward_operator(function_space)` — returns `pygeoinf.LinearOperator` with:
  - `mapping(f)` → delegates to cached `solver.forward(f)`
  - `adjoint_mapping(d)` → vectorized representer `x ↦ Σ d_i k_i(x)` via `provider.evaluate_all`
  - `dual_mapping(yp)` → assembles `VolumeKernel3D` sum for Lebesgue duality chain
  - Satisfies `A @ A* = G` (Gram matrix) to machine precision

**Tests created/changed:**
- `test_returns_linear_operator`
- `test_domain_is_function_space`
- `test_codomain_has_correct_dim`
- `test_application_matches_solver_forward`
- `test_affine_mapping_returns_gaussian_measure`
- `test_affine_mapping_domain_is_euclidean`
- `test_affine_mapping_expectation_is_zero`
- `test_affine_mapping_covariance_equals_gram_squared`
- `test_A_Astar_equals_gram`
- `test_forward_operator_different_calls_are_independent`

**Review Status:** APPROVED

**Git Commit Message:**
```
feat(solvers): add DenseVolumeSolaSolver.forward_operator() for pygeoinf integration

- forward_operator(space) returns pygeoinf.LinearOperator backed by cached K matrix
- adjoint: vectorized representer A*(d)(x) = sum_i d_i k_i(x) via evaluate_all
- A @ A* = gram() to machine precision (tested)
- GaussianMeasure.affine_mapping(A) produces valid data-space Gaussian measure
- pygeoinf NOT modified; hooks via existing LinearOperator callable constructor
- 10 new integration tests; 905/905 pass
- pygeoinf3d-reference.md updated with full fast-path architecture section

Plan: pygeoinf3D/docs/agent-docs/active-plans/sola-structured-fast-path-plan.md
Phase: 5 of 5
Related: pygeoinf3D/docs/agent-docs/active-plans/sola-structured-fast-path-phase-5-complete.md
```
