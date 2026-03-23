## Phase 2 Complete: Volumetric Interaction Backend

Added a reusable `VolumeInteractionBackend` that assembles and caches the `(n_kernels × n_pts)` kernel-evaluation matrix on a structured quadrature mesh, making repeated Gram/cross-Gram assembly a single matrix multiply. A default loop-based `evaluate_all` was added to `FunctionProvider3D`, with a vectorized override in `GaussianProvider3D` (no Python loop over kernels). The reviewer's recommendation to replace the O(N) `np.array_equal` mesh check with an O(1) attribute check in `cross_gram` was also adopted.

**Files created/changed:**
- `pygeoinf3d/providers/base.py` — added `evaluate_all(points)` with loop fallback
- `pygeoinf3d/providers/function_providers/gaussian.py` — added vectorized `evaluate_all` override
- `pygeoinf3d/solvers/volume_interactions.py` — new module; `VolumeInteractionBackend` class
- `pygeoinf3d/solvers/__init__.py` — added `VolumeInteractionBackend` export + `__all__`
- `tests/solvers/test_volume_interaction_backend.py` — 6 new tests
- `docs/agent-docs/references/living/pygeoinf3d-reference.md` — updated for Phase 2

**Functions created/changed:**
- `FunctionProvider3D.evaluate_all(points)` — concrete method, default loop, returns `(n, n_pts)`
- `GaussianProvider3D.evaluate_all(points)` — vectorized via broadcasting over `(n_kernels, n_pts, 3)`; no Python loop
- `VolumeInteractionBackend.__init__` — owns `_pts`, `_weights` from domain cache; `_K_cache = None`
- `VolumeInteractionBackend.interaction_matrix()` — lazy-compute K via `provider.evaluate_all`; cached
- `VolumeInteractionBackend.gram()` — `K @ (w[:, None] * K.T)` (BLAS-friendly)
- `VolumeInteractionBackend.cross_gram(other)` — O(1) domain/resolution/location attribute check, then cross product

**Tests created/changed:**
- `test_gaussian_provider_evaluate_all_matches_get_function_loop`
- `test_interaction_matrix_matches_kernel_by_kernel_quadrature`
- `test_gram_matches_direct_integration`
- `test_volume_interaction_backend_falls_back_for_provider_without_evaluate_all`
- `test_cross_gram_matches_direct`
- `test_cross_gram_raises_for_incompatible_backends`

**Review Status:** APPROVED (after adding cross_gram tests and fixing `__init__.py` lint)

**Git Commit Message:**
```
feat(solvers): add VolumeInteractionBackend with batched kernel evaluation

- Add evaluate_all() concrete method to FunctionProvider3D (loop fallback)
- Add GaussianProvider3D.evaluate_all(): vectorized, no Python loop over kernels
- Add VolumeInteractionBackend: instance-owned pts/weights/_K_cache
- gram() = K @ diag(w) @ K.T (single BLAS call)
- cross_gram(other) with O(1) domain/resolution/location validation
- __all__ added to solvers/__init__.py; cross_gram tested for correctness + error

Plan: pygeoinf3D/docs/agent-docs/active-plans/sola-structured-fast-path-plan.md
Phase: 2 of 5
Related: pygeoinf3D/docs/agent-docs/active-plans/sola-structured-fast-path-phase-2-complete.md
```
