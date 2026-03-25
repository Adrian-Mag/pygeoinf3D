## Phase 2 Complete: Use Cache In Evaluation

Updated `VolumeKernel3D` evaluation to use the cached structured quadrature path only when the cache is complete, with the existing integration-backed implementation retained as the always-correct fallback. The cache stores preweighted kernel values so repeated application only evaluates the target function and takes a dot product.

**Files created/changed:**
- pygeoinf3D/pygeoinf3d/spaces/forms.py

**Functions created/changed:**
- `VolumeKernel3D._mapping_impl`
- `VolumeKernel3D.cache_built`
- `VolumeKernel3D.build_cache`

**Tests created/changed:**
- `TestCachedMatchesUncached.test_default_resolution`
- `TestCachedMatchesUncached.test_nodes_location`
- `TestCachedMatchesUncached.test_second_g_after_cache_built`
- `TestKernelNotReevaluated.test_kernel_evaluated_once_at_build_time`
- `TestBuildCacheTypeError.test_raises_when_structured_mesh_missing`

**Review Status:** APPROVED

**Git Commit Message:**
feat(spaces): use cached quadrature in VolumeKernel3D evaluation

- Add a guarded fast path in _mapping_impl for prebuilt quadrature caches
- Restrict cache building to structured domains with mesh and weight support
- Preserve integrate_volume fallback for uncached and unsupported cases

Plan: pygeoinf3D/docs/agent-docs/completed-plans/volume-kernel-caching-plan.md
Phase: 2 of 3
Related: pygeoinf3D/docs/agent-docs/completed-plans/volume-kernel-caching-phase-2-complete.md