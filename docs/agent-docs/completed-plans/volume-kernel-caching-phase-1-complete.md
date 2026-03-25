## Phase 1 Complete: Add Explicit Cache API

Added explicit cache state and lifecycle methods to `VolumeKernel3D` so standalone volume forms can precompute a structured quadrature backend and later discard it explicitly. Cache invalidation is wired into the in-place arithmetic methods that mutate the stored kernel.

**Files created/changed:**
- pygeoinf3D/pygeoinf3d/spaces/forms.py

**Functions created/changed:**
- `VolumeKernel3D.__init__`
- `VolumeKernel3D.build_cache`
- `VolumeKernel3D.clear_cache`
- `VolumeKernel3D.cache_built`
- `VolumeKernel3D.__imul__`
- `VolumeKernel3D.__iadd__`
- `VolumeKernel3D.__isub__`

**Tests created/changed:**
- `TestClearCache.test_cache_built_property`
- `TestRebuildAfterClear`
- `TestInPlaceArithmeticInvalidatesCache.test_imul_invalidates`
- `TestInPlaceArithmeticInvalidatesCache.test_iadd_invalidates`
- `TestInPlaceArithmeticInvalidatesCache.test_isub_invalidates`

**Review Status:** APPROVED

**Git Commit Message:**
feat(spaces): add explicit VolumeKernel3D cache API

- Add build_cache, clear_cache, and cache_built on VolumeKernel3D
- Invalidate cached quadrature state on in-place kernel mutation
- Keep standalone volume forms basis-free while enabling reuse

Plan: pygeoinf3D/docs/agent-docs/completed-plans/volume-kernel-caching-plan.md
Phase: 1 of 3
Related: pygeoinf3D/docs/agent-docs/completed-plans/volume-kernel-caching-phase-1-complete.md