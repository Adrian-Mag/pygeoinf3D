## Phase 1 Complete: Quadrature Cache Primitives

Added internal per-instance quadrature caching to `BoxDomain.structured_mesh()` and `BoxDomain.volume_weights()`, keyed on `('mesh'|'weights', location, (nx, ny, nz))`. Public semantics are completely unchanged; repeated calls with identical arguments now return the same array object instead of recomputing the grid.

**Files created/changed:**
- `pygeoinf3D/pygeoinf3d/geometries/box/domain.py`
- `pygeoinf3D/tests/geometries/test_box_domain.py`

**Functions created/changed:**
- `BoxDomain.__init__` — added `self._quad_cache: dict = {}`
- `BoxDomain.structured_mesh` — check/populate `_quad_cache` before computing mesh
- `BoxDomain.volume_weights` — check/populate `_quad_cache` before computing weights

**Tests created/changed:**
- `TestBoxDomainQuadratureCache::test_structured_mesh_cache_returns_same_object`
- `TestBoxDomainQuadratureCache::test_structured_mesh_cache_same_object_nodes`
- `TestBoxDomainQuadratureCache::test_structured_mesh_cache_different_resolutions_are_independent`
- `TestBoxDomainQuadratureCache::test_structured_mesh_cache_different_locations_are_independent`
- `TestBoxDomainQuadratureCache::test_volume_weights_cache_returns_same_object`
- `TestBoxDomainQuadratureCache::test_volume_weights_cache_same_object_nodes`
- `TestBoxDomainQuadratureCache::test_volume_weights_cache_different_resolutions_are_independent`
- `TestBoxDomainQuadratureCache::test_volume_weights_cache_different_locations_are_independent`
- `TestBoxDomainQuadratureCache::test_cache_is_per_instance`
- `TestBoxDomainQuadratureCache::test_cache_is_per_instance_weights`
- `TestBoxDomainQuadratureCache::test_integrate_volume_matches_cached_mesh_dot_product`

**Review Status:** APPROVED (after adding two missing coverage tests flagged in code review)

**Git Commit Message:**
```
feat(geometries): add quadrature cache to BoxDomain mesh and weights

- Add _quad_cache dict to BoxDomain.__init__ keyed on (kind, location, (nx,ny,nz))
- structured_mesh returns cached array object on repeated calls
- volume_weights returns cached array object on repeated calls
- 11 new cache tests covering identity reuse, location/resolution independence,
  per-instance isolation, and integrate_volume consistency
- 870 existing tests still passing

Plan: pygeoinf3D/docs/agent-docs/active-plans/sola-structured-fast-path-plan.md
Phase: 1 of 5
Related: pygeoinf3D/docs/agent-docs/active-plans/sola-structured-fast-path-phase-1-complete.md
```
