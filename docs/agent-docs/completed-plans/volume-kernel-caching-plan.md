## Plan: Volume Kernel Caching

Add explicit caching to `VolumeKernel3D` so a single volume form can be compiled against a structured quadrature mesh and then applied repeatedly to many functions with one batched function evaluation and a dot product. This keeps the form mathematically basis-free while reusing the same mesh/weight infrastructure already present in the box-domain and SOLA fast paths.

**Phases 3**
1. **Phase 1: Add Explicit Cache API**
    - **Objective:** Add explicit cache state and cache-management methods to `VolumeKernel3D`.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/pygeoinf3d/spaces/forms.py`
    - **Tests to Write:** cache build/clear behavior; stale-cache invalidation after in-place arithmetic
    - **Steps:**
        1. Add cache fields for quadrature points and preweighted kernel values.
        2. Add explicit methods such as `build_cache(resolution=20, location="cell_centers")`, `clear_cache()`, and a read-only cache-status property.
        3. Ensure in-place mutation methods invalidate any existing cache.

2. **Phase 2: Use Cache In Evaluation**
    - **Objective:** Make repeated form application fast when a cache has been built, while preserving the current integration-based fallback path.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/pygeoinf3d/spaces/forms.py`
    - **Tests to Write:** cached apply matches uncached apply; repeated apply avoids kernel re-evaluation on the cached mesh
    - **Steps:**
        1. Update `_mapping_impl` to use cached quadrature points and preweighted kernel values when present.
        2. Keep `domain.integrate_volume(self._kernel.pointwise_mul(g))` as the always-correct fallback.
        3. Restrict caching to structured domains that provide compatible `structured_mesh` and `volume_weights` methods.

3. **Phase 3: Add Regression Coverage And Reference Update**
    - **Objective:** Validate the new cache path and document it in the living reference.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/tests/spaces/test_volume_kernel_caching.py`; `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`
    - **Tests to Write:** cache invalidation after `*=`, `+=`, `-=`; compatibility with existing `SOLAOperator3D` workflows
    - **Steps:**
        1. Add focused unit tests for standalone cached `VolumeKernel3D` use.
        2. Confirm existing operator-level cache behavior remains unchanged.
        3. Update the living reference to describe the new explicit cache API.

**Open Questions 3**
1. Should cache construction remain explicit only, or auto-build on first use for structured domains? Recommended: explicit only.
2. Should the first implementation accept any `StructuredDomain3D` with compatible mesh/weight methods, or only `BoxDomain`? Recommended: compatible structured domains, validated conservatively.
3. Should the cache store `k(x_q)` and `w_q` separately or preweighted `w_q k(x_q)` only? Recommended: preweighted only for the minimal fast path.