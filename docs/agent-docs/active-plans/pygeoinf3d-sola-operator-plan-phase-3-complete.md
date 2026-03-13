## Phase 3 Complete: Add Measurement Providers And Lazy Access

Extended `SOLAOperator3D` to accept either an eager sequence of `VolumeKernel3D` kernels or a provider-backed lazy kernel source with internal caching. This phase also completed the public API rename from `measurements` to `kernels`, added a minimal provider abstraction plus adapter, updated the public exports and living reference, and kept the operator basis-free on the domain side.

**Files created/changed:**
- pygeoinf3d/providers/base.py
- pygeoinf3d/providers/__init__.py
- pygeoinf3d/operators/sola.py
- pygeoinf3d/__init__.py
- tests/operators/test_sola.py
- docs/agent-docs/references/living/pygeoinf3d-reference.md
- docs/agent-docs/active-plans/pygeoinf3d-sola-operator-plan.md

**Functions created/changed:**
- `KernelProvider3D.dim`
- `KernelProvider3D.get_kernel`
- `FunctionToKernelAdapter.__init__`
- `FunctionToKernelAdapter.dim`
- `FunctionToKernelAdapter.get_kernel`
- `SOLAOperator3D.__init__`
- `SOLAOperator3D._get_kernel`
- `SOLAOperator3D._forward`
- `SOLAOperator3D._dual_mapping`
- `SOLAOperator3D.kernels`

**Tests created/changed:**
- `test_sola_accepts_volume_kernel_provider`
- `test_sola_lazy_provider_access_matches_list_kernels`
- `test_sola_provider_cache_returns_stable_kernels`
- `test_sola_rejects_zero_kernels`
- `test_sola_kernels_property_returns_copy`
- `test_sola_rejects_non_volume_kernel_kernel`
- `test_sola_rejects_wrong_domain_kernel`

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
feat(operators): add provider-backed SOLA kernels and lazy access

- Add KernelProvider3D and FunctionToKernelAdapter for indexed lazy kernels
- Extend SOLAOperator3D to support provider-backed cached kernel access
- Rename the SOLA public API from measurements to kernels and update tests/docs

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-sola-operator-plan.md
Phase: 3 of 4
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-sola-operator-plan-phase-3-complete.md