## Phase 2 Complete: Add First Volumetric Function Providers

Added the first concrete volumetric `Function3D` provider families for direct SOLA kernel construction: Gaussian, bump, and boxcar. This phase established stable parameter semantics, batch evaluation, normalization behavior, and adapter compatibility, with the provider test files passing cleanly after review.

**Files created/changed:**
- pygeoinf3d/providers/function_providers/__init__.py
- pygeoinf3d/providers/function_providers/gaussian.py
- pygeoinf3d/providers/function_providers/bump.py
- pygeoinf3d/providers/function_providers/boxcar.py
- tests/providers/test_function_providers.py
- tests/providers/test_concrete_function_providers.py
- docs/agent-docs/references/living/pygeoinf3d-reference.md

**Functions created/changed:**
- `GaussianProvider3D.__init__`
- `GaussianProvider3D.dim`
- `GaussianProvider3D.get_function`
- `BumpProvider3D.__init__`
- `BumpProvider3D.dim`
- `BumpProvider3D.get_function`
- `BoxcarProvider3D.__init__`
- `BoxcarProvider3D.dim`
- `BoxcarProvider3D.get_function`
- `_bump1d_integral`

**Tests created/changed:**
- `TestGaussianProvider3DConstruction`
- `TestGaussianProvider3DEvaluation`
- `TestGaussianProvider3DAdapterCompatibility`
- `TestBumpProvider3DConstruction`
- `TestBumpProvider3DEvaluation`
- `TestBumpProvider3DAdapterCompatibility`
- `TestBoxcarProvider3DConstruction`
- `TestBoxcarProvider3DEvaluation`
- `TestBoxcarProvider3DAdapterCompatibility`
- `TestFunctionProvidersPackageImport.test_package_has_all`

**Review Status:** APPROVED

**Git Commit Message:**
feat(providers): add volumetric Function3D provider families

- Add GaussianProvider3D, BumpProvider3D, and BoxcarProvider3D under function_providers
- Cover construction, evaluation, normalization, and FunctionToKernelAdapter compatibility with concrete provider tests
- Update provider exports and the living reference for the new volumetric kernel workflow

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-volumetric-kernel-providers-plan.md
Phase: 2 of 4
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-volumetric-kernel-providers-plan-phase-2-complete.md
