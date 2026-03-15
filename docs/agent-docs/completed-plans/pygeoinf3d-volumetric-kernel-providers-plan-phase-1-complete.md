## Phase 1 Complete: Add Function-Provider Package Skeleton

Created the `pygeoinf3d/providers/function_providers/` package scaffold and locked its initial public surface with import/export and protocol-contract tests. This phase establishes the package boundary for the upcoming volumetric kernel families without introducing any concrete providers yet.

**Files created/changed:**
- pygeoinf3d/providers/function_providers/__init__.py
- pygeoinf3d/providers/__init__.py
- tests/providers/test_function_providers.py
- docs/agent-docs/references/living/pygeoinf3d-reference.md

**Functions created/changed:**
- `TestFunctionProvidersPackageImport.test_package_importable`
- `TestFunctionProvidersPackageImport.test_package_has_all`
- `TestFunctionProvidersPackageImport.test_function_provider_3d_re_exported`
- `TestFunctionProvidersPackageImport.test_accessible_via_parent_providers`
- `TestFunctionProvidersPackageImport.test_all_symbols_importable`
- `_constant_function`
- `_StubProvider.__init__`
- `_StubProvider.dim`
- `_StubProvider.get_function`
- `TestFunctionProvider3DContract.test_dim_is_positive_int`
- `TestFunctionProvider3DContract.test_get_function_returns_function3d`
- `TestFunctionProvider3DContract.test_get_function_all_indices`
- `TestFunctionProvider3DContract.test_get_function_out_of_range_raises`
- `TestFunctionProvider3DContract.test_get_function_negative_raises`
- `TestFunctionProvider3DContract.test_returned_function_is_callable`
- `TestFunctionProvider3DContract.test_returned_function_has_domain`

**Tests created/changed:**
- `test_package_importable`
- `test_package_has_all`
- `test_function_provider_3d_re_exported`
- `test_accessible_via_parent_providers`
- `test_all_symbols_importable`
- `test_dim_is_positive_int`
- `test_get_function_returns_function3d`
- `test_get_function_all_indices`
- `test_get_function_out_of_range_raises`
- `test_get_function_negative_raises`
- `test_returned_function_is_callable`
- `test_returned_function_has_domain`

**Review Status:** APPROVED

**Git Commit Message:**
feat(providers): add function_providers package scaffold

- Create pygeoinf3d.providers.function_providers package and re-export FunctionProvider3D
- Expose the new subpackage from pygeoinf3d.providers with a locked __all__ surface
- Add contract and import/export tests for future volumetric function providers

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-volumetric-kernel-providers-plan.md
Phase: 1 of 4
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-volumetric-kernel-providers-plan-phase-1-complete.md