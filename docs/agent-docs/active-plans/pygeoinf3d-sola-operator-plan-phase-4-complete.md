## Phase 4 Complete: Document Extension Boundaries

Locked the SOLA public API with explicit export tests and finalized the living reference so the extension boundary is clear: forward and dual remain representation-independent, while adjoint support still depends on `domain.from_dual` being able to realize the returned dual element. The final phase also synchronized top-level and subpackage re-exports, removed a tracked bytecode artifact from version control, and kept the full test suite green.

**Files created/changed:**
- pygeoinf3d/__init__.py
- pygeoinf3d/operators/__init__.py
- tests/operators/test_sola.py
- docs/agent-docs/references/living/pygeoinf3d-reference.md

**Functions created/changed:**
- `TestPhase4PublicAPIExports.test_top_level_exports_sola_operator`
- `TestPhase4PublicAPIExports.test_top_level_exports_function_provider`
- `TestPhase4PublicAPIExports.test_top_level_exports_kernel_provider`
- `TestPhase4PublicAPIExports.test_top_level_exports_function_to_kernel_adapter`
- `TestPhase4PublicAPIExports.test_operators_subpackage_exports_sola_operator`
- `TestPhase4PublicAPIExports.test_operators_dunder_all_contains_sola_operator`
- `TestPhase4PublicAPIExports.test_sola_operator_in_dunder_all`
- `TestPhase4PublicAPIExports.test_function_provider_in_dunder_all`
- `TestPhase4PublicAPIExports.test_kernel_provider_in_dunder_all`
- `TestPhase4PublicAPIExports.test_function_to_kernel_adapter_in_dunder_all`

**Tests created/changed:**
- `test_top_level_exports_sola_operator`
- `test_top_level_exports_function_provider`
- `test_top_level_exports_kernel_provider`
- `test_top_level_exports_function_to_kernel_adapter`
- `test_operators_subpackage_exports_sola_operator`
- `test_operators_dunder_all_contains_sola_operator`
- `test_sola_operator_in_dunder_all`
- `test_function_provider_in_dunder_all`
- `test_kernel_provider_in_dunder_all`
- `test_function_to_kernel_adapter_in_dunder_all`

**Review Status:** APPROVED

**Git Commit Message:**
docs(operators): lock SOLA public API and document extension boundary

- Add export-lock tests for SOLAOperator3D and provider abstractions
- Document the forward/dual vs adjoint extension boundary in the living reference
- Remove tracked bytecode artifact and synchronize re-exports with tests

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-sola-operator-plan.md
Phase: 4 of 4
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-sola-operator-plan-phase-4-complete.md