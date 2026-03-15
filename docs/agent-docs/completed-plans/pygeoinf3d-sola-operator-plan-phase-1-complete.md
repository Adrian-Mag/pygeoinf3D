## Phase 1 Complete: Add Volume SOLA Core

Implemented the first pygeoinf3D SOLA operator as a true `pygeoinf.LinearOperator` subclass for stacked `VolumeKernel3D` measurements. This phase establishes the representation-independent forward map, constructor validation, public exports, and focused operator tests while intentionally deferring explicit dual and adjoint mappings to Phase 2.

**Files created/changed:**
- `pygeoinf3d/operators/sola.py`
- `pygeoinf3d/operators/__init__.py`
- `pygeoinf3d/__init__.py`
- `tests/operators/test_sola.py`
- `docs/agent-docs/references/living/pygeoinf3d-reference.md`
- `docs/agent-docs/active-plans/pygeoinf3d-sola-operator-plan.md`

**Functions created/changed:**
- `SOLAOperator3D.__init__`
- `SOLAOperator3D._forward`
- `SOLAOperator3D.measurements`

**Tests created/changed:**
- `test_sola_forward_volume_kernels_matches_analytic_integrals`
- `test_sola_forward_respects_custom_codomain_realization`
- `test_sola_rejects_mismatched_kernel_counts`
- `test_sola_forward_constant_kernel_constant_function`
- `test_sola_forward_single_linear_measurement`
- `test_sola_is_linear_operator`
- `test_sola_rejects_non_volume_kernel_measurement`
- `test_sola_rejects_wrong_domain_measurement`
- `test_sola_rejects_non_lebesgue_domain`
- `test_sola_measurements_property_returns_copy`

**Review Status:** APPROVED

**Git Commit Message:**
feat(operators): add phase-1 SOLAOperator3D volume measurement core

- Add SOLAOperator3D as a LinearOperator subclass with basis-free forward evaluation
- Export the new operator and document its phase-1 forward-only limitations
- Add focused operator tests for analytic measurements and codomain realization

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-sola-operator-plan.md
Phase: 1 of 4
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-sola-operator-plan-phase-1-complete.md