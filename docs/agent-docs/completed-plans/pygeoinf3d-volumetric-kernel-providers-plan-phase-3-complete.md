## Phase 3 Complete: Add Visualization Utilities For Volumetric Kernels

Added a minimal slice-based visualization layer for `Function3D` objects in `pygeoinf3d.plot`, separating deterministic slice sampling from matplotlib rendering. This phase gives users a direct way to inspect provider-generated volumetric kernels on axis-aligned slices while keeping the API small, testable, and independent of the mathematical core.

**Files created/changed:**
- pygeoinf3d/plot.py
- tests/test_plot.py
- docs/agent-docs/references/living/pygeoinf3d-reference.md

**Functions created/changed:**
- `_parse_axis`
- `_get_other_axes`
- `_validate_resolution`
- `evaluate_function3d_on_axis_slice`
- `plot_function3d_axis_slice`

**Tests created/changed:**
- `test_evaluate_slice_z_center`
- `test_evaluate_slice_x_axis`
- `test_plot_slice_smoke`
- `test_invalid_input`
- `test_resolution_validation`
- `test_default_coord`

**Review Status:** APPROVED

**Git Commit Message:**
feat(plot): add slice visualization for Function3D

- Add pygeoinf3d.plot with axis-aligned slice sampling and matplotlib rendering helpers
- Cover slice evaluation, coordinate defaults, validation, and plotting contract with focused tests
- Update the living reference for the new plotting module and plot test coverage

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-volumetric-kernel-providers-plan.md
Phase: 3 of 4
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-volumetric-kernel-providers-plan-phase-3-complete.md
