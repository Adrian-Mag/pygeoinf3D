## Phase 2 Complete: Wire plotting helpers to the fast path

Phase 2 wires KL-generated tensor-product samples into the shared plotting backends so the existing volume, isosurface, and axis-slice helpers automatically use the vectorized evaluator. Generic `Function3D` inputs still follow the original sampling path unchanged, and the implementation was reviewed and approved with only minor non-blocking style/type-check recommendations.

**Files created/changed:**
- `pygeoinf3D/pygeoinf3d/sampling/kl_sampler.py`
- `pygeoinf3D/pygeoinf3d/plot.py`
- `pygeoinf3D/tests/test_plot.py`
- `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`

**Functions created/changed:**
- `_attach_kl_fast_paths`
- `KLSampler3D.sample`
- `KLSampler3D.sample_from_xis`
- `evaluate_function3d_on_regular_grid`
- `evaluate_function3d_on_axis_slice`

**Tests created/changed:**
- `test_kl_sample_has_fast_path_attributes`
- `test_generic_function3d_has_no_fast_path`
- `test_kl_sample_from_xis_has_fast_path_attributes`
- `test_kl_fast_grid_numerically_matches_generic_path`
- `test_kl_fast_slice_numerically_matches_generic_path`
- `test_volume_dropdown_uses_kl_fast_path`
- `test_isosurface_uses_kl_fast_path`
- `test_axis_slider_uses_kl_fast_path`

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
feat(plot): auto-dispatch KL fast path in plot helpers

- tag tensor-product KL samples with private fast grid and slice evaluators
- route shared grid and slice sampling helpers through the KL fast path when available
- add plot integration tests and update the living reference for automatic dispatch

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-kl-plot-fast-path-plan.md
Phase: 2 of 3
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-kl-plot-fast-path-phase-2-complete.md