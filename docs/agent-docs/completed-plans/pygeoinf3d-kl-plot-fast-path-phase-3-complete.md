## Phase 3 Complete: Add plotting knob and document the workflow

Phase 3 exposes `plot_n_modes` across the public plotting helpers as a visualization-only KL mode cap. The new option is threaded only into the KL tensor-product fast path, generic `Function3D` inputs ignore it without changing results, and the implementation was reviewed and approved with only minor documentation and typing recommendations.

**Files created/changed:**
- `pygeoinf3D/pygeoinf3d/sampling/kl_sampler.py`
- `pygeoinf3D/pygeoinf3d/plot.py`
- `pygeoinf3D/tests/test_plot.py`
- `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`

**Functions created/changed:**
- `_attach_kl_fast_paths`
- `_validate_plot_n_modes`
- `evaluate_function3d_on_regular_grid`
- `evaluate_function3d_on_axis_slice`
- `plot_function3d_axis_slice`
- `plot_function3d_axis_slider`
- `plot_function3d_volume_dropdown`
- `plot_function3d_isosurface`
- `compare_function3d_axis_sliders`

**Tests created/changed:**
- `test_plot_n_modes_validation`
- `test_plot_n_modes_changes_kl_sampling_budget_only_for_plotting`
- `test_plot_n_modes_full_equals_default`
- `test_plot_n_modes_threaded_through_volume_dropdown`
- `test_plot_n_modes_threaded_through_isosurface`
- `test_plot_n_modes_threaded_through_axis_slider`
- updated Phase 2 fast-path tracking tests for the new callable signatures

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
feat(plot): add plot_n_modes visualization cap for KL plots

- expose plot_n_modes across slice, volume, isosurface, and comparison plot helpers
- thread the visualization-only mode cap through KL fast-path dispatch while leaving generic functions unchanged
- extend plot tests and living reference for validation and KL-only truncation behavior

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-kl-plot-fast-path-plan.md
Phase: 3 of 3
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-kl-plot-fast-path-phase-3-complete.md