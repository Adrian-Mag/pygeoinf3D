## Phase 1 Complete: Add fast KL grid evaluator

Phase 1 added a tensor-product fast path to `KLSampler3D` for regular 3D grids and axis-aligned slices, together with reproducible `sample_from_xis` construction and validation tests against the existing `Function3D` evaluation path. The implementation was reviewed and approved, with only minor non-blocking recommendations around extra validation consistency.

**Files created/changed:**
- `pygeoinf3D/pygeoinf3d/sampling/kl_sampler.py`
- `pygeoinf3D/tests/sampling/test_kl_sampler.py`
- `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`
- `pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-kl-plot-fast-path-plan.md`

**Functions created/changed:**
- `_parse_tp_axis`
- `KLSampler3D.has_tensor_product_structure`
- `KLSampler3D.sample_from_xis`
- `KLSampler3D.evaluate_on_regular_grid`
- `KLSampler3D.evaluate_on_axis_slice`
- `KLSampler3D._resolve_plot_n_modes`
- `KLSampler3D._build_basis_matrices`

**Tests created/changed:**
- `TestKLSamplerTensorProductStructure.test_tp_structure_detected_for_inv_laplacian`
- `TestKLSamplerTensorProductStructure.test_tp_structure_detected_for_neumann`
- `TestKLSamplerTensorProductStructure.test_no_tp_structure_for_generic_operator`
- `TestKLSamplerSampleFromXis.test_returns_function3d`
- `TestKLSamplerSampleFromXis.test_zero_xis_gives_mean`
- `TestKLSamplerSampleFromXis.test_matches_sample_with_known_xis_batch`
- `TestKLSamplerSampleFromXis.test_wrong_xis_shape_raises`
- `TestKLSamplerFastGridEvaluator.test_fast_regular_grid_matches_function_evaluation`
- `TestKLSamplerFastGridEvaluator.test_fast_regular_grid_output_shapes`
- `TestKLSamplerFastGridEvaluator.test_fast_regular_grid_with_tuple_resolution`
- `TestKLSamplerFastGridEvaluator.test_fast_evaluator_respects_plot_n_modes_cap`
- `TestKLSamplerFastGridEvaluator.test_plot_n_modes_zero_raises`
- `TestKLSamplerFastGridEvaluator.test_plot_n_modes_exceeds_n_modes_raises`
- `TestKLSamplerFastGridEvaluator.test_fast_grid_requires_tp_structure`
- `TestKLSamplerFastGridEvaluator.test_fast_grid_respects_mean`
- `TestKLSamplerFastSliceEvaluator.test_fast_axis_slice_matches_function_evaluation`
- `TestKLSamplerFastSliceEvaluator.test_fast_axis_slice_output_shapes_default_coord`
- `TestKLSamplerFastSliceEvaluator.test_fast_axis_slice_integer_axis`
- `TestKLSamplerFastSliceEvaluator.test_fast_axis_slice_tuple_resolution`
- `TestKLSamplerFastSliceEvaluator.test_fast_axis_slice_respects_plot_n_modes_cap`
- `TestKLSamplerFastSliceEvaluator.test_fast_slice_requires_tp_structure`
- `TestKLSamplerFastSliceEvaluator.test_fast_slice_respects_mean`

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
feat(kl-plot): add fast KL grid evaluators

- add tensor-product fast-path evaluation for regular grids and axis slices
- add deterministic sample_from_xis helper and plot_n_modes-ready mode resolution
- extend KL sampler tests and living reference for the new fast APIs

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-kl-plot-fast-path-plan.md
Phase: 1 of 3
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-kl-plot-fast-path-phase-1-complete.md
