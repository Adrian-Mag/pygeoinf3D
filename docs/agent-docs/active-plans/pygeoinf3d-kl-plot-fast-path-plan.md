## Plan: KL Plot Fast Path

Add a KL-specific fast evaluation path for structured plotting grids in pygeoinf3D, then wire the existing plot helpers to use it automatically and expose a `plot_n_modes` knob for visualization-only truncation. This speeds up plotting high-mode KL samples by replacing the current Python-loop evaluation path with vectorized tensor-product evaluation while preserving the existing generic `Function3D` API.

**Phases 3**
1. **Phase 1: Add fast KL grid evaluator**
    - **Objective:** Implement and validate a vectorized evaluator for KL samples on regular 3D grids and axis-aligned slice grids.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/pygeoinf3d/sampling/kl_sampler.py`; potentially `pygeoinf3D/pygeoinf3d/providers/tensor_product.py`; `pygeoinf3D/tests/sampling/test_kl_sampler.py`
    - **Tests to Write:** `test_fast_regular_grid_matches_function_evaluation`; `test_fast_axis_slice_matches_function_evaluation`; `test_fast_evaluator_respects_plot_n_modes_cap`
    - **Steps:**
        1. Write failing tests that compare fast structured-grid evaluation against the current `Function3D` evaluation on the same points.
        2. Implement a KL/tensor-product-specific fast path that evaluates retained modes in vectorized form on regular grids and 2D slices.
        3. Run the targeted KL sampler tests again and confirm the new path is numerically consistent.

2. **Phase 2: Wire plotting helpers to the fast path**
    - **Objective:** Update the plot sampling helpers so KL-generated samples use the fast evaluator automatically while generic `Function3D` instances keep the existing behavior.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/pygeoinf3d/plot.py`; `pygeoinf3D/tests/test_plot.py`
    - **Tests to Write:** `test_volume_dropdown_uses_kl_fast_path`; `test_isosurface_uses_kl_fast_path`; `test_axis_slider_uses_kl_fast_path`
    - **Steps:**
        1. Write failing plot tests covering volume, isosurface, and axis-slice sampling for KL samples.
        2. Add fast-path detection and dispatch inside the regular-grid and slice evaluators used by the plotting API.
        3. Re-run the targeted plot tests and verify output equivalence with the existing path.

3. **Phase 3: Add plotting knob and document the workflow**
    - **Objective:** Expose `plot_n_modes` as a visualization-only control and document how to use it for faster plotting of high-mode KL samples.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/pygeoinf3d/plot.py`; `pygeoinf3D/tests/test_plot.py`; `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`; optionally `pygeoinf3D/demos/kl_sampler_prior_demo.ipynb`
    - **Tests to Write:** `test_plot_n_modes_validation`; `test_plot_n_modes_changes_kl_sampling_budget_only_for_plotting`
    - **Steps:**
        1. Write failing tests for the new `plot_n_modes` option and its validation rules.
        2. Implement `plot_n_modes` in the relevant plotting helpers so it only affects KL-specific fast-path evaluation.
        3. Update the living reference and, if it materially helps the demo, the notebook usage example.

**Open Questions**
1. `plot_n_modes` will apply only to the KL-specific fast path for samples produced by `KLSampler3D`; generic `Function3D` inputs will ignore it or raise a validation error.
2. The first implementation will prioritize vectorized NumPy evaluation over explicit multiprocessing; parallelism can be revisited later if profiling still shows a bottleneck.
