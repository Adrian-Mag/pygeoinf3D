## Plan Complete: Noisy Path-Form Inversion Notebook

Created a full end-to-end demo notebook (`noisy_path_form_least_norm_inversion_demo.ipynb`) demonstrating minimum-norm inversion on a 3D unit-cube domain with 10 random straight-ray path-integral measurements, a Sobolev model space, and noisy synthetic data. The notebook covers geometry setup, forward operator construction, noisy data generation, inversion, diagnostics, and model comparison visualizations across 4 phases.

**Phases Completed:** 4 of 4
1. ✅ Phase 1: Notebook scaffold and path geometry
2. ✅ Phase 2: Forward operator and noisy data model
3. ✅ Phase 3: Noisy minimum-norm inversion and diagnostics
4. ✅ Phase 4: Visualize results and update package reference

**All Files Created/Modified:**
- `pygeoinf3D/demos/noisy_path_form_least_norm_inversion_demo.ipynb` (new)
- `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md` (updated)

**Key Functions/Classes Added:**
- `true_model_callable` / `u_true` — 3-Gaussian `Function3D` ground-truth model
- `forms` — 10 `PathIntegralFunctional(sobolev, path, n_quad=300)` objects
- `G` — `SOLAOperator3D(sobolev, EuclideanSpace(10), forms)` forward operator
- `noise_measure` — `GaussianMeasure` with σ²I covariance
- `forward_problem` — `LinearForwardProblem(G, data_error_measure=noise_measure)`
- `u_hat` — recovered `Function3D` via data-space minimum-norm branch

**Test Coverage:**
- Total cells: 21 cells (11 code, 10 markdown)
- All code cells execute without errors: ✅
- Key assertions: source/receiver geometry checks, type checks, diagnostic prints

**Recommendations for Next Steps:**
- Implement `Sobolev3D.from_components` to enable the discrepancy-principle branch of `LinearMinimumNormInversion`, allowing proper noise-level regularization (chi-squared ≈ N_paths)
- Extend to non-straight (curved) rays using `Path3D` with intermediate waypoints
- Add a sensitivity/resolution kernel visualization using `BackusGilbert` methods
