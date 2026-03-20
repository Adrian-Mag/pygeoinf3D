## Phase 3 Complete: Noisy Minimum-Norm Inversion

Implemented the noisy minimum-norm inversion using the data-space branch of `LinearMinimumNormInversion` (`u = G^T (GG^T)^{-1} d`), which is tractable without a matrix representation of the infinite-dimensional Sobolev model space. Added informative diagnostics contrasting the fit chi-squared (≈ 0: exact interpolation) with the noise chi-squared (≈ N_PATHS: noise reference). Fixed a misleading diagnostic label and `galerkin=False` → `galerkin=True`.

**Files created/changed:**
- `pygeoinf3D/demos/noisy_path_form_least_norm_inversion_demo.ipynb`

**Functions created/changed:**
- `plain_fp` — `LinearForwardProblem(G)` (data-space path, no data_error_measure)
- `inversion` — `LinearMinimumNormInversion(plain_fp)`
- `inverse_op` — minimum-norm operator via `EigenSolver(galerkin=True, rtol=1e-12)` on the 10×10 Gram matrix
- `u_hat` — `Function3D` recovered model
- `d_hat`, `residual`, `chi_sq_fit`, `chi_sq_noise` — diagnostic quantities

**Tests created/changed:**
- Cell `#VSC-2310427e`: prints type of `u_hat` (confirmed `Function3D`), evaluates both `u_hat` and `u_true` at (0.25,0.25,0.25)
- Cell `#VSC-e3d7aacf`: prints fit residual (≈ machine precision), chi-squared (fit ≈ 0, noise ≈ 6.1, expected ≈ 10), relative misfit

**Review Status:** APPROVED (after targeted fixes to diagnostics and galerkin flag)

**Git Commit Message:**
```
feat(demo): phase 3 — noisy minimum-norm inversion and diagnostics

- Implement LinearMinimumNormInversion via data-space branch: u = G^T(GG^T)^{-1}d
- EigenSolver(galerkin=True, rtol=1e-12) pseudo-inverts 10×10 Gram matrix
- u_hat confirmed as Function3D; u_hat(0.25,0.25,0.25) ≈ 0.114 vs u_true ≈ 1.005
- Add two-part chi-squared diagnostics: fit (≈0) vs noise (≈N_PATHS)
- Note: data_error_measure/discrepancy-principle branch requires Sobolev3D.from_components (not yet implemented)

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-noisy-path-form-inversion-notebook-plan.md
Phase: 3 of 4
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-noisy-path-form-inversion-notebook-phase-3-complete.md
```
