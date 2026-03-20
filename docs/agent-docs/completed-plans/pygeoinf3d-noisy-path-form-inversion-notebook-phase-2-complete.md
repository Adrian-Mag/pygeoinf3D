## Phase 2 Complete: Build Forward Operator and Noisy Data Model

Added 7 new cells (3 code, 4 markdown) covering the true model, path-integral forward operator, and noisy synthetic data generation. All cells execute without errors; `G`, `d_clean`, `d_noisy`, `noise_measure`, and `forward_problem` are live in kernel scope and ready for Phase 3.

**Files created/changed:**
- `pygeoinf3D/demos/noisy_path_form_least_norm_inversion_demo.ipynb`

**Functions created/changed:**
- `true_model_callable` — sum of 3 Gaussians, handles `(3,)` and `(n,3)` inputs via `np.atleast_2d`
- `u_true` — `Function3D(box, evaluate_callable=true_model_callable, name="true model")`
- `forms` — list of 10 `PathIntegralFunctional(sobolev, p, n_quad=300)`
- `G` — `SOLAOperator3D(sobolev, EuclideanSpace(N_PATHS), forms)`
- `d_clean` — clean forward data, shape `(10,)`
- `sigma` — noise level = 2% of data range ≈ 0.004660
- `d_noisy` — noisy data, SNR ≈ 28.1 dB
- `noise_measure` — `GaussianMeasure.from_covariance_matrix(G.codomain, sigma² I)`
- `forward_problem` — `LinearForwardProblem(G, data_error_measure=noise_measure)`

**Tests created/changed:**
- Cell `#VSC-2ddf43ff`: prints `u_true(0.25,0.25,0.25) ≈ 1.005` as a quick sanity check
- Cell `#VSC-4100e6a7`: prints `G.domain` and `G.codomain` to confirm operator shape
- Cell `#VSC-806967c7`: prints clean data, sigma, SNR
- Cell `#VSC-27d444cf`: bar chart comparing clean vs noisy measurements

**Review Status:** APPROVED

**Git Commit Message:**
```
feat(demo): phase 2 — forward operator and noisy data

- Add Function3D true model (sum of 3 Gaussians, handles (3,) and (n,3) inputs)
- Build PathIntegralFunctional (n_quad=300, sobolev domain) for 10 paths
- Stack with SOLAOperator3D(sobolev, EuclideanSpace(10), forms)
- Generate clean data, 2% noise (sigma≈0.00466, SNR≈28 dB), noisy data
- Construct GaussianMeasure noise model and LinearForwardProblem
- Add clean-vs-noisy bar chart

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-noisy-path-form-inversion-notebook-plan.md
Phase: 2 of 4
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-noisy-path-form-inversion-notebook-phase-2-complete.md
```
