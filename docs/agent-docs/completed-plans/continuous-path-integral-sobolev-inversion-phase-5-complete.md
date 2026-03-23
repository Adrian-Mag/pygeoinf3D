## Phase 5 Complete: Spectral Sobolev Representers + End-to-End Demo

Extended `Sobolev3D.from_dual` with a 3-branch spectral representer implementation and unlocked adjoint computation for `SOLAOperator3D` with path-integral forms when the domain is a `Sobolev3D`. An end-to-end minimum-norm inversion demo with path-integral measurements is included.

**Files created/changed:**
- `pygeoinf3d/spaces/sobolev.py`
- `pygeoinf3d/operators/sola.py`
- `tests/spaces/test_sobolev.py`
- `tests/operators/test_sola.py`
- `demos/path_integral_minimum_norm_demo.py` (new)
- `docs/agent-docs/references/living/pygeoinf3d-reference.md`

**Functions created/changed:**
- `Sobolev3D.from_dual` — 3 new branches: `SumLinearForm` (recursive), general `LinearForm` (spectral representer `r_ℓ = Σ_k ℓ(Φ_k) ν_k Φ_k`)
- `SOLAOperator3D.__init__` — accepts `Sobolev3D` domain via `base_space` duck-typing

**Tests created/changed:**
- `TestSobolev3DFromDualSpectral.test_from_dual_path_integral_returns_function3d`
- `TestSobolev3DFromDualSpectral.test_from_dual_path_integral_representer_identity`
- `TestSobolev3DFromDualSpectral.test_from_dual_path_integral_representer_theorem`
- `TestSobolev3DFromDualSpectral.test_from_dual_sum_linear_form_returns_function3d`
- `TestSobolev3DFromDualSpectral.test_from_dual_sum_linear_form_representer_identity`
- `TestSobolev3DFromDualSpectral.test_from_dual_non_spectral_inverse_weight_raises`
- `TestSobolev3DFromDualSpectral.test_sola_adjoint_with_sobolev_domain`
- `TestSobolev3DFromDualSpectral.test_sola_adjoint_consistency`
- `TestSOLAOperator3DWithPathFunctionals.test_adjoint_path_functionals_raises_clear_error` — updated match pattern to `"Sobolev3D"` (formerly `"Phase 5"`)

**Review Status:** APPROVED (after revision — stale docstrings, long lines, living reference table fixed)

**Git Commit Message:**
```
feat(spaces,operators): spectral Sobolev representers for path-integral forms

- extend Sobolev3D.from_dual: 3 branches (VolumeKernel3D/SumLinearForm/spectral)
- spectral representer: r_ℓ = Σ_k ℓ(Φ_k) · ν_k · Φ_k via SpectralOperator3D
- SOLAOperator3D now accepts Sobolev3D domain (duck-typed via base_space)
- adjoint fully functional for path-integral forms on Sobolev3D domain
- 8 new representer/adjoint tests; 379 total passing
- add path_integral_minimum_norm_demo.py end-to-end demo
- update living reference; archive continuous-path-integral plan

Plan: pygeoinf3D/docs/agent-docs/active-plans/continuous-path-integral-sobolev-inversion-plan.md
Phase: 5 of 5
Related: pygeoinf3D/docs/agent-docs/active-plans/continuous-path-integral-sobolev-inversion-phase-5-complete.md
```
