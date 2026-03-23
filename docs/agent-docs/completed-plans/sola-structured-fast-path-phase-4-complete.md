## Phase 4 Complete: Spectral Mixed-Form Extension

Extended `SpectralGramSolver` to support `VolumeKernel3D` forms alongside `PathIntegralFunctional` forms, including fully mixed stacks. A new `_build_L_volume` static method assembles the interaction matrix rows for volume forms via structured quadrature; path and volume rows are then interleaved into the full `L` matrix before Gram computation. `SolverRouter` was updated to accept Sobolev spaces with volume-only and mixed form stacks.

**Files created/changed:**
- `pygeoinf3d/solvers/spectral_gram.py` — added `VolumeKernel3D` import, volume-form validation, `_build_L_volume()`, mixed-stack partitioning in `__init__`, `resolution` parameter
- `pygeoinf3d/solvers/router.py` — extended routing to accept Sobolev + volume stacks and Sobolev + mixed stacks
- `tests/solvers/test_spectral_gram_volume.py` — new test file with 5 tests

**Functions created/changed:**
- `SpectralGramSolver.__init__` — added form-type validation (path + volume OK; anything else raises `TypeError`); added form partitioning logic; `resolution` kwarg for volume quadrature
- `SpectralGramSolver._build_L_volume(volume_forms, provider, domain, resolution)` — static method; assembles `L[i,k] = (k_vals * w) @ Phi.T` for each volume kernel against all eigenfunctions
- `SolverRouter.recommend` — relaxed form-type guard to accept `VolumeKernel3D` in Sobolev stacks

**Tests created/changed:**
- `test_spectral_interaction_coefficients_for_volume_form_match_direct_quadrature`
- `test_spectral_gram_solver_volume_stack_gram_matches_direct_quadrature`
- `test_spectral_gram_solver_supports_mixed_path_and_volume_stack`
- `test_router_accepts_sobolev_volume_stack`
- `test_router_accepts_sobolev_mixed_stack`

**Review Status:** APPROVED

**Git Commit Message:**
```
feat(solvers): extend SpectralGramSolver to support VolumeKernel3D and mixed stacks

- Accept VolumeKernel3D forms (homogeneous or mixed with PathIntegralFunctional)
- Add _build_L_volume(): interaction matrix rows via structured quadrature
- Partition forms by type; interleave rows into full L matrix before Gram
- Add resolution kwarg for volume quadrature grid
- SolverRouter accepts Sobolev + volume-only and Sobolev + mixed stacks
- 5 new tests covering L coefficients, Gram accuracy, mixed-stack solve, routing

Plan: pygeoinf3D/docs/agent-docs/active-plans/sola-structured-fast-path-plan.md
Phase: 4 of 5
Related: pygeoinf3D/docs/agent-docs/active-plans/sola-structured-fast-path-phase-4-complete.md
```
