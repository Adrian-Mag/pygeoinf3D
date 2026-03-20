## Phase 3 Complete: Notebook Update & Scaling Benchmark

Updated the demo notebook to use `SpectralGramSolver` instead of the generic `LinearMinimumNormInversion` path, and added a scaling benchmark section showing performance from N=10 to N=200.

**Files created/changed:**
- `demos/noisy_path_form_least_norm_inversion_demo.ipynb`

**Functions created/changed:**
- Imports cell: added `SpectralGramSolver` import
- Inversion cell: replaced generic inversion with `SpectralGramSolver` build + solve
- Diagnostics cell: uses `solver.forward(u_hat)` instead of `G(u_hat)`
- Comparison cell: total sensitivity via `solver.solve(np.ones(N_PATHS))`
- Summary markdown: updated with performance discussion
- New scaling benchmark section (markdown + code cell)

**Tests created/changed:**
- N/A (notebook cells verified by execution)

**Review Status:** APPROVED — all 26 cells execute correctly, 805 tests pass

**Git Commit Message:**
```
feat(solvers): add SpectralGramSolver for 10000x faster minimum-norm inversion

Plan: pygeoinf3D/docs/agent-docs/active-plans/spectral-gram-fast-path-plan.md
Phase: 3 of 3
```
