## Plan: Cross-Gram Fast Path for `T @ G.adjoint`

When two `SOLAOperator3D` instances share the same `Sobolev3D` domain, the product
`T @ G.adjoint` (or `G @ T.adjoint`) reduces to a dense `N_p × N_d` matrix via the
spectral identity $C_{TG}[k,i] = L_T \operatorname{diag}(\nu) L_G^\top$. Precomputing
this cross-Gram matrix eliminates the two dominant bottlenecks identified by profiling
(spectral `SumLinearForm` synthesis at 970 ms/col and 125 serial volume integrations at
2600 ms/col), collapsing `p_cov.matrix()` from ~160 s to a few matrix multiplications.

**Phases**

1. **Phase 1: Extract `_build_L_matrix` static helper from `SpectralGramSolver`**
   - **Objective:** Refactor the L-matrix assembly logic out of `__init__` into a
     reusable `@staticmethod _build_L_matrix(sobolev_space, forms, resolution) →
     (L: ndarray[m,N], nu: ndarray[N,])`. The existing `__init__` delegates to it.
     Allows the cross-Gram path to build `L_T` and `L_G` without constructing full
     `SpectralGramSolver` instances (no eigendecomposition, no data-covariance logic).
     Mixed form stacks (path + volume) must be supported.
   - **Files/Functions to Modify/Create:**
     - `pygeoinf3D/pygeoinf3d/solvers/spectral_gram.py`:
       add `@staticmethod _build_L_matrix(sobolev_space, forms, *, resolution=20)`.
       Refactor `__init__` to call it instead of duplicating the L-matrix logic.
   - **Tests to Write:**
     - `TestBuildLMatrix::test_path_forms_matches_interaction_matrix`
     - `TestBuildLMatrix::test_vol_forms_matches_interaction_matrix`
     - `TestBuildLMatrix::test_mixed_forms_matches_interaction_matrix`
     - `TestBuildLMatrix::test_rejects_non_spectral_sobolev` (raises TypeError loudly)
   - **Steps:**
     1. Write the four failing tests in `tests/operators/test_cross_gram.py`.
     2. Add `_build_L_matrix` to `spectral_gram.py`, moving path-row and vol-row logic into it; return `(L, nu)`.
     3. Refactor `__init__` to call `L, nu = self._build_L_matrix(...)` and continue as now.
     4. Run tests; confirm all pass.

2. **Phase 2: Cross-Gram `__matmul__` fast path in `SOLAOperator3D`**
   - **Objective:** Detect `T @ G.adjoint` (and `G @ T.adjoint`) where both operands are
     `SOLAOperator3D` on the same `Sobolev3D` domain, compute
     $C_{TG} = L_T \operatorname{diag}(\nu) L_G^\top$ via `_build_L_matrix`, and return
     a `MatrixLinearOperator(G.codomain, T.codomain, C_TG, galerkin=True)`.
     Gate on `isinstance(domain, Sobolev3D)` with a `SpectralOperator3D` inverse weight.
     Fail loudly (raise `TypeError`/`ValueError`) if the gate passes but L-matrix
     construction fails for any reason. Cache result keyed on `id(other_sola)` to avoid
     recomputation when the same pair is used multiple times.
   - **Files/Functions to Modify/Create:**
     - `pygeoinf3D/pygeoinf3d/operators/sola.py`:
       - `__init__`: add `self._cross_gram_cache: dict[int, MatrixLinearOperator] = {}`.
       - Add `_try_build_fast_cross_gram_operator(self, other_sola) -> MatrixLinearOperator | None`.
       - Extend `__matmul__`: detect cross case (`other._adjoint_base is other_sola, other_sola is SOLAOperator3D, same domain`) and call `_try_build_fast_cross_gram_operator`.
   - **Tests to Write:**
     - `TestCrossGramFastPath::test_T_at_G_adjoint_returns_matrix_linear_operator`
     - `TestCrossGramFastPath::test_G_at_T_adjoint_returns_matrix_linear_operator`
     - `TestCrossGramFastPath::test_cross_gram_value_matches_slow_path` (rtol=1e-5)
     - `TestCrossGramFastPath::test_self_gram_still_uses_existing_fast_path` (no regression)
     - `TestCrossGramFastPath::test_cross_gram_is_cached` (same object returned twice)
     - `TestCrossGramFastPath::test_lebesgue_domain_skips_fast_path` (returns generic op)
   - **Steps:**
     1. Write the six failing tests.
     2. Implement `_try_build_fast_cross_gram_operator` calling `_build_L_matrix` for both operators, computing `L_T @ np.diag(nu) @ L_G.T`, wrapping in `MatrixLinearOperator` with `galerkin=True`.
     3. Extend `__matmul__` to call it in the cross case.
     4. Check cache logic; run tests; confirm all pass including the 184 existing tests.

3. **Phase 3: End-to-end notebook verification**
   - **Objective:** Re-run the profiling cell in `sola_rays.ipynb` to confirm `T @ G.adjoint` now returns a `MatrixLinearOperator` and that `p_cov.matrix()` wall time drops dramatically. Update the profiling cell results comment.
   - **Files to Modify:**
     - `pygeoinf3D/demos/box/sola/sola_rays.ipynb`: update profiling cell to add cross-Gram timing and verify the new operator type.
   - **Steps:**
     1. Run full test suite; confirm 184+ tests pass.
     2. Re-execute profiling cell; record new timings.
     3. Add a line to the profiling cell printing `type(T @ G.adjoint)` and the construction time.

**Open Questions**
(resolved before implementation)
1. ~~Should `X` be materialized as dense?~~ No — only `SOLAOperator3D` fast path.
2. ~~Mixed form stacks allowed?~~ Yes, both paths and volumes.
3. ~~Silent fallback or loud failure?~~ Fail loudly.
