## Plan Complete: Vectorize Hot Paths via Batch Eigenfunction Evaluation

Replaced Python loops over eigenfunctions with vectorized batch evaluation in four hot paths across pygeoinf3D. These changes leverage the `evaluate_all_eigenfunctions()` method added in the spectral-gram-fast-path plan to eliminate per-mode Python function calls in spectral sums, coefficient computation, and evaluate closures.

**Phases Completed:** 4 of 4
1. ✅ Phase 1: PathIntegralFunctional — cache quadrature nodes in `__init__`
2. ✅ Phase 2: KLSampler3D — vectorized general evaluate via batch eigenfunction eval
3. ✅ Phase 3: GeneralizedBoxLaplacian._apply — vectorized coefficient computation and synthesis
4. ✅ Phase 4: Sobolev3D.from_dual Branch 3 — vectorized representer with ultra-fast PathIntegralFunctional path

**All Files Created/Modified:**
- `pygeoinf3d/spaces/forms.py` (cached nodes in PathIntegralFunctional)
- `pygeoinf3d/sampling/kl_sampler.py` (vectorized evaluate closure)
- `pygeoinf3d/operators/generalized_box_laplacian.py` (vectorized _apply)
- `pygeoinf3d/spaces/sobolev.py` (vectorized from_dual Branch 3)
- `tests/spaces/test_path_integral_functional.py` (5 new tests)
- `tests/sampling/test_kl_vectorized.py` (5 new tests)
- `tests/operators/test_generalized_box_laplacian_vectorized.py` (5 new tests)
- `tests/spaces/test_from_dual_vectorized.py` (11 new tests)
- `docs/agent-docs/references/living/pygeoinf3d-reference.md` (updated)

**Key Functions/Classes Changed:**
- `PathIntegralFunctional.__init__` — caches `_t_arr`, `_pts`, `_eval_scale`
- `PathIntegralFunctional._mapping_impl` — uses cached values
- `KLSampler3D._build_vectorized_evaluate()` — new method using `_eval_1d_basis` + einsum
- `KLSampler3D.sample()` / `sample_from_xis()` — dispatch to vectorized path
- `GeneralizedBoxLaplacian._apply()` — matmul-based coefficient + synthesis
- `Sobolev3D.from_dual` Branch 3 — three-tier dispatch (ultra-fast / fast synthesis / loop fallback)

**Test Coverage:**
- Total tests written: 26
- All tests passing: ✅ (831 total)

**Performance Notes:**
- PathIntegralFunctional: eliminates repeated `arc_length_nodes` calls (O(n_quad) per eval → 0)
- KLSampler3D: ~4-5x speedup at 50+ modes for arbitrary point evaluation
- GeneralizedBoxLaplacian: eliminates N Python function calls per apply + per evaluate
- Sobolev3D.from_dual: ultra-fast path eliminates ALL Python loops for PathIntegralFunctional representers

**Recommendations for Next Steps:**
- Profile the notebook end-to-end with the new fast paths
- Consider extending vectorized evaluate to the `_attach_kl_fast_paths` grid/slice evaluators
- Tikhonov regularization extension for SpectralGramSolver
