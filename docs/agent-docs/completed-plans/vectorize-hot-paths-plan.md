## Plan: Vectorize Hot Paths via Batch Eigenfunction Evaluation

Leverage the `evaluate_all_eigenfunctions()` added in Phase 1 of the spectral-gram plan to replace Python loops over eigenfunctions in four hot paths: PathIntegralFunctional node caching, KLSampler3D general evaluate, GeneralizedBoxLaplacian._apply, and Sobolev3D.from_dual Branch 3.

**Phases: 4**

1. **Phase 1: PathIntegralFunctional — cache quadrature nodes**
    - **Objective:** Eliminate repeated `arc_length_nodes` computation on every form evaluation
    - **Files/Functions to Modify/Create:** `pygeoinf3d/spaces/forms.py` (`PathIntegralFunctional.__init__`, `_mapping_impl`)
    - **Tests to Write:** `test_path_integral_cached_nodes`, `test_path_integral_same_results`
    - **Steps:**
        1. Write tests verifying current behavior (golden results)
        2. Cache `t_arr` and `pts` in `__init__` as `self._t_arr` and `self._pts`
        3. Pre-compute `self._eval_scale = self._scale * self._path.arclength`
        4. Simplify `_mapping_impl` to use cached values
        5. Run tests to verify identical results

2. **Phase 2: KLSampler3D — vectorized general evaluate**
    - **Objective:** Replace Python loop in `evaluate` closure with batch eigenfunction evaluation
    - **Files/Functions to Modify/Create:** `pygeoinf3d/sampling/kl_sampler.py` (`KLSampler3D.sample`, `sample_from_xis`)
    - **Tests to Write:** `test_kl_sample_vectorized_matches_original`, `test_kl_sample_batch_eval_speed`
    - **Steps:**
        1. Write test capturing current sample values at specific seed
        2. In `sample()`, when `has_tensor_product_structure`, build vectorized evaluate: `weights @ provider.evaluate_all_eigenfunctions(pts) + mean(pts)`
        3. Apply same optimization to `sample_from_xis()`
        4. Run tests to verify identical values

3. **Phase 3: GeneralizedBoxLaplacian._apply — vectorized spectral projection**
    - **Objective:** Replace both Python loops (coefficient computation + evaluate closure) with batch eval + matrix ops
    - **Files/Functions to Modify/Create:** `pygeoinf3d/operators/generalized_box_laplacian.py` (`_apply`)
    - **Tests to Write:** `test_apply_vectorized_matches_original`, `test_apply_vectorized_speed`
    - **Steps:**
        1. Write test capturing current operator application results
        2. Use `provider.evaluate_all_eigenfunctions(self._pts)` to get `Phi_matrix` `(N, n_pts)`
        3. Compute coefficients as `Phi_matrix @ (f_vals * self._weights)` — single matmul
        4. Build vectorized evaluate closure: `(mu * c) @ provider.evaluate_all_eigenfunctions(pts)`
        5. Run tests to verify identical results

4. **Phase 4: Sobolev3D.from_dual Branch 3 — vectorized representer**
    - **Objective:** Replace spectral loop with vectorized batch eval
    - **Files/Functions to Modify/Create:** `pygeoinf3d/spaces/sobolev.py` (`from_dual` Branch 3)
    - **Tests to Write:** `test_from_dual_vectorized_matches_original`, `test_from_dual_speed`
    - **Steps:**
        1. Write test capturing current from_dual results
        2. Detect if `inverse_weight._provider` is a `BoxLaplacianSpectrumProvider`
        3. If so, use `provider.evaluate_all_eigenfunctions(pts)` and vectorized synthesis
        4. Fall back to existing loop otherwise
        5. Run tests

**Open Questions**
1. Should the vectorized path be the only path, or keep the loop as fallback for non-tensor-product providers? → Keep fallback.
2. Should `PathIntegralFunctional.copy()` and algebra methods share cached nodes? → Yes, via shallow copy of arrays.
