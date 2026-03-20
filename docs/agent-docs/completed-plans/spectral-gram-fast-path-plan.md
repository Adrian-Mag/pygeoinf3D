## Plan: Spectral Gram Fast Path for Minimum-Norm Inversion

The current minimum-norm inversion pipeline for `PathIntegralFunctional` on `Sobolev3D` is extremely slow (~O(N³) scaling) due to nested Python loops and lambda-tree traversal in the generic framework. This plan adds a vectorized "spectral Gram matrix" fast path that computes the Gram matrix `GG^T` analytically using pre-computed spectral interaction matrices, reducing cost from hours to seconds for N=100.

**Phases: 3**

1. **Phase 1: Batch Eigenfunction Evaluation on BoxLaplacianSpectrumProvider**
    - **Objective:** Add `evaluate_all_eigenfunctions(points)` method that returns `(N_modes, n_points)` array using existing `_eval_1d_basis` helper, avoiding per-mode Function3D calls.
    - **Files/Functions to Modify/Create:**
        - `pygeoinf3D/pygeoinf3d/providers/tensor_product.py` — add `evaluate_all_eigenfunctions` method to `BoxLaplacianSpectrumProvider`
        - `pygeoinf3D/tests/providers/test_tensor_product.py` — add tests
    - **Tests to Write:**
        - `test_evaluate_all_eigenfunctions_matches_individual` — verify batch output matches loop over `get_eigenfunction(k)(pts)` for all modes
        - `test_evaluate_all_eigenfunctions_shapes` — verify output shape `(N_modes, n_points)`
        - `test_evaluate_all_eigenfunctions_dirichlet` — verify Dirichlet BC variant
    - **Steps:**
        1. Write failing tests that call `provider.evaluate_all_eigenfunctions(pts)` and compare against per-mode evaluation
        2. Implement `evaluate_all_eigenfunctions` using `_eval_1d_basis` from `bessel_sobolev_box.py` (import or extract to shared location)
        3. Run tests to confirm passing

2. **Phase 2: SpectralGramSolver Utility**
    - **Objective:** Create a standalone solver class that pre-computes the spectral interaction matrix `L[i,k] = ℓ_i(Φ_k)` for all path forms and modes, builds `GG^T = L diag(ν) L^T` via numpy matmul, and returns the minimum-norm solution `u = G^T(GG^T)^{-1}d` as a `Function3D`.
    - **Files/Functions to Modify/Create:**
        - `pygeoinf3D/pygeoinf3d/solvers/__init__.py` — new package
        - `pygeoinf3D/pygeoinf3d/solvers/spectral_gram.py` — `SpectralGramSolver` class
        - `pygeoinf3D/tests/solvers/__init__.py` — new test package
        - `pygeoinf3D/tests/solvers/test_spectral_gram.py` — tests
    - **Tests to Write:**
        - `test_gram_matrix_matches_generic` — verify Gram matrix matches `(G @ G.adjoint).matrix()` from the generic framework (for small N)
        - `test_minimum_norm_solution_matches_generic` — verify the fast solution matches the generic `LinearMinimumNormInversion` result
        - `test_residual_near_zero` — verify `||G(u_hat) - d|| ≈ 0` for exact data
        - `test_representer_evaluation` — verify the returned Function3D evaluates correctly
    - **Steps:**
        1. Write failing tests
        2. Implement `SpectralGramSolver.__init__` — pre-compute quadrature points, L matrix, eigenvalues
        3. Implement `SpectralGramSolver.gram_matrix()` — `L @ diag(ν) @ L^T`
        4. Implement `SpectralGramSolver.solve(data)` — returns `Function3D` using spectral representer with vectorized einsum evaluation
        5. Run tests to confirm passing

3. **Phase 3: Update Notebook and Benchmark**
    - **Objective:** Replace the slow inversion in the demo notebook with `SpectralGramSolver`, add timing comparison, and run at N=10, 20, 40, 60, 100.
    - **Files/Functions to Modify/Create:**
        - `pygeoinf3D/demos/noisy_path_form_least_norm_inversion_demo.ipynb` — update inversion cell, add benchmark cells
    - **Tests to Write:** (none — notebook demo)
    - **Steps:**
        1. Add import for `SpectralGramSolver`
        2. Replace the slow `LinearMinimumNormInversion` cell with `SpectralGramSolver`
        3. Add a scaling benchmark cell that times N=10,20,40,60,100
        4. Verify all cells execute correctly

**Open Questions:**
1. Should `_eval_1d_basis` be extracted to a shared module or imported cross-module? (Answer: import from `bessel_sobolev_box` for now, refactor later)
