## Plan Complete: SpectralGramSolver Fast Path

Implemented a vectorized spectral Gram matrix solver (`SpectralGramSolver`) that replaces the generic framework's nested Python loops with batch eigenfunction evaluation and numpy matmuls. This achieves 1,000–100,000× speedups for minimum-norm inversion of path-integral data on 3D box domains, enabling interactive-speed inversions with hundreds of paths.

**Phases Completed:** 3 of 3
1. ✅ Phase 1: Batch Eigenfunction Evaluation — `evaluate_all_eigenfunctions()` and `get_all_eigenvalues()` on `BoxLaplacianSpectrumProvider`
2. ✅ Phase 2: SpectralGramSolver Class — Pre-computes interaction matrix L, Gram via matmul, eigendecomposition solver
3. ✅ Phase 3: Notebook Update — Replaced generic inversion with fast solver, added scaling benchmark

**All Files Created/Modified:**
- `pygeoinf3d/providers/tensor_product.py` (added `_eval_1d_basis`, `evaluate_all_eigenfunctions`, `get_all_eigenvalues`)
- `pygeoinf3d/solvers/__init__.py` (new)
- `pygeoinf3d/solvers/spectral_gram.py` (new — `SpectralGramSolver` class)
- `tests/providers/test_tensor_product.py` (added `TestEvaluateAllEigenfunctions`)
- `tests/solvers/__init__.py` (new)
- `tests/solvers/test_spectral_gram.py` (new — 7 tests)
- `demos/noisy_path_form_least_norm_inversion_demo.ipynb` (updated for fast solver)
- `docs/agent-docs/references/living/pygeoinf3d-reference.md` (updated)

**Key Functions/Classes Added:**
- `_eval_1d_basis(xs, n_modes, bc_type, low, high)` — vectorized 1D basis evaluation returning `(n_points, n_modes)` matrix
- `BoxLaplacianSpectrumProvider.evaluate_all_eigenfunctions(points)` — batch 3D eigenfunction evaluation returning `(dim, n_points)` array
- `BoxLaplacianSpectrumProvider.get_all_eigenvalues()` — returns all eigenvalues as numpy array
- `SpectralGramSolver.__init__(sobolev, forms, rcond)` — builds interaction matrix L and Gram matrix
- `SpectralGramSolver.solve(data)` — solves minimum-norm problem via eigendecomposition
- `SpectralGramSolver.forward(f)` — evaluates forward operator

**Test Coverage:**
- Total tests written: 13 (6 eigenfunction + 7 solver)
- All tests passing: ✅
- Total suite: 805 tests in 5.76s

**Performance Results:**
| N paths | Old time (est.) | New time | Speedup |
|---------|----------------|----------|---------|
| 10      | 17s            | 0.010s   | 1,700×  |
| 20      | 110s           | 0.025s   | 4,400×  |
| 40      | 783s           | 0.074s   | 10,600× |
| 100     | ~8.5h          | 0.236s   | ~100,000× |
| 200     | infeasible     | 0.497s   | — |
