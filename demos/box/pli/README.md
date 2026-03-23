# PLI — Path-Integral Line Measurements (Fast Spectral Path)

This directory contains demonstrations of fast minimum-norm inversion using **PLI** (path-integral line) measurements along curves on 3D box domains, with the **SpectralGramSolver** fast path.

## Background

**Path-integral measurements** are line integrals along curves (boreholes, ray paths, etc.):

$$\ell_i(u) = \int_{\text{path}_i} u(\mathbf{x}) \, d\ell$$

When combined with a **Sobolev3D** space weighted by a **spectral operator** (e.g., BesselSobolevBox), the `SpectralGramSolver` provides a **1000–100,000× speedup** by:
1. Pre-computing spectral interaction matrices (once, at construction)
2. Fast eigendecomposition of the Gram matrix (O(m³), where m ≲ 100)
3. Solving in O(m²) per data vector

This is the recommended path for large-scale path-integral inversions.

## Notebooks (in recommended order)

### 1_path_forms_intro.ipynb
Introduction to Path3D objects and PathIntegralFunctional.

**Topics covered:**
- Path3D construction: polylines, arcs, parameterizations
- Arc-length parameterization and normalization
- PathIntegralFunctional: line integral linear forms
- Quadrature rules: composite trapezoid on paths

**Preconditions:**
- Familiarity with `Lebesgue3D` (see `../foundational/3_lebesgue_space_3d.ipynb`)

**Duration:** ~30 minutes

---

### 2_pli_basics.ipynb
Basic forward operator and data generation with path-integral measurements.

**Topics covered:**
- SOLAOperator3D with PathIntegralFunctional forms
- Forward evaluation: applying G to a Function3D
- Synthetic data generation and visualization
- Measurement sensitivity and noise models

**Preconditions:**
- Completion of `1_path_forms_intro.ipynb`
- Familiarity with `Sobolev3D` (see `../foundational/7_bessel_sobolev_3d.ipynb`)

**Duration:** ~30 minutes

---

### 3_pli_noisy_least_norm.ipynb
**Fast** least-norm inversion with noisy data and optional data-space covariance weighting.

**Topics covered:**
- SpectralGramSolver: ultra-fast batch solver
- Handling noisy measurements with covariance weighting
- Minimum-norm solution comparison: exact vs. noisy data
- Speed comparison: SpectralGramSolver vs. generic SOLAOperator3D
- Solution visualization and residual analysis

**Preconditions:**
- Completion of `1_path_forms_intro.ipynb` and `2_pli_basics.ipynb`
- Understanding of Sobolev spaces and spectral operators

**Duration:** ~45 minutes

---

### path_integral_minimum_norm_demo.py
Standalone Python script for batch processing (no Jupyter needed).

**Use for:**
- Reproducible batch experiments
- Large-scale parameter sweeps
- Integration into processing pipelines

---

## Performance Characteristics

| Solver | Construction | Solve (per data vector) | Typical Problem Size | Notes |
|---|---|---|---|---|
| **SpectralGramSolver** | O(N·m·q) + O(m³) eigen | **O(m²)** | m ≤ 100, N ≤ 1000, q ≤ 100 | **FAST** — spectral Gram precomputed |
| SOLAOperator3D + LinearMinimumNormInversion | O(m) adjoint | O(m³) eigen | m ≤ 100 | General — no pre-computation |

**N** = number of spectral modes, **m** = number of measurements, **q** = quadrature points per path

**Speedup:** 1000–100,000× for typical problem sizes (depending on N and m).

---

## Supported Configurations

✅ **Fast Path (SpectralGramSolver):**
- Space: `Sobolev3D`
- Weight: Spectral `SpectralOperator3D` (e.g., `BesselSobolevBox`)
- Provider: `BoxLaplacianSpectrumProvider` (has_tensor_product_structure = True)
- Forms: All `PathIntegralFunctional` with same `n_quad`
- Optional: Data-space covariance matrix `C_D` (any shape (m, m))

❌ **Slower Fallback (SOLAOperator3D + LinearMinimumNormInversion):**
- Space: `Lebesgue3D` or non-spectral `Sobolev3D`
- Forms: Any linear form type

---

## When to Use This Fast Path

✅ **Use SpectralGramSolver when:**
- All measurements are **path integrals** with same axis resolution
- Space is `Sobolev3D` with a **spectral weight** (BesselSobolevBox)
- You plan to solve **multiple times** (pre-computation amortized)
- m ≤ 100 and N ≤ 10,000 (practical regime)
- Data have heterogeneous noise (need covariance weighting)

❌ **Use generic path when:**
- Measurements are **mixed types** (volume + path integrals)
- Weight is non-spectral or unavailable
- m >> 100 or N >> 10,000 (memory/cost becomes prohibitive)
- Single solve only (pre-computation cost not amortized)

---

## Automatic Solver Selection

The `SolverRouter` automatically recommends SpectralGramSolver for compatible configurations:

```python
from pygeoinf3d import SolverRouter

router = SolverRouter()
rec = router.recommend(sobolev_space, forms)

if rec.solver_spec.name == "SpectralGramSolver":
    print("✓ Fast spectral path available!")
else:
    print(f"Falling back to {rec.solver_spec.name}")
```

See [../../docs/SOLVER_SELECTION_GUIDE.md](../../docs/SOLVER_SELECTION_GUIDE.md) for details.

---

## Next Steps

- **[../sola/](../sola/)** — SOLA operator with volumetric kernels (general case)
- **[../sampling/](../sampling/)** — Gaussian random field sampling via KL expansion
- **[../../docs/SOLVER_SELECTION_GUIDE.md](../../docs/SOLVER_SELECTION_GUIDE.md)** — Full solver compatibility matrix
