# Box Geometry Demonstrations

This directory contains demonstrations of inversion and sampling on **3D rectangular (box) domains**. Organized by measurement type and method.

## Directory Structure

```
box/
├── sola/          (volumetric kernel measurements)
├── pli/           (path-integral line measurements)
└── sampling/      (Gaussian random field sampling)
```

### [sola/](sola/) — SOLA Operator with Volumetric Kernels

The **SOLA** (Standard Orthonormal Linearized Adjoint) operator for least-norm inversion with smooth spatial kernel measurements (e.g., smooth Gaussian blobs).

**Use when:**
- Measurements are smooth volumetric functions (e.g., seismic wavelets, gravity kernels)
- You want a general-purpose measurement operator
- You plan to use pygeoinf's generic inversion framework

**Speed:** Slower than spectral solvers; no pre-computation.

---

### [pli/](pli/) — Path-Integral Line Measurements

Fast minimum-norm inversion using **PLI** (path-integral line) measurements along curves (boreholes, ray paths, etc.).

**Use when:**
- Measurements are line integrals along paths
- You have a Sobolev3D space with **spectral weight** (BesselSobolevBox)
- You want the **1000–100,000× speedup** of SpectralGramSolver
- Data have heterogeneous noise (covariance weighting)

**Speed:** **Fast** — SpectralGramSolver pre-computes Gram matrix; O(m³) eigendecomposition once, then O(m²) per solve.

---

### [sampling/](sampling/) — Gaussian Random Field Sampling

Sampling from Gaussian measures via **KL** (Karhunen–Loève) expansion.

**Use when:**
- You need random realizations of a spatial prior or posterior
- The covariance operator is spectral (Laplacian, inverse Laplacian, Sobolev weight)
- You want vectorized fast-path sampling

**Speed:** Fast with tensor-product structure (BoxLaplacianSpectrumProvider).

---

## Quick Solver Selection

| Measurement Type | Space | Operator | Solver | Speed | Notes |
|---|---|---|---|---|---|
| **Volume kernels** | Lebesgue3D | Any | SOLAOperator3D | General | Generic framework; linear in m, N |
| **Path integrals** | Sobolev3D | **Spectral** (e.g., BesselSobolevBox) | **SpectralGramSolver** | **FAST** | 1000–100,000× speedup; O(m³) setup |
| **Path integrals** | Sobolev3D | Non-spectral | SOLAOperator3D | General | Falls back to generic framework |
| **Sampling** | Any | SpectralOperator3D | **KLSampler3D** | **FAST** | Vectorized einsum with tensor-product |

---

## Learning Paths

### Path A: Start with SOLA (Generic case)
1. `../foundational/1_box_domain_intro.ipynb`
2. `../foundational/3_lebesgue_space_3d.ipynb`
3. `../foundational/4_kernel_functionals_3d.ipynb`
4. [`sola/sola_volumetric_kernels.ipynb`](sola/)

**Time:** ~1 hour | **Solver:** SOLAOperator3D + LinearMinimumNormInversion

---

### Path B: Full Fast Path-Integral Stack (Recommended)
1. `../foundational/1_box_domain_intro.ipynb`
2. `../foundational/5_providers_3d.ipynb`
3. `../foundational/6_laplacian_3d.ipynb`
4. `../foundational/7_bessel_sobolev_3d.ipynb`
5. [`pli/1_path_forms_intro.ipynb`](pli/)
6. [`pli/2_pli_basics.ipynb`](pli/)
7. [`pli/3_pli_noisy_least_norm.ipynb`](pli/)

**Time:** ~2 hours | **Solver:** SpectralGramSolver (fast)

---

### Path C: Sampling for Priors
1. `../foundational/1_box_domain_intro.ipynb`
2. `../foundational/5_providers_3d.ipynb`
3. `../foundational/6_laplacian_3d.ipynb`
4. [`sampling/kl_sampler_gaussian_prior.ipynb`](sampling/)

**Time:** ~1 hour | **Sampler:** KLSampler3D

---

## Choosing Your Path

**If you:**
- Want a quick introduction → **Path A** (SOLA, generic framework)
- Plan to repeat inversions on large problems → **Path B** (PLI, spectral fast path)
- Need spatial priors → **Path C** (Sampling)
- Want everything → all three paths (~3 hours)

---

## Automatic Solver Selection

Use `SolverRouter` to automatically select the best solver:

```python
from pygeoinf3d import SolverRouter, SOLVER_REGISTRY

router = SolverRouter()
rec = router.recommend(sobolev_space, forms)

print(f"Solver: {rec.solver_spec.name}")
print(f"Reason: {rec.reason}")
for warning in rec.warnings:
    print(f"WARNING: {warning}")
```

For details, see [../docs/SOLVER_SELECTION_GUIDE.md](../../docs/SOLVER_SELECTION_GUIDE.md).
