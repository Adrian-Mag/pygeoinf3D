# Sampling — Gaussian Random Field Sampling via KL Expansion

This directory contains a demonstration of sampling from Gaussian measures on Sobolev spaces using the **KL** (Karhunen–Loève) sampler.

## Background

The **Karhunen–Loève expansion** decomposes a Gaussian random field into uncorrelated modes:

$$u(\mathbf{x}) = m(\mathbf{x}) + \sum_{k=0}^{K-1} \sqrt{\lambda_k} \, \xi_k \, \Phi_k(\mathbf{x}), \quad \xi_k \sim \mathcal{N}(0, 1)$$

where:
- $m(\mathbf{x})$ = mean (prior expectation)
- $(\lambda_k, \Phi_k)$ = eigenvalue/eigenfunction pairs of the covariance operator
- $K$ = truncation (number of modes, trade-off between accuracy and cost)

When the covariance operator is **spectral** (diagonalized in an explicit basis), the KLSampler3D can:
1. Draw random standard normals $\xi_k$
2. Evaluate eigenfunctions at desired points via batch matrix operations
3. Form the random field in seconds

## Notebooks

### kl_sampler_gaussian_prior.ipynb

Demonstrates sampling from a Gaussian prior on a Sobolev space.

**Topics covered:**
- Setting up a Gaussian measure with spectral covariance
- KLSampler3D: batch-vectorized sampling
- Drawing single and multiple samples
- Visualization of sample statistics (mean, covariance)
- Fast path (tensor-product) vs. loop fallback

**Preconditions:**
- Familiarity with `Sobolev3D` (see `../foundational/7_bessel_sobolev_3d.ipynb`)
- Understanding of spectral operators (see `../foundational/6_laplacian_3d.ipynb`)

**Duration:** ~30 minutes

---

## Performance Characteristics

| Provider | Structure | Evaluation | Cost per Sample |
|---|---|---|---|
| BoxLaplacianSpectrumProvider | Tensor-product ✓ | Vectorized einsum | **O(K · p)** fast |
| Generic SpectralOperator3D | Non-tensor-product | Loop over modes | **O(K · p)** slower |

**K** = number of modes, **p** = number of evaluation points

**Speedup (tensor-product):** 2–10× faster than loop fallback for typical K ≤ 1000.

---

## When to Use KL Sampling

✅ **Use KL sampling when:**
- You need random realizations of a Gaussian prior
- The covariance operator is spectral (Laplacian, inverse Laplacian, Sobolev weight)
- You want vectorized fast-path evaluation
- Budget for N ≤ 1000 modes (typical regularization region)

❌ **Alternatives:**
- **Non-Gaussian priors** → use custom sampling or rejection methods
- **Non-spectral covariance** → use generic pygeoinf.GaussianMeasure with covariance_factor
- **Expensive matrix operations** → consider reduced-rank approximations

---

## Integration with Inversion

KL sampling is often used for:
1. **Prior predictions** — sample before any data to visualize prior uncertainty
2. **Posterior sampling** — after inversion (different framework; requires e.g., MCMC or ensemble methods)
3. **Sensitivity analysis** — sample from prior, propagate through forward, analyze output range

---

## Supported Operators

✅ **Tensor-product fast path:**
- `InverseBoxLaplacian` with `BoxLaplacianSpectrumProvider`
- `BesselSobolevBox` with `BoxLaplacianSpectrumProvider`

Available for:
- `BoxDomain` (3D rectangular domains)

❌ **Loop fallback (slower):**
- Any `SpectralOperator3D` without `has_tensor_product_structure`

---

## Automatic Sampler Selection

The `SolverRouter.recommend_sampler()` method recommends KLSampler3D for spectral operators:

```python
from pygeoinf3d import SolverRouter

router = SolverRouter()
rec = router.recommend_sampler(covariance_operator)

if rec.solver_spec.name == "KLSampler3D":
    print("✓ KL sampler is optimal for this operator")
```

---

## Next Steps

- **[../sola/](../sola/)** — SOLA operator with volumetric kernels
- **[../pli/](../pli/)** — Path-integral measurements and fast spectral inversion
- **[../../docs/SOLVER_SELECTION_GUIDE.md](../../docs/SOLVER_SELECTION_GUIDE.md)** — Full comparison of all solvers
