# pygeoinf3D Demonstration Notebooks

This directory contains a collection of Jupyter notebooks and scripts that demonstrate the key features of pygeoinf3D, organized by foundational concepts and concrete geometry implementations.

## Quick Start with SolverRouter

Before choosing a demo, check the automatic solver selection:

```python
from pygeoinf3d import SolverRouter, Sobolev3D, BesselSobolevBox, BoxDomain, Lebesgue3D
from pygeoinf3d.spaces.forms import PathIntegralFunctional
import numpy as np

router = SolverRouter()

# Example 1: Path-integral inversion on Sobolev3D
box = BoxDomain(np.zeros(3), np.ones(3))
leb = Lebesgue3D(box)
provider = ...  # BoxLaplacianSpectrumProvider
sobolev = BesselSobolevBox(leb, provider, order=1.5)
forms = [...]  # list of PathIntegralFunctional

rec = router.recommend(sobolev, forms)
print(f"Recommended solver: {rec.solver_spec.name}")
# Output: "SpectralGramSolver"
```

See [docs/SOLVER_SELECTION_GUIDE.md](../docs/SOLVER_SELECTION_GUIDE.md) for details.

---

## Directory Structure

### 1. **foundational/** — Geometry-agnostic foundation

These notebooks introduce the core abstractions in pygeoinf3D: Hilbert spaces, functions, operators, and basis-independent representations. They are independent of specific geometries and can serve as a gentle introduction to the library.

- `1_box_domain_intro.ipynb` — Introduction to BoxDomain and coordinate systems
- `2_functions_3d.ipynb` — Function3D objects and callable evaluation
- `3_lebesgue_space_3d.ipynb` — L² spaces on 3D domains
- `4_kernel_functionals_3d.ipynb` — Integral kernels and volume functionals
- `5_providers_3d.ipynb` — Function and kernel providers for lazy evaluation
- `6_laplacian_3d.ipynb` — The Laplacian operator on boxes
- `7_bessel_sobolev_3d.ipynb` — Sobolev space structure for smoothing

**Recommend starting here if:**
- You're new to pygeoinf3D
- You want to understand the abstraction layers
- You're considering building a new concrete geometry

---

### 2. **box/** — Box geometry methods

Demonstrations of inversion and sampling on 3D rectangular domains. Organized by measurement type and inversion method.

#### **box/sola/** — SOLA operator with volumetric kernels (Euclidean measurements)

The **SOLA** (Standard Orthonormal Linearized Adjoint) operator-based workflow for least-norm inversion with volumetric kernel measurements.

- `sola_volumetric_kernels.ipynb` — Least-norm inversion using Gaussian volumetric kernels on a Lebesgue3D space

**When to use SOLA:**
- Measurements are smooth spatial kernels (e.g., Gaussian blobs)
- You want a basis-free forward operator that works with any linear form
- You plan to use pygeoinf's generic LinearMinimumNormInversion solver

---

#### **box/pli/** — Path-integral line measurements

Fast minimum-norm inversion and exploratory analysis using **PLI** (path-integral line) measurements along curves in the domain.

- `1_path_forms_intro.ipynb` — Introduction to Path3D and PathIntegralFunctional
- `2_pli_basics.ipynb` — Basic path-integral measurements and forward operators
- `3_pli_noisy_least_norm.ipynb` — Least-norm inversion with noisy data and optional covariance weighting
- `path_integral_minimum_norm_demo.py` — Standalone Python script for batch processing

**When to use PLI:**
- Measurements are line integrals along paths (boreholes, raytracing, etc.)
- You have a Sobolev3D space with spectral weight (e.g., BesselSobolevBox)
- You want the **fast SpectralGramSolver** (1000–100,000× speedup) for repeated inversions
- Covariance weighting is needed for heterogeneous noise

**Fast path availability:**
- Sobolev3D + BesselSobolevBox → **SpectralGramSolver** ✓ (fast)
- Sobolev3D + non-spectral weight → SOLAOperator3D (slower generic fallback)

---

#### **box/sampling/** — Gaussian random field sampling

Sampling from Gaussian measures via Karhunen–Loève expansion.

- `kl_sampler_gaussian_prior.ipynb` — KL sampler for spatial priors on Sobolev spaces

**When to use KL sampling:**
- You need random realizations of a Gaussian process or prior
- The covariance operator is a SpectralOperator3D (e.g., InverseBoxLaplacian, BesselSobolevBox)
- Tensor-product structure is available (BoxLaplacianSpectrumProvider) for fast vectorized sampling

---

## Recommended Learning Paths

### Path A: Inversion with SOLA + Generic Framework
1. `foundational/1_box_domain_intro.ipynb`
2. `foundational/2_functions_3d.ipynb`
3. `foundational/3_lebesgue_space_3d.ipynb`
4. `foundational/4_kernel_functionals_3d.ipynb`
5. `box/sola/sola_volumetric_kernels.ipynb`

### Path B: Fast Path-Integral Inversion (Full Stack)
1. `foundational/1_box_domain_intro.ipynb`
2. `foundational/2_functions_3d.ipynb`
3. `foundational/3_lebesgue_space_3d.ipynb`
4. `foundational/5_providers_3d.ipynb`
5. `foundational/6_laplacian_3d.ipynb`
6. `foundational/7_bessel_sobolev_3d.ipynb`
7. `box/pli/1_path_forms_intro.ipynb`
8. `box/pli/2_pli_basics.ipynb`
9. `box/pli/3_pli_noisy_least_norm.ipynb`

### Path C: Gaussian Sampling for Priors
1. `foundational/1_box_domain_intro.ipynb`
2. `foundational/5_providers_3d.ipynb`
3. `foundational/6_laplacian_3d.ipynb`
4. `box/sampling/kl_sampler_gaussian_prior.ipynb`

---

## Adding a New Geometry

When adding a new geometry (e.g., spherical shell `sphere/`):

1. Create `sphere/` directory with subdirectories:
   ```
   demos/sphere/
   ├── sola/
   ├── pli/
   ├── sampling/
   └── README.md
   ```

2. Implement fundamental classes:
   - `SphereDomain`, `Lebesgue3D(sphere_domain)`
   - Basis providers (e.g., spherical harmonics)
   - Spectral operators (e.g., Laplace-Beltrami eigendecomposition)

3. Follow the same compartmentalization pattern:
   - **sola/**: volumetric and surface measurements
   - **pli/**: arc-length path integrals on the surface
   - **sampling/**: KL sampling for priors

4. Update this README with the new geometry section and learning path.

---

## File Organization Summary

```
demos/
├── README.md (this file)
├── test.ipynb (deprecated; for cleanup)
│
├── foundational/
│   ├── README.md
│   ├── 1_box_domain_intro.ipynb
│   ├── 2_functions_3d.ipynb
│   ├── 3_lebesgue_space_3d.ipynb
│   ├── 4_kernel_functionals_3d.ipynb
│   ├── 5_providers_3d.ipynb
│   ├── 6_laplacian_3d.ipynb
│   └── 7_bessel_sobolev_3d.ipynb
│
└── box/
    ├── README.md
    ├── sola/
    │   ├── README.md
    │   └── sola_volumetric_kernels.ipynb
    ├── pli/
    │   ├── README.md
    │   ├── 1_path_forms_intro.ipynb
    │   ├── 2_pli_basics.ipynb
    │   ├── 3_pli_noisy_least_norm.ipynb
    │   └── path_integral_minimum_norm_demo.py
    └── sampling/
        ├── README.md
        └── kl_sampler_gaussian_prior.ipynb
```

---

## Running Demos

### Using Jupyter
```bash
cd demos/
jupyter notebook foundational/1_box_domain_intro.ipynb
```

### Using IPython
```bash
cd demos/box/pli/
ipython -c "%run path_integral_minimum_norm_demo.py"
```

### In Python/IPython REPL
```python
%cd demos/box/pli/
%run path_integral_minimum_norm_demo.py
```
