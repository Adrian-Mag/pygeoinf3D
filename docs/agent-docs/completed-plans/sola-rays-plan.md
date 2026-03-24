# Plan: SOLA Demo with Sobolev Model Space and Straight-Ray Data

Create `sola_rays.ipynb` and companion `sola_rays.py` in `pygeoinf3D/demos/box/sola/`. This mirrors the structure of `sola_noise.ipynb/py` but replaces the Lebesgue L² model space with a Sobolev H² space and Gaussian sensitivity kernels with path-integral measurements from straight seismic rays.

**Approved configuration:**
- 30 rays (N_d = 30), 10 stations randomly at z=1 surface
- Sobolev order s=2 (BesselSobolevBox order=2 / order=-2)
- 27 target properties on 3×3×3 grid (N_p = 27)
- Both notebook + script

**Key differences from `sola_noise`:**

| Feature          | `sola_noise`                     | `sola_rays`                                   |
|------------------|----------------------------------|-----------------------------------------------|
| Model space      | `Lebesgue3D` (L²)                | `Sobolev3D` (H² via `BesselSobolevBox`)        |
| Forward forms    | Gaussian sensitivity kernels     | `PathIntegralFunctional` (straight rays)      |
| Operator         | `DenseVolumeSolaSolver`          | Explicit spectral Gram matrix                 |
| Gram matrix      | Push-through identity            | `forms_G[i](sobolev.from_dual(forms_G[j]))`   |
| Representers     | N/A (Gaussian; have pointvals)   | `sobolev.from_dual(forms_G[i])` → Function3D  |
| Geometry         | No spatial structure             | Stations at z=1 surface, sources in volume    |

---

## Phases (3 phases)

### Phase 1: Implement `sola_rays.ipynb`
- **Objective:** Build the full notebook mirroring `sola_noise.ipynb` with markdown + code cells
- **Files to create:** `pygeoinf3D/demos/box/sola/sola_rays.ipynb`
- **Steps:**

  1. **Cell: Setup** — build `BoxDomain`, `Lebesgue3D`, `BoxLaplacianSpectrumProvider(truncation=(8,8,8), bc_type="neumann")`, `BesselSobolevBox(order=2)` and `(order=-2)`, `Sobolev3D`. Print eigenvalue spectrum summary.

  2. **Cell: Ray geometry** — N_d = 30 rays; N_stations = 10 stations randomly placed at z=1 surface (x,y uniform in [0.1,0.9]); N_sources = 30 sources randomly in the volume (uniform in [0.1,0.9]³); each source assigned to a random station; build `Path3D.from_polyline([source, station])` for each ray. Visualise ray geometry with Plotly scatter3d (ray lines + station markers + source markers).

  3. **Cell: Forward forms & SOLA operator** — `PathIntegralFunctional(sobolev, path, n_quad=200)` for each ray; build `SOLAOperator3D(sobolev, EuclideanSpace(N_d), forms_G)` for the summary print; print operator domain/codomain.

  4. **Cell: Target forms** — N_p = 27 targets on 3×3×3 grid (linspace 0.1→0.9); `GaussianProvider3D(box, centers=target_grid, sigmas=0.10, normalized=True)`; wrap each as `VolumeKernel3D(base_space, gaussian_fn)`. Visualise target locations.

  5. **Cell: True model + synthetic data** — 3-Gaussian true model (same as sola_noise); compute `d_bar = [form(m_true) for form in forms_G]`; noise `σ = 0.05 * max(|d_bar|)`; `C_D = σ² * I`; `d_tilde = d_bar + noise`. Print SNR.

  6. **Cell: Representers + Gram matrix** — compute `representers = [sobolev.from_dual(f) for f in forms_G]`; `C_GG[i,j] = forms_G[i](representers[j])`; verify symmetry and positive definiteness. Print condition number. Also compute `C_TG[k,i] = forms_T[k](representers[i])`. Compute SOLA matrix `X = C_TG @ solve(C_GG + C_D, I)`. Print shape.

  7. **Cell: Property inference** — `p_mean = X @ d_tilde`; `p_cov = X @ C_D @ X.T`; `p_std = sqrt(diag(p_cov))`; print table of true vs estimated vs ±2σ. Plot with compare_function3d_axis_sliders (SOLA mean | 2σ).

  8. **Cell: Unimodularity** — rank-1 correction `X_uni` to enforce `X_uni @ [forms_G[i](ones) for i] = ones`; three comparison plots (unconstrained mean, unconstrained 2σ, unimodular mean | 2σ side-by-side).

  9. **Cell: Resolving kernels** — for 5 selected targets: `resolving_k = Function3D(box, evaluate_callable=lambda x: sum(X_uni[k,i] * representers[i](x) for i))`. Compare resolving kernel vs target Gaussian side-by-side.

### Phase 2: Create companion script `sola_rays.py`
- **Objective:** Script companion matching structure of `sola_noise.py` with argparse, `--no-viz`, `--out-dir`, `--n-sources`, `--resolution` flags and HTML figure output
- **Files to create:** `pygeoinf3D/demos/box/sola/sola_rays.py`
- **Key test:** `python sola_rays.py --no-viz` exits 0; unimodularity check prints `[1. 1. 1. 1. 1.]`
- **Figure outputs:**
  - `ray_geometry.html` — Plotly scatter3d of rays, stations, sources
  - `true_model.html` — z-slider of true model
  - `property_estimate.html` — SOLA mean | 2σ
  - `unimodular_mean.html` — unconstrained vs unimodular mean
  - `unimodular_uncertainty.html` — unconstrained vs unimodular 2σ
  - `unimodular_estimate.html` — unimodular mean | 2σ
  - `resolving_kernels.html` — 5 resolving kernels vs target Gaussians

### Phase 3: Update living reference + `.europa.yml`
- **Objective:** Document new demo in `pygeoinf3d-reference.md`; add `sola_rays_figures/**` to `pull_include`
- **Files to modify:** `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`, `.europa.yml`

---

## Key API Reference

```python
# Imports needed
from pygeoinf3d import BoxDomain, Lebesgue3D, Function3D, SOLAOperator3D
from pygeoinf3d.providers.tensor_product import BoxLaplacianSpectrumProvider
from pygeoinf3d.operators.bessel_sobolev_box import BesselSobolevBox
from pygeoinf3d.spaces.sobolev import Sobolev3D
from pygeoinf3d.spaces.forms import PathIntegralFunctional, VolumeKernel3D
from pygeoinf3d.core.paths import Path3D
from pygeoinf3d.providers.function_providers import GaussianProvider3D
from pygeoinf import EuclideanSpace

# Space construction
box = BoxDomain(np.zeros(3), np.ones(3))
base_space = Lebesgue3D(box)
spec_provider = BoxLaplacianSpectrumProvider(box, truncation=(8,8,8), bc_type="neumann")
W     = BesselSobolevBox(base_space, spec_provider, shift=1.0, order=2.0)
W_inv = BesselSobolevBox(base_space, spec_provider, shift=1.0, order=-2.0)
sobolev = Sobolev3D(base_space, W, W_inv)

# Ray paths (straight rays: source → station)
path = Path3D.from_polyline(np.array([source, station]))
forms_G = [PathIntegralFunctional(sobolev, p, n_quad=200) for p in paths]

# Target forms (Gaussian volume kernels)
gaussian_fn = GaussianProvider3D(box, centers=..., sigmas=0.10, normalized=True).get_function(k)
forms_T = [VolumeKernel3D(base_space, gaussian_fn_k) for k in range(N_p)]

# SOLA computation
representers = [sobolev.from_dual(f) for f in forms_G]
C_GG = np.array([[forms_G[i](representers[j]) for j in range(N_d)] for i in range(N_d)])
C_TG = np.array([[forms_T[k](representers[i]) for i in range(N_d)] for k in range(N_p)])
X = C_TG @ np.linalg.solve(C_GG + C_D, np.eye(N_d))
```

## Open Questions (resolved)
1. ~~Ray count~~ → 30
2. ~~Sobolev order~~ → s=2
3. ~~Station layout~~ → random at z=1 surface
4. ~~Notebook + script~~ → both
