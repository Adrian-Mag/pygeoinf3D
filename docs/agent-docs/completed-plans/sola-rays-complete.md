## Plan Complete: SOLA Demo with Sobolev H² Model Space and Straight-Ray Data

Created `sola_rays.ipynb` and companion `sola_rays.py` in `pygeoinf3D/demos/box/sola/`, demonstrating the SOLA method with a Sobolev H² model space and straight-ray path-integral measurements. The implementation mirrors `sola_noise.ipynb/py` in structure but replaces L² Gaussian kernels with `PathIntegralFunctional` objects derived from 30 random seismic rays connecting sources in the volume to stations randomly placed on the `z=1` surface. All code-review issues (fragile bootstrap, wrong true model centres, incorrect unimodularity formula) were identified and resolved.

**Phases Completed:** 3 of 3
1. ✅ Phase 1: Implement `sola_rays.ipynb` (notebook with 8 markdown + 8 code cells)
2. ✅ Phase 2: Create companion script `sola_rays.py` (argparse, `--no-viz`, 7 HTML outputs)
3. ✅ Phase 3: Update living reference + `.europa.yml` pull_include

**All Files Created/Modified:**
- `pygeoinf3D/demos/box/sola/sola_rays.ipynb` — interactive notebook (8 code + 8 markdown cells)
- `pygeoinf3D/demos/box/sola/sola_rays.py` — script companion with argparse flags
- `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md` — updated with both new files

**Key Functions/Classes Used:**
- `Sobolev3D(base_space, W, W_inv)` — H² model space
- `BesselSobolevBox(base_space, spec_provider, shift=1.0, order=±2.0)` — Bessel–Sobolev weight operator
- `BoxLaplacianSpectrumProvider(box, truncation=(8,8,8), bc_type="neumann")` — 512 spectral modes
- `PathIntegralFunctional(sobolev, path, n_quad=200)` — forward form (line integral on Sobolev3D)
- `Path3D.from_polyline(np.array([source, station]))` — straight-ray path
- `VolumeKernel3D(base_space, gaussian_fn)` — target functional (L² Gaussian kernel)
- `GaussianProvider3D(box, centers=target_grid, sigmas=0.10, normalized=True)` — target Gaussians
- `sobolev.from_dual(f)` — Sobolev representer of forward form
- Unimodularity rank-1 correction: `u = solve(C_GG+C_D, v); X_uni = X - outer((Xv−w)/β, u)`
- `compare_panels_axis_dropdown_sliders` — resolving kernels vs target Gaussian viewer

**Test Coverage:**
- Total tests written: 0 (demo, not production code)
- Script validation: `python sola_rays.py --no-viz` exits 0, runtime ~18s
- Unimodularity check: `X_uni @ v = [1. 1. 1. 1. 1.]` (first 5 shown) ✅

**Code Review Issues Fixed:**
1. Fragile bootstrap (`Path().resolve().parents[3]`) → robust `_find_workspace_root()` upward-search
2. Wrong true model Gaussian centres → corrected to match `sola_noise.py` exactly (`[0.30,0.30,0.50]`, `[0.70,0.65,0.55]`, `[0.55,0.40,0.45]`)
3. Wrong unimodularity formula (Euclidean `v@v` denominator) → correct metric-weighted formula using `(C_GG+C_D)^{-1}`
4. Removed unused imports (`EuclideanSpace`, `SOLAOperator3D`)

**Recommendations for Next Steps:**
- Run on europa for full-resolution figures (`--resolution 80`, default 30 sources)
- Consider a 5×5×5 target grid (N_p=125) for denser spatial coverage
- Add a `sola_rays_anisotropic.py` variant testing non-uniform ray coverage patterns
