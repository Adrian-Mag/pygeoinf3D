## Plan: pygeoinf3D Demo Suite — 6 Core Notebooks

Create six standalone Jupyter notebooks in `demos/`, each a 3D analogue of the corresponding intervalinf numbered demo. The six notebooks form a self-contained tutorial progression from domains to operators. The `5_gradient` notebook is excluded because `GradientOperator3D` is not yet implemented.

**Phases (6 of 6)**

1. **Phase 1: Box Domain Demo** — `demos/1_box_domain_3d_demo.ipynb`
    - **Objective:** Introduce `BoxDomain` and `AxisAlignedBoundingBox` — the geometric foundation
    - **Files/Functions:** `demos/1_box_domain_3d_demo.ipynb`
    - **Steps:**
        1. Construct `BoxDomain` variants (unit cube, scaled, shifted)
        2. Show `contains()`, `volume`, `extents`, `center`, `bounding_box()`
        3. Visualise `sample_interior()` and `sample_boundary()` as 3D scatter plots
        4. Demo `integrate_volume()` on known functions, verify vs analytic result
        5. Demo `structured_mesh()` and `volume_weights()`

2. **Phase 2: Functions Demo** — `demos/2_functions_3d_demo.ipynb`
    - **Objective:** Show `Function3D` creation, evaluation, arithmetic, and plotting
    - **Files/Functions:** `demos/2_functions_3d_demo.ipynb`
    - **Steps:**
        1. Create `Function3D` from callable, show pointwise evaluation
        2. Demo arithmetic: `+`, `-`, `*` scalar and `pointwise_mul()`
        3. Integrate via `box.integrate_volume()`; verify analytically
        4. Visualise z-slices with `plot_function3d_axis_slice` at multiple levels

3. **Phase 3: Lebesgue Space Demo** — `demos/3_lebesgue_space_3d_demo.ipynb`
    - **Objective:** Show `Lebesgue3D` inner product, norm, basis functions, and orthonormality
    - **Files/Functions:** `demos/3_lebesgue_space_3d_demo.ipynb`
    - **Steps:**
        1. Create `Lebesgue3D(box)` in basis-free mode; confirm `dim == 0`
        2. Compute `inner_product(f, g)` and `norm(f)`; verify Parseval/Cauchy–Schwarz
        3. Create `TensorProductProvider` (Dirichlet and Neumann); confirm `dim > 0`
        4. Extract basis functions via `prov.get_function(i)`, confirm orthonormality
        5. Manual projection: `c_i = inner_product(phi_i, f)`, reconstruct, plot error
        6. Visualise Gram matrix

4. **Phase 4: Kernel Functionals Demo** — `demos/3.1_kernel_functionals_3d_demo.ipynb`
    - **Objective:** Show `VolumeKernel3D` — kernel-based linear forms — and Riesz duality
    - **Files/Functions:** `demos/3.1_kernel_functionals_3d_demo.ipynb`
    - **Steps:**
        1. Construct `VolumeKernel3D` from Gaussian kernel; evaluate on test functions
        2. Verify linearity numerically
        3. Build a `VolumeKernel3D` sum (SOLA-style) and apply to a 3D model
        4. Show Riesz map: `space.riesz(f)` produces the dual element of `f`

5. **Phase 5: Providers Demo** — `demos/4_providers_3d_demo.ipynb`
    - **Objective:** Survey `TensorProductProvider`, `GaussianProvider3D`, `BumpProvider3D`, `FunctionToKernelAdapter`
    - **Files/Functions:** `demos/4_providers_3d_demo.ipynb`
    - **Steps:**
        1. `TensorProductProvider`: Dirichlet vs Neumann, visualise first 9 eigenfunctions
        2. `GaussianProvider3D`: regular grid of sensing kernels, visualise as overlapping slices
        3. `BumpProvider3D`: bump functions at a 2×2×2 target grid
        4. `FunctionToKernelAdapter`: wrap custom callable into kernel provider
        5. Show all providers satisfy `get_function(i)` → `Function3D` and `dim` property

6. **Phase 6: Laplacian Demo** — `demos/6_laplacian_3d_demo.ipynb`
    - **Objective:** Show the spectral Laplacian family — `BoxLaplacian`, `InverseBoxLaplacian`, `GeneralizedBoxLaplacian`
    - **Files/Functions:** `demos/6_laplacian_3d_demo.ipynb`
    - **Steps:**
        1. `BoxLaplacian`: extract and plot eigenvalue spectrum, visualise first eigenfunctions
        2. Verify Δφ_n ≈ -λ_n φ_n numerically on a test function
        3. `InverseBoxLaplacian`: show 1/λ_n spectra, confirm inverse relationship
        4. `GeneralizedBoxLaplacian(power=-1, alpha)` for several α values; compare spectra
        5. Draw prior samples via `KLSampler3D` for different α; compare visual roughness

**Open Questions**
1. Keep/supersede the partial `generalized_laplacian_covariance_demo.ipynb` in Phase 6, or reference it?
2. Gradient demo placeholder notebook for future `GradientOperator3D` implementation?
