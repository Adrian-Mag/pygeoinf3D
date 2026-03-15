## Plan Complete: pygeoinf3D Demo Suite — 6 Core Notebooks

Six standalone Jupyter notebooks were implemented in `demos/`, forming a self-contained tutorial progression that mirrors the intervalinf numbered demo series. Each notebook was written, API-verified against the live package, and executed to completion without errors. All API discrepancies discovered during implementation have been documented.

**Phases Completed:** 6 of 6
1. ✅ Phase 1: Box Domain Demo — `demos/1_box_domain_3d_demo.ipynb`
2. ✅ Phase 2: Functions Demo — `demos/2_functions_3d_demo.ipynb`
3. ✅ Phase 3: Lebesgue Space Demo — `demos/3_lebesgue_space_3d_demo.ipynb`
4. ✅ Phase 4: Kernel Functionals Demo — `demos/3.1_kernel_functionals_3d_demo.ipynb`
5. ✅ Phase 5: Providers Demo — `demos/4_providers_3d_demo.ipynb`
6. ✅ Phase 6: Laplacian Demo — `demos/6_laplacian_3d_demo.ipynb`

**All Files Created/Modified:**
- `demos/1_box_domain_3d_demo.ipynb`
- `demos/2_functions_3d_demo.ipynb`
- `demos/3_lebesgue_space_3d_demo.ipynb`
- `demos/3.1_kernel_functionals_3d_demo.ipynb`
- `demos/4_providers_3d_demo.ipynb`
- `demos/6_laplacian_3d_demo.ipynb`
- `docs/agent-docs/references/living/pygeoinf3d-reference.md` (updated)

**Key APIs Demonstrated Per Notebook:**

| Notebook | Key Classes/Functions |
|---|---|
| 1_box_domain | `BoxDomain`, `contains`, `integrate_volume`, `structured_mesh`, `volume_weights`, `sample_interior`, `boundary_components` |
| 2_functions | `Function3D`, `+`, `-`, `pointwise_mul`, `integrate_volume`, `plot_function3d_axis_slice` |
| 3_lebesgue_space | `Lebesgue3D`, `inner_product`, `norm`, `TensorProductProvider`, manual projection |
| 3.1_kernel_functionals | `VolumeKernel3D`, `to_dual`, `riesz`, linearity/SOLA multi-kernel |
| 4_providers | `TensorProductProvider`, `GaussianProvider3D`, `BumpProvider3D`, `FunctionToKernelAdapter` |
| 6_laplacian | `BoxLaplacian`, `InverseBoxLaplacian`, `GeneralizedBoxLaplacian`, `KLSampler3D` |

**Confirmed API Corrections (discovered during implementation):**

| Spec / Assumption | Actual API |
|---|---|
| `BoxDomain(low=..., high=...)` | Positional only: `BoxDomain(np.zeros(3), np.ones(3))` |
| `Function3D(box, callable)` | `Function3D(box, evaluate_callable=fn)` |
| Callable `fn(x,y,z) → float` | Must be vectorised: `fn(pts:(N,3)) → (N,) array` |
| `from pygeoinf3d.visualization import ...` | `from pygeoinf3d.plot import plot_function3d_axis_slice` |
| `space.riesz(f)` = primal→dual | `space.to_dual(f)` = primal→dual; `space.riesz(vk)` = dual→primal |
| `OrthonormalProjectionBackend` works with TensorProductProvider | TensorProductProvider has `get_eigenfunction` not `get_basis_function` → `to_components` fails; use manual projection loops |
| `sample_boundary()` works | Raises `NotImplementedError` → use face-by-face manual sampling |
| `BoxLaplacian(provider)` | `BoxLaplacian(space, provider)` — requires both |
| `KLSampler3D(operator)` | `KLSampler3D(operator, n_modes=N)` — `n_modes` required |
| `GeneralizedBoxLaplacian` formula: (α+λ)^p | Formula is (-αΔ)^s so eigenvalues = (α·λ_k)^s; power must be integer |

**Test Coverage:**
- Total notebooks with embedded output: 6
- All notebooks execute cleanly (0 errors)

**Recommendations for Next Steps:**
- Implement `5_gradient_3d_demo.ipynb` once `GradientOperator3D` is implemented
- Fix `OrthonormalProjectionBackend` to accept `TensorProductProvider` (currently `get_basis_function` name mismatch)
- Add `sample_boundary()` implementation to `BoxDomain`
- Create an index notebook or README linking all demos together
