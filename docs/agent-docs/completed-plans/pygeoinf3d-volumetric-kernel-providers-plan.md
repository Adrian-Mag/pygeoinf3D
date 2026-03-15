## Plan: pygeoinf3D Volumetric Kernel Providers

Add user-facing volumetric kernel convenience classes for `SOLAOperator3D` by introducing indexed `Function3D` providers under `pygeoinf3d/providers/function_providers/`, with each provider in its own file. The plan mirrors the successful `intervalinf` provider pattern, keeps the current volumetric-only adjoint boundary intact, and adds slice-based visualization as the initial user-facing inspection layer.

**Phases 4**
1. **Phase 1: Add Function-Provider Package Skeleton**
    - **Objective:** Create the `pygeoinf3d/providers/function_providers/` package and establish the shared provider conventions for user-facing volumetric kernel families.
    - **Files/Functions to Modify/Create:** `pygeoinf3d/providers/function_providers/__init__.py`, one shared support file if needed, `pygeoinf3d/providers/__init__.py`, `pygeoinf3d/__init__.py`, tests under `tests/providers/`
    - **Tests to Write:** import/export tests, provider construction tests, generic `get_function(i)` contract tests
    - **Steps:**
        1. Write failing tests for the new `function_providers` package layout and public re-exports.
        2. Create the package with one-provider-per-file structure and minimal shared conventions aligned with `FunctionProvider3D`.
        3. Run provider tests to confirm the package surface is stable before adding concrete families.

2. **Phase 2: Add First Volumetric Function Providers**
    - **Objective:** Introduce the first user-facing volumetric kernel families as `Function3D`-producing providers that are easy to promote into `VolumeKernel3D` and use in `SOLAOperator3D`.
    - **Files/Functions to Modify/Create:** `pygeoinf3d/providers/function_providers/gaussian.py`, `pygeoinf3d/providers/function_providers/bump.py`, `pygeoinf3d/providers/function_providers/boxcar.py`, `pygeoinf3d/providers/function_providers/__init__.py`, tests under `tests/providers/`
    - **Tests to Write:** pointwise evaluation tests, batch evaluation tests, parameter validation tests, provider indexing tests, compatibility tests with `FunctionToKernelAdapter`
    - **Steps:**
        1. Write failing tests for Gaussian, bump, and boxcar function providers on `BoxDomain` / `Lebesgue3D`.
        2. Implement each provider in its own file, returning concrete `Function3D` instances with stable parameter semantics.
        3. Run provider and focused SOLA tests to verify the generated functions can be wrapped into volumetric kernels cleanly.

3. **Phase 3: Add Visualization Utilities For Volumetric Kernels**
    - **Objective:** Give users straightforward ways to inspect provider-generated volumetric kernels through slice-based plots.
    - **Files/Functions to Modify/Create:** new `pygeoinf3d/plot.py` or `pygeoinf3d/visualization.py`, optional lightweight plotting helpers in provider files, `pygeoinf3d/__init__.py`, tests under `tests/`
    - **Tests to Write:** slice sampling correctness tests, plotting API smoke tests, figure/axes return contract tests
    - **Steps:**
        1. Write failing tests for a minimal plotting API that evaluates a `Function3D` on planar slices of a structured 3D domain.
        2. Implement slice plotting utilities aligned with existing `pygeoinf` / `intervalinf` plotting conventions.
        3. Run plotting and provider tests to confirm the visualization layer is ergonomic but separate from the mathematical core.

4. **Phase 4: Document SOLA Kernel Workflows**
    - **Objective:** Document the new provider families, the recommended path from provider to `VolumeKernel3D` to `SOLAOperator3D`, and the current volumetric-only boundary.
    - **Files/Functions to Modify/Create:** `docs/agent-docs/references/living/pygeoinf3d-reference.md`, optionally `README.md`, export tests if public symbols are added
    - **Tests to Write:** export-lock tests for any newly public provider symbols
    - **Steps:**
        1. Write or extend failing export tests if the new providers become part of the top-level public API.
        2. Update the living reference to describe the new `function_providers` package, visualization entry points, and the SOLA usage path.
        3. Run provider, export, and SOLA tests to keep documentation and public API synchronized.

**Open Questions 1**
1. Default MVP assumes slice-based visualization first; full 3D voxel or isosurface views can be added later if needed.