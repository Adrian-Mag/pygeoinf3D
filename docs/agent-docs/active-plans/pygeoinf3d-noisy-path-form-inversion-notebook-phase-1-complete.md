## Phase 1 Complete: Scaffold Notebook And Path Geometry

Created the new noisy path-form inversion notebook scaffold and validated the Phase 1 cells end-to-end. The notebook now contains the introduction, low-order Sobolev model-space construction, randomized boundary-to-top acquisition geometry, geometry diagnostics, and interactive path visualization, with the living reference updated to list the new demo as a Phase 1 scaffold.

**Files created/changed:**
- `pygeoinf3D/demos/noisy_path_form_least_norm_inversion_demo.ipynb`
- `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`

**Functions created/changed:**
- Notebook setup cells for `BoxDomain`, `Lebesgue3D`, `BoxLaplacianSpectrumProvider`, `BesselSobolevBox`, and `Sobolev3D`
- Notebook geometry cells for random boundary-face source sampling, top-face receiver sampling, and `Path3D.from_polyline(..., domain=box)` construction
- Notebook visualization cell for `plot_paths3d_interactive`

**Tests created/changed:**
- Notebook Phase 1 cells executed successfully through the geometry diagnostics and interactive path plot
- Geometry assertions verify all sources avoid the top face and all receivers lie on the top face
- Review confirmed the notebook remains within Phase 1 scope and uses the current Sobolev/path APIs correctly

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
feat(demos): scaffold noisy path-form inversion notebook

- Add a new pygeoinf3D notebook for noisy boundary-to-top path acquisition geometry
- Build the low-order Sobolev3D model space and validate Phase 1 geometry cells
- Update the living reference to list the new notebook as a Phase 1 scaffold

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-noisy-path-form-inversion-notebook-plan.md
Phase: 1 of 4
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-noisy-path-form-inversion-notebook-phase-1-complete.md