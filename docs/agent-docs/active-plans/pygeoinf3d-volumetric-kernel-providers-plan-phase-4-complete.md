## Phase 4 Complete: Document SOLA Kernel Workflows

Updated the package documentation to describe the implemented volumetric-kernel workflow from provider families through `FunctionToKernelAdapter` into `SOLAOperator3D`, and clarified the current volume-kernel-only boundary. The README now reflects the current package surface instead of the original skeleton state, and the living reference includes a canonical end-to-end example aligned with the current API.

**Files created/changed:**
- README.md
- docs/agent-docs/references/living/pygeoinf3d-reference.md

**Functions created/changed:**
- `FunctionToKernelAdapter` documentation
- `SOLAOperator3D` workflow documentation
- `evaluate_function3d_on_axis_slice` workflow documentation
- `plot_function3d_axis_slice` workflow documentation

**Tests created/changed:**
- No test files changed
- Verified full suite: `424 passed in 5.42s`

**Review Status:** APPROVED

**Git Commit Message:**
docs(pygeoinf3d): document volumetric SOLA workflow

- Refresh the README to match the implemented box-domain, provider, and SOLA surface
- Add a canonical provider-to-kernel-to-SOLA workflow and volumetric boundary notes to the living reference
- Reconfirm the published test count with the full pygeoinf3D suite

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-volumetric-kernel-providers-plan.md
Phase: 4 of 4
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-volumetric-kernel-providers-plan-phase-4-complete.md