## Phase 1 Complete: Create And Validate Least-Norm Demo Notebook

Added a runnable notebook under `pygeoinf3D/demos/` showing a noiseless minimum-norm inversion driven by `pygeoinf` and instantiated with a `pygeoinf3d.SOLAOperator3D`. The notebook was executed end-to-end, producing a `Function3D` least-norm solution, a near-machine-precision residual, and slice plots for the true model, recovered model, and a representative kernel.

**Files created/changed:**
- pygeoinf3D/demos/least_norm_inversion_demo.ipynb
- pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md
- pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-least-norm-demo-plan.md

**Functions created/changed:**
- No library functions changed
- Notebook workflow covers `LinearForwardProblem`, `LinearMinimumNormInversion`, `EigenSolver`, `FunctionToKernelAdapter`, and `SOLAOperator3D`

**Tests created/changed:**
- No test files changed
- Notebook execution validated successfully
- Residual norm: `6.422e-16`
- Explicit pseudoinverse verification mismatch: `9.992e-16`

**Review Status:** APPROVED with minor recommendations addressed

**Git Commit Message:**
feat(demos): add least-norm SOLA inversion notebook

- Add a runnable pygeoinf3D demo notebook for noiseless minimum-norm inversion with Gaussian volumetric kernels
- Validate the notebook end-to-end with pygeoinf's LinearForwardProblem and LinearMinimumNormInversion workflow
- Update the living reference to include the new demo and its inversion path

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-least-norm-demo-plan.md
Phase: 1 of 1
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-least-norm-demo-phase-1-complete.md
