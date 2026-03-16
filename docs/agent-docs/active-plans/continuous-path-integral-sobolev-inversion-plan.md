## Plan: Continuous Path-Integral Sobolev Inversion

Build a continuous-first path-integral inversion stack in `pygeoinf3D` centered on an explicit `Sobolev3D` wrapper, generalized measurement operators, and path functionals that are dual objects first and primal representers second. The first representer backend will use the existing box spectral infrastructure, but the design will record that this is only the initial realization path and not the long-term general definition.

**Phases 5**
1. **Phase 1: Add `Sobolev3D` Wrapper**
    - **Objective:** Implement an explicit `Sobolev3D` space in `pygeoinf3D` rather than relying on a bare weighted-space wrapper.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/pygeoinf3d/spaces/`, `pygeoinf3D/pygeoinf3d/__init__.py`, `pygeoinf3D/tests/spaces/`, `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`
    - **Tests to Write:** constructor validation, `to_dual` / `from_dual` round-trip, norm / inner-product consistency, compatibility with `BesselSobolevBox`, correct Riesz semantics for later functional representers
    - **Steps:**
        1. Write failing tests for an explicit `Sobolev3D` wrapper over `Lebesgue3D` plus a positive self-adjoint weight operator.
        2. Implement `Sobolev3D` as a first-class 3D space with explicit access to its base space and weight operator.
        3. Export the class through `pygeoinf3d.spaces` and the package root.
        4. Run focused tests and update the living reference.
2. **Phase 2: Add General Path Geometry Input Layer**
    - **Objective:** Support paths defined by smooth parameterizations, polylines, or point samples with interpolation.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/pygeoinf3d/core/` or `pygeoinf3D/pygeoinf3d/geometries/`, tests, living reference
    - **Tests to Write:** polyline construction, smooth parameterized path construction, point-sample interpolation construction, path containment/domain checks, arclength quadrature consistency, reparameterization sanity tests
    - **Steps:**
        1. Write failing tests for the three approved path input modes: continuous parameterization, polylines, and ordered point samples.
        2. Implement a unified path object API normalized to an arclength-based interface.
        3. Add interpolation support for point-defined paths with a conservative default suitable for a first release.
        4. Run focused tests and update the living reference.
3. **Phase 3: Add Path-Integral Dual Functionals**
    - **Objective:** Introduce general path-integral linear functionals as dual objects, not volume kernels.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/pygeoinf3d/spaces/forms.py` or adjacent dual-functional module, tests, living reference
    - **Tests to Write:** path-integral evaluation on analytic functions, weighted and unweighted path integrals, correct separation between dual functional and primal representer, domain/type validation
    - **Steps:**
        1. Write failing tests for a path-integral functional that evaluates by line integration over a normalized path object.
        2. Implement a general, non-seismology-specific dual functional API.
        3. Keep the functional separate from any Sobolev representer realization.
        4. Run focused tests and update the living reference.
4. **Phase 4: Generalize `SOLAOperator3D`**
    - **Objective:** Extend `SOLAOperator3D` so it works with general continuous linear functionals, not only `VolumeKernel3D`.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/pygeoinf3d/operators/sola.py`, provider contracts if needed, tests, living reference
    - **Tests to Write:** forward mapping with path functionals, dual mapping, adjoint correctness under arbitrary finite-dimensional data codomains, backward compatibility for existing volume-kernel workflows
    - **Steps:**
        1. Write failing tests showing `SOLAOperator3D` accepts both existing volume kernels and new path functionals.
        2. Generalize type contracts away from `VolumeKernel3D`-only assumptions while preserving the basis-free forward design.
        3. Ensure adjoints route through domain `from_dual` semantics rather than assuming an `L^2` representer.
        4. Run focused tests and update the living reference.
5. **Phase 5: Add Spectral Sobolev Representers And End-To-End Minimum-Norm Demo**
    - **Objective:** Realize path duals as volumetric Sobolev representers using the box spectral backend, and demonstrate least-norm inversion.
    - **Files/Functions to Modify/Create:** `Sobolev3D` representer logic, demo notebook/script, integration tests, living reference
    - **Tests to Write:** representer identity `ℓ_γ(v) = <r_γ, v>_{H^s}`, spectral coefficient formula checks, end-to-end noiseless minimum-norm inversion using `LinearForwardProblem` and `LinearMinimumNormInversion`
    - **Steps:**
        1. Write failing tests for inverse-Riesz realization of path duals in `Sobolev3D`.
        2. Implement the first representer backend using `BesselSobolevBox` on `BoxDomain`.
        3. Document explicitly that box spectral realization is the first backend only, leaving room for future PDE/FEM/general-domain representer backends.
        4. Build a demo using the existing `pygeoinf` inversion stack.
        5. Run focused tests and update the living reference.

**Open Questions 2**
1. For point-sampled paths, should the first interpolation default be piecewise linear or spline-based? Proposed first implementation: piecewise linear, with smoother interpolation deferred.
2. Should the first `Sobolev3D` convenience API accept a prebuilt weight operator only, or also provide a helper constructor from `BesselSobolevBox` ingredients (`provider`, `shift`, `order`)?
