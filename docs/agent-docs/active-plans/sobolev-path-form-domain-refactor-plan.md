## Plan: Sobolev Path Form Domain Refactor

Refactor path-integral measurements so their declared `LinearForm.domain` matches the actual Hilbert space of continuity, while preserving a usable route for `SOLAOperator3D` to support both `Lebesgue3D` and `Sobolev3D` domains according to the measurement stack. The work is phased so the domain semantics are corrected first, then operator plumbing and provider-facing contracts are updated incrementally with tests at each step.

**Phases 4 phases**
1. **Phase 1: Re-anchor Path Forms**
    - **Objective:** Make `PathIntegralFunctional` genuinely Sobolev-based and generalize `SumLinearForm` so dual sums are not hard-wired to `Lebesgue3D`.
    - **Files/Functions to Modify/Create:** `pygeoinf3d/spaces/forms.py` (`PathIntegralFunctional`, `SumLinearForm`), `tests/spaces/test_path_integral_functional.py`, `tests/spaces/test_sobolev.py`
    - **Tests to Write:** constructor rejects `Lebesgue3D` for path forms; constructor accepts `Sobolev3D`; sum-linear-form accepts Sobolev domain; existing Sobolev representer tests updated to use Sobolev-domain path forms
    - **Steps:**
        1. Add failing tests for Sobolev-only `PathIntegralFunctional` construction and domain-generic `SumLinearForm` behaviour.
        2. Refactor `PathIntegralFunctional` to require `Sobolev3D` and update internal space/type handling.
        3. Refactor `SumLinearForm` to validate against a generic Hilbert-space domain rather than only `Lebesgue3D`.
        4. Run the targeted space-level tests and confirm they pass.
2. **Phase 2: Re-anchor Sobolev Dual Semantics**
    - **Objective:** Make `Sobolev3D.to_dual` and `Sobolev3D.from_dual` operate in the Sobolev dual space instead of the base Lebesgue space.
    - **Files/Functions to Modify/Create:** `pygeoinf3d/spaces/sobolev.py`, `pygeoinf3d/spaces/forms.py`, `tests/spaces/test_sobolev.py`
    - **Tests to Write:** `to_dual` returns Sobolev-domain duals; `from_dual` accepts Sobolev-domain volume/path/sum forms; legacy Lebesgue-domain assumptions removed
    - **Steps:**
        1. Add failing tests for Sobolev-domain dual objects.
        2. Refactor `Sobolev3D.to_dual` to return dual forms anchored on `self`.
        3. Refactor `Sobolev3D.from_dual` domain checks and representer branches to consume Sobolev-domain duals.
        4. Run targeted Sobolev tests and confirm they pass.
3. **Phase 3: Tighten SOLA Domain Contracts**
    - **Objective:** Make `SOLAOperator3D` enforce that measurement forms live on the operator domain, with path forms requiring a Sobolev domain.
    - **Files/Functions to Modify/Create:** `pygeoinf3d/operators/sola.py`, `tests/operators/test_sola.py`, `demos/path_integral_minimum_norm_demo.py`
    - **Tests to Write:** path-form stacks reject `Lebesgue3D` operators; mixed Sobolev-domain stacks succeed; dual mapping returns domain-consistent sum forms
    - **Steps:**
        1. Add failing operator tests for domain-form identity and path-form Sobolev requirements.
        2. Refactor constructor validation and dual assembly to use the actual operator domain.
        3. Update the path-integral demo to construct forms on Sobolev spaces.
        4. Run targeted operator tests and confirm they pass.
4. **Phase 4: Finalize Public Contracts**
    - **Objective:** Update provider/documentation/reference surfaces to reflect the corrected domain semantics and any remaining constraints.
    - **Files/Functions to Modify/Create:** `pygeoinf3d/providers/base.py`, `pygeoinf3d/README.md`, `pygeoinf3d/docs/agent-docs/references/living/pygeoinf3d-reference.md`, relevant export files and demos if required
    - **Tests to Write:** provider/doc-facing contract tests if needed; public API assertions for updated signatures and error messages
    - **Steps:**
        1. Add failing documentation/export tests if the package locks these contracts.
        2. Update provider contracts or explicitly document remaining Lebesgue-only provider mode.
        3. Refresh README and living reference to describe the corrected path-form/SOLA model.
        4. Run the relevant targeted tests and confirm they pass.

**Open Questions 3 questions**
1. Should `VolumeKernel3D` remain constructible on both `Lebesgue3D` and `Sobolev3D`, or should Sobolev volume measurements use a distinct form class?
2. Do we want a temporary compatibility shim in `Sobolev3D.from_dual` for legacy base-space duals, or should the break be immediate?
3. Should provider-backed `SOLAOperator3D` remain Lebesgue-only for now, or be generalized to return forms on the operator domain?