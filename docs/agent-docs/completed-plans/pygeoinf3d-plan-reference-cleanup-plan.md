## Plan: pygeoinf3D Plan And Reference Cleanup

Audit pygeoinf3D and adjacent pygeoinf planning metadata, archive plans that are finished but still marked active, and update references so the remaining active work is explicit. The cleanup keeps active-plans focused on genuinely unfinished work and introduces a legacy bucket for superseded design notes.

**Phases 1**
1. **Phase 1: Archive Finished Plans And Normalize References**
    - **Objective:** Move completed pygeoinf3D plans and their phase-complete files into completed-plans, create a legacy folder for outdated reference notes, and update living references to reflect the true active state.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/docs/agent-docs/active-plans/`, `pygeoinf3D/docs/agent-docs/completed-plans/`, `pygeoinf3D/docs/agent-docs/references/legacy/`, `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`
    - **Tests to Write:** none
    - **Steps:**
        1. Move finished plan files and matching phase-complete files from active-plans to completed-plans.
        2. Create `references/legacy/` in pygeoinf3D and move outdated forward-looking notes there.
        3. Update the living reference so it distinguishes archived work from the remaining active architecture plan.
        4. Verify the final directory state and summarize which plan remains active and how it connects to prior work.

**Open Questions 1**
1. Treat `inverse-box-laplacian-plan.md` and `spectral-operator-kl-sampler-plan.md` as finished-in-substance archival items because code, tests, and living-reference coverage are already present.