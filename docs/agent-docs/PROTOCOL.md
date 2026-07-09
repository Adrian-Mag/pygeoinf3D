# Agent-Docs Protocol

Version 2026-07-09. This directory is the durable record of agent-assisted work in this
repository: what was planned, why changes happened, and how the code is organised. It is
**provider-neutral** — any coding agent (Claude Code, Codex, Cascade, …) and any human
follows the same rules. This file is the contract; if a tool's instructions conflict with
it, this file wins.

## Layout

```
docs/agent-docs/
  PROTOCOL.md          this file
  active-plans/        plans currently being executed
  completed-plans/     finished plans + phase-completion summaries (append-only)
  references/living/   curated, up-to-date architecture/API references
  references/legacy/   archived references — never consult unless explicitly asked
```

## Plans

- One plan per substantial unit of work (feature, refactor, campaign, paper section).
- Filename: `<kebab-case-name>-plan.md`, created in `active-plans/` when the plan is approved.
- If the plan came from a harness planning mode (e.g. Claude plan mode), persisting it here
  verbatim is the **first execution step** — harness plan storage is ephemeral.
- Template:

  ```markdown
  ## Plan: <title>
  **Status:** active | completed | abandoned   **Created:** YYYY-MM-DD

  ### Goal
  <2–4 sentences>
  ### Design notes
  <key decisions and why — the "why" is what future readers need>
  ### Files / components
  ### Phases
  <numbered; each phase independently committable>
  ### Open questions
  ```

- **Phase completion:** when a phase lands, write
  `completed-plans/<name>-phase-N-complete.md`: what was done, deviations from the plan,
  evidence (tests run, builds, figures).
- **Plan completion:** move the plan file to `completed-plans/`, set Status, append a short
  closing note (what changed vs the original plan, follow-ups spawned).
- **Abandonment:** move to `completed-plans/` prefixed `abandoned-`, with a note saying why.
- **Never delete a plan.** The history is the point.

## Living references

- `references/living/<topic>-reference.md`: scope, architecture, core classes/functions
  with construction examples, invariants, file map.
- **Read all living references before exploring the code they cover** (they are cheaper
  and more reliable than re-deriving architecture from source).
- **Update the affected living reference in the same unit of work as the code change.**
  The update is reviewed together with the code — never bulk-generated after the fact,
  which is how references stop being trustworthy.
- Never consult `references/legacy/`.

## Commit traceability

Feature/fix/refactor commits reference the plan that directed them:

```
<type>(<scope>): <subject>

- bullets describing the changes

Plan: docs/agent-docs/active-plans/<name>-plan.md
Phase: <N> of <M>                                            (omit if single-phase)
Related: docs/agent-docs/completed-plans/<name>-phase-N-complete.md   (when it exists)
```

- Types `feat` / `fix` / `refactor` require a `Plan:` trailer; `docs` / `chore` / `test`
  do not. Merge, revert, and fixup commits are exempt.
- The `Plan:` path is workspace-relative and points at the plan's location **at commit
  time** (plans later move to `completed-plans/`; historical trailers are not rewritten).
- A **warn-only** `commit-msg` hook (`.githooks/commit-msg`) checks the trailer. Activate
  it once per clone:

  ```bash
  git config core.hooksPath .githooks
  ```

## Hygiene

Monthly, or when a session notices drift:

- `active-plans/` entries untouched for >30 days: complete, abandon, or consciously
  re-activate each one.
- Spot-check one living reference against the code it covers; fix drift in place.
- Plans exist in `completed-plans/` for every `Plan:` trailer in recent history.
