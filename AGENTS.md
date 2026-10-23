# Windsurf/Cascade Instructions for pygeoinf3D

<!-- BEGIN two-machine-workflow: managed by thesis/scripts/apply-agent-rules.py from thesis/docs/agent-docs/references/living/two-machine-workflow.md; edit the source and re-run -->

## Two-machine workflow: laptop + europa (author rules, 2026-09-16)

The author works on two machines the way a two-person team would:

- **laptop**: repos in `~/PhD/<repo>`. Laptop agents edit code here.
- **europa** (office PC, main machine): repos in `/disks/data/phd/<repo>`. These are full git clones,
  where the author and europa agents edit. Heavy jobs always run on europa.

Each machine has its own clone. **Git is the only way code moves between the two clones.** Never
copy code between them with rsync or scp. Never work on the same branch on both machines at once.

### Start of every session (both machines)
1. Run `git status`, then `git fetch <sync remote>` (see the table below). If the current branch is
   behind, run `git pull --ff-only`. If it has diverged, or the tree holds uncommitted changes you
   did not make, stop and ask the author.
2. Work on a task branch (`<area>/<task>`), not directly on `main`, unless the author says otherwise.
   Check with the author before taking a branch the other machine may be using.

### While working
3. Make small, coherent commits (WIP commits are fine). Follow the repo's commit conventions.
4. Code and small text files go through git. Large outputs, data, pickles, generated PDFs and
   build products stay gitignored and live on europa.

5. **Branch collision check.** A branch whose tip on the sync remote moved in the last 12 hours
   and was not pushed by you is in use on the other machine: ask the author before checking it
   out or committing to it.

### Before stopping or handing over to the other machine
6. Commit, then `git push <sync remote> <branch>` without force. Tell the author the branch name.
7. Merge into `main` only when the author asks, and only after the thesis or paper builds cleanly.

### Git network permission
This is a narrow exception to the offline and no-GitHub rules elsewhere in this file.
- **Allowed:** `git fetch`, `git pull --ff-only` and `git push` to this repo's sync remote in the
  table. Never `--force`, never delete remote branches, never push tags without the author's
  request.
- **Not allowed:** other remotes, `gh` commands (except `gh auth status`), uploads, or anything
  else the offline rules forbid.

| Repo | Sync remote | Notes |
|---|---|---|
| thesis | `origin` (Adrian-Mag/thesis) | private |
| BGP_Paper | `origin` (Adrian-Mag/BGP_Paper) | private |
| PLI_paper | `origin` (Adrian-Mag/PLI_Paper) | private |
| Inferences | `origin` (Adrian-Mag/Inferences) | private |
| Inferences/intervalinf, intervalinf-stabilize worktree | `origin` (Adrian-Mag/intervalinf) | **public**; work branches approved by the author |
| Inferences/pygeoinf and its worktrees | `backup` (Adrian-Mag/pygeoinf-backup) | private; **never push to `origin` (da380, third party)** |
| Every other repo (pygeoinf3D, AxiSEM3D and its sub-repos, Ascension, Adrian-Mag.github.io, control-testing, ...) | none | commit only; the author syncs |

`backup/wip-<date>-<machine>-<branch>` branches are written automatically by the hourly snapshot
script (`thesis/scripts/backup-snapshot.sh`). Never push, delete or base work on them.

### Jobs on europa (europa CLI v2, 2026-09-17)
- **One working tree on europa per project: the git clone `/disks/data/phd/<project>`.** Jobs from
  either machine run there and write their outputs there. `/disks/data/PhD` (capital) was retired on
  2026-09-17; never recreate or use it.
- **From the laptop:** commit, push the branch to the sync remote, then `europa submit '<cmd>'`. It
  syncs europa's clone to your exact pushed commit (refusing on unpushed commits, uncommitted
  changes, a running job, other processes working in the clone, or divergence) and starts the job
  there. If europa's clone is on another branch it refuses unless you pass `--switch`; ask the author
  first, because someone may be using that branch on europa. `europa push` is retired.
- **Library repos are not synced by submit.** A job imports europa's own checkouts listed as
  `libraries` in `.europa.yml`. For thesis, BGP_Paper and PLI_paper those are the worktrees
  `Inferences/intervalinf-stabilize`, `Inferences/pygeoinf-intervalinf-runtime`, `Inferences/pygeoinf3D` and `SOLA_DLI`, not the plain `intervalinf` and `pygeoinf` checkouts. Submit warns when they differ from
  the laptop's and records their commits in `europa_job.info`. To run library changes, push them and
  bring that worktree up to date on europa by git first.
- **On europa itself:** `europa claim`, `submit`, `status` and `release` inside
  `/disks/data/phd/<project>` (local mode).
- One job slot for the whole machine. Always `claim` before submitting and `release` afterwards.
- **Python env:** both machines' jobs map `inferences2` in `.europa.yml` to
  `/disks/data/envs/inferences-main`, the exact copy of the laptop `inferences` env.

### Results and watchers
- **Big outputs stay on europa**, in the clone (backed up nightly); the laptop pulls a specific
  path with `europa pull <path>` when it needs one. Text outputs (logs, STATUS/REPORT notes, small JSON)
  are pulled or committed if the project tracks them.
- **Every job longer than about an hour gets a watcher agent on europa**, launched with
  `europa-watch` from a mission written from the watcher template. A watcher runs under
  `claude -p`, never ends its turn while the job lives, waits with `europa-wait` (never a bare
  `sleep`, a background task or Monitor), and does no git. Details: the `europa` and
  `long-missions` skills (source `thesis/tools/claude-skills/`, installed on both machines by
  `thesis/scripts/install-claude-skills.sh`) and
  `references/living/two-machine-operations.md`.

### Not synced between machines
- **Zotero:** undecided (2026-09-16). Agents must not add, edit or delete Zotero items on either
  machine; read-only lookups are fine.
- **Claude memory:** separate on each machine. **Skills and CLI:** the `europa` and
  `long-missions` skills and the `europa`, `europa-watch` and `europa-wait` commands are versioned in `thesis/tools/`
  and installed on both machines by `thesis/scripts/install-claude-skills.sh` (`--check` reports
  drift); other skills are separate.

<!-- END two-machine-workflow -->


Historical Copilot-oriented package instructions are archived in `AGENTS_copilot.md`.

## Package Role

`pygeoinf3D` is the 3D extension package for concrete function spaces, domains, regions, and operators built on top of `pygeoinf`.

## Architectural Rules

- `Region3D` is a thin wrapper over `pygeoinf.subsets.Subset` and handles pure geometry semantics.
- `Domain3D` is not a `HilbertSpace`.
- `Domain3D` owns an ambient `EuclideanSpace(3)` and a geometric region.
- `StructuredDomain3D` is an optional layer for domains with natural grids, coordinate systems, or spectral structure.
- Function spaces inherit from `pygeoinf.HilbertSpace` and are defined on a `Domain3D`.

## Intended Scope

- 3D geometric domains embedded in `EuclideanSpace(3)`.
- Region semantics via `pygeoinf.subsets.Subset` or thin `Region3D` wrappers.
- Computational geometry on top of domain regions.
- Lebesgue and Sobolev spaces of scalar functions on 3D domains.
- Spectral and quadrature-based operators for boxes, parallelepipeds, and balls first.

## Orientation

Before exploring source files, read living references when present:

```text
pygeoinf3D/docs/agent-docs/references/living/*-reference.md
```

Never use `references/legacy/` unless explicitly requested.

## Plan Directory

Use:

```text
pygeoinf3D/docs/agent-docs/
```

Expected subdirectories are `active-plans/`, `completed-plans/`, `references/`, and `theory/`.

## Dependencies

Core dependencies are numpy, scipy, and matplotlib. Optional dependencies include pyshtools and Cartopy for spherical-harmonic and geographic workflows.

## After Changes

Update affected living references under `pygeoinf3D/docs/agent-docs/references/living/` when they exist.
