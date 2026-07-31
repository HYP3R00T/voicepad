# Upstream sources

This skill is a project-level adaptation for HYP3R00T's issue-driven
development workflow. It was written after reviewing the sources below; no
upstream scripts or executable code are included.

## Blueprint

- Repository: <https://github.com/owainlewis/blueprint>
- Audited commit: `fe856861a743184c6fd4e939787191193a73940e`
- Relevant files:
  - `skills/task-to-pr/SKILL.md`
  - `skills/test/SKILL.md`
  - `skills/review/SKILL.md`
  - `README.md`
- Concepts adapted: one focused task per PR, worktree isolation, evidence-based
  testing, review loops, current-head CI, current-comment handling, and human
  merge.
- License: MIT; reproduced in `LICENSE`.

## Factory

- Repository: <https://github.com/owainlewis/factory>
- Audited commit: `d0a48df55dc301044ebd1860c9f62abcd301c648`
- Relevant files:
  - `docs/poller.md`
  - `docs/worker.md`
  - `docs/github-ingest/design.md`
  - `docs/workflows/design.md`
- Concepts adapted: trusted workflow versus untrusted issue context, live-state
  revalidation, idempotent delivery, repository identity checks, owned
  worktrees, safe preservation of unpublished work, and human review gates.
- License: MIT; reproduced in `LICENSE`.

## Earlier candidate review

An earlier unversioned `issue-to-pr` candidate was reviewed for fit. It had
useful GitHub and Project workflow detail, but no bundled provenance or license
and required `.github/agent-workflow.toml`. It was not copied as the published
source. This adaptation instead discovers live Project metadata and requires no
repository workflow configuration file.
