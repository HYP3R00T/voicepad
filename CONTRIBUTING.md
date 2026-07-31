# Contributing to VoicePad

Thank you for helping improve VoicePad. Keep contributions focused, explain the outcome they deliver, and verify the behavior they change.

## Before starting

Search existing issues and pull requests first.

- Use the bug form for reproducible incorrect behavior.
- Use the feature form for a new user-facing capability.
- Use the maintenance form for refactoring, tooling, dependencies, documentation infrastructure, or other accepted engineering work.
- Discuss substantial work in an issue before implementation.
- Do not report vulnerabilities publicly; follow [SECURITY.md](SECURITY.md).

An issue is ready for implementation when it has:

- one clear outcome;
- relevant context and constraints;
- testable acceptance criteria;
- a verification plan;
- explicit dependencies; and
- related work that is out of scope.

Newly tracked issues start in `Backlog`. Approval of an issue does not guarantee that every proposed implementation will be merged.

## Refine and deliver issues with a coding agent

Trusted VoicePad checkouts include three manually invoked skills for Agent Skills-compatible coding agents. Reload project skills through the client when the checkout gained or changed a skill after the session started.

Use `refine-issue` to inspect one backlog issue against live repository evidence, resolve material readiness gaps, structure the issue, and move it to `Ready`:

```text
refine-issue https://github.com/HYP3R00T/voicepad/issues/<number>
```

If a product or technical decision remains, the skill leaves the issue unchanged in `Backlog` and asks only for that decision. It never implements the issue, creates related issues, changes repository files, or invokes a delivery skill.

After refinement and approval, invoke `issue-to-pr` separately with the same full issue URL:

```text
issue-to-pr https://github.com/HYP3R00T/voicepad/issues/<number>
```

The delivery skill validates live issue and Project state, creates or resumes an isolated worktree, implements and verifies the accepted issue, and opens or updates one pull request. It stops at `In Review`; it never refines backlog work, merges, enables auto-merge, publishes packages, marks the issue `Done`, or removes the implementation worktree.

Each invocation authorizes only its named lifecycle step for the linked issue. Review the resulting pull request and merge it manually.

After issues have been finalized, use `cleanup-worktrees` without arguments to audit all local worktrees and issue branches for the current repository:

```text
cleanup-worktrees
```

The cleanup skill removes only inactive, ordinarily clean, fully integrated agent-managed issue artifacts. Removing a qualifying worktree also removes all ignored files and directories contained within it; staged, unstaged, and ordinary untracked content still blocks removal. The skill preserves the primary checkout, active issues, dirty or unpublished work, and unrelated or manually created worktrees and branches. It reads GitHub lifecycle and pull-request state but never finalizes issues or changes Project status.

## Set up the project

VoicePad requires Python 3.13 or newer, [Mise](https://mise.jdx.dev/), and [uv](https://docs.astral.sh/uv/).

```sh
mise install
uv sync --upgrade
prek install --hook-type pre-commit --overwrite
prek install --hook-type commit-msg --overwrite
```

Use `uv run` for project commands. The virtual environment does not need to be activated manually.

For Linux audio development, install the PortAudio runtime and development packages documented by your distribution. Optional GPU and model tests require suitable local hardware and model assets.

## Create a branch

Create one branch from the latest `origin/main` for each accepted issue:

```sh
git fetch origin
git switch -c issue-<number>-<short-description> origin/main
```

Use an isolated worktree when another task or uncommitted work already occupies the main checkout. Never overwrite or clean unrelated work.

## Make a focused change

- Deliver the smallest cohesive outcome satisfying the issue.
- Keep unrelated refactoring and formatting out of the pull request.
- Add regression tests for confirmed defects.
- Test important failure, cancellation, cleanup, and recovery behavior.
- Update user or contributor documentation when behavior or workflow changes.
- Do not commit credentials, `.env`, user recordings, private logs, model binaries, caches, or generated `site/` output.
- Do not convert or add model artifacts unless an approved issue explicitly requires it and licensing permits it.

If implementation reveals a materially different requirement, stop and update or split the issue before expanding the change.

## Verify the change

Run the focused checks while developing, then run the full gate before requesting review:

```sh
uv run ruff check
uv run ruff format --check
uv run ty check
uv run pytest packages --cov=voicepad --cov=voicepad_core --cov-fail-under=70
uv run zensical build --clean
```

Run `prek run --all-files` before committing. Report hardware, model, network, or platform checks as passed, failed, or unavailable; do not represent unavailable checks as proof.

## Commit and open a pull request

Use [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<optional-scope>): <description>
```

Complete the pull-request template. Link the accepted issue and use `Closes #<number>` only when the pull request fully resolves it. Keep the evidence current when commits change.

The `protect-main` ruleset requires:

- a pull request;
- current `CI (uv)` success;
- resolved review conversations; and
- merge or squash integration.

VoicePad is currently solo-maintained, so another person's approval is not required. Merge remains an explicit human action after reviewing the final diff and evidence. Agents must not merge or enable auto-merge unless the maintainer explicitly changes this policy.

After merge, verify that the reviewed and checked head was integrated, close the resolved issue, update its Project status to `Done`, and preserve any dirty worktree or unpushed branch.

## Publication freeze

Publishing `voicepad` or `voicepad-core` to PyPI is frozen during the backend migration. The release workflow must remain disabled, and no publishing credential may be added or used without a separate, explicit maintainer decision.

Ordinary pull requests and merges must never publish packages, create release tags, or create public releases.

## Code of Conduct

Participation is governed by the [Code of Conduct](.github/CODE_OF_CONDUCT.md).
