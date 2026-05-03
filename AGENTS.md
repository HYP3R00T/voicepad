# AGENTS.md

Instructions for coding agents working in this repository.

## Project overview

- Repo: `HYP3R00T/voicepad` — a minimal, reusable Python devcontainer template.
- Stack: Python >= 3.13, `uv`, `ruff`, `ty`, `pytest`, `zensical`.

## Environment setup

```sh
mise install
uv sync --upgrade
prek install --hook-type pre-commit --overwrite
prek install --hook-type commit-msg --overwrite
```

GPU support (CUDA DLLs) is bundled with `torch` — no separate CUDA installation needed.
Dev container runs `scripts/setup.sh` automatically on create.

## Key files

| Concern | Files |
|---|---|
| Python / tooling | `pyproject.toml`, `ruff.toml`, `ty.toml`, `mise.toml` |
| Docs | `zensical.toml`, `docs/index.md` |
| Scripts | `scripts/setup.sh`, `scripts/enter_project.sh` |
| CI | `.github/workflows/ci.yml`, `.github/workflows/docs.yml` |

## Commands

IMPORTANT - Activate the virtual environment first

```sh
# Full quality pass (run before PR)
uv run ruff check; uv run ruff format --check; uv run ty check; uv run pytest packages --cov=voicepad --cov=voicepad_core --cov-fail-under=70

# Individual
uv run ruff check              # lint
uv run ruff format             # format (apply)
uv run ruff format --check     # format (check only)
uv run ty check                # type check
uv run pytest packages --cov=voicepad --cov=voicepad_core --cov-fail-under=70  # tests

# Docs
uv run zensical build --clean
uv run zensical serve
```

Coverage threshold: **70%** (enforced in CI and `mise.toml` `test` task).

## Expectations

- **Code:** typed, explicit Python; `ruff` is the formatting/lint source of truth; avoid new tools unless justified; keep template files generic.
- **Tests:** add or update tests for behavior changes; prefer focused unit tests over broad integration scaffolding.
- **Docs:** update `docs/` if behavior or config changes; don't hand-edit `site/` (build artifact).
- **Commits:** use conventional commits (`cz commit` if available); PRs should include a short summary of commands run.
- **Secrets:** never commit credentials; `.env` is gitignored and local-only.

## Agent behavior

- Prefer minimal diffs; don't refactor unrelated files.
- If tools are missing, run `mise install` and `uv sync` before trying workarounds.
- Keep CI workflows and local guidance in sync when changing related behavior.

## Behavioral guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```txt
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
