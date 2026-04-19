# AGENTS.md

Instructions for coding agents working in this repository.

## Project overview

- Repo: `HYP3R00T/voicepad` — a minimal, reusable Python devcontainer template.
- Stack: Python >= 3.13, `uv`, `ruff`, `ty`, `pytest`, `zensical`.

## Environment setup

```sh
mise install
uv sync --extra gpu --upgrade
prek install --hook-type pre-commit --overwrite
prek install --hook-type commit-msg --overwrite
```

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
ruff check && ruff format --check && ty check && pytest packages --cov=packages/* --cov-fail-under=80

# Individual
ruff check              # lint
ruff format             # format (apply)
ruff format --check     # format (check only)
ty check                # type check
pytest packages --cov=packages/* --cov-fail-under=80  # tests

# Docs
zensical build --clean
zensical serve
```

Coverage threshold: **80%** (enforced in CI and `mise.toml` `test` task).

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
