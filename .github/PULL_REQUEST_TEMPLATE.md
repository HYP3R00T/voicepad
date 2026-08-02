# Pull request

## Summary

<!-- Describe the focused outcome and why it is needed. Do not provide only a file list. -->

## Related issue and design

<!-- Use "Closes #123" only when this PR fully resolves the issue. Link approved design sections when applicable. -->

## Behavior and implementation

<!-- Explain observable behavior, important implementation decisions, and failure or recovery behavior. -->

## Acceptance evidence

<!-- Map each issue criterion to concrete evidence. Add or remove rows as needed. -->

| Acceptance criterion | Evidence | Result |
|---|---|---|
| | | Pass / Fail / Unverified |

## Verification

<!-- List exact commands and real-surface checks with results. Mark unavailable hardware, model, or platform checks honestly. -->

```text
uv run ruff check
uv run ruff format --check
uv run ty check
uv run pytest packages --cov=voicepad --cov=voicepad_core --cov-fail-under=70
pnpm --dir docs install --frozen-lockfile
pnpm --dir docs build
```

## Risks, limitations, and review hotspots

<!-- Identify compatibility, migration, privacy, concurrency, resource, or operational risks. Point reviewers to the most consequential code. Write "None" only after considering them. -->

## Documentation impact

<!-- Link or describe documentation changes. If none are needed, explain why. -->

## Checklist

- [ ] This pull request contains one focused outcome and no unrelated changes.
- [ ] The source issue is approved and its acceptance criteria remain accurate.
- [ ] Tests cover changed behavior and important failure paths.
- [ ] Required local checks pass, and unavailable checks are disclosed.
- [ ] Documentation is updated or the lack of documentation impact is explained.
- [ ] The diff contains no credentials, private recordings, model binaries, caches, or generated site output.
- [ ] This change does not publish packages, enable the PyPI workflow, add publishing credentials, or create a release.
- [ ] I reviewed the final diff and can explain every change, including AI-assisted work.
