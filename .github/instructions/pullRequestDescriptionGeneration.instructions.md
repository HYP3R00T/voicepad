Generate the pull request description by strictly following the repository pull request template. Do not add, remove, or rename sections.

Populate only the following sections, in this exact order:

## Summary

- Write one or two clear sentences
- State what changed and why
- Avoid implementation details
- Use plain, direct language

## Context (optional)

- Include this section only if the reason for the change is not obvious from the diff
- Provide brief background, constraints, or motivation
- Omit entirely if unnecessary

## Affected Responsibility

- Select exactly one primary responsibility
- Mark it with [x]
- Do not select multiple items unless the change truly spans responsibilities

Available responsibilities:

- Service Logic
- Runtime Topology
- Provisioning
- Change Automation

## General rules:

- Assume the reviewer has no prior context
- Do not repeat commit messages
- Do not include commentary, explanations of the template, or placeholders
- Output valid Markdown only
