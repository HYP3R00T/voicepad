# Documentation Authoring Guidelines

## Documentation Principles

All documentation must adhere to the following four principles to ensure it remains accurate, scalable, and maintainable.

### 1. The "No Fiction" Rule: Docs Describe What Exists

Documentation must reflect the current state of the codebase. It is strictly forbidden to document features, architectures, or workflows that are "planned" or "in progress" within the repository documentation.

**Key Points:**

- Only document what exists now
- Mark aspirational features clearly (e.g., with a warning admonition)
- If code changes, documentation must change at the same time
- Treat documentation as code and keep it synchronized with every PR
- Do not write documentation for a feature until the feature PR is ready to merge
- Code and documentation ship together

**Why:** If documentation describes a system that doesn't exist yet, it erodes trust. A reader following a guide that fails because the code isn't there yet will stop reading.

### 2. Single Responsibility Principle: One Document, One Purpose

Apply the Single Responsibility Principle to writing. A document should answer one specific question or address one specific audience.

**Key Points:**

- If you have to use "and" in the document title, it likely needs to be split
- Examples:
    - ❌ `DevOps-Guide.md` (Too broad)
    - ✅ `setup-guide.md`, `deployment-guide.md`, `monitoring-guide.md` (Focused)

**Why:** Large, multi-purpose files are hard to search, hard to link to, and intimidating to read. Small, focused files are easy to maintain and deprecate.

### 3. Folder Scaling: Files Scale to Folders, Not Mega-Files

When a topic becomes complex, do not make the file longer; promote it to a directory instead.

**The Rule of Thumb:** If a markdown file exceeds ~3-4 distinct logical sections or requires a Table of Contents to navigate, it should likely become a folder.

**The Transformation:**

1. A file like `gitops.md` becomes too long
2. Create folder `gitops/`
3. Break content into focused files: `gitops/overview.md`, `gitops/reconciliation.md`, `gitops/secrets.md`
4. Create `gitops/index.md` as the map for that folder

**Why:** This mirrors code modularity. It allows different parts of a system to evolve at different speeds without causing merge conflicts in a massive monolithic document.

### 4. Source of Truth Rule: Link to Code Properly

Documentation explains the Why and the Context. The code explains the How. Do not duplicate code into documentation unless it is a small, immutable snippet for illustration.

**DO - Referencing Source Code Files:**

- ✅ Inline code style: `See services/api/routes.py for health check implementation`
- ✅ GitHub permalink: `https://github.com/HYP3R00T/voicepad/blob/main/path/to/file.py`
- ✅ GitHub permalink with commit SHA: `https://github.com/HYP3R00T/voicepad/blob/a1b2c3d/path/to/file.py`

**DO NOT - Referencing Source Code Files:**

- ❌ Relative paths from docs: `[file.py](../../services/api/file.py)`
- ❌ Copy-pasting large code blocks that will go stale

**Why relative paths don't work:**

- Documentation sites don't have access to source code at relative paths
- Rendered documentation can't properly resolve paths outside the docs tree
- Not version-specific—can't verify what the code looked like when docs were written
- Break when documentation is consumed anywhere other than the raw repository view

## Authoring Instructions

### Using Admonitions Effectively

Admonitions are highlighted blocks that draw attention to important information. Use them strategically to highlight constraints, best practices, and critical information.

**Format:**

```markdown
???+ type "Optional Title"
    Content goes here.
    Support multiple lines.
```

**Available Types:**

- `note` - General information and context
- `tip` - Best practices and recommendations
- `info` - Status updates and version information
- `warning` - Actions that could cause problems
- `danger` - Destructive or serious security concerns
- `example` - Example usage and patterns

**Key Constraint:** Use admonitions sparingly. They should highlight what matters, not clutter the document. Every admonition should serve a purpose.

For comprehensive admonition documentation, see the [Zensical Admonitions Guide](https://zensical.org/docs/authoring/admonitions/).

### Using Tables

Tables organize structured data effectively. Keep them simple and clear.

**Basic Structure:**

```markdown
| Column | Type | Purpose   |
| ------ | ---- | --------- |
| name   | str  | Item name |
```

**Constraints:**

- Keep to 3-5 columns ideal
- Avoid complex nested structures
- Use consistent formatting

For more details, refer to the [Zensical Markdown Reference](https://zensical.org/docs/authoring/markdown/).

### Internal and External Links

Point readers to resources rather than duplicating content. Link to:

- Other documentation files within the project
- Official Zensical guides and references
- External resources and frameworks
- GitHub permalinks for code references

## Before Committing

Ensure all documentation changes follow these principles and constraints:

- Documentation reflects current state of codebase (no fiction)
- Each file has a single, clear purpose
- File is appropriately sized (promote to folder if needed)
- Code is referenced properly (links, not relative paths)
- Admonitions are used meaningfully, not excessively
- All links are valid and point to appropriate resources
- If creating a new document, `zensical.toml` is updated to include it in navigation
