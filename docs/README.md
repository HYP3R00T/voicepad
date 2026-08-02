# VoicePad documentation

The VoicePad landing page and documentation are a self-contained Astro and
Tailwind CSS project.

```bash
pnpm --dir docs install
pnpm --dir docs dev
pnpm --dir docs build
```

- `content/` contains product and architecture documentation in Markdown.
- `src/pages/index.astro` is the product landing page.
- `src/` contains layouts, components, navigation data, and styles.
- `public/` contains static assets.
- `dist/` is generated and must not be edited or committed.

The production build is served from the root of the GitHub Pages custom domain
`https://voicepad.hyperoot.dev`.
