export interface DocLink {
  slug: string
  label: string
  description: string
}

export interface DocGroup {
  label: string
  links: DocLink[]
}

export const docGroups: DocGroup[] = [
  {
    label: 'Start here',
    links: [
      { slug: 'index', label: 'Overview', description: 'What VoicePad is and the hardware it targets.' },
      { slug: 'getting-started', label: 'Getting started', description: 'Prepare the deployment and make your first recording.' },
      { slug: 'interface', label: 'Interface', description: 'TUI states, controls, history, and live text.' },
    ],
  },
  {
    label: 'Configure',
    links: [
      { slug: 'configuration/index', label: 'Configuration', description: 'The strict schema-1 application configuration.' },
      { slug: 'configuration/global-hotkey', label: 'Global shortcut', description: 'Record from any application on Wayland.' },
      { slug: 'configuration/gpu', label: 'NVIDIA GPU', description: 'CUDA admission and memory requirements.' },
      { slug: 'configuration/input-device', label: 'Input device', description: 'Shared Linux microphone selection.' },
      { slug: 'configuration/models', label: 'Deployment', description: 'The official Parakeet runtime and artifacts.' },
      { slug: 'configuration/output-paths', label: 'Output paths', description: 'WAV, Markdown, and artifact locations.' },
    ],
  },
  {
    label: 'Deep dive',
    links: [
      {
        slug: 'designs/transcription-pipeline',
        label: 'Transcription pipeline',
        description: 'Resident inference, semantic chunking, and authoritative assembly.',
      },
    ],
  },
]

export const flatDocs = docGroups.flatMap((group) => group.links)

export function docHref(slug: string, base: string): string {
  if (slug === 'index') return `${base}/docs/`
  const route = slug.endsWith('/index') ? slug.slice(0, -'/index'.length) : slug
  return `${base}/docs/${route}/`
}
