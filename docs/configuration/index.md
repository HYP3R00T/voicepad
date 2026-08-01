# Configuration

VoicePad owns a strict schema-1 JSON configuration at:

```text
~/.config/voicepad/config-v2.json
```

Inspect or initialize it with:

```bash
voicepad config path
voicepad config show
voicepad config init
```

Supported fields are:

```json
{
  "schema": 1,
  "deployment_id": "parakeet-v3.transformers-fp16-cuda",
  "recordings_path": "~/.config/voicepad/data/recordings",
  "markdown_path": "~/.config/voicepad/data/markdown",
  "artifact_cache_path": "~/.cache/voicepad-v2/artifacts",
  "recording_prefix": "recording",
  "input_device_index": null,
  "copy_complete_text": true,
  "proper_nouns": [
    {"canonical": "VoicePad", "aliases": ["voice pad"]}
  ]
}
```

Unknown fields and obsolete schemas fail with an actionable error. VoicePad does
not silently migrate or overwrite old configuration.

Proper-noun aliases are deterministic word/phrase replacements after timestamp
assembly. They are not decoder hotwords and never use fuzzy matching.
