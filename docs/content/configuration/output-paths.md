# Output paths

Defaults:

```text
WAV recordings:  ~/.config/voicepad/data/recordings
Markdown results: ~/.config/voicepad/data/markdown
Artifact cache:   ~/.cache/voicepad-v2/artifacts
```

Capture writes to an operation-owned float WAV spool and then publishes a PCM
WAV. Inference reads committed disk ranges and cannot block microphone
persistence. A failed finalization retains recoverable spool audio.

VoicePad refuses to overwrite existing WAV or Markdown files. Legacy model
caches are ignored and never migrated or deleted automatically.
