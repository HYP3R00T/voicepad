# voicepad-core

Local audio and transcription pipeline primitives for VoicePad.

The package owns audio capture, durable WAV persistence, canonical waveform
preparation, artifact verification, inference sessions, and bounded
transcription processing. Application configuration and presentation belong to
the `voicepad` package.

## Package vocabulary

| Package | Responsibility |
|---|---|
| `audio` | Load, capture, and durably persist audio |
| `preprocessing` | Normalize audio for downstream processing |
| `vad` | Detect speech and segment it into speech regions and natural pauses |
| `planning` | Plan bounded, overlapping inference chunks |
| `deployments` | Declare supported model and runtime combinations |
| `artifacts` | Acquire and verify immutable model artifacts |
| `inference` | Own the resident model engine and transcribe one prepared chunk |
| `pipeline` | Coordinate components and assemble authoritative results |

The pipeline supports two execution modes:

- **Batch** transcription processes a complete audio input.
- **Incremental** transcription processes committed ranges while a recording is
  still being persisted.

The current production deployment is documented in
[`docs/content/designs/transcription-pipeline.md`](../../docs/content/designs/transcription-pipeline.md).

Package publication is currently frozen.
