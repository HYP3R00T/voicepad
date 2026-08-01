# voicepad-core

Local audio and transcription pipeline primitives for VoicePad.

The package owns audio capture, durable WAV persistence, canonical waveform
preparation, artifact verification, inference sessions, and bounded
transcription processing. Application configuration and presentation belong to
the `voicepad` package.

The current production deployment is documented in
[`docs/designs/transcription-pipeline.md`](../../docs/designs/transcription-pipeline.md).

Package publication is currently frozen.
