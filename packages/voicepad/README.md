# voicepad

Textual TUI and Typer CLI for VoicePad's verified NVIDIA transcription pipeline.

```bash
voicepad prepare
voicepad                 # TUI
voicepad record start
voicepad transcribe recording.wav
```

The application requires Linux x86_64 and a supported NVIDIA CUDA GPU. It does
not provide CPU ASR or silent runtime fallback. Configuration is application
owned and strict; see the project documentation for schema 1.

Package publication is currently frozen.
