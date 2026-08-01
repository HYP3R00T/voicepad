# VoicePad

**Private, local-first NVIDIA dictation for Linux.**

VoicePad continuously persists microphone audio and transcribes it locally with
NVIDIA's official Parakeet TDT 0.6B v3 model through PyTorch FP16 CUDA. Audio,
transcripts, and model artifacts remain on the machine.

## Current target

- Linux x86_64
- NVIDIA CUDA GPU in the tested 4 GB physical class or larger
- Python 3.13+
- A microphone available through the shared Linux sound server

There is no CPU ASR fallback and no silent precision or model fallback.

## Development quick start

```bash
mise install
uv sync --upgrade
uv run voicepad prepare
uv run voicepad
```

Press **Space** to record and **Space** again to stop. The verified model remains
resident between recordings.

CLI alternatives:

```bash
uv run voicepad record start --duration 10
uv run voicepad transcribe path/to/recording.wav
```

Package publication is currently frozen. See
[`docs/designs/transcription-pipeline.md`](docs/designs/transcription-pipeline.md)
for the complete architecture and validation evidence.

## Quality gate

```bash
uv run ruff check
uv run ruff format --check
uv run ty check
uv run pytest packages --cov=voicepad --cov=voicepad_core --cov-fail-under=70
uv run zensical build --clean
```

## License

[MIT](LICENSE)
