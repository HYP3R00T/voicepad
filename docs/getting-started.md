# Getting started

## Requirements

VoicePad currently requires Linux x86_64, a supported NVIDIA driver, and an
NVIDIA CUDA GPU. The maintained hardware baseline is an RTX 3050 Laptop GPU with
4 GB physical VRAM. CPU-only ASR, AMD GPUs, and macOS are not supported.

## Install the project

```bash
mise install
uv sync --upgrade
```

No separate CUDA toolkit is required; the locked PyTorch dependencies include
the reviewed CUDA runtime libraries.

## Prepare the deployment

```bash
uv run voicepad prepare
```

The first preparation downloads and verifies approximately 2.5 GB of official
Parakeet files and the official Silero wheel. Later startup is offline-capable.

## Use the TUI

```bash
uv run voicepad
```

Wait for the NVIDIA device to show as ready. Press **Space** to begin recording
and **Space** again to stop. VoicePad finalizes the WAV, drains bounded
transcription work, writes Markdown, and copies only a complete result.

## Use the CLI

```bash
uv run voicepad record start --duration 10
uv run voicepad transcribe path/to/existing.wav
```

Existing input audio is opened read-only. Output collisions fail rather than
overwrite prior data.
