---
icon: lucide/home
---

# Welcome to VoicePad

**Your private, local-first dictation studio.**

VoicePad is an open-source terminal UI for voice recording and transcription. Everything runs on your machine - no cloud, no API keys, no subscriptions. Speak, transcribe, done.

![VoicePad Interface](assets/sample_1.png)

## Why VoicePad

### Private by design

There is no server. Not "we promise we don't look" -- there is literally no server to send your data to. No bills to pay means no subscription model, which means no reason to charge you. Your audio stays on your machine because there is nowhere else for it to go.

### Record from anywhere, without switching apps

Open a terminal, run `uvx voicepad`, and leave it. From that point, press your hotkey in any application -- your browser, your editor, your notes app -- and VoicePad starts recording. Press it again to stop. The transcription is processed and copied to your clipboard automatically. When you close the terminal, VoicePad is gone. No background service, no startup entry, no idle resource drain.

### Minimal footprint

It is a terminal UI, not an Electron app. No bundled JavaScript engine, no browser runtime, no 300 MB of overhead just to show a button. When VoicePad is sitting idle, it is barely there.

## Features

- **No server, no telemetry** - Nothing leaves your machine. I literally cannot collect your data - there is no server to receive it.
- **No API keys, no accounts** - Just run it. No sign-up, no usage limits, no subscription.
- **Global hotkey** - Trigger recording system-wide from any app. Transcription auto-copies to clipboard when done.
- **GPU-accelerated transcription** - NVIDIA CUDA support included. No separate CUDA install needed.
- **Multiple Whisper models** - From `tiny` to `large-v3`, `turbo`, and `distil` variants. Pick the right balance of speed and accuracy for your hardware.
- **Streaming transcription** - Text appears live as you speak.
- **History and re-transcription** - Browse past recordings and re-run transcription with a better model at any time. Your original audio is always there.
- **Completely offline** - After the initial model download, no internet connection needed.
- **Auto-save** - Every recording and transcription saved automatically.
- **Open source** - MIT licensed, fully auditable. [Read the code](https://github.com/HYP3R00T/voicepad) or contribute.

!!! note "Resource usage benchmarks coming soon"
    Formal benchmarks across models and hardware are in progress and will be published here once available.

## Quick Start

```bash
uvx voicepad
```

No installation needed. Downloads and runs VoicePad in an isolated environment using [uv](https://docs.astral.sh/uv/). The default model downloads on first run, then you are ready to record.

!!! tip "Full setup guide"
    For installation options, first-run walkthrough, and microphone setup, see [Getting Started](getting-started.md).

## System Requirements

| | |
|---|---|
| **Python** | 3.13 or newer |
| **OS** | Windows, Linux |
| **GPU** | NVIDIA GPU required for GPU-accelerated transcription (tested on RTX 30/40 series) |
| **Microphone** | Any input device recognised by your OS |

!!! note "NVIDIA GPUs only"
    GPU acceleration has been tested exclusively with NVIDIA GPUs. AMD and Apple Silicon are not supported. CPU-only mode is available as a fallback but is significantly slower.

## Support the Project ❤️

VoicePad is built with love for the open-source and privacy-focused community. If you enjoy using the tool, consider supporting its continued development!

- ⭐ [Star the project on GitHub](https://github.com/HYP3R00T/voicepad)
- ☕ [Sponsor me](https://github.com/sponsors/HYP3R00T) to help keep the coffee flowing and the updates coming.
