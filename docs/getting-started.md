---
icon: lucide/rocket
---

# Getting Started

## Requirements

| Requirement | Details |
|---|---|
| Python | 3.13 or newer |
| Microphone | Any input device recognised by your OS |
| NVIDIA GPU | Required for GPU-accelerated transcription. See [GPU Acceleration](configuration/gpu.md) |
| uv | Required for installation. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) |

!!! warning "NVIDIA GPUs only"
    GPU acceleration has been tested exclusively with NVIDIA GPUs (RTX 30/40 series). AMD and Apple Silicon GPUs are not supported. VoicePad can run in CPU-only mode, but transcription will be significantly slower.

## Install uv

VoicePad is installed and run using [uv](https://docs.astral.sh/uv/), the fast Python package manager from [Astral](https://astral.sh/).

=== "Windows"

    ```powershell
    winget install --id=astral-sh.uv -e
    ```

=== "Linux / macOS"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

After installing, restart your terminal so the `uv` and `uvx` commands are available.

## Installation

### Option 1: Run with uvx (recommended)

[`uvx`](https://docs.astral.sh/uv/concepts/tools/) runs VoicePad in a temporary, isolated environment. Nothing is permanently installed on your system.

```bash
uvx voicepad
```

On first run, it downloads the package. VoicePad then walks you through a short onboarding flow to confirm the system microphone and choose a starter Whisper model before the first model download begins. Subsequent runs start immediately from cache.

### Option 2: Install as a persistent tool

If you want VoicePad permanently available in your shell:

```bash
uv tool install voicepad
voicepad
```

This installs VoicePad into an isolated environment managed by uv and adds the `voicepad` command to your PATH.

## First Run

When you run VoicePad for the first time:

1. The interface opens and shows **initialising** in the header
2. The onboarding flow confirms the system-default microphone
3. The onboarding flow asks you to choose a starter Whisper model
4. VoicePad downloads that model from HuggingFace (one time only)
5. The model loads into GPU memory (or CPU if no NVIDIA GPU is detected)
6. The status changes to **ready**. You can now record

!!! tip "Simple model list by default"
    The onboarding flow shows a short curated list of starter models so new users are not overwhelmed. Advanced models are still available by editing `voicepad.yaml` directly.

!!! tip "Subsequent launches are instant"
    The model is cached locally after the first download. Future launches skip the download and start in seconds.

## Your First Recording

1. Open VoicePad: `uvx voicepad`
2. Make sure you are on the **Record** tab (selected by default)
3. Press ++space++ to start recording. The status changes to **recording**
4. Speak clearly into your microphone
5. Press ++space++ again to stop. Transcription begins immediately
6. The transcribed text appears in the transcription panel within seconds

The recording is saved as a WAV file and the transcription as a markdown file. Both go to `~/.config/voicepad/data/` by default. See [Configuration](configuration/index.md) to change this.

## Check Your Microphone

If nothing is captured, select the microphone as the default input in your
desktop's **Sound** settings. On Linux, VoicePad uses that shared input so it can
record at the same time as OBS and other PipeWire/PulseAudio applications.

## Next Steps

- [User Interface](interface.md): learn the full terminal interface
- [Configuration](configuration/index.md): change output paths, model, and device settings
- [GPU Acceleration](configuration/gpu.md): get the most out of your NVIDIA GPU
