---
icon: lucide/home
---

# Welcome to VoicePad

**Your private, local-first dictation studio.**

Welcome! VoicePad is an open-source, lightning-fast dictation tool built from the ground up to respect your privacy. Everything happens on your machine, securely and privately. No data mining, no cloud APIs, and absolutely no subscriptions. You own your voice, and with VoicePad, you keep it that way.

![VoicePad Interface](assets/sample_1.png)

## Why VoicePad?

Most transcription tools send your audio to a cloud API. That means your voice data (meetings, notes, ideas, sensitive conversations) leaves your machine and lives on someone else's server.

VoicePad takes the opposite approach:

- **100% local processing**: every byte of audio stays on your hardware.
- **Secure by design**: no cloud, no tracking, no data leaks.
- **GPU-accelerated**: near-instant transcription powered by your NVIDIA GPU.
- **Open source**: MIT licensed, fully auditable, and you are welcome to contribute!

## More Than Just Text

VoicePad doesn't just give you text; it safely stores your original audio recordings alongside your markdown notes. This is a massive advantage:

- **No vendor lock-in**: You always have your original audio files. If you ever want to use a different tool in the future, your audio is ready to go.
- **Re-transcribe anytime**: If the transcription missed a highly nuanced word or if it crashed due to memory issues, you don't have to speak again. Just select the recording in the history and hit re-transcribe!
- **Listen back**: Sometimes you need to hear the original tone or emotion that text just can't capture.

## Features

| Feature | Details |
|---|---|
| Interactive terminal interface | Record, review history, and configure settings in one place |
| Streaming transcription | Live text appears while you speak |
| Whisper model support | All faster-whisper models, from tiny through large-v3, turbo, and distil variants |
| GPU acceleration | NVIDIA CUDA support with no separate CUDA install needed |
| Auto-save | WAV recordings and markdown transcriptions saved automatically |
| Flexible configuration | YAML config file or in-app settings panel |

## Quick Start

The fastest way to try VoicePad. No installation needed:

```bash
uvx voicepad
```

This downloads and runs VoicePad in an isolated environment using [uv](https://docs.astral.sh/uv/). The interface opens immediately, the default transcription model downloads on first run, and you can start recording right away.

## Compatibility

| Requirement | Details |
|---|---|
| Python | 3.13 or newer |
| Operating system | Windows, Linux |
| GPU | **NVIDIA GPU required** for GPU-accelerated transcription (tested on RTX 30/40 series) |
| Audio | Any microphone or input device recognised by your OS |

!!! note "NVIDIA GPUs only"
    VoicePad's GPU acceleration has been tested exclusively with NVIDIA GPUs. AMD and Apple Silicon GPUs are **not supported**. CPU-only mode is available as a fallback but is significantly slower.

## Support the Project ❤️

VoicePad is built with love for the open-source and privacy-focused community. If you enjoy using the tool, consider supporting its continued development!

- ⭐ [Star the project on GitHub](https://github.com/HYP3R00T/voicepad)
- ☕ [Sponsor me](https://github.com/sponsors/HYP3R00T) to help keep the coffee flowing and the updates coming.
