---
icon: lucide/home
---

# VoicePad

VoicePad is a local-first dictation studio for people who want fast transcription without accounts, cloud APIs, or workflow friction.

You press a hotkey, speak, and get live text plus a saved recording and markdown transcript on your own machine.

## Why VoicePad

Most dictation tools trade privacy for convenience, or speed for control. VoicePad is built to keep all three in the same place.

- Local-first: audio and transcripts stay on your machine
- Fast: GPU-accelerated Whisper transcription with live streaming feedback
- Practical: hotkey capture, saved history, markdown output, and re-transcription built in
- Configurable: simple defaults for most people, deeper control in `voicepad.yaml` when you want it

## Quick Start

```bash
uvx voicepad
```

!!! note "Requirements"
    Requires `uv` to be installed. Runs in an isolated environment.

On first run, VoicePad walks you through a short onboarding flow so you can pick a microphone and a starter model before the first download begins.

## What You Get

### Local Recording and Transcription

Record with a single keypress and transcribe locally with Whisper-compatible models. No browser tab, no remote upload, no API key setup.

### Live Streaming Feedback

Text appears while you speak, so you can see whether the recording is going in the right direction before the session ends.

### Markdown-Native History

Every session is stored as a WAV file plus a markdown transcript, which makes the output easy to archive, search, edit, and re-use.

### Curated Model Choices

The interface offers four models with distinct purposes: the default, a lightweight option, an English-focused option, and an alternate NVIDIA backend.

### Re-Transcription

You can revisit older recordings and re-run them with a different model later instead of treating the first pass as permanent.

## Who It Is For

VoicePad works especially well for:

- developers who want a keyboard-first dictation workflow
- writers and note-takers who want local ownership of recordings and transcripts
- GPU users who want faster-than-real-time transcription without cloud services
- power users who want plain files and editable configuration instead of a locked-down app

## System Requirements

| Requirement | Details |
|---|---|
| Python | 3.13 or newer |
| OS | Windows, Linux |
| Microphone | Any input device recognised by your OS |
| NVIDIA GPU | Recommended for fast transcription |

See [Getting Started](getting-started.md) for installation and first-run setup, or [Configuration](configuration/index.md) if you want to understand how the app is wired.

## Should We Build a Custom Docs Site?

Probably not yet.

Right now the bigger issue is writing and information design, not the documentation engine itself. The current system is already good enough for:

- clean navigation
- fast iteration
- markdown authoring
- a searchable static docs site

A custom Astro site starts making sense when we genuinely need one of these:

- a strong marketing-style landing page with custom layouts and richer visual storytelling
- interactive demos or embedded product walkthroughs
- a branded docs experience that the current theme cannot express well
- docs and website merging into one intentional product surface

If the goal is simply "make the docs feel clearer and less generic," we should keep improving the content and structure first. That is much cheaper and will probably get us 80% of the value.

## Start Here

[Get Started](getting-started.md){ .md-button .md-button--primary }
[Open Configuration](configuration/index.md){ .md-button }
