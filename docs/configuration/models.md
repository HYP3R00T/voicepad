---
icon: lucide/brain
---

# Transcription Models

VoicePad offers four models. Each one serves a distinct use case.

## Default Model

The default model is **turbo**, an optimized version of Whisper large-v3. It is the recommended model for most users with an NVIDIA GPU.

## Available Models

| Model | Purpose | Language | Approximate VRAM |
|---|---|---|---|
| `turbo` | Recommended default | Multilingual | ~3 GB |
| `small` | Lightweight fallback | Multilingual | ~1 GB |
| `distil-large-v3.5` | Fast English dictation | English only | ~3 GB |
| `parakeet-tdt-0.6b-v3` | NVIDIA architecture, Sherpa-ONNX int8 export | 25 European languages | ~670 MB weights, plus runtime memory |

## Which Model Should I Use?

| Your Hardware | Recommended Model |
|---|---|
| NVIDIA GPU, 4 GB VRAM | `turbo` |
| NVIDIA GPU, more than 4 GB VRAM | `turbo` |
| NVIDIA GPU, less than 4 GB VRAM | `small` |
| CPU only | `small` |

`parakeet-tdt-0.6b-v3` uses Sherpa's pinned int8 export. VoicePad verifies the encoder, decoder, joiner, and tokens before Sherpa runs them through CUDA.

Parakeet uses greedy decoding because it is reliable on short dictation. Sherpa's modified beam decoder produced severe truncation during local testing, so `proper_nouns` currently applies only to Faster Whisper models. VoicePad does not rewrite Parakeet transcripts afterward.

## Changing the Model

Open the **Settings** tab, select a model, and press **Save**. VoicePad reloads the model immediately.

The first time you use a model, VoicePad downloads and caches its artifact. Later sessions use the local copy.
