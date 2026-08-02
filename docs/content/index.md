# VoicePad

VoicePad is a private, local-first dictation application for Linux systems with
an NVIDIA GPU. It records durable WAV audio and transcribes it with NVIDIA's
official Parakeet TDT 0.6B v3 model through PyTorch FP16 CUDA.

## Current production target

- Linux x86_64
- NVIDIA CUDA GPU in the tested 4 GB physical class or larger
- Official `nvidia/parakeet-tdt-0.6b-v3` safetensors
- Official Silero VAD 6.2.1 on CPU
- Python 3.13 and `uv`

VoicePad provides no silent CPU ASR fallback. Unsupported hardware fails before
recording begins.

## Commands

```bash
voicepad prepare
voicepad transcribe recording.wav
voicepad record start
voicepad
```

Running `voicepad` without a command opens the TUI. Press **Space** to record or
stop. The model remains resident between recordings.

Audio is persisted independently of inference. Existing WAV and Markdown files
are never overwritten automatically.
