---
icon: lucide/zap
---

# GPU Acceleration

VoicePad uses your NVIDIA GPU to run Whisper transcription significantly faster than CPU. On an RTX 3050 (4 GB VRAM), the `turbo` model transcribes a 60-second recording in under 3 seconds.

!!! warning "NVIDIA GPUs only"
    GPU acceleration has been tested exclusively with **NVIDIA GPUs**. AMD and Apple Silicon GPUs are **not supported**.

## What You Need

- An NVIDIA GPU (GeForce, Quadro, or Tesla)
- An up-to-date NVIDIA driver (version 520 or newer)

CUDA runtime libraries are bundled with VoicePad automatically. You do **not** need to install the CUDA toolkit, cuDNN, or any NVIDIA developer tools.

## Recommended Models by GPU

| GPU | VRAM | Recommended Model |
|---|---|---|
| RTX 3050, RTX 4060 | 4 GB | `turbo` |
| RTX 3060, RTX 4070 | 6-8 GB | `turbo` or `large-v3` |
| RTX 3080, RTX 4090 | 10+ GB | `large-v3` |

## Verifying GPU Is Active

When VoicePad starts, the header bar shows the active device:

```text
model: turbo  device: cuda
```

If it shows `device: cpu`, VoicePad did not detect a usable NVIDIA GPU and fell back to CPU automatically.

## Performance

Measured on RTX 3050 (4 GB VRAM), `turbo` model:

| Audio Duration | Transcription Time |
|---|---|
| 10 seconds | ~0.5 s |
| 60 seconds | ~2-3 s |
| 5 minutes | ~10-15 s |

GPU acceleration provides roughly **4-10x speedup** over CPU.

## Troubleshooting

**VoicePad shows `device: cpu` but I have an NVIDIA GPU**

1. Check your NVIDIA driver version by running `nvidia-smi` in a terminal
2. Make sure the driver is version 520 or newer
3. Reinstall VoicePad: `uv tool install voicepad --reinstall`

### Out of memory error

Switch to a smaller model from the **Settings** tab. Try `small` or `distil-small.en` if your GPU has limited VRAM.
