# GPU Acceleration

Voicepad supports NVIDIA GPU-accelerated transcription for **4-5x faster** processing compared to CPU-only mode. GPU acceleration is optional and works seamlessly with automatic CPU fallback.

## Overview

**GPU Mode:** ~2-3 seconds per minute of audio (requires NVIDIA GPU + `[gpu]` extra)

**CPU Mode:** ~8-10 seconds per minute of audio (default, works everywhere)

Transcription automatically detects GPU availability and falls back to CPU if needed—no manual configuration required.

## Key Features

- ✅ **Zero global system changes** — CUDA libraries isolated in your virtual environment
- ✅ **Works with any NVIDIA GPU** — GTX 10-series and newer supported
- ✅ **Auto-fallback** — Seamlessly uses CPU if GPU unavailable
- ✅ **No CUDA Toolkit required** — Everything provided by `[gpu]` extra
- ✅ **Conflict-free** — Doesn't interfere with system CUDA or other projects

## Quick Links

- **[Quick Start](quick-start.md)** — Install and verify GPU works in 5 minutes
- **[Troubleshooting](troubleshooting.md)** — Fix common issues and check performance
- **[Advanced Setup](advanced.md)** — Hardware requirements, deep configuration, technical details

## Installation

The simplest way to enable GPU acceleration:

```bash
pip install voicepad-core[gpu]
```

Then verify it works:

```bash
voicepad config system
```

Look for `[OK] Compatible` next to "faster-whisper GPU" in the output.

## Next Steps

- Not sure if your GPU is supported? Start with [Requirements](advanced.md#hardware)
- Want to get GPU working? Go to [Quick Start](quick-start.md)
- Having issues? Check [Troubleshooting](troubleshooting.md)
