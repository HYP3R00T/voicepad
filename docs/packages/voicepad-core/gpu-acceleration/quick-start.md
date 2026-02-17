# Quick Start: GPU Acceleration

Get GPU acceleration working in 5 minutes.

## Installation

### Install with GPU Support

```bash
pip install voicepad-core[gpu]
```

This installs:

- Base voicepad-core (~60MB)
- CUDA 12 libraries (~500MB in your virtual environment)

**Total:** ~560MB (isolated to your venv, no system changes)

### Verify GPU is Working

Run a test transcription:

```bash
voicepad record start
```

Check the output for `Device: cuda` (GPU working) or `Device: cpu` (using CPU fallback—see [Troubleshooting](troubleshooting.md)).

## How GPU Acceleration Works

### Two Modes

#### CPU Mode (Default)

- Works on all systems
- ~8-10 seconds per minute of audio
- Included with base package

#### GPU Mode (With [gpu] extra)

- Requires NVIDIA GPU
- ~2-3 seconds per minute of audio
- 4-5x faster than CPU
- Install: `pip install voicepad-core[gpu]`

### Automatic Detection

At transcription time, voicepad checks for CUDA libraries:

1. Attempts to import CUDA modules (`nvidia.cublas.lib`, `nvidia.cudnn.lib`)
2. If available → uses GPU automatically
3. If missing → falls back to CPU with helpful guidance

**No manual configuration needed.** The `transcription_device: auto` setting in your config does this automatically.

### Virtual Environment Isolation

CUDA libraries live **only in your virtual environment**, not globally:

```sh
your-venv/Lib/site-packages/nvidia/cublas/  # CUDA here
your-venv/Lib/site-packages/nvidia/cudnn/   # CUDA here
```

**Benefits:**

- ✅ Different projects can use different CUDA versions
- ✅ No conflicts with system CUDA
- ✅ Clean uninstall: just `pip uninstall nvidia-cublas-cu12 nvidia-cudnn-cu12`
- ✅ Works even if system has CUDA 13 while venv uses CUDA 12

## Common First Steps

Check if your GPU is detected:

```bash
voicepad config system
```

Look for:

```sh
GPU:
   NVIDIA Driver: [OK] Detected
   CUDA Devices: 1 device(s) available
   faster-whisper GPU: [OK] Compatible
```

Get a model recommendation for your system:

```bash
voicepad config recommend
```

See [Advanced Setup](advanced.md) for hardware requirements and detailed configuration.
