# GPU Acceleration

Voicepad supports GPU-accelerated transcription using NVIDIA GPUs for 4-5x faster processing.

## Quick Start

### Install with GPU Support

```bash
pip install voicepad-core[gpu]
```

This installs:

- Base voicepad (~60MB)
- CUDA 12 libraries (~500MB in virtual environment)

### Verify GPU is Working

Record and transcribe:

```bash
voicepad record start
```

Check the transcription output for:

- `Device: cuda` (GPU working)
- `Device: cpu` (fallback mode - see troubleshooting)

---

## How It Works

### Two Modes

#### CPU Mode (Default)

- Works on all systems
- ~8-10s per minute of audio
- Installed with base package

#### GPU Mode (Optional)

- Requires NVIDIA GPU
- ~2-3s per minute of audio
- Install: `pip install voicepad-core[gpu]`

### CUDA Library Detection

Voicepad checks for CUDA libraries at transcription time:

1. Try to import `nvidia.cublas.lib` and `nvidia.cudnn.lib`
2. If available → use GPU
3. If missing → auto-fallback to CPU with helpful message

???+ note "System CUDA Not Used"
    Voicepad only uses Python CUDA packages (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`), not system-installed CUDA.

    **What this means:**
    - Even if you have CUDA 12/13 installed globally, voicepad won't use it
    - You must install the `[gpu]` extra for GPU support
    - Each virtual environment has isolated CUDA libraries

    **Why this approach:**
    - ✅ No version conflicts between projects
    - ✅ Works regardless of system CUDA version
    - ✅ Consistent behavior across systems

---

## Requirements

### Hardware

| GPU Series | Compute Capability | Supported? | Recommended Model |
|------------|-------------------|------------|-------------------|
| GTX 10 (1050, 1080) | 6.1 | ✅ Yes | small (2GB VRAM) |
| RTX 20 (2060, 2080) | 7.5 | ✅ Yes | medium (3GB VRAM) |
| RTX 30 (3050, 3090) | 8.6 | ✅ Yes | medium (3GB VRAM) |
| RTX 40 (4060, 4090) | 8.9 | ✅ Yes | large (5GB VRAM) |
| RTX 50 (5080, 5090) | 9.0 | ✅ Yes | large (5GB VRAM) |
| AMD GPUs | N/A | ❌ No | Use CPU mode |
| Apple Silicon (M1/M2/M3) | N/A | ❌ No | Use CPU mode |

### Software

- **Windows or Linux** (macOS Intel with eGPU may work)
- **NVIDIA GPU Driver** (any recent version)
- **Python 3.13+**
- **Virtual environment recommended** (venv, conda, uv)

???+ warning "No System CUDA Required"
    You do NOT need to install CUDA Toolkit globally.
    The `[gpu]` extra provides everything needed in your virtual environment.

---

## Troubleshooting

### GPU Detected But Using CPU

**Symptom:**

```sh
[!] GPU requested but CUDA libraries not available
  Missing: CUDA libraries

  For 4x faster transcription, install GPU support:
    pip install voicepad-core[gpu]
```

Cause: Python CUDA packages not installed

Solution:

```bash
pip install voicepad-core[gpu]
```

---

### "CUDA 13 Incompatible" Error

Symptom: System has CUDA 13, but ctranslate2 only supports CUDA 12

Solution: The `[gpu]` extra installs CUDA 12 in your virtual environment automatically. No conflict occurs because:

- Your system keeps CUDA 13 globally
- Your venv uses CUDA 12 packages
- They don't interfere with each other

---

### Still Using CPU After Installing [gpu]

1. **Verify installation:**

   ```bash
   python -c "import nvidia.cublas.lib; import nvidia.cudnn.lib; print('GPU packages installed')"
   ```

2. **Check GPU is detected:**

   ```bash
   nvidia-smi
   ```

3. **Check configuration:**

   ```bash
   voicepad config system
   ```

4. **Check logs:** Look for `CUDA libraries detected` in output

---

## Performance Expectations

### RTX 3050 (4GB VRAM)

- **Model:** medium
- **CPU time:** ~8-10s per minute
- **GPU time:** ~2-3s per minute
- **Speedup:** 4-5x

### GTX 1050 (2-4GB VRAM)

- **Model:** small
- **CPU time:** ~6-8s per minute
- **GPU time:** ~2-3s per minute
- **Speedup:** 3-4x

---

## Configuration

### Checking Current Settings

```bash
voicepad config show
```

Look for:

```yaml
transcription_device: auto  # or cuda/cpu
transcription_compute_type: auto  # or float16/int8
transcription_model: medium
```

### Force CPU Mode

If you have GPU installed but want to use CPU:

```yaml
# voicepad.yaml
transcription_device: cpu
transcription_compute_type: int8
```

---

## Technical Details

### What Gets Installed

**Base package (`voicepad-core`):**

- faster-whisper (transcription engine)
- CPU support included
- ~60MB total

**GPU extra (`voicepad-core[gpu]`):**

- `nvidia-cublas-cu12>=12.0.0,<13.0.0` (~200MB)
- `nvidia-cudnn-cu12>=9.0.0,<10.0.0` (~300MB)
- Total: ~560MB

### Virtual Environment Isolation

CUDA libraries are installed **only in your virtual environment**:

```sh
your-venv/
└── Lib/
    └── site-packages/
        └── nvidia/
            ├── cublas/
            │   └── bin/
            │       └── cublas64_12.dll  # Windows
            └── cudnn/
                └── bin/
                    └── cudnn64_9.dll    # Windows
```

**Benefits:**

- No global system changes
- Different projects can use different CUDA versions
- Uninstall is clean (`pip uninstall nvidia-cublas-cu12 nvidia-cudnn-cu12`)

---

## FAQ

**Q: Do I need to install CUDA Toolkit?**
A: No. The `[gpu]` extra provides everything needed.

**Q: Will this conflict with my system CUDA?**
A: No. Virtual environment CUDA libraries are isolated.

**Q: Can I use AMD GPUs?**
A: Not currently. faster-whisper only supports NVIDIA CUDA.

**Q: What about Apple Silicon (M1/M2)?**
A: GPU acceleration not available. CPU mode works well on Apple Silicon.

**Q: How do I uninstall GPU support?**
A: `pip uninstall nvidia-cublas-cu12 nvidia-cudnn-cu12`

**Q: Can I use this with uvx?**
A: Yes! `uvx --with voicepad-core[gpu] voicepad record start`

---

## Related

- [Configuration Guide](../configuration/index.md)
- [System Requirements](../getting-started/requirements.md)
- [Troubleshooting](../troubleshooting/index.md)
