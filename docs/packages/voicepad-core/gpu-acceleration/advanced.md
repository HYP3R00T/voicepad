# Advanced Setup

Hardware requirements, configuration, and technical details.

## Hardware Requirements

### GPU Support

| GPU Series | Compute Capability | Supported? | Recommended Model |
|------------|-------------------|------------|-------------------|
| GTX 10 (1050, 1080) | 6.1 | ✅ Yes | small (2GB VRAM) |
| RTX 20 (2060, 2080) | 7.5 | ✅ Yes | medium (3GB VRAM) |
| RTX 30 (3050, 3090) | 8.6 | ✅ Yes | medium (3GB VRAM) |
| RTX 40 (4060, 4090) | 8.9 | ✅ Yes | large (5GB VRAM) |
| RTX 50 (5080, 5090) | 9.0 | ✅ Yes | large (5GB VRAM) |
| AMD GPUs | N/A | ❌ No | Use CPU mode |
| Apple Silicon (M1/M2/M3) | N/A | ❌ No | Use CPU mode |

**Minimum:** GTX 10-series or equivalent (Compute Capability 6.1+)

### Software Requirements

- Windows or Linux (macOS Intel with eGPU may work)
- NVIDIA GPU Driver (any recent version)
- Python 3.13+
- Virtual environment recommended

## Configuration

### Check Current Settings

```bash
voicepad config show
```

Look for transcription settings:

```yaml
transcription_device: auto        # auto, cuda, or cpu
transcription_compute_type: auto  # auto, float16, int8, or float32
transcription_model: medium
```

### Force CPU Mode

To use CPU instead of GPU (if GPU is available):

```yaml
# voicepad.yaml
transcription_device: cpu
transcription_compute_type: int8
```

### Optimize for Your GPU

Recommendation: Run `voicepad config recommend` to get system-specific suggestions.

For manual tuning:

- **Limited VRAM (<2GB):** Use `small` model with `int8` compute type
- **Moderate VRAM (2-4GB):** Use `medium` model with `auto` or `float16`
- **Plenty VRAM (>4GB):** Use `large-v3` model with `float16`

## Technical Details

### What Gets Installed

**Base package (`voicepad-core`):**

- faster-whisper (transcription engine)
- CPU support included
- ~60MB total

**GPU extra (`[gpu]`):**

- `nvidia-cublas-cu12>=12.0.0` (~200MB)
- `nvidia-cudnn-cu12>=9.0.0` (~300MB)
- Total: ~560MB (added to your venv)

### Virtual Environment Isolation

CUDA libraries are installed **only in your virtual environment**, not system-wide:

```sh
your-venv/
└── Lib/site-packages/nvidia/
    ├── cublas/bin/cublas64_12.dll     # Windows
    └── cudnn/bin/cudnn64_9.dll        # Windows
```

**How detection works:**

At transcription time, voicepad attempts to import:

```python
import nvidia.cublas.lib
import nvidia.cudnn.lib
```

If both imports succeed → GPU mode enabled. If either fails → fallback to CPU.

### Why Virtual Environment Isolation?

✅ No global system changes
✅ Different projects can use different CUDA versions without conflict
✅ Works even if system CUDA version differs from venv CUDA version
✅ Clean uninstall: `pip uninstall nvidia-cublas-cu12 nvidia-cudnn-cu12`
✅ Reproducible builds across different systems

### CUDA Compute Type Reference

| Type | Quality | Speed | VRAM | Use Case |
|------|---------|-------|------|----------|
| float32 | ⭐⭐⭐⭐⭐ Excellent | Slow | High | Maximum accuracy |
| float16 | ⭐⭐⭐⭐ Good | Fast | Medium | GPU default (balanced) |
| int8 | ⭐⭐⭐ Good | Fast | Low | CPU default, limited VRAM |

Back to: [Quick Start](quick-start.md) | [Troubleshooting](troubleshooting.md)
