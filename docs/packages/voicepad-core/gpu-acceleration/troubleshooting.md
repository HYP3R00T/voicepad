# Troubleshooting

Solve common GPU acceleration issues.

## GPU Detected But Using CPU

**Symptom:** Output shows `Device: cpu` even though you installed `[gpu]`

**Solution:**

1. Verify CUDA packages are installed:

   ```bash
   python -c "import nvidia.cublas.lib; import nvidia.cudnn.lib; print('✓ GPU packages OK')"
   ```

2. Check your venv has GPU support:

   ```bash
   pip list | grep nvidia
   ```

   Should show `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`

3. Verify GPU is detected by system:

   ```bash
   nvidia-smi
   ```

   Should list your GPU

4. If still using CPU, reinstall GPU extra:

   ```bash
   pip install --upgrade voicepad-core[gpu]
   ```

## "CUDA 13 Incompatible" Error

**Symptom:** System has CUDA 13 installed, but voicepad mentions CUDA 12

**Why this happens:** faster-whisper requires CUDA 12, but your system has CUDA 13

**Solution:** No solution needed! The `[gpu]` extra installs CUDA 12 **in your virtual environment**, separate from system CUDA 13. They don't conflict.

- System keeps CUDA 13 globally
- Your venv uses CUDA 12 packages
- Both coexist without problems

## GPU Not Detected

**Symptom:** `voicepad config system` shows GPU as not available

**Causes & Solutions:**

| Issue | Check | Solution |
|-------|-------|----------|
| Old GPU | Compute Capability < 6.0 | Upgrade GPU (GTX 10-series minimum) |
| Old driver | `nvidia-smi --query-gpu=compute_cap` | Update NVIDIA drivers |
| No CUDA packages | `pip list \| grep nvidia` | Run `pip install voicepad-core[gpu]` |

## Performance Benchmarks

### RTX 3050 (4GB VRAM)

- **Model:** medium (recommended)
- **CPU time:** ~8-10s per minute of audio
- **GPU time:** ~2-3s per minute of audio
- **Speedup:** 4-5x

### GTX 1050 (2-4GB VRAM)

- **Model:** small (recommended for limited VRAM)
- **CPU time:** ~6-8s per minute of audio
- **GPU time:** ~2-3s per minute of audio
- **Speedup:** 3-4x

**Key:** Speedup depends on model size. Smaller models are faster on GPU; larger models benefit more but need more VRAM.

## FAQ

**Q: Do I need CUDA Toolkit installed globally?**
A: No. The `[gpu]` extra provides everything in your venv.

**Q: Will GPU support conflict with my system CUDA?**
A: No. Virtual environment isolation prevents conflicts.

**Q: Can I use AMD GPUs?**
A: Not currently. faster-whisper only supports NVIDIA CUDA.

**Q: What about Apple Silicon (M1/M2)?**
A: GPU acceleration not available, but CPU mode works well.

**Q: How do I disable GPU support?**
A: Set `transcription_device: cpu` in `voicepad.yaml`, or uninstall: `pip uninstall nvidia-cublas-cu12 nvidia-cudnn-cu12`

**Q: Can I use `uvx voicepad` with GPU?**
A: Yes! `uvx --with voicepad-core[gpu] voicepad record start`

See [Quick Start](quick-start.md) for installation or [Advanced Setup](advanced.md) for detailed requirements.
