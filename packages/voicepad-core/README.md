# voicepad-core

Core Python library for voice recording, GPU-accelerated transcription, and system diagnostics.

## Install

```bash
pip install voicepad-core
```

**For GPU support (4-5x faster):**

```bash
pip install voicepad-core[gpu]
```

**Requirements:** Python 3.13+

## Quick Start

```python
from voicepad_core import AudioRecorder, transcribe_audio, get_config

# Load configuration
config = get_config()

# Record audio
recorder = AudioRecorder(config)
audio_file = recorder.start_recording()
# Press Ctrl+C to stop

# Transcribe the audio file
output_file = config.markdown_path / "transcript.md"
stats = transcribe_audio(audio_file, output_file, config)
print(f"Transcribed: {stats['word_count']} words")
```

## Key Components

### Audio Recording

```python
from voicepad_core import AudioRecorder, get_config

config = get_config()
recorder = AudioRecorder(config)
audio_file = recorder.start_recording()  # Press Ctrl+C to stop
```

### Transcription

```python
from voicepad_core import transcribe_audio, get_config

config = get_config()
stats = transcribe_audio(
    audio_file=Path("recording.wav"),
    output_file=Path("transcript.md"),
    config=config
)
```

### System Diagnostics

```python
from voicepad_core import gpu_diagnostics, get_ram_info, get_cpu_info

gpu = gpu_diagnostics()
ram = get_ram_info()
cpu = get_cpu_info()

print(f"GPU available: {gpu.faster_whisper_gpu.success}")
print(f"Available RAM: {ram.available_gb} GB")
```

### Model Recommendations

```python
from voicepad_core import (
    get_model_recommendation,
    get_available_models,
    get_ram_info,
    get_cpu_info,
    gpu_diagnostics
)
from voicepad_core.diagnostics.models import SystemInfo

system_info = SystemInfo(
    ram=get_ram_info(),
    cpu=get_cpu_info(),
    gpu_diagnostics=gpu_diagnostics()
)
recommendation = get_model_recommendation(system_info, get_available_models())
print(f"Recommended model: {recommendation.recommended_model}")
```

## Configuration

Create `voicepad.yaml`:

```yaml
recordings_path: data/recordings
markdown_path: data/markdown
input_device_index: null
transcription_model: tiny
transcription_device: auto
transcription_compute_type: auto
```

## Documentation

- [API Reference](https://voicepad.readthedocs.io/packages/voicepad-core/) - Complete API docs
- [GPU Acceleration](https://voicepad.readthedocs.io/packages/voicepad-core/gpu-acceleration.md) - GPU setup guide
- [Main README](https://github.com/HYP3R00T/voicepad#readme) - Project overview

## Supported Models

All OpenAI Whisper models are supported:

- **tiny** - Fastest (39M)
- **base** - Fast (74M)
- **small** - Balanced (244M)
- **medium** - Accurate (769M)
- **large-v3** - Most accurate (1.5B)
- **turbo** - Latest generation (809M)

Use `get_available_models()` to list all models.

## Requirements

- **Python** 3.13+
- **Audio device** for recording
- **GPU (optional)** for 4-5x faster transcription
