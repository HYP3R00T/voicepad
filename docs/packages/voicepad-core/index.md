# voicepad-core

Core Python library for voice recording, transcription, and system diagnostics. Use this package to integrate audio recording and GPU-accelerated transcription into your own Python applications.

## Installation

```bash
pip install voicepad-core
```

**Requirements:** Python 3.13+

**Base Dependencies:**

- `faster-whisper` - OpenAI Whisper model via ONNX runtime
- `pydantic` - Data validation and config models
- `sounddevice` - Audio input device access
- `soundfile` - WAV file I/O
- `utilityhub-config` - Configuration management

**GPU Support (Optional):**

```bash
# Install with CUDA 12 libraries for GPU acceleration
pip install voicepad-core[gpu]
```

This installs `ctranslate2[cuda12]` and `pydantic-core` wheel for NVIDIA GPU support, enabling 4-5x faster transcription.

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

## Public API

### Configuration

#### `Config` Model

The `Config` Pydantic model defines all settings for voicepad:

```python
from voicepad_core import Config

class Config(BaseModel):
    recordings_path: Path = Path("data/recordings")
    markdown_path: Path = Path("data/markdown")
    input_device_index: int | None = None
    recording_prefix: str = "recording"
    transcription_model: str = "tiny"
    transcription_device: str = "auto"  # "auto", "cuda", or "cpu"
    transcription_compute_type: str = "auto"  # "auto", "float16", "int8", "float32"
```

**Fields:**

- `recordings_path` - Directory where audio files are saved (created if missing)
- `markdown_path` - Directory where markdown transcriptions are saved
- `input_device_index` - Audio device index (None = system default)
- `recording_prefix` - Filename prefix for recordings (default: "recording")
- `transcription_model` - Whisper model name (default: "tiny")
- `transcription_device` - Device for transcription (auto-selects CUDA if available)
- `transcription_compute_type` - Precision level (auto-selects float16 on CUDA, int8 on CPU)

#### `get_config(cwd=None, app_name="voicepad")`

Load configuration from `voicepad.yaml` using precedence rules:

```python
from voicepad_core import get_config

config = get_config()
print(config.recordings_path)
print(config.transcription_model)
```

**Precedence (highest to lowest):**

1. CLI arguments (handled by `voicepad` package)
2. Environment variables (e.g., `VOICEPAD_TRANSCRIPTION_MODEL=small`)
3. Project config (`./voicepad.yaml`)
4. Global config (`~/.config/voicepad/voicepad.yaml`)
5. Built-in defaults

**Parameters:**

- `cwd` - Search for config starting in this directory (default: current working directory)
- `app_name` - Application name for config namespace (default: "voicepad")

#### `get_config_with_metadata(cwd=None, app_name="voicepad")`

Load configuration and return metadata about the config source:

```python
from voicepad_core import get_config_with_metadata

config, metadata = get_config_with_metadata()
for field_name, source_info in metadata.per_field.items():
    print(f"{field_name}: from {source_info.source}")
```

### Audio Recording

#### `AudioRecorder` Class

Record audio from the configured input device:

```python
from voicepad_core import AudioRecorder, AudioRecorderError, get_config

config = get_config()
recorder = AudioRecorder(config)

try:
    # Start recording (returns Path to output file)
    audio_file = recorder.start_recording(prefix="my_recording", duration=None)

    # Check if still recording
    while recorder.is_recording():
        print("Recording...")
        import time
        time.sleep(1)

    # Stop recording manually
    audio_file = recorder.stop_recording()
    print(f"Saved to: {audio_file}")

except AudioRecorderError as e:
    print(f"Error: {e}")
```

**Methods:**

- `start_recording(prefix=None, duration=None) -> Path` — Start recording
    - `prefix` (str, optional) - Custom filename prefix
    - `duration` (float, optional) - Auto-stop after N seconds
    - Returns: Path to output WAV file
- `stop_recording() -> Path` — Stop recording and save file
    - Returns: Path to saved WAV file, or None if no recording is active
- `is_recording() -> bool` — Check if currently recording

**Exceptions:**

- `AudioRecorderError` - Raised on recording failures (device not available, permissions, etc.)

### Transcription

#### `transcribe_audio()`

Transcribe an audio file to markdown text:

```python
from voicepad_core import transcribe_audio, TranscriptionError, get_config
from pathlib import Path

config = get_config()

# Transcribe file
audio_file = Path("recording.wav")
output_file = Path("transcript.md")

try:
    stats = transcribe_audio(audio_file, output_file, config)

    print(f"Device used: {stats['device']}")
    print(f"Duration: {stats['duration']:.1f}s")
    print(f"Language: {stats['language']}")
    print(f"Words: {stats['word_count']}")
    print(f"Segments: {stats['segment_count']}")

except TranscriptionError as e:
    print(f"Transcription failed: {e}")
```

**Signature:**

```python
def transcribe_audio(
    audio_file: Path,
    output_file: Path,
    config: Config
) -> dict[str, Any]:
```

**Parameters:**

- `audio_file` - Path to WAV file to transcribe
- `output_file` - Path where markdown transcription will be saved
- `config` - Configuration with model, device, and compute type settings

**Returns:** Dictionary with transcription statistics:

```python
{
    "device": "cuda" or "cpu",           # Device used for transcription
    "duration": 42.5,                    # Audio duration in seconds
    "language": "en",                    # Detected language code
    "language_probability": 0.95,        # Confidence of language detection (0-1)
    "word_count": 512,                   # Total words transcribed
    "segment_count": 8,                  # Number of speech segments
    "fallback_info": {                   # Only present if GPU fallback occurred
        "fallback_occurred": True,
        "missing_components": ["ctranslate2-cuda"],
        "device_attempted": "cuda"
    }
}
```

**Exceptions:**

- `TranscriptionError` - Raised on transcription failures
- GPU fallback - Automatically falls back to CPU if requested device is unavailable

**Output Format:**

The markdown file contains the full transcription with metadata:

```markdown
# Transcription
**Language:** English (98.5%)
**Date:** 2026-02-18 10:30:45

---

This is the transcribed text from the audio file. Segments are separated by paragraphs...

---

**Statistics**
- **Duration:** 42.5 seconds
- **Segments:** 8
- **Words:** 512
```

#### `resolve_auto_settings()`

Resolve "auto" values in transcription settings based on system capabilities:

```python
from voicepad_core import resolve_auto_settings, get_config, gpu_diagnostics

config = get_config()
gpu_report = gpu_diagnostics()

# Resolve auto settings
resolved_device = resolve_auto_settings(
    device="auto",
    gpu_diagnostics=gpu_report
)
# Returns "cuda" if GPU available, else "cpu"

resolved_compute = resolve_auto_settings(
    compute_type="auto",
    device=resolved_device
)
# Returns "float16" if cuda, "int8" if cpu
```

**Exceptions:**

- `TranscriptionError` - Raised if resolution fails

### System Diagnostics

Use system diagnostics to check hardware capabilities and get recommendations:

```python
from voicepad_core import gpu_diagnostics, get_ram_info, get_cpu_info

# Check GPU availability
gpu = gpu_diagnostics()
print(f"NVIDIA Driver: {gpu.nvidia_smi.success}")
print(f"CUDA Devices: {gpu.ctranslate2_cuda.cuda_device_count}")
print(f"Faster-Whisper GPU: {gpu.faster_whisper_gpu.success}")

# Check system resources
ram = get_ram_info()
print(f"Available RAM: {ram.available_gb:.1f} GB")

cpu = get_cpu_info()
print(f"CPU Cores: {cpu.count}")
```

#### `gpu_diagnostics() -> GPUDiagnosticsReport`

Run comprehensive GPU diagnostics:

```python
from voicepad_core import gpu_diagnostics

report = gpu_diagnostics()

# Examing results
if report.faster_whisper_gpu.success:
    print("GPU transcription available")
else:
    print("GPU not available, using CPU")
```

**Returns:** `GPUDiagnosticsReport` with:

- `nvidia_smi` - NVIDIA driver check result
- `ctranslate2_cuda` - CUDA device detection result
- `faster_whisper_gpu` - Model loading test result

#### `get_ram_info() -> RAMInfo`

Get RAM information:

```python
from voicepad_core import get_ram_info

ram = get_ram_info()
print(f"Total: {ram.total_gb:.1f} GB")
print(f"Available: {ram.available_gb:.1f} GB")
```

#### `get_cpu_info() -> CPUInfo`

Get CPU information:

```python
from voicepad_core import get_cpu_info

cpu = get_cpu_info()
print(f"Cores: {cpu.count}")
print(f"Model: {cpu.model_name}")
```

#### `get_model_recommendation(system_info, available_models) -> ModelRecommendation`

Get a model recommendation based on system capabilities:

```python
from voicepad_core import (
    get_model_recommendation,
    get_available_models,
    get_ram_info,
    get_cpu_info,
    gpu_diagnostics
)
from voicepad_core.diagnostics.models import SystemInfo

# Gather system info
system_info = SystemInfo(
    ram=get_ram_info(),
    cpu=get_cpu_info(),
    gpu_diagnostics=gpu_diagnostics()
)

available = get_available_models()

# Get recommendation
recommendation = get_model_recommendation(system_info, available)
print(f"Recommended model: {recommendation.recommended_model}")
print(f"Reason: {recommendation.reason}")
print(f"Alternatives: {recommendation.alternative_models}")
```

**Returns:** `ModelRecommendation` with:

- `recommended_model` - Best model for the system
- `recommended_device` - Recommended device (cuda/cpu/auto)
- `recommended_compute_type` - Recommended compute type
- `reason` - Explanation for the recommendation
- `alternative_models` - List of alternative models to try

#### `get_available_models() -> list[str]`

Get list of all available Whisper models:

```python
from voicepad_core import get_available_models

models = get_available_models()
# Returns: ["tiny", "tiny.en", "base", "base.en", "small", ...]
```

### Checking Functions

Individual GPU checks (called by `gpu_diagnostics()`):

```python
from voicepad_core import (
    check_nvidia_smi,
    check_ctranslate2_gpu,
    check_faster_whisper_gpu
)

# Check NVIDIA driver
result = check_nvidia_smi()
print(f"NVIDIA Driver: {result.success}")

# Check CUDA devices via CTranslate2
result = check_ctranslate2_gpu()
print(f"CUDA Devices: {result.cuda_device_count}")

# Test Whisper model loading on GPU
result = check_faster_whisper_gpu()
print(f"Whisper GPU: {result.success}")
```

## Configuration

Configuration files use YAML format. Create `voicepad.yaml` in your project or global config directory:

```yaml
# Recording paths
recordings_path: data/recordings
markdown_path: data/markdown

# Audio input device (null = system default)
input_device_index: null

# Recording filename prefix
recording_prefix: recording

# Transcription settings
transcription_model: tiny                    # tiny, small, medium, large-v3, etc.
transcription_device: auto                  # auto, cuda, cpu
transcription_compute_type: auto            # auto, float16, int8, float32
```

**Config Location Precedence:**

1. `./voicepad.yaml` (current working directory)
2. `~/.config/voicepad/voicepad.yaml` (home directory)
3. Built-in defaults

**Supported Models:**

All [OpenAI Whisper](https://github.com/openai/whisper) models:

- Tiny (fastest): `tiny`, `tiny.en`
- Base: `base`, `base.en`
- Small: `small`, `small.en`, `distil-small.en`
- Medium: `medium`, `medium.en`, `distil-medium.en`
- Large: `large-v1`, `large-v2`, `large-v3`, `distil-large-v2`, `distil-large-v3`
- Turbo (latest): `turbo`, `large-v3-turbo`

**Device Options:**

- `auto` - Automatically detect and use CUDA if available, otherwise CPU
- `cuda` - Force CUDA GPU usage (fails if CUDA unavailable)
- `cpu` - Force CPU usage

**Compute Type Options:**

- `auto` - float16 on CUDA, int8 on CPU
- `float16` - Full 16-bit floating point (best quality, more VRAM)
- `int8` - 8-bit integer quantization (less VRAM, minimal quality loss)
- `float32` - Full 32-bit floating point (rarely needed)

## Examples

### Example 1: Basic Recording and Transcription

```python
from voicepad_core import AudioRecorder, transcribe_audio, get_config
from pathlib import Path

config = get_config()

# Record audio
recorder = AudioRecorder(config)
print("Recording... (press Ctrl+C to stop)")
audio_file = recorder.start_recording()

# Transcribe
output = config.markdown_path / f"{audio_file.stem}.md"
stats = transcribe_audio(audio_file, output, config)

print(f"Saved to: {output}")
print(f"Words: {stats['word_count']}")
```

### Example 2: Check System and Get Recommendations

```python
from voicepad_core import (
    gpu_diagnostics,
    get_model_recommendation,
    get_available_models,
    get_ram_info,
    get_cpu_info
)
from voicepad_core.diagnostics.models import SystemInfo

system_info = SystemInfo(
    ram=get_ram_info(),
    cpu=get_cpu_info(),
    gpu_diagnostics=gpu_diagnostics()
)

recommendation = get_model_recommendation(system_info, get_available_models())

print(f"Recommended: {recommendation.recommended_model}")
print(f"Device: {recommendation.recommended_device}")
print(f"Reason: {recommendation.reason}")
```

### Example 3: Fixed Duration Recording

```python
from voicepad_core import AudioRecorder, get_config

config = get_config()
recorder = AudioRecorder(config)

# Record for 30 seconds
audio_file = recorder.start_recording(duration=30)
print(f"Recording for 30 seconds... Saved to: {audio_file}")
```

## See Also

- [voicepad CLI](../voicepad/index.md) - Command-line interface
- [GPU Acceleration](gpu-acceleration.md) - GPU setup and optimization
- [Configuration](index.md#configuration) - All config options
