# VoicePad

Voice recording and GPU-accelerated transcription tool. Effortlessly capture and transcribe audio with intelligent defaults and GPU support.

## Features

- 🎙️ **Simple Recording**: Record audio with one command, press Ctrl+C to stop
- 🚀 **GPU-Accelerated Transcription**: Fast transcription using faster-whisper with optional CUDA support
- ⚙️ **Smart Configuration**: Set defaults once, use them everywhere
- 📝 **Automatic Output**: Saves both audio and markdown transcriptions to configured directories
- 🎯 **Auto Model Selection**: System diagnostics recommend optimal model for your hardware
- 🔧 **System Inspection**: Check GPU availability, supported models, and system capabilities

This is a **two-package architecture**:

- **`voicepad`** - Command-line interface for recording and configuration (in [packages/voicepad/](packages/voicepad/))
- **`voicepad-core`** - Python library with recording, transcription, and diagnostics (in [packages/voicepad-core/](packages/voicepad-core/))

## Installation

### Using `uvx` (Recommended)

Run VoicePad instantly without manual installation:

```bash
# List audio input devices and set a default
uvx voicepad config input

# Check system capabilities and get model recommendations
uvx voicepad config system

# Start recording (output directory created automatically)
uvx voicepad record start
```

### Local Installation

For development or local use:

```bash
git clone https://github.com/HYP3R00T/voicepad.git
cd voicepad
uv sync
uv run voicepad config input
```

## System Requirements

- **Python**: 3.13+
- **Audio Device**: Microphone or audio input device
- **GPU (Optional)**: NVIDIA GPU for 4-5x faster transcription

## Quick Start

### Record Audio

```bash
# Start recording (press Ctrl+C to stop)
voicepad record start

# Display current recording configuration
voicepad record info

# Record with auto-transcription disabled
voicepad record start --no-transcribe

# Record for a fixed duration (seconds)
voicepad record start --duration 30

# Use a custom filename prefix
voicepad record start --prefix my_recording
```

### Configure Input Device

```bash
# List available audio input devices
voicepad config input

# Edit voicepad.yaml to set a default input device:
# input_device_index: 2
```

### Check System Capabilities

```bash
# Display RAM, CPU, and GPU information
voicepad config system

# Get model recommendations based on your system
voicepad config recommend

# View current transcription configuration
voicepad config transcription

# List all available Whisper models
voicepad config models
```

## Configuration

Configuration is managed through `voicepad.yaml` in the working directory or `~/.config/voicepad/voicepad.yaml` globally.

### Configuration File Structure

```yaml
# Paths for saving recordings and transcriptions
recordings_path: data/recordings
markdown_path: data/markdown

# Audio device (null for default system audio input)
input_device_index: null

# Filename prefix for recordings
recording_prefix: recording

# Transcription settings
transcription_model: tiny              # See available models below
transcription_device: auto             # auto, cuda, or cpu
transcription_compute_type: auto       # auto, float16, int8, or float32
```

### Configuration Precedence (highest to lowest)

1. CLI command arguments
2. Environment variables (e.g., `VOICEPAD_TRANSCRIPTION_MODEL=small`)
3. Project config (`./voicepad.yaml`)
4. Global config (`~/.config/voicepad/voicepad.yaml`)
5. Built-in defaults

### GPU Acceleration

Install GPU support for 4-5x faster transcription:

```bash
# Install with GPU support
pip install voicepad-core[gpu]

# Verify GPU is available
voicepad config system
```

See the [GPU Acceleration Guide](docs/packages/voicepad-core/gpu-acceleration.md) for detailed setup instructions.

## Available Whisper Models

Use `voicepad config models` to list available models. Smaller models are faster but less accurate; larger models are slower but more accurate.

| Model | Size | Speed (CPU) | Accuracy | VRAM (GPU) | Language |
|-------|------|-------------|----------|-----------|----------|
| tiny | 39M | ⚡⚡⚡⚡⚡ Very Fast | ⭐ Low | <1 GB | Multi |
| base | 74M | ⚡⚡⚡⚡ Fast | ⭐⭐ | <1 GB | Multi |
| small | 244M | ⚡⚡⚡ Moderate | ⭐⭐⭐ | 1-2 GB | Multi |
| medium | 769M | ⚡⚡ Slow | ⭐⭐⭐⭐ | 2-3 GB | Multi |
| large-v2 | 1.5B | ⚡ Very Slow | ⭐⭐⭐⭐⭐ Excellent | ~4.7 GB | Multi |
| large-v3 | 1.5B | ⚡ Very Slow | ⭐⭐⭐⭐⭐ Excellent | ~4.7 GB | Multi |
| turbo | 809M | ⚡⚡⚡ Moderate | ⭐⭐⭐⭐⭐ | 3-4 GB | Multi |
| distil-small.en | 134M | ⚡⚡⚡ Moderate | ⭐⭐⭐ | <1 GB | English Only |
| distil-medium.en | 394M | ⚡⚡ Slow | ⭐⭐⭐⭐ | 1-2 GB | English Only |
| distil-large-v2 | 756M | ⚡⚡ Slow | ⭐⭐⭐⭐⭐ | 3-4 GB | English Only |

**Tip**: Run `voicepad config recommend` to get a model recommendation based on your system resources.

## Package Documentation

- **[voicepad](docs/packages/voicepad/index.md)** - CLI commands and usage guide
- **[voicepad-core](docs/packages/voicepad-core/index.md)** - Python library API reference
- **[GPU Acceleration](docs/packages/voicepad-core/gpu-acceleration.md)** - GPU setup and optimization

## Development

### Code Standards

Follow the project coding standards:

- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- **Type Hints**: Required for all functions and class attributes
- **Formatting**: [PEP 8](https://peps.python.org/pep-0008/)
- **Validation**: Use Pydantic for data models

See [.github/copilot-instructions.md](.github/copilot-instructions.md) for complete guidelines.

### Formatting and Linting

```bash
# Format all code
ruff format

# Check for linting issues
ruff check

# Type check Python code
ty check
```

## Project Structure

```text
voicepad (root)
├── packages/
│   ├── voicepad/                      # CLI package (Typer)
│   │   └── src/voicepad/
│   │       ├── cli/
│   │       │   ├── record.py          # record start, record info
│   │       │   └── config.py          # config input, system, recommend, etc.
│   │       └── __main__.py            # CLI entry point
│   │
│   └── voicepad-core/                 # Core library (Pydantic + faster-whisper)
│       └── src/voicepad_core/
│           ├── config/
│           │   └── settings.py        # Config model and loading
│           ├── recorder.py            # AudioRecorder class
│           ├── transcription.py       # transcribe_audio() function
│           └── diagnostics/
│               ├── system.py          # System info (RAM, CPU)
│               ├── gpu.py             # GPU detection and checks
│               ├── models.py          # Recommendation logic
│               └── recommendations.py # get_model_recommendation()
│
└── docs/
    ├── packages/voicepad/             # CLI documentation
    ├── packages/voicepad-core/        # Library documentation
    └── packages/voicepad-core/gpu-acceleration.md
```

## License

See [LICENSE](LICENSE) file for details.
