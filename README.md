# VoicePad

Voice recording and GPU-accelerated transcription tool. Run anywhere with `uvx voicepad`!

## Features

- 🎙️ **Voice Activity Detection (VAD)**: Automatic silence detection using Silero VAD
- 🎯 **Continuous Recording**: Record until you press Enter
- 🚀 **GPU-Accelerated Transcription**: Fast transcription using faster-whisper with auto-detection
- ⚙️ **Smart Configuration**: Set defaults once, use them everywhere
- 🎬 **Real-Time Streaming**: Watch transcription appear as it processes
- 📝 **Flexible Output**: Save to configured markdown directory
- 🖥️ **Terminal UI**: Interactive TUI built with Textual
- 🔌 **Standalone Tool**: Use with `uvx` - no manual installation required
- 🎮 **Auto GPU Detection**: Detects GPU and recommends optimal model

## Installation

### Quick Start (Recommended)

Run VoicePad instantly without installation using `uvx`:

```bash
# Check system status and GPU availability
uvx voicepad config init

# Start the terminal UI
uvx voicepad

# Record audio
uvx voicepad audio record

# Transcribe audio file
uvx voicepad transcribe transcribe recording.wav
```

### Development Installation

For development or local modifications:

```bash
git clone https://github.com/HYP3R00T/voicepad.git
cd voicepad
uv sync
```

## System Requirements

- Python 3.12+
- Audio input device
- NVIDIA GPU (optional, for GPU acceleration)
- NVIDIA drivers 545+ (for CUDA 12.4 support)

### GPU Support

VoicePad automatically includes PyTorch with CUDA 12.4 support and will use your GPU when available.

If ROCm or Apple MPS is detected, voicepad will fall back to CPU (faster-whisper currently supports CUDA only) and warn you.

Check your system status:

```bash
uvx voicepad config init
```

If you have an NVIDIA GPU but it's not detected, ensure your drivers are up to date:

- Windows: Download from [NVIDIA website](https://www.nvidia.com/download/index.aspx)
- Linux: Use your distribution's driver manager or `nvidia-driver-XXX`

## Quick Start

### Record Audio

```bash
# Record audio (stops when you press Enter)
uvx voicepad audio record --device 0

# List available audio devices
uvx voicepad audio devices
```

### Record and Transcribe Simultaneously

```bash
# Record and transcribe in real-time (NEW!)
# Transcription happens every 30 seconds while you're still recording
uvx voicepad audio record-and-transcribe

# With custom model
uvx voicepad audio record-and-transcribe --model small

# Change poll interval for transcription (default: 30s)
uvx voicepad audio record-and-transcribe --poll-interval 15

# Use specific device
uvx voicepad audio record-and-transcribe --compute-device cuda
```

### Transcribe Audio

```bash
# Transcribe with auto-detected settings (recommended)
uvx voicepad transcribe transcribe recording.wav

# Stream transcription output in real-time
uvx voicepad transcribe transcribe recording.wav --stream

# Use different model sizes
uvx voicepad transcribe transcribe recording.wav --model tiny    # Fastest
uvx voicepad transcribe transcribe recording.wav --model small   # Balanced (default)
uvx voicepad transcribe transcribe recording.wav --model medium  # Better quality

# Specify device
uvx voicepad transcribe transcribe recording.wav --device cuda  # GPU
uvx voicepad transcribe transcribe recording.wav --device cpu   # CPU

# Specify compute type (auto chooses float16 on CUDA, int8 on CPU)
uvx voicepad transcribe transcribe recording.wav --compute-type int8

# Specify language (auto-detect by default)
uvx voicepad transcribe transcribe recording.wav --language en

# Save to specific file
uvx voicepad transcribe transcribe recording.wav --output transcript.txt
```

### Launch TUI

```bash
# Start the terminal UI
uvx voicepad
```

### Configuration

Set your preferred transcription model and device as defaults:

```bash
# Show current configuration
uvx voicepad config show

# Set default model (auto-detect recommended)
uvx voicepad config set-model auto      # Auto-detect best model for your GPU
uvx voicepad config set-model small     # Or use specific model

# Set default device
uvx voicepad config set-device cuda     # Use GPU
uvx voicepad config set-device auto     # Auto-detect

# Verify configuration and create directories
uvx voicepad config verify

# Initialize system (check GPU and get recommendations)
uvx voicepad config init
```

All transcription commands will use your configured defaults. CLI arguments override config:

```bash
# Uses your config settings
uvx voicepad transcribe transcribe audio.wav

# Overrides config with --model large-v3
uvx voicepad transcribe transcribe audio.wav --model large-v3
```

## Configuration File

Create or edit `voicepad.yaml` in your project or `~/.config/voicepad/voicepad.yaml` globally:

```yaml
recordings_path: data/recordings
markdown_path: data/markdown

transcription:
  model: auto              # Supported: tiny, tiny.en, base, base.en, small, small.en, distil-small.en, medium, medium.en, distil-medium.en, large-v1, large-v2, large-v3, large, distil-large-v2, distil-large-v3, large-v3-turbo, turbo
  device: auto             # cuda, cpu, auto
  compute_type: auto       # float16, int8, int8_float16
  language: null           # null for auto-detect, or language code like en, es, fr
```

Auto compute type picks `float16` on CUDA GPUs and `int8` on CPU (including ROCm/MPS fallback).

**Precedence** (higher overrides lower):

1. Defaults (in code)
2. Global config (`~/.config/voicepad/voicepad.yaml`)
3. Project config (`./voicepad.yaml`)
4. Environment variables (`VOICEPAD_TRANSCRIPTION_MODEL=small`)
5. CLI arguments (`--model large-v3`)

## Project Structure

```text
voicepad/
├── src/voicepad/
│   ├── audio/           # Audio recording and VAD
│   │   ├── scanner.py   # Core recording logic
│   │   ├── utils.py     # Audio utilities
│   │   ├── cli.py       # Audio CLI commands
│   │   └── legacy.py    # Historical implementations
│   ├── config/          # Configuration management
│   │   ├── settings.py  # Config models
│   │   └── cli.py       # Config CLI commands
│   ├── transcription/   # Audio-to-text conversion
│   │   ├── transcriber.py  # Whisper transcription
│   │   └── cli.py          # Transcription CLI
│   ├── ui/              # Terminal UI
│   │   ├── voicepad_ui.py  # Main TUI
│   │   └── components/     # UI components
│   ├── system_utils.py  # GPU and system checks
│   └── main.py          # CLI entry point
```

## Development

### Code Standards

- Follow PEP 8 and project coding standards (see `.github/copilot-instructions.md`)
- Use `snake_case` for functions/variables, `PascalCase` for classes
- Type hints required for all functions
- Use Pydantic for data validation

### Linting and Formatting

```bash
# Format code
ruff format

# Lint code
ruff check
```

## Available Commands

### Audio Commands

- `audio record` - Record audio with VAD
- `audio list-devices` - List audio devices

### Configuration Commands

- `config init` - Check system capabilities and get GPU recommendations
- `config show` - Display current configuration
- `config set-model` - Set default transcription model
- `config set-device` - Set default device (cuda/cpu/auto)
- `config verify` - Verify configuration and create directories

### Transcription Commands

- `transcribe transcribe` - Transcribe audio file
  - `--stream` / `-s` - Stream output in real-time
  - `--model` / `-m` - Override configured model
  - `--device` / `-d` - Override configured device
  - `--language` / `-l` - Specify language (auto-detect by default)

## Available Models

| Model | Size | Speed | Accuracy | VRAM (fp16) |
| --- | --- | --- | --- | --- |
| tiny | 39M | ⚡⚡⚡⚡⚡ | ⭐ | <1 GB |
| base | 74M | ⚡⚡⚡⚡ | ⭐⭐ | <1 GB |
| small | 244M | ⚡⚡⚡ | ⭐⭐⭐ | 1-2 GB |
| medium | 769M | ⚡⚡ | ⭐⭐⭐⭐ | 2-3 GB |
| large-v2 | 1550M | ⚡ | ⭐⭐⭐⭐⭐ | ~4.7 GB |
| large-v3 | 1550M | ⚡ | ⭐⭐⭐⭐⭐ | ~4.7 GB |
| turbo | 809M | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ~3-4 GB |
| distil-large-v3 | 756M | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ~3-4 GB |

Supported model names (no custom paths): tiny, tiny.en, base, base.en, small, small.en, distil-small.en, medium, medium.en, distil-medium.en, large-v1, large-v2, large-v3, large, distil-large-v2, distil-large-v3, large-v3-turbo, turbo.

**Recommendation**: Use `--model auto` (default) to let voicepad choose the best model for your GPU.

## How It Works

1. **Recording**: Uses VAD (Voice Activity Detection) to capture audio continuously
2. **Processing**: Saves recordings with timestamps to configured directory
3. **Transcription**: Uses faster-whisper with GPU acceleration (when available)
4. **Output**: Generates text transcriptions alongside audio files

## Performance

- **GPU Mode**: ~10-50x faster than CPU (depending on GPU)
- **CPU Mode**: Fully functional, slower but works everywhere
- **Model Sizes**: Trade-off between speed and accuracy

## Documentation

For more detailed information, see:

- [Configuration Guide](docs/CONFIG.md) - Complete config system documentation
- [GPU Detection Guide](docs/GPU_DETECTION.md) - How GPU detection works, VRAM requirements
- [Model Comparison](docs/MODELS.md) - Detailed model information and recommendations

## License

See LICENSE file for details.
