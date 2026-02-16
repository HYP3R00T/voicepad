# voicepad-core

Core audio recording library for Python projects. Provides device management, configuration, and background recording capabilities.

## Overview

`voicepad-core` is a lightweight library that handles low-level audio operations. It can be imported by other Python packages to add voice recording functionality without CLI overhead.

**Key Features:**

- List and select audio input devices
- Record audio in background threads
- Configure recording and output paths
- Thread-safe recording with stop signals
- Type-safe API with Pydantic models

## Installation

```bash
pip install voicepad-core
```

**Requirements:** Python 3.13+

**Dependencies:** `sounddevice`, `soundfile`, `pydantic`, `utilityhub-config`

## Public API

The library exports the following functions and classes from `voicepad_core`:

### Configuration

- `get_config()` - Get current configuration settings
- `get_config_with_metadata()` - Get config with source metadata
- `Config` - Pydantic model for configuration

### Device Management

- `get_input_devices()` - List all available audio input devices
- `get_device_by_index(index)` - Get device info by OS index
- `AudioDevice` - TypedDict for device information

### Recording

- `record_voice(device_index, output_dir, prefix)` - Interactive recording with user control
- `capture_audio_background(device_index, output_dir, prefix, stop_event)` - Background recording with threading

### Path Utilities

- `get_timestamp()` - Generate timestamp string (YYYYMMDD_HHMMSS)
- `get_recording_path(output_dir, prefix)` - Generate WAV file path
- `get_transcript_path(output_dir, prefix)` - Generate markdown file path

## Usage Example

```python
from voicepad_core import get_input_devices, record_voice

# List available devices
devices = get_input_devices()
for device in devices:
    print(f"{device['index']}: {device['name']}")

# Record audio interactively
record_voice(
    device_index=0,
    output_dir="recordings",
    prefix="meeting"
)
# Saves to: recordings/meeting_20260216_143022.wav
```

## Configuration

The library uses `utilityhub-config` for configuration management. Settings can be defined in `voicepad.yaml`:

```yaml
recordings_path: "data/recordings/"
markdown_path: "data/markdown/"
```

Configuration is loaded automatically. Default paths are used if no config file exists.

???+ note "Thread-Safe Recording"
    Use `capture_audio_background()` with `threading.Event` for applications that need non-blocking recording. The function returns when the stop event is set.

## Source Code

Implementation details: `packages/voicepad-core/src/voicepad_core/` in the repository

- Configuration: See `voicepad_core/config/settings.py` for Pydantic models
- Recording engine: See `voicepad_core/voice/recorder.py` for audio capture logic
- Utilities: See `voicepad_core/voice/utils.py` for path generation
