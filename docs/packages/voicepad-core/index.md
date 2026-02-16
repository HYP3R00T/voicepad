# voicepad-core

Configuration and GPU diagnostics library for Voicepad. Provides config loading and GPU readiness checks.

## Overview

`voicepad-core` is a lightweight library that handles configuration and GPU diagnostics. It can be imported by other Python packages to reuse config loading or GPU readiness checks without CLI overhead.

**Key Features:**

- Load configuration settings for Voicepad
- Report GPU readiness for faster-whisper and CTranslate2
- Provide type-safe models for diagnostics output

## Installation

```bash
pip install voicepad-core
```

**Requirements:** Python 3.13+

**Dependencies:** `faster-whisper`, `pydantic`, `sounddevice`, `soundfile`, `utilityhub-config`

## Public API

The library exports the following functions and classes from `voicepad_core`:

### Configuration

- `get_config()` - Get current configuration settings
- `get_config_with_metadata()` - Get config with source metadata
- `Config` - Pydantic model for configuration

### GPU Diagnostics

- `gpu_diagnostics()` - Run all GPU checks and return a report
- `check_nvidia_smi()` - Validate NVIDIA driver availability
- `check_ctranslate2_gpu()` - Detect CUDA devices via CTranslate2
- `check_faster_whisper_gpu()` - Attempt to load a Whisper model on GPU
- `GPUDiagnosticsReport` - Pydantic model with GPU diagnostics results

## Usage Example

```python
from voicepad_core import get_config, gpu_diagnostics

config = get_config()
print(config.recordings_path)

report = gpu_diagnostics()
print(report.model_dump_json(indent=2))
```

## Configuration

The library uses `utilityhub-config` for configuration management. Settings can be defined in `voicepad.yaml`:

```yaml
recordings_path: "data/recordings/"
markdown_path: "data/markdown/"
```

Configuration is loaded automatically. Default paths are used if no config file exists.

## CLI Diagnostics

The package includes a `voicepad-core` console script that prints a JSON GPU diagnostics report.
