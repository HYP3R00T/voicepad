# voicepad CLI

Command-line interface for recording audio and managing configuration. Built with Typer for a clean, user-friendly experience.

## Installation

```bash
pip install voicepad
```

**Requirements:** Python 3.13+

**Dependencies:** `voicepad-core`, `typer`

## Quick Start

```bash
# List audio input devices and configured default
voicepad config input

# Start recording (press Ctrl+C to stop)
voicepad record start

# Check system capabilities (RAM, CPU, GPU)
voicepad config system
```

## Command Reference

### `voicepad record start`

Start recording audio from the configured input device. The recording will automatically transcribe after stopping unless `--no-transcribe` is used.

**Usage:**

```bash
voicepad record start [OPTIONS]
```

**Options:**

- `--prefix TEXT, -p TEXT` — Custom filename prefix (overrides config)
- `--duration FLOAT, -d FLOAT` — Record for a fixed number of seconds (optional)
- `--transcribe / --no-transcribe` — Enable/disable auto-transcription (default: enabled)

**Examples:**

```bash
# Record indefinitely (press Ctrl+C to stop)
voicepad record start

# Record for 30 seconds with transcription disabled
voicepad record start --duration 30 --no-transcribe

# Record with custom filename prefix
voicepad record start --prefix important_meeting
```

**Output:**

- Audio file saved to: `recordings_path` directory
- Transcription saved to: `markdown_path` directory (if `--transcribe` is enabled)
- Filename format: `{prefix}_{timestamp}.wav`

---

### `voicepad record info`

Display the current recording configuration, including paths, device, and filename prefix.

**Usage:**

```bash
voicepad record info
```

**Output:**

```sh
Current Recording Configuration
============================================================
Input device index: default (system)
Recordings directory: data/recordings
Filename prefix: recording
============================================================
[OK] Recordings directory exists
   5 recording(s) found

Tip: Use 'voicepad config input' to view and configure audio devices
```

---

### `voicepad config input`

List available audio input devices and display the configured default input device.

**Usage:**

```bash
voicepad config input
```

**Output:**

```sh
Configured default input device:
  2: Microphone (High Definition Audio Device)

Available audio input devices:
------------------------------------------------------------
[0] Speakers (Realtek High Definition Audio) (2 in, 48000Hz)
[1] Line In (Realtek High Definition Audio) (2 in, 48000Hz)
[2] Microphone (High Definition Audio Device) (1 in, 16000Hz) (configured)
------------------------------------------------------------
Config file: /home/user/.config/voicepad/voicepad.yaml
Set input_device_index in that file to persist the default.
```

**How to configure:**

Edit `voicepad.yaml` (in working directory or `~/.config/voicepad/voicepad.yaml`):

```yaml
input_device_index: 2  # Use the device index shown above
```

---

### `voicepad config system`

Display comprehensive system information including RAM, CPU, and GPU diagnostics. Useful for troubleshooting and understanding what models your system supports.

**Usage:**

```bash
voicepad config system
```

**Output:**

```sh
System Information
============================================================

RAM:
   Total: 16.0 GB
   Available: 12.5 GB
   [OK] Sufficient RAM for transcription

CPU:
   Cores: 8
   Model: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz

GPU:
   NVIDIA Driver: [OK] Detected
   CUDA Devices: 1 device(s) available
   faster-whisper GPU: [OK] Compatible
============================================================
```

**What it checks:**

- **RAM**: Total and available memory (8GB+ recommended)
- **CPU**: Core count and model name
- **NVIDIA Driver**: Whether NVIDIA GPU drivers are installed
- **CUDA**: Detection of CUDA compute capability
- **faster-whisper GPU**: Compatibility with GPU acceleration

---

### `voicepad config recommend`

Get a model recommendation based on your system capabilities (RAM, CPU, GPU). Shows the recommended model, device, compute type, and alternative options.

**Usage:**

```bash
voicepad config recommend
```

**Output:**

```sh
Model Recommendation Based on System Capabilities
============================================================

Recommended Configuration:
   Model: tiny
   Device: auto
   Compute Type: auto

Reason:
   Limited RAM reduces effectiveness of larger models. "tiny" model
   offers a good balance of speed and accuracy for systems with 4-8 GB RAM.

Alternative Models:
   - tiny.en
   - base
   - base.en

To apply this configuration, add to voicepad.yaml:
   transcription_model: tiny
   transcription_device: auto
   transcription_compute_type: auto
============================================================
```

---

### `voicepad config transcription`

View the current transcription configuration and compare it with system recommendations.

**Usage:**

```bash
voicepad config transcription
```

**Output:**

```sh
Transcription Configuration
============================================================

Current Settings:
   Model: tiny
   Device: auto
   Compute Type: auto

Recommended Settings:
   [OK] Your configuration matches the recommendation!
============================================================
Config file: /home/user/.config/voicepad/voicepad.yaml
Set input_device_index in that file to persist the default.
```

---

### `voicepad config models`

List all available Whisper models, organized by size and type.

**Usage:**

```bash
voicepad config models
```

**Output:**

```sh
Available Whisper Models
============================================================

Tiny:
   - tiny
   - tiny.en

Base:
   - base
   - base.en

Small:
   - small
   - small.en
   - distil-small.en (English-only)

Medium:
   - medium
   - medium.en
   - distil-medium.en (English-only)

Large:
   - distil-large-v2 (English-only)
   - distil-large-v3 (English-only)
   - large-v1
   - large-v2
   - large-v3
   - large-v3-turbo

Turbo:
   - turbo

============================================================
Total: 17 models available

Tip: Use 'voicepad config recommend' to get a model recommendation
     based on your system capabilities.
```

**Model Naming:**

- Models ending with `.en` are **English-only** (slightly smaller and faster)
- Models starting with `distil-` are **distilled versions** (faster, with minimal accuracy loss)
- `turbo` is a **latest generation model** (fast and accurate, requires more VRAM)

---

## Configuration

Configuration is stored in `voicepad.yaml`. The CLI looks for it in the following order:

1. `./voicepad.yaml` (current working directory)
2. `~/.config/voicepad/voicepad.yaml` (home config directory)
3. Built-in defaults (if no config file exists)

### Configuration Fields

See [voicepad-core Configuration](../voicepad-core/index.md#configuration) for complete details on all available settings.

## Usage Examples

### Example 1: Record and Transcribe a Meeting

```bash
# Set your preferred microphone (list devices first)
voicepad config input

# Record the meeting (will auto-transcribe)
voicepad record start --prefix team_meeting

# Output:
# - data/recordings/team_meeting_20260218_103045.wav
# - data/markdown/team_meeting_20260218_103045.md
```

### Example 2: Check System Before Recording

```bash
# See what model works best for your system
voicepad config system

# Get specific recommendations
voicepad config recommend

# Check your current transcription config
voicepad config transcription

# Then record
voicepad record start
```

### Example 3: Record Without Transcribing

```bash
# Record audio only, skip transcription
voicepad record start --no-transcribe

# Transcribe later using the voicepad-core library
```

## See Also

- [voicepad-core](../voicepad-core/index.md) - Python library API and programmatic usage
- [GPU Acceleration](../voicepad-core/gpu-acceleration.md) - GPU setup guide
