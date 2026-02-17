---
icon: lucide/settings
---

# Configuration

Voicepad uses YAML-based configuration files to customize recording, transcription, and output settings. This guide explains the configuration system and how to use it.

## Configuration File Location

Voicepad looks for `voicepad.yaml` in the following locations (in order of precedence):

1. **Project directory** - `./voicepad.yaml`
2. **Home config** - `~/.config/voicepad/voicepad.yaml` (Linux/macOS) or `%APPDATA%\voicepad\voicepad.yaml` (Windows)
3. **Built-in defaults** - Used if no config file exists

!!! tip "Recommended Setup"
    Use project-level config (`./voicepad.yaml`) for project-specific settings, and home config for personal defaults.

## Creating a Configuration File

Create `voicepad.yaml` in your project directory:

```yaml
# File Paths
recordings_path: data/recordings
markdown_path: data/markdown

# Audio Input
input_device_index: 2
recording_prefix: recording

# Transcription Settings
transcription_model: medium
transcription_device: auto
transcription_compute_type: auto

# VAD Chunking
vad_enabled: true
vad_min_chunk_duration: 60.0
vad_threshold: 0.5
vad_min_silence_duration_ms: 1000
vad_speech_pad_ms: 400
```

## Configuration Sections

### File Paths

Control where audio and transcription files are saved.

```yaml
recordings_path: data/recordings   # Audio files (.wav)
markdown_path: data/markdown       # Transcription files (.md)
```

Both paths:

- Can be relative (to current directory) or absolute
- Are created automatically if they don't exist
- Support environment variable expansion

**See:** [Paths Reference](reference/paths.md)

### Audio Input

Configure your microphone and recording settings.

```yaml
input_device_index: 2           # Device index (null = system default)
recording_prefix: recording     # Filename prefix
```

**Find your device index:**

```bash
voicepad config input
```

**See:** [Audio Settings Reference](reference/audio.md)

### Transcription Settings

Choose the Whisper model and processing device.

```yaml
transcription_model: medium              # Model size/accuracy
transcription_device: auto               # auto, cuda, cpu
transcription_compute_type: auto         # Precision level
```

**Model Options:** `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`

**See:** [Transcription Reference](reference/transcription.md) | [Model Guide](features/transcription-models.md)

### VAD Chunking

Configure smart audio chunking for long recordings.

```yaml
vad_enabled: true                        # Enable VAD chunking
vad_min_chunk_duration: 60.0             # Minimum seconds before split
vad_threshold: 0.5                       # Speech detection sensitivity
vad_min_silence_duration_ms: 1000        # Silence needed for split
vad_speech_pad_ms: 400                   # Padding around speech
```

**When to enable:**

- ✅ Meetings, lectures, interviews (5+ minutes)
- ✅ Want real-time transcription progress
- ❌ Short recordings (< 5 minutes)
- ❌ Need exact audio file timing

**See:** [VAD Chunking Guide](features/vad-chunking.md) | [VAD Reference](reference/vad.md)

## Environment Variables

Override any setting using environment variables with the `VOICEPAD_` prefix:

```bash
# Override model
export VOICEPAD_TRANSCRIPTION_MODEL=tiny

# Override device
export VOICEPAD_TRANSCRIPTION_DEVICE=cpu

# Record with override
voicepad record start
```

Variable names use uppercase with underscores, for example:

- `vad_enabled` → `VOICEPAD_VAD_ENABLED=true`
- `transcription_model` → `VOICEPAD_TRANSCRIPTION_MODEL=small`

## CLI Overrides

Command-line arguments take precedence over config files:

```bash
# Override prefix for this recording only
voicepad record start --prefix meeting

# Override VAD settings
voicepad record start --vad --min-chunk-duration 120
```

## Precedence Order

Settings are resolved in this order (highest to lowest):

1. **CLI arguments** - `--prefix meeting`
2. **Environment variables** - `VOICEPAD_TRANSCRIPTION_MODEL=tiny`
3. **Project config** - `./voicepad.yaml`
4. **Home config** - `~/.config/voicepad/voicepad.yaml`
5. **Built-in defaults**

## Viewing Current Configuration

See what settings are currently active:

```bash
# Show recording config
voicepad record info

# Show transcription config
voicepad config transcription
```

## Example Configurations

### Minimal Configuration

Use defaults for everything except microphone:

```yaml
input_device_index: 2
```

### High-Quality Recording

Best accuracy, slower processing:

```yaml
transcription_model: large-v3
transcription_device: cuda
transcription_compute_type: float16
vad_enabled: false
```

### Fast Processing

Quick transcription, lower accuracy:

```yaml
transcription_model: tiny
transcription_device: cpu
transcription_compute_type: int8
vad_enabled: true
vad_min_chunk_duration: 30.0
```

### Meeting Recording

Optimized for meetings with background transcription:

```yaml
recording_prefix: meeting
transcription_model: medium
vad_enabled: true
vad_min_chunk_duration: 60.0
vad_threshold: 0.45  # Slightly more sensitive
```

## Validation and Errors

Voicepad validates your configuration on startup:

**Valid ranges:**

- `vad_min_chunk_duration`: 10.0 - 600.0 seconds
- `vad_threshold`: 0.0 - 1.0
- `vad_min_silence_duration_ms`: 100 - 5000 ms
- `vad_speech_pad_ms`: 0 - 2000 ms

**Invalid config example:**

```yaml
vad_min_chunk_duration: 5.0  # ❌ Too low, minimum is 10 seconds
```

Error message:

```text
ValidationError: vad_min_chunk_duration must be >= 10 seconds, got 5.0
```

## Next Steps

- **[Configuration Reference](reference/index.md)** - Detailed documentation for all settings
- **[VAD Settings](reference/vad.md)** - Deep dive into VAD parameters
- **[Transcription Settings](reference/transcription.md)** - Model and device options
- **[CLI Reference](cli/record.md)** - Command-line overrides
