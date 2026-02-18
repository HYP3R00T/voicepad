---
icon: lucide/settings-2
---

# config commands

Configuration and system information commands.

## voicepad config show {#show}

Show current configuration and where values are loaded from.

### Synopsis

```bash
voicepad config show
```

### Description

Displays all configuration fields with their current values and indicates where each value is loaded from (defaults, environment variables, or configuration files). This is useful for understanding your active configuration and troubleshooting configuration issues.

### Output Example

```text
                    Voicepad Configuration
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Field                     ┃ Value       ┃ Source                   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ transcription_model       │ medium      │ file (voicepad.yaml)     │
│ transcription_device      │ cuda        │ file (voicepad.yaml)     │
│ transcription_compute_... │ float16     │ file (voicepad.yaml)     │
│ input_device_index        │ 2           │ file (voicepad.yaml)     │
│ sample_rate               │ 16000       │ default                  │
│ chunk_duration            │ 30          │ environment variable     │
│ output_directory          │ ./output    │ default                  │
└───────────────────────────┴─────────────┴──────────────────────────┘

Config file: /home/user/.config/voicepad/voicepad.yaml
Set input_device_index in that file to persist the default.
```

### Configuration Sources

Values can come from multiple sources, shown in priority order (highest to lowest):

1. **Environment variables** - Settings from `VOICEPAD_*` environment variables
2. **Configuration file** - Settings from `voicepad.yaml` (project or user config)
3. **Default** - Built-in default values

### Use Cases

- **Configuration audit** - See all active settings at a glance
- **Troubleshooting** - Identify which config file or environment variable is being used
- **Debugging** - Verify that configuration changes are being loaded correctly
- **Documentation** - Understand which values have been customized vs defaults

### Exit Codes

- `0` - Always succeeds

---

## voicepad config input {#input}

List available audio input devices and show the configured default.

### Synopsis

```bash
voicepad config input
```

### Description

Displays all audio input devices detected on your system along with the currently configured default device. Use this to find the correct device index for your microphone.

### Output Example

```text
Configured default input device:
  2: Microphone (High Definition Audio Device)

Available audio input devices:
------------------------------------------------------------
[0] Speakers (Realtek High Definition Audio) (2 in, 48000Hz)
[1] Line In (Realtek High Definition Audio) (2 in, 48000Hz)
[2] Microphone (High Definition Audio Device) (1 in, 16000Hz) (configured)
[3] USB Microphone (USB Audio Device) (1 in, 48000Hz)
------------------------------------------------------------
Config file: /home/user/.config/voicepad/voicepad.yaml
Set input_device_index in that file to persist the default.
```

### How to Configure

Edit `voicepad.yaml`:

```yaml
input_device_index: 3  # Use USB microphone
```

Or use null for system default:

```yaml
input_device_index: null
```

### Exit Codes

- `0` - Always succeeds

---

## voicepad config system {#system}

Display comprehensive system information including RAM, CPU, and GPU diagnostics.

### Synopsis

```bash
voicepad config system
```

### Description

Shows system capabilities for transcription, including:

- RAM (total and available)
- CPU (cores and model)
- GPU (NVIDIA driver, CUDA, model compatibility)

### Output Example

```text
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

### GPU Status Indicators

- **[OK]** - Component available and working
- **[MISSING]** - Component not found
- **[ERROR]** - Component found but not working

### Exit Codes

- `0` - Always succeeds

---

## voicepad config transcription {#transcription}

View current transcription configuration and compare with recommendations.

### Synopsis

```bash
voicepad config transcription
```

### Description

Shows the active transcription settings and indicates if they match the recommended configuration for your system.

### Output Example

```text
Transcription Configuration
============================================================

Current Settings:
   Model: medium
   Device: auto
   Compute Type: auto

Recommended Settings:
   [OK] Your configuration matches the recommendation!
============================================================
Config file: /home/user/.config/voicepad/voicepad.yaml
```

### Exit Codes

- `0` - Always succeeds

---

## voicepad config recommend {#recommend}

Get a model recommendation based on system capabilities.

### Synopsis

```bash
voicepad config recommend
```

### Description

Analyzes your system (RAM, CPU, GPU) and recommends the best transcription model and settings.

### Output Example

```text
Model Recommendation Based on System Capabilities
============================================================

Recommended Configuration:
   Model: medium
   Device: cuda
   Compute Type: float16

Reason:
   GPU available for 4-5x faster transcription. 16GB RAM supports
   medium model with good accuracy/speed balance.

Alternative Models:
   - small (faster, slightly less accurate)
   - large-v3 (slower, best accuracy)

To apply this configuration, add to voicepad.yaml:
   transcription_model: medium
   transcription_device: cuda
   transcription_compute_type: float16
============================================================
```

### Recommendation Logic

**With GPU (CUDA available):**

- 8+ GB RAM → `medium` model
- 4-8 GB RAM → `small` model
- < 4 GB RAM → `tiny` model

**Without GPU (CPU only):**

- 8+ GB RAM → `small` model (medium too slow on CPU)
- 4-8 GB RAM → `tiny` model
- < 4 GB RAM → `tiny` model

### Exit Codes

- `0` - Always succeeds

---

## voicepad config models {#models}

List all available Whisper models.

### Synopsis

```bash
voicepad config models
```

### Description

Displays all Whisper models organized by size and type.

### Output Example

```text
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

### Model Types

**Multilingual models:**

- No suffix (e.g., `tiny`, `medium`) - Support 99 languages

**English-only models:**

- `.en` suffix (e.g., `tiny.en`, `medium.en`) - English only, slightly faster

**Distilled models:**

- `distil-` prefix (e.g., `distil-medium.en`) - Faster with minimal accuracy loss

### Exit Codes

- `0` - Always succeeds

---

## Common Workflows

### Initial Setup

```bash
# 1. Check system capabilities
voicepad config system

# 2. Get recommendation
voicepad config recommend

# 3. Find microphone
voicepad config input

# 4. Edit voicepad.yaml with recommended settings

# 5. Verify configuration
voicepad config show
voicepad config transcription
voicepad record info
```

### Troubleshooting Slow Transcription

```bash
# 1. Check if GPU is available
voicepad config system

# 2. If no GPU, get CPU recommendation
voicepad config recommend

# 3. Check current settings
voicepad config show
voicepad config transcription

# 4. Try a smaller model
# Edit voicepad.yaml:
#   transcription_model: tiny
```

### Model Comparison

```bash
# See all options
voicepad config models

# Get recommendation
voicepad config recommend

# Test different models by editing voicepad.yaml
```

## See Also

- **[Configuration Guide](../configuration.md)** - YAML configuration
- **[Transcription Models](../features/transcription-models.md)** - Model comparison
- **[Transcription Reference](../reference/transcription.md)** - Setting details
- **[GPU Acceleration](../packages/voicepad-core/gpu-acceleration/index.md)** - GPU setup
