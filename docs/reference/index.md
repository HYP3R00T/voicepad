---
icon: lucide/folder-tree
---

# Configuration Reference

Detailed documentation for all configuration settings.

## Setting Categories

### [File Paths](paths.md)

Control where audio and transcription files are saved.

- `recordings_path` - Audio output directory
- `markdown_path` - Transcription output directory

### [Audio Settings](audio.md)

Configure microphone input and recording behavior.

- `input_device_index` - Microphone selection
- `recording_prefix` - Filename prefix

### [Transcription Settings](transcription.md)

Choose model and processing device.

- `transcription_model` - Whisper model selection
- `transcription_device` - CPU/GPU selection
- `transcription_compute_type` - Precision level

### [VAD Settings](vad.md)

Fine-tune voice activity detection and chunking.

- `vad_enabled` - Enable/disable chunking
- `vad_min_chunk_duration` - Minimum chunk size
- `vad_threshold` - Speech detection sensitivity
- `vad_min_silence_duration_ms` - Silence detection
- `vad_speech_pad_ms` - Speech padding

## Configuration File

All settings are defined in `voicepad.yaml`:

```yaml
# Paths
recordings_path: data/recordings
markdown_path: data/markdown

# Audio
input_device_index: 2
recording_prefix: recording

# Transcription
transcription_model: medium
transcription_device: auto
transcription_compute_type: auto

# VAD
vad_enabled: true
vad_min_chunk_duration: 60.0
vad_threshold: 0.5
vad_min_silence_duration_ms: 1000
vad_speech_pad_ms: 400
```

## See Also

- **[Configuration Guide](../configuration.md)** - How to use configuration
- **[CLI Reference](../cli/index.md)** - Command-line overrides
