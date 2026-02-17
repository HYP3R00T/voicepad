---
icon: lucide/list
---

# Features

Voicepad provides powerful features for voice recording and transcription.

## Core Features

### [Recording](recording.md)

High-quality audio capture from any input device with customizable settings.

- 16kHz mono WAV format
- Configurable input devices
- Fixed or unlimited duration
- Automatic file management

### [VAD Chunking](vad-chunking.md)

Smart audio splitting at natural speech boundaries using AI-powered voice detection.

- Real-time chunk detection
- Configurable minimum duration
- Adjustable sensitivity
- Natural pause detection

### [Background Transcription](background-transcription.md)

Real-time transcription processing while recording continues.

- Live markdown updates
- Model caching for speed
- Thread-safe file writing
- Single merged output file

### [Transcription Models](transcription-models.md)

Multiple Whisper model options for different accuracy/speed tradeoffs.

- Tiny to Large-v3 models
- English-only variants
- Distilled models
- GPU acceleration support

## Feature Comparison

| Feature | Without VAD | With VAD |
|---------|-------------|----------|
| **Transcription timing** | After recording | During recording |
| **Markdown updates** | Once, at end | Live, per chunk |
| **Wait time** | Full transcription | ~0 seconds |
| **Memory usage** | Entire recording | Per-chunk buffer |
| **Best for** | Short recordings | Long recordings (10+ min) |

## Quick Links

- **[Getting Started](../getting-started.md)** - First-time setup
- **[Configuration](../configuration.md)** - Customize settings
- **[CLI Reference](../cli/index.md)** - Command usage
