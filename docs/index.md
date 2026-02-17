---
icon: lucide/rocket
---

# Voicepad

Voice recording with GPU-accelerated transcription and smart audio chunking.

## What is Voicepad?

Voicepad is a command-line tool for recording audio and transcribing it using OpenAI's Whisper models. It features VAD (Voice Activity Detection) chunking that automatically splits long recordings at natural speech boundaries, enabling real-time background transcription.

**Key Features:**

- 🎙️ **High-Quality Recording** - 16kHz mono WAV files from any audio input device
- ✂️ **Smart Chunking** - AI-powered splitting at natural speech boundaries
- ⚡ **Background Transcription** - Real-time processing while you record
- 🚀 **GPU Acceleration** - 4-5x faster with CUDA support
- 📝 **Markdown Output** - Clean, formatted transcriptions with metadata
- ⚙️ **Flexible Configuration** - YAML-based settings for all options

## Quick Start

```bash
# Install
pip install voicepad

# First recording
voicepad record start

# With smart chunking (for long recordings)
voicepad record start --vad --min-chunk-duration 60
```

**New user?** Start with the [Getting Started Guide](getting-started.md).

## Documentation

### For Users

- **[Getting Started](getting-started.md)** - Installation and first recording
- **[Configuration](configuration.md)** - Customize all settings
- **[Features](features/index.md)** - Smart chunking, background transcription, models
- **[CLI Reference](cli/index.md)** - Complete command documentation
- **[Guides](guides/index.md)** - How-to guides for common tasks

### For Developers

- **[voicepad-core](packages/voicepad-core/index.md)** - Python library API
- **[Architecture](packages/voicepad-core/architecture.md)** - Technical design
- **[GPU Acceleration](packages/voicepad-core/gpu-acceleration/index.md)** - CUDA setup

### Configuration Reference

- **[Paths](reference/paths.md)** - Output directories
- **[Audio](reference/audio.md)** - Input device settings
- **[Transcription](reference/transcription.md)** - Model and device options
- **[VAD Settings](reference/vad.md)** - Chunking parameters

## Feature Highlights

### VAD Chunking

Split long recordings at natural speech boundaries for background processing:

```yaml
vad_enabled: true
vad_min_chunk_duration: 60.0
vad_threshold: 0.5
```

**Learn more:** [VAD Chunking Guide](features/vad-chunking.md)

### Background Transcription

See transcription update in real-time while recording:

```text
[Recording...] → [Chunk 1 transcribed] → [Chunk 2 transcribed] → [Recording...]
```

**Learn more:** [Background Transcription](features/background-transcription.md)

### Multiple Whisper Models

Choose the right balance of speed vs. accuracy:

- **tiny** - Fastest, good for quick notes
- **medium** - Balanced, recommended default
- **large-v3** - Best accuracy, slower

**Learn more:** [Transcription Models](features/transcription-models.md)

## Resources

- **[GitHub Repository](https://github.com/HYP3R00T/voicepad)** - Source code and issues
- **[Zensical Documentation](https://zensical.org/docs/)** - Documentation framework
- **[Writing Documentation](dev/writing-documentation.md)** - Contribution guide
