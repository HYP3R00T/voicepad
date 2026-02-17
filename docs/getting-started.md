---
icon: lucide/rocket
---

# Getting Started

Welcome to Voicepad! This guide will help you set up and make your first recording with transcription.

## Installation

Install Voicepad using pip:

```bash
pip install voicepad
```

**Requirements:** Python 3.13 or higher

!!! tip "GPU Acceleration Available"
    For 4-5x faster transcription, install with GPU support:
    ```bash
    pip install voicepad[gpu]
    ```
    See [GPU Acceleration Guide](packages/voicepad-core/gpu-acceleration/index.md) for setup.

## First-Time Setup

### 1. Check Your Microphone

List available audio devices:

```bash
voicepad config input
```

This shows all audio input devices on your system. Note the index number of your microphone.

### 2. Create Configuration File

Create a `voicepad.yaml` file in your project directory:

```yaml
# Audio input (use device index from previous step)
input_device_index: 2

# Output directories
recordings_path: data/recordings
markdown_path: data/markdown

# Transcription settings
transcription_model: medium
transcription_device: auto
```

!!! note "Configuration Location"
    You can also place this file at `~/.config/voicepad/voicepad.yaml` for global configuration.

### 3. Verify System Capabilities

Check if your system is ready:

```bash
voicepad config system
```

This displays RAM, CPU, and GPU information to help you choose the right model.

## Your First Recording

### Simple Recording

Start recording (press **Ctrl+C** to stop):

```bash
voicepad record start
```

This will:

1. ✅ Record audio from your configured microphone
2. ✅ Save audio to `data/recordings/recording_TIMESTAMP.wav`
3. ✅ Automatically transcribe after stopping
4. ✅ Save transcription to `data/markdown/recording_TIMESTAMP.md`

### Recording with VAD Chunking

For long recordings, use VAD (Voice Activity Detection) chunking:

```bash
voicepad record start --vad --min-chunk-duration 60
```

This enables:

- 🎯 Real-time transcription while recording
- 📝 Live markdown updates
- 🔄 Smart chunk splitting at speech boundaries
- ⚡ Reduced wait time at the end

!!! tip "VAD Chunking Recommended For"
    - Meetings longer than 5 minutes
    - Lectures and presentations
    - Interviews
    - Any recording where you want to see transcription progress in real-time

## Understanding the Output

### Audio Files

Located in `data/recordings/`:

- Format: WAV (16-bit PCM, 16kHz mono)
- Naming: `{prefix}_{timestamp}.wav`
- Example: `recording_20260218_033000.wav`

### Transcription Files

Located in `data/markdown/`:

- Format: Markdown with metadata
- Naming: Same as audio file with `.md` extension
- Content: Full transcription with timestamps and statistics

Example transcription output:

```markdown
# Transcription: recording_20260218_033000.wav

**Status:** Recording complete

---

## Chunk 1 (0:00 - 1:12)

This is the transcribed text from the first chunk of audio...

## Chunk 2 (1:12 - 2:24)

Continuing with the second chunk...

---

**Recording Complete**
- Total chunks: 2
- Total duration: 2:24
```

## Common Tasks

### Record with Custom Filename

```bash
voicepad record start --prefix meeting_notes
```

### Record for Fixed Duration

```bash
voicepad record start --duration 300  # 5 minutes
```

### Record Without Transcription

```bash
voicepad record start --no-transcribe
```

### Check Current Configuration

```bash
voicepad record info
```

## Next Steps

Now that you've made your first recording, explore:

- **[Configuration Guide](configuration.md)** - Customize all settings
- **[VAD Chunking](features/vad-chunking.md)** - Learn about smart audio splitting
- **[CLI Reference](cli/record.md)** - Complete command options
- **[Transcription Models](features/transcription-models.md)** - Choose the right model

## Troubleshooting

### No Audio Recorded

- Check microphone selection with `voicepad config input`
- Verify microphone permissions in your OS
- Test microphone in another application

### Slow Transcription

- Consider using a smaller model (e.g., `tiny` instead of `medium`)
- Install GPU support: `pip install voicepad[gpu]`
- See [GPU Acceleration](packages/voicepad-core/gpu-acceleration/index.md)

### Import Errors

- Ensure Python 3.13+ is installed: `python --version`
- Reinstall: `pip install --force-reinstall voicepad`

!!! info "Need Help?"
    Visit the [GitHub repository](https://github.com/HYP3R00T/voicepad) to report issues or ask questions.
