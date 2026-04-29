# voicepad-core

Core library powering VoicePad — audio recording, GPU-accelerated transcription via faster-whisper, and system diagnostics.

## Install

```bash
pip install voicepad-core
```

## Quick Start

```python
from voicepad_core import AudioRecorder, transcribe_audio, get_config

config = get_config()

recorder = AudioRecorder(config)
audio_file = recorder.start_recording()  # Ctrl+C to stop

transcribe_audio(audio_file, config.markdown_path / "transcript.md", config)
```

## Documentation

**[voicepad.hyperoot.dev](https://voicepad.hyperoot.dev)**
