# voicepad

Terminal interface for VoicePad — record audio and transcribe it locally with a single keypress.

## Install

```bash
# Run without installing
uvx voicepad

# Install permanently
uv tool install voicepad
```

## Usage

```bash
voicepad
```

That's it. The interactive TUI opens, loads the Whisper model, and you're ready to record.

| Key | Action |
|---|---|
| `Space` | Start / stop recording |
| `c` | Copy transcription to clipboard |
| `Tab` | Switch tabs (Record / History / Settings) |
| `q` | Quit |

## Configuration

Settings are stored at `~/.config/voicepad/voicepad.yaml` and can be changed from the **Settings** tab inside the app.

```yaml
transcription_model: turbo
transcription_device: auto
transcription_compute_type: auto
input_device_index: null
recordings_path: ~/.config/voicepad/data/recordings
markdown_path: ~/.config/voicepad/data/markdown

# Streaming / chunking
min_chunk_s: 15.0
max_chunk_s: 29.0
overlap_s: 0.5
silence_threshold_ms: 1000
min_fresh_speech_duration_s: 0.25

# Inference cleanup / prompting
initial_prompt: "Hello. This is a transcription with proper punctuation, capitalization, and grammar."
text_postprocessing_enabled: false
no_speech_threshold: 0.6
hallucination_max_repetitions: 3

# VAD tuning
vad_threshold: 0.5
vad_min_speech_duration_ms: 250
vad_speech_pad_ms: 30
```

## Documentation

**[voicepad.hyperoot.dev](https://voicepad.hyperoot.dev)**
