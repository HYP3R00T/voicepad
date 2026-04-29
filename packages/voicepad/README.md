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
transcription_model: turbo       # default model
transcription_device: auto       # auto, cuda, or cpu
input_device_index: null         # null = system default mic
recordings_path: ~/.config/voicepad/data/recordings
markdown_path: ~/.config/voicepad/data/markdown
```

## Documentation

**[voicepad.hyperoot.dev](https://voicepad.hyperoot.dev)**
