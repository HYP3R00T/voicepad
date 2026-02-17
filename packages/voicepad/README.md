# voicepad CLI

Simple command-line interface for recording audio and managing transcription configuration.

## Install

```bash
pip install voicepad
```

**Requirements:** Python 3.13+

## Quick Start

```bash
# List audio input devices
voicepad config input

# Start recording (press Ctrl+C to stop)
voicepad record start

# Check system capabilities
voicepad config system
```

## Example: Record and Transcribe

```bash
# Record a meeting (will auto-transcribe)
voicepad record start --prefix team_meeting

# Output:
# - data/recordings/team_meeting_20260218_103045.wav
# - data/markdown/team_meeting_20260218_103045.md
```

## Documentation

- [CLI Command Reference](https://voicepad.readthedocs.io/packages/voicepad/) - Full documentation
- [voicepad-core Library](https://voicepad.readthedocs.io/packages/voicepad-core/) - Python API
- [Main README](https://github.com/HYP3R00T/voicepad#readme) - Project overview

## Configuration

Edit `voicepad.yaml` to set defaults:

```yaml
recordings_path: data/recordings
markdown_path: data/markdown
input_device_index: null
transcription_model: tiny
transcription_device: auto
transcription_compute_type: auto
```

See the [full documentation](https://voicepad.readthedocs.io/packages/voicepad/) for all configuration options.
