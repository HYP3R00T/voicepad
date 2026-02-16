# voicepad

Command-line interface for voice recording. Provides user-friendly commands for audio capture without writing code.

## Overview

`voicepad` is a CLI wrapper around `voicepad-core` built with Typer. It exposes recording functionality through simple terminal commands.

**Key Features:**

- List audio input devices
- Record audio from command line
- Select specific input device
- Configure output directory
- Customize recording file prefix

## Installation

```bash
pip install voicepad
```

**Requirements:** Python 3.13+

**Dependencies:** `voicepad-core`, `typer`

## Command Structure

```sh
voicepad
└── voice                    Voice recording operations
    ├── list-devices        List audio input devices
    └── record              Record audio to file
```

This hierarchical structure allows future expansion with additional command groups.

## Commands

### `voicepad voice list-devices`

List all available audio input devices.

```bash
voicepad voice list-devices
```

**Output:**

```sh
Available audio input devices:
------------------------------------------------------------
  AudioDevice(index=0, name='Microphone (Realtek Audio)', channels=2, sample_rate=48000)
  AudioDevice(index=1, name='Webcam Microphone', channels=1, sample_rate=44100)
------------------------------------------------------------
```

### `voicepad voice record`

Record audio interactively. Press Enter to stop recording.

```bash
voicepad voice record                           # Use defaults
voicepad voice record -d 1                      # Specific device
voicepad voice record -d 2 -o audio             # Custom output directory
```

**Options:**

| Option | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--device-index` | `-d` | int | 0 | Audio input device index |
| `--output-dir` | `-o` | Path | `data/recordings` | Output directory for WAV files |

**Output:** `{output_dir}/recording_{YYYYMMDD_HHMMSS}.wav`

## Example Workflow

```bash
# List available devices
voicepad voice list-devices

# Record using selected device (press Enter to stop)
voicepad voice record -d 1 -o recordings

# Result: recordings/recording_20260216_143530.wav
```

## Source Code

Implementation details: `packages/voicepad/src/voicepad/` in the repository

- CLI commands: See `voicepad/cli/voice.py` for command implementations
- Entry point: See `voicepad/main.py` for Typer app setup
