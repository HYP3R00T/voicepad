---
icon: lucide/folder-open
---

# Output Paths

VoicePad saves two files for each recording:

- A **WAV file** containing the raw audio
- A **markdown file** containing the transcription

## Default Locations

By default, both are saved under `~/.config/voicepad/data/`:

```sh
~/.config/voicepad/data/recordings/    # WAV files
~/.config/voicepad/data/markdown/      # Markdown transcriptions
```

On Windows, `~` expands to your user profile directory (e.g. `C:\Users\YourName`).

These directories are created automatically on first recording.

## Changing Output Paths

Open the **Settings** tab and update the **recordings_path** or **markdown_path** fields. Press **Save** to apply.

You can also edit `~/.config/voicepad/voicepad.yaml` directly:

```yaml
recordings_path: ~/Documents/voicepad/recordings
markdown_path: ~/Documents/voicepad/transcriptions
```

Both paths support `~` expansion.

## Filename Format

Each file is named with a prefix and a timestamp:

```sh
recording_20260429_143022.wav
recording_20260429_143022.md
```

The default prefix is `recording`. You can change it by setting `recording_prefix` in the config file.
