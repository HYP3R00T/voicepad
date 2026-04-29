---
icon: lucide/layout-dashboard
---

# User Interface

Running `voicepad` (or `uvx voicepad`) opens the interactive terminal interface. It is built with [Textual](https://textual.textualize.io/) and runs entirely in your terminal.

## Layout

The interface has a header bar at the top, three tabs in the middle, and a keyboard shortcut bar at the bottom.

![VoicePad Interface](assets/sample_1.png)

### Header Bar

The header shows the app name, version, current status, and the active Whisper model with its compute device (`cuda` or `cpu`).

| Status Colour | Meaning |
|---|---|
| Green | Ready to record |
| Red | Recording in progress |
| Yellow | Transcription running |

## Keyboard Shortcuts

| Key | Action |
|---|---|
| ++space++ | Start or stop recording |
| ++c++ | Copy the current transcription to clipboard |
| ++q++ | Quit VoicePad |
| ++i++ | Open the info panel |
| ++ctrl+p++ | Open the command palette |
| ++tab++ | Switch between tabs |
| ++escape++ | Close a modal or dismiss |

## Tabs

### Record

The main tab. Press ++space++ to start recording. While recording, a live timer shows in the status bar. Text appears in real time via streaming transcription as you speak.

When you stop recording, the final transcription appears in the panel. Press ++c++ or use the **copy** button to copy the text to your clipboard.

### History

Browse all past recordings. Select a recording from the left panel to view its full markdown transcription on the right.

The **retranscribe** button re-runs the Whisper model on the selected WAV file. This is useful when you switch to a more accurate model after recording.

### Settings

Configure VoicePad without editing any files. Changes are saved to your global config file when you press **Save**.

You can change the following settings:

| Setting | Description |
|---|---|
| Recordings path | Where WAV files are saved |
| Markdown path | Where markdown transcriptions are saved |
| Transcription model | Whisper model to use |
| Input device | Microphone to record from |

See [Configuration](configuration/index.md) for more details on each setting.

## Info Panel

Press ++i++ to open the info panel. It shows the version, privacy guarantees, and links to the GitHub repository and sponsor page.
