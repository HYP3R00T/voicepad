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

### Global Shortcuts

These shortcuts work in any tab:

| Key | Action |
|---|---|
| ++q++ | Quit VoicePad |
| ++i++ | Open the info panel |
| ++ctrl+p++ | Open the command palette |
| ++tab++ | Switch between tabs |
| ++escape++ | Close a modal or dismiss |

### Tab-Specific Shortcuts

Different shortcuts are available depending on which tab you're in:

#### Record Tab

| Key | Action |
|---|---|
| ++space++ | Start or stop recording |
| ++c++ | Copy the current transcription to clipboard |

#### History Tab

| Key | Action |
|---|---|
| ++t++ | Retranscribe the selected recording |
| ++d++ | Delete the selected recording |
| ++v++ | Open the selected recording in your audio player |
| ++m++ | Open the selected markdown file in your editor |
| ++o++ | Toggle sort order (ascending/descending) |

#### Settings Tab

| Key | Action |
|---|---|
| ++s++ | Save settings |
| ++p++ | Open config directory in file explorer |

## Tabs

### Record

The main tab. Press ++space++ to start recording. While recording, a live timer shows in the status bar. Text appears in real time via streaming transcription as you speak.

When you stop recording, the final transcription appears in the panel. Press ++c++ or use the **copy** button to copy the text to your clipboard.

### History

Browse all past recordings. Select a recording from the left panel to view its full markdown transcription on the right.

**Available actions:**

- **Retranscribe** (++t++): Re-run the Whisper model on the selected WAV file. Useful when you switch to a more accurate model after recording.
- **Delete** (++d++): Remove the selected recording and its transcription.
- **Open recording** (++v++): Open the WAV file in your default audio player.
- **Open markdown** (++m++): Open the transcription file in your default text editor.
- **Sort** (++o++): Toggle between ascending and descending order.

### Settings

Configure VoicePad without editing any files. Changes are saved to your global config file when you press ++s++ or click **Save**.

**Available settings:**

| Setting | Description |
|---|---|
| Recordings path | Where WAV files are saved |
| Markdown path | Where markdown transcriptions are saved |
| VAD model path | Where Voice Activity Detection model is stored |
| Transcription model | Whisper model to use |
| Input device | Microphone to record from |
| Theme | UI color theme |
| Global hotkey | System-wide keyboard shortcut for recording |

**Quick actions:**

- Press ++p++ to open the config directory in your file explorer
- The config file path is shown at the top of the settings panel

See [Configuration](configuration/) for more details on each setting.

## Info Panel

Press ++i++ to open the info panel. It shows the version, privacy guarantees, and links to the GitHub repository and sponsor page.
