# User interface

The TUI keeps one production transcription path while restoring the polished
three-tab workflow.

## Tabs

- **Record** shows live elapsed recording time, provisional chunk text,
  authoritative final text, processing metadata, and copy control. The first
  provisional update normally appears at the first confirmed pause after about
  25 seconds; later updates follow each processed chunk.
- **History** lists schema-1 Markdown results and displays the selected record.
- **Settings** controls recording, Markdown, and artifact-cache paths; filename
  prefix; theme; automatic copy; explicit proper-noun aliases; and the
  compositor-managed global shortcut. Saving reloads the verified resident
  deployment when necessary.

The theme selector includes the built-in Textual palettes, with `tokyo-night` as
the default. The header intentionally reports the compact runtime identity
`Parakeet v3 · NVIDIA CUDA · FP16` instead of the full physical GPU name.

## States

- **Loading** — verifies artifacts, admits CUDA, loads and warms Parakeet.
- **Ready** — the model is resident and recording can begin immediately.
- **Recording** — canonical audio is continuously persisted while CPU VAD and
  bounded inference may proceed concurrently.
- **Finalizing** — capture stops, the WAV is finalized, remaining descriptors
  are processed, and one authoritative result is assembled.
- **Error** — the requested contract could not be preserved; there is no CPU
  fallback.

## Keys

| Key | Action |
|---|---|
| **Space** | Start or stop recording |
| **C** | Copy the last complete transcription |
| **S** | Save settings from the Settings tab |
| **Q** | Quit safely |

The local `voicepad toggle` command sends the same start/stop action to a running
TUI over a user-only Unix socket. Desktop environments may bind that command to
a global shortcut. External toggles switch to the Record tab and update one
native recording, transcribing, or completion status card while another
application has focus. See [Global desktop shortcut](configuration/global-hotkey.md).

While recording, the header displays elapsed time in seconds and then minutes.
Incomplete text is displayed and persisted with metadata, but is not copied
automatically. The WAV remains the durable source for later retranscription.
