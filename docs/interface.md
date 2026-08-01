# User interface

The TUI intentionally exposes one production path.

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
| **Q** | Quit safely |

The local `voicepad toggle` command sends the same start/stop action to a running
TUI over a user-only Unix socket. Desktop environments may bind that command to
a global shortcut.

Incomplete text is displayed and persisted with metadata, but is not copied
automatically. The WAV remains the durable source for later retranscription.
