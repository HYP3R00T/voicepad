---
icon: lucide/keyboard
---

# Global Hotkey

VoicePad can start or stop recording from another application when the operating system sends it a toggle request.

## Windows

Set `global_hotkey` in the Settings tab or in `voicepad.yaml`:

```yaml
global_hotkey: win+space
```

VoicePad registers this shortcut with Windows after the transcription model is ready. If another application has already claimed the combination, VoicePad records the registration failure in its application log.

## Linux

Wayland compositors manage global shortcuts. VoicePad therefore provides a command that safely contacts the running TUI instead of reading privileged keyboard devices:

```bash
voicepad toggle
```

Install VoicePad as a persistent tool so the command is available to your desktop environment:

```bash
uv tool install voicepad
```

Then:

1. Start VoicePad with `voicepad`.
2. Open your desktop's keyboard-shortcut settings.
3. Add a custom shortcut with the command `voicepad toggle`.
4. Assign your preferred combination, such as `Super+Space`.
5. Press the shortcut once to record and again to stop and transcribe.

The TUI remains the owner of the recording session. A successful toggle switches the visible TUI to the Record tab and updates its status from **ready** to **recording**, then **transcribing**, and finally **ready**. The second toggle also enables automatic clipboard copying after transcription.

### COSMIC

Open **COSMIC Settings**, select **Keyboard**, then **Keyboard Shortcuts**, and add a custom shortcut:

| Field | Value |
|---|---|
| Name | `VoicePad Toggle` |
| Command | `voicepad toggle` |
| Shortcut | `Super+Space` |

COSMIC uses `Super+Space` for input-source switching by default. Replace or disable that existing binding when COSMIC reports the conflict.

The command must be available to the desktop session. For a development checkout that has not been installed as a tool, use the absolute executable path instead:

```text
/absolute/path/to/voicepad/.venv/bin/voicepad toggle
```

The `global_hotkey` value in `voicepad.yaml` controls native Windows registration. On Linux, the compositor shortcut controls which keys launch `voicepad toggle`.

## Local Control

The toggle command communicates only with the VoicePad process owned by the current user. On Linux, the local socket is created in `XDG_RUNTIME_DIR` with mode `0600` and is removed when VoicePad exits.

If VoicePad is not running, the command exits with an error:

```text
VoicePad toggle failed: Could not reach the running VoicePad app...
```

Start the TUI and wait for its status to become **ready** before using the global shortcut.
