---
icon: lucide/keyboard
---

# Global desktop shortcut

VoicePad can start or stop recording while another application has focus. On
Linux and Wayland, the desktop compositor owns the key combination; VoicePad
does not read privileged keyboard devices.

The shortcut runs:

```bash
voicepad toggle
```

That lightweight command contacts the running TUI over a user-only Unix socket
and requests the same record/stop action as **Space**. The resident TUI remains
responsible for the microphone, durable WAV recording, and transcription.

## COSMIC on Wayland

1. Start VoicePad and wait for **ready**.
2. Open the **Settings** tab.
3. Under **global desktop shortcut**, confirm that the shortcut is configured.
4. If setup is required, select **open keyboard settings** and add a custom
   shortcut using the displayed absolute toggle command.
5. Assign a combination such as **Super+Space**.

Press the shortcut once from any application to record. Press it again to stop,
finalize the WAV, and transcribe. VoicePad switches its TUI to the Record tab and
shows a small recording/transcribing/completion overlay above other applications.
The overlay is centered near the bottom of the enabled monitor containing the
pointer, rather than across the combined virtual desktop. Disabled displays are
ignored.

COSMIC uses **Super+Space** for input-source switching by default. Replace or
disable that existing binding if COSMIC reports a conflict.

## Other Linux desktops

Open your desktop environment's custom keyboard-shortcut settings and bind the
absolute command shown in VoicePad's Settings tab. Compositor-managed shortcuts
are the supported Wayland mechanism.

## Local control security

The control socket is created under `XDG_RUNTIME_DIR` with mode `0600`, accepts
only the toggle command, and is removed when VoicePad exits. It cannot start a
separate background model process: the already-running TUI must own the resident
model and reach **ready** first.

If VoicePad is not running, the command exits with an actionable error instead
of starting a recording elsewhere.
