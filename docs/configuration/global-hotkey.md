# Desktop shortcut

A running TUI exposes a user-only Unix control socket. The command:

```bash
voicepad toggle
```

requests the same record/stop action as **Space**. Configure this command as a
desktop-environment shortcut if a global hotkey is desired.

The socket is permissioned for the current user, rejects unsupported commands,
and is removed during normal shutdown. VoicePad does not install or register a
system-wide Linux key listener itself.
