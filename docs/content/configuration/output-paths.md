# Output paths

Defaults:

```text
WAV recordings:  ~/.config/voicepad/data/recordings
Markdown results: ~/.config/voicepad/data/markdown
Session logs:     ~/.config/voicepad/logs
Artifact cache:   ~/.cache/voicepad-v2/artifacts
```

Capture writes to an operation-owned float WAV spool and then publishes a PCM
WAV. Inference reads committed disk ranges without owning the persistence
lifecycle: a failed live transcription read is reported but does not terminate
the audio writer. If native microphone shutdown fails, VoicePad still attempts
to publish the maximum valid WAV. A failed finalization retains the operation's
recoverable spool instead of deleting it.

Every process creates a private session log before application startup. Logs
record recording paths and lifecycle transitions, wall-clock and persisted
durations, frame counts, capture/writer failures, finalization, and
transcription outcomes. They do not record transcript, token, or word text.

VoicePad refuses to overwrite existing WAV or Markdown files. Legacy model
caches are ignored and never migrated or deleted automatically.
