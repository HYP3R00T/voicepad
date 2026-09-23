# Output paths

Defaults:

```text
WAV recordings:  ~/.config/voicepad/data/recordings
Markdown results: ~/.config/voicepad/data/markdown
Session logs:     ~/.local/state/voicepad/logs
Recording logs:   ~/.local/state/voicepad/logs/scopes/recording
Artifact cache:   ~/.cache/voicepad/artifacts
```

Capture writes to an operation-owned float WAV spool and then publishes a PCM
WAV. If native microphone shutdown fails, VoicePad still attempts to publish
the maximum valid WAV. A failed finalization retains the operation's recoverable
spool instead of deleting it.

Every process creates a private, UtilityHub-managed JSON session log before
application startup. Each recording also creates a correlated JSON log that
contains its capture, persistence, preprocessing, chunking, inference, and
completion records. Recording records also propagate to the session log.

The host application owns log-file and recording-scope lifecycles. VoicePad
Core supplies the stage-level facts. Log directories use mode `0700`, and log
files use mode `0600`.

Logs record recording paths, wall-clock and persisted durations, frame and
sample ranges, deployment identifiers, transformations, timings, outcomes, and
typed failures. They do not record audio samples, transcript text, token text,
or word text.

VoicePad refuses to overwrite existing WAV or Markdown files. Model artifacts
can be deleted and downloaded again when needed.
