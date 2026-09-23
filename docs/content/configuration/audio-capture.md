# Audio capture

VoicePad uses `sounddevice` (PortAudio) for microphone capture. On Linux,
configured `input_device_index` values are ignored: select the desired microphone
in system sound settings. VoicePad uses PortAudio's default input endpoint.

Before capture, VoicePad resolves that endpoint and validates its requested
16 kHz mono float32 format, without creating a recording file if validation
fails. The recording log identifies the endpoint name/index, host API, advertised
default rate, and requested format.

A shared endpoint such as ALSA `default` does not identify the physical microphone
behind it; check system audio routing. The requested capture rate is not
necessarily the hardware's native rate.

## Signal health

The WAV writer measures audio before PCM16 conversion; no analysis runs in the
microphone callback and no gain, normalization, or noise reduction is applied.

- The TUI recording status shows the latest writer block's RMS in dBFS.
- Recording logs summarize whole-recording peak/RMS and sample counts at PCM16
  limits or with non-finite values.
- Any sample at or beyond a PCM16 limit triggers a **possible clipping** warning,
  not proof of hardware clipping. Peaks above 0 dBFS are possible in float capture.
- After at least two seconds, cumulative RMS below −60 dBFS triggers a
  **mostly near-silent** warning. It clears if subsequent audio raises cumulative
  RMS. This is not speech detection; quiet surroundings can legitimately trigger it.
- Non-finite samples receive a separate warning rather than being labeled silence.

Warnings appear in CLI output and TUI notifications when recording finishes, and
in transcription Markdown metadata when transcribing a live recording. They are
advisory: they do not stop capture or by themselves mark transcription incomplete.
An empty recording has no level measurements. Live readings may lag by the writer
queue and do not identify the physical device behind a shared endpoint.

## Dropped or substituted audio

VoicePad counts PortAudio callbacks reporting input overflow (discarded audio)
or input underflow (potentially substituted silence). It keeps saving subsequent
audio instead of aborting on the first event. The TUI shows **audio gaps** while
recording, and callback status summaries are logged when capture stops rather
than logging each status from the audio callback.

Any reported input discontinuity marks a live transcription **incomplete**.
Warnings, including callback-event counts, are retained in transcription Markdown;
the result is not automatically copied. Capture-only `--no-transcribe` preserves
the WAV, warns that it is partial, and exits with code 2, as does incomplete CLI
transcription. Captured samples are not padded with guessed silence.

Event counts are not counts of lost samples. PortAudio does not provide exact
lost durations or locations here, so VoicePad leaves them unknown. Elapsed and
persisted durations remain in logs, but their difference is not labeled missing
audio. Device loss that stops the stream remains a fatal capture error; VoicePad
attempts to finalize the already captured audio. Some backends can substitute
silence without reporting device loss, so this is not universal unplug detection.

## Writer backpressure and recovery

Audio submission never waits for queue space. A full bounded queue raises a
capture error; VoicePad attempts to finalize the blocks already accepted rather
than silently dropping them. Live reads can wait for space, but release the
writer's state lock and have a bounded queue/read wait. A read racing with stop
uses the finalized WAV or reports failure if finalization did not succeed.

Stopping rejects new submissions and drains the accepted queue without needing
to enqueue a stop marker. Shutdown, writer failures, and finalization failures
wake waiting readers. Native microphone callbacks only remember errors; error
logging is deferred until capture stops.

Hidden `.<recording-stem>-live-*.wav` spools are kept when they contain committed
audio, the writer fails, or shutdown times out. Clean, empty aborted spools are
removed. A retained spool may contain only part of a recording, especially after
a disk failure; do not assume queued or unwritten samples were saved. No automatic
recovery is performed. Blocked native disk I/O cannot be forcibly cancelled, but
the writer is requested to exit after I/O resumes and accepted work is drained.
