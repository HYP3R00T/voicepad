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
