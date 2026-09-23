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
