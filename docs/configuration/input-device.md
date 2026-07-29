---
icon: lucide/mic
---

# Input Device

On Linux, VoicePad always uses the system's default microphone. Select that
microphone in your desktop's **Sound** settings.

This lets the Linux audio server (PipeWire or PulseAudio) share the microphone
with applications such as OBS. VoicePad does not open raw ALSA devices such as
`hw:1,0`, because those devices can become unavailable when another application
is recording.

## Changing the Microphone on Linux

1. Open your desktop's **Sound** settings
2. Choose the microphone under **Input**
3. Confirm that its input level moves when you speak
4. Start or restart a VoicePad recording

VoicePad follows changes to the system default when it opens the next recording.
An existing numeric `input_device_index` in `voicepad.yaml` is ignored on Linux
and can be removed or set to `null`.

On other operating systems, the VoicePad **Settings** tab continues to offer
the input devices reported by the operating system.

## Recording Alongside OBS

Configure both applications to use the desktop-managed microphone:

1. Set the microphone as the default input in the desktop's **Sound** settings
2. In OBS, select the same PipeWire/PulseAudio microphone source
3. Leave VoicePad on **System default**

OBS and VoicePad can then record the microphone at the same time through the
audio server.

## Troubleshooting

Run this command to confirm VoicePad's Linux input policy:

```sh
voicepad config input
```

If VoicePad cannot open the microphone:

- Make sure the microphone is visible and selected in the desktop's **Sound** settings
- Check that VoicePad is not muted in the desktop's per-application audio controls
- Ensure OBS uses a PipeWire/PulseAudio source rather than a raw ALSA `hw:*` source
- Reconnect the microphone, then start a new recording
