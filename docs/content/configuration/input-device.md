# Input device

On Linux, VoicePad records through the shared system-default microphone managed
by PipeWire or PulseAudio. Select the desired default input in the desktop sound
settings.

Live capture requests mono 16 kHz float audio so persisted sample positions,
VAD positions, model ranges, and output timestamps share one exact coordinate
system. The final user WAV is PCM and is published without overwriting an
existing file.
