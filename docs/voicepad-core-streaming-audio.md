# Live recording and chunking

VoicePad uses integer sample positions as exact addresses in one recording.

```text
time in seconds = sample position / sample rate
```

At 16 kHz, sample `480000` is exactly 30 seconds from the start.
A window records its absolute `start_sample`; its end is start plus its length.

## Live path

```text
Microphone callback
  -> bounded writer queue
  -> temporary float WAV
  -> atomic PCM WAV finalization

Streaming transcriber
  -> requests unconsumed samples + overlap
  -> VAD chooses a boundary
  -> 15–30 second chunk
  -> inference
```

The complete recording lives on disk.
Only one maximum-size chunk, its overlap, and the bounded writer queue occupy
working memory. A delayed backlog is read and drained in multiple bounded
sample ranges.

The queue accepts at most 256 callback buffers.
If disk writing cannot keep pace, recording fails loudly instead of silently
dropping audio or growing memory without limit.

## Current boundaries

- The TUI uses continuous disk persistence.
- Every microphone session requires a WAV destination.
- Live transcription currently assumes mono capture.
- Finalization converts the temporary float WAV to PCM16 in bounded blocks.
