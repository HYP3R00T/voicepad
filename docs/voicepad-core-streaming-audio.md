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
  -> provisional preview inference

Recording stops
  -> stop streaming without transcribing the remaining tail
  -> load the completed WAV
  -> transcribe the complete recording
  -> accept the final result only when it covers the accumulated previews
```

The complete recording lives on disk. During capture, only one maximum-size
chunk, its overlap, and the bounded writer queue occupy working memory. When
recording stops, the completed file is loaded for the final full-recording
pass. If that pass fails, returns no text, or returns substantially less text
than the accumulated chunks, VoicePad retains the streaming transcript as a
fallback. This protects long recordings from a backend returning a nonempty
but incomplete result.

The queue accepts at most 256 callback buffers.
If disk writing cannot keep pace, recording fails loudly instead of silently
dropping audio or growing memory without limit.

## Current boundaries

- The TUI uses continuous disk persistence.
- Every microphone session requires a WAV destination.
- Live transcription currently assumes mono capture.
- Finalization converts the temporary float WAV to PCM16 in bounded blocks.
