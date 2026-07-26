# VoicePad Core learning map

This is our temporary reading order.

We will learn one concept at a time.
The order follows the data, not the folder names.

## The whole path

```text
App starts
  -> configuration is loaded
  -> model and runtime are prepared
  -> microphone recording starts
  -> raw audio is saved independently
  -> speech chunks are detected
  -> each chunk is prepared for the selected backend
  -> the backend transcribes it
  -> the result is cleaned and joined
  -> the app displays and stores the text
```

## Reading order

1. **Audio vocabulary**
   `audio/types.py` and [Core contract boundaries](voicepad-core-contracts.md)
   Learn the difference between captured audio and backend requirements.

2. **Audio capture and storage**
   `audio/microphone.py`, `audio/persistence.py`, `audio/file.py`, and
   [Live recording and chunking](voicepad-core-streaming-audio.md)
   Follow samples from the microphone into a durable WAV file.

3. **Configuration**
   `config/types.py`
   See which choices belong to the user and which belong to a backend.

4. **Model catalogue**
   `models.py`
   See how a model name selects a model family and runtime.

5. **Backend boundary**
   `inference/contracts.py`, `inference/types.py`
   Learn the exact inputs and outputs every backend must provide.

6. **Preprocessing**
   `preprocessing/preprocessor.py`
   Convert source audio into the waveform required by a backend.

7. **Runtime ownership**
   `inference/runtime.py`, `inference/composition.py`
   Load, retain, select, and release a model runtime.

8. **Backend implementations**
   `inference/backends/faster_whisper.py`, `inference/backends/parakeet.py`
   Compare two implementations of the same contract.

9. **One-shot inference**
   `inference/engine.py`
   Follow one complete transcription request.

10. **Speech detection**
    `vad/silero.py`
    Decide which sample ranges contain speech.

11. **Streaming loop**
    `streaming/transcriber.py`, `streaming/types.py`
    See how recording, chunks, inference, and partial results repeat.

12. **Postprocessing**
    `postprocessing/`
    Normalize, deduplicate, and reject likely hallucinations.

13. **Public API and application**
    `voicepad_core/__init__.py`, then `packages/voicepad/src/voicepad/`
    See how the CLI and TUI use the completed core.

## Progress

- [ ] Audio vocabulary — current
- [ ] Audio capture and storage
- [ ] Configuration
- [ ] Model catalogue
- [ ] Backend boundary
- [ ] Preprocessing
- [ ] Runtime ownership
- [ ] Backend implementations
- [ ] One-shot inference
- [ ] Speech detection
- [ ] Streaming loop
- [ ] Postprocessing
- [ ] Public API and application
