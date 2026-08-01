# VoicePad Core contract boundaries

This document records where model-related parameters belong in the current implementation.

!!! note "Approved replacement design"
    The [transcribe.cpp GGUF transcription pipeline](designs/transcribe-cpp-pipeline.md)
    describes the approved target architecture. This page remains authoritative
    for current behavior until the migration is implemented.

## Contract layers

| Question | Contract | Examples |
|---|---|---|
| What waveform must preprocessing produce? | `WaveformSpec` | Sample rate, channels, peak normalization |
| How should the model runtime open? | `RuntimeOptions` | Device, precision, CPU fallback |
| What should this inference call do? | `TranscriptionRequest` | Language, beam size, timestamps, context |
| Which features can the backend perform? | `BackendCapabilities` | Translation, biasing, beam search, VAD |
| What model artifact should be loaded? | `Model` | Hugging Face repository, files, backend, precision |
| How does one runtime implement a feature? | Backend implementation | CUDA providers, Sherpa decoding method, native argument names |

`WaveformSpec` deliberately describes only waveform transformations.
It is not a container for every model option.

## Adding a parameter

1. Put preprocessing requirements in `WaveformSpec`.
2. Put model-loading choices in `RuntimeOptions`.
3. Put per-call semantic choices in `TranscriptionRequest`.
4. Add a capability flag when support differs by backend.
5. Keep implementation-only tuning inside the backend.
6. Use typed backend-specific options if a useful setting cannot be shared.

Avoid an unrestricted options dictionary.
It would hide unsupported settings and weaken validation.

## `audio/types.py` issue register

| Item | Current decision |
|---|---|
| Sample dtype and tensor layout are implicit. | Both current backends consume mono `float32`. Extend `WaveformSpec` before adding a backend that does not. |
| `channels` accepts values that preprocessing cannot yet produce. | Keep the contract honest, but reject non-mono targets in preprocessing until multichannel inference exists. |
| Shared NumPy arrays could be changed accidentally. | `RawAudio` and `AudioWindow` mark samples read-only without copying. Transformations create new arrays. |
| Numeric type and finite sample values are not validated at capture. | Register as a robustness improvement. Validate only if real inputs show this failure, to avoid duplicate full-buffer scans. |
