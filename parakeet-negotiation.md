# Inference Negotiation

This file records our current agreement.

## Scope

- Branch: `spike/parakeet-backend`
- Primary target: native Windows.
- Hardware: NVIDIA RTX 3050 Laptop, 4 GB VRAM.
- Linux with NVIDIA may follow.
- WSL is out of scope.
- Audio and transcripts stay local.
- The terminal theme stays unchanged.

## Ownership

- VoicePad owns audio capture and storage.
- Audio storage never depends on transcription success.
- VoicePad owns VAD, chunking, overlap, and cleanup.
- VoicePad owns model selection and runtime lifecycle.
- VoicePad owns request, result, and capability contracts.
- VoicePad owns Parakeet decoding and proper-noun bias.
- Native runtimes only execute model graphs.

## Chosen Parakeet Path

- Model: [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- Artifact: [Handy-compatible INT8 ONNX bundle](https://blob.handy.computer/parakeet-v3-int8.tar.gz)
- Runtime: ONNX Runtime with CUDA.
- Fallback: ONNX Runtime CPU provider.
- Input: mono, float32, 16 kHz audio.
- Bundle: encoder, decoder, preprocessor, and vocabulary.

## Architecture

- Models declare artifacts and required files.
- Drivers isolate each runtime family.
- One runtime stays active at a time.
- Switching models unloads the old runtime first.
- Applications use one public inference coordinator.
- Faster Whisper remains a temporary driver.
- New families should not change capture or streaming code.

## Proper Nouns

- Proper nouns enter a semantic context object.
- Faster Whisper receives hotwords.
- Parakeet receives decoder-logit bias.
- Completed text is never rewritten.

## Validation

- Bundle checksum verified.
- CUDA provider verified with CPU fallback disabled.
- Test audio: 24.21 seconds.
- Load time: 4.34 seconds.
- Inference time: 2.47 seconds.
- Real-time factor: 0.10.
- Measured GPU memory increase: 714 MiB.
- Fixture transcript was correct.
- Compare against the current Faster Whisper baseline.

## Removal Rule

- Keep Faster Whisper until Parakeet reaches practical parity.
- Remove legacy download and global model caches now.
- Remove Faster Whisper dependencies only after parity.
