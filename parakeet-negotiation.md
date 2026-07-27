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
- Backend libraries own model decoding.

## Chosen Parakeet Path

- Base model: [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- Artifact: pinned Sherpa-ONNX int8 export.
- Runtime: Sherpa-ONNX CUDA 12 with cuDNN 9.
- Precision: int8.
- Decoder: greedy search.
- Parakeet must run on CUDA.
- CPU fallback is rejected.
- Input: mono, float32, 16 kHz audio.

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
- Parakeet does not claim context biasing.
- Sherpa modified beam search is rejected until it passes real dictation tests.
- Completed text is never rewritten.

## Validation

- Unit tests verify CUDA-only provider selection.
- The real Silero and Parakeet models load together.
- The reference transcript is correct on the RTX 3050.
- Short real dictation is correct with greedy search.
- Reference inference takes about 0.66 seconds.
- Compare against the current Faster Whisper baseline.

## Removal Rule

- Keep Faster Whisper until Parakeet reaches practical parity.
- Remove legacy download and global model caches now.
- Remove Faster Whisper dependencies only after parity.
