# Transcription deployment

VoicePad currently exposes one production deployment:

```text
parakeet-v3.transformers-fp16-cuda
```

It combines NVIDIA's official Parakeet TDT 0.6B v3 safetensors, Transformers,
PyTorch FP16 CUDA, immutable artifact hashes, strict device admission, and the
bounded chunk policy.

There is no model dropdown because unvalidated model/runtime combinations are
not user settings. A future deployment must define its own artifacts, adapter,
capabilities, resource profile, and quality evidence before it appears here.
