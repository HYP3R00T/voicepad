# NVIDIA GPU requirements

The production deployment requires a CUDA-capable NVIDIA GPU and Linux x86_64.
Startup validates:

- PyTorch CUDA availability;
- stable NVIDIA device UUID;
- FP16 compute capability;
- physical and currently free memory;
- complete CUDA placement after model load.

The maintained baseline is an RTX 3050 Laptop GPU in the 4 GB physical class.
PyTorch exposes 3,953,393,664 bytes on that device, so admission uses a
3,900,000,000-byte floor plus a free-memory safety margin.

VoicePad does not fall back to CPU, lower precision, ONNX ASR, or another model.
Close other GPU-heavy programs if admission reports insufficient free memory.
A supported NVIDIA driver is a system prerequisite; a separate CUDA toolkit is
not required.
