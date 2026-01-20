"""System utilities for GPU and dependency checking.

This module provides utilities to check system capabilities before running
GPU-accelerated tasks like transcription using Python-only detection.
"""

import logging
import sys
from dataclasses import dataclass
from typing import Literal

import typer

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "rocm", "mps", "cpu"]


@dataclass
class GPUInfo:
    """Information about GPU availability and capabilities."""

    torch_available: bool
    device_type: DeviceType
    device_name: str | None
    total_memory_gb: float | None
    faster_whisper_available: bool
    cuda_version: str | None


@dataclass
class ModelRecommendation:
    """Recommended faster-whisper model configuration."""

    model_size: str
    compute_type: str
    notes: str


def detect_device_capabilities() -> tuple[DeviceType, str | None, float | None, str | None]:
    """Detect device capabilities using PyTorch only.

    Returns:
        tuple: (device_type, device_name, total_memory_gb, cuda_version)
    """
    try:
        import torch

        # Check CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            device_name = props.name
            total_memory_gb = props.total_memory / (1024**3)  # bytes to GB
            cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
            return "cuda", device_name, total_memory_gb, cuda_version

        # Check ROCm (AMD GPU)
        # torch.version may not exist in some installations, so access it safely.
        version = getattr(torch, "version", None)
        if version is not None and getattr(version, "hip", False):
            # ROCm detected but faster-whisper doesn't support it
            return "rocm", "AMD GPU (ROCm)", None, None

        # Check Apple MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS available but faster-whisper doesn't support it
            return "mps", "Apple Silicon GPU", None, None

        # CPU-only
        return "cpu", None, None, None

    except ImportError:
        return "cpu", None, None, None


def recommend_faster_whisper_model(
    device_type: DeviceType,
    gpu_memory_gb: float | None = None,
) -> ModelRecommendation:
    """Recommend faster-whisper model based on device capabilities.

    Args:
        device_type: Type of device (cuda, rocm, mps, cpu)
        gpu_memory_gb: Total GPU memory in GB (for CUDA only)

    Returns:
        ModelRecommendation: Recommended model size, compute type, and notes
    """
    if device_type == "cuda" and gpu_memory_gb:
        # NVIDIA GPU with CUDA - use GPU acceleration
        # VRAM requirements based on official benchmarks (RTX 3070 Ti 8GB):
        # - large-v3 fp16: ~4.7GB VRAM
        # - large-v2 fp16: ~4.7GB VRAM, int8: ~2.9GB VRAM
        # - medium fp16: ~2-3GB VRAM
        # - small fp16: ~1-2GB VRAM
        # - base/tiny fp16: <1GB VRAM

        if gpu_memory_gb >= 6:
            # Sufficient for large models with headroom
            return ModelRecommendation(
                model_size="large-v3",
                compute_type="float16",
                notes="Best accuracy. Latest OpenAI model. Try 'turbo' for 2x speed with similar quality.",
            )
        elif gpu_memory_gb >= 3.5:
            # 3.5-6 GB: medium is safe, large models need int8
            return ModelRecommendation(
                model_size="medium",
                compute_type="float16",
                notes="Balanced speed/accuracy. Use 'large-v2' + int8 if you need better accuracy.",
            )
        elif gpu_memory_gb >= 1.5:
            return ModelRecommendation(
                model_size="small",
                compute_type="float16",
                notes="Fast with good accuracy. Best for quick transcription.",
            )
        elif gpu_memory_gb >= 1:
            return ModelRecommendation(
                model_size="base",
                compute_type="float16",
                notes="Faster transcription. Use 'tiny' for maximum speed (lower accuracy).",
            )
        else:  # < 1 GB
            return ModelRecommendation(
                model_size="tiny",
                compute_type="int8",
                notes="Fastest model. Limited accuracy. Consider CPU if quality matters.",
            )

    # CPU fallback for all non-CUDA devices (including ROCm and MPS)
    # faster-whisper only supports CUDA for GPU acceleration
    return ModelRecommendation(
        model_size="small",
        compute_type="int8",
        notes="Optimized for CPU. Use 'tiny'/'base' for speed, 'medium' for better accuracy.",
    )


def check_gpu_capabilities() -> GPUInfo:
    """Check GPU capabilities and library availability.

    Returns:
        GPUInfo: GPU and library availability information
    """
    # Check PyTorch availability
    torch_available = False
    try:
        import torch  # noqa: F401

        torch_available = True
    except ImportError:
        pass

    # Detect device capabilities
    device_type, device_name, total_memory_gb, cuda_version = detect_device_capabilities()

    # Check faster-whisper availability
    faster_whisper_available = False
    try:
        import faster_whisper  # noqa: F401

        faster_whisper_available = True
    except ImportError:
        pass

    return GPUInfo(
        torch_available=torch_available,
        device_type=device_type,
        device_name=device_name,
        total_memory_gb=total_memory_gb,
        faster_whisper_available=faster_whisper_available,
        cuda_version=cuda_version,
    )


def print_system_status() -> None:
    """Print system status including GPU capabilities and model recommendations."""
    gpu_info = check_gpu_capabilities()

    typer.echo("=" * 70)
    typer.echo("Voicepad System Status")
    typer.echo("=" * 70)

    # Python version
    typer.echo(f"Python: {sys.version.split()[0]}")

    # PyTorch status
    if gpu_info.torch_available:
        typer.secho("✓ PyTorch: Available", fg=typer.colors.GREEN)
    else:
        typer.secho("✗ PyTorch: Not available", fg=typer.colors.RED)
        typer.echo("  Please reinstall: uvx --reinstall voicepad")
        return

    # Device type and details
    typer.echo(f"\nDevice Type: {gpu_info.device_type.upper()}")

    if gpu_info.device_type == "cuda":
        # NVIDIA GPU with CUDA support
        typer.secho(f"✓ NVIDIA GPU: {gpu_info.device_name}", fg=typer.colors.GREEN)
        if gpu_info.total_memory_gb:
            typer.echo(f"  VRAM: {gpu_info.total_memory_gb:.2f} GB")
        typer.echo(f"  CUDA Version: {gpu_info.cuda_version}")

    elif gpu_info.device_type == "rocm":
        # AMD GPU with ROCm
        typer.secho("✓ AMD GPU: Detected (ROCm)", fg=typer.colors.YELLOW)
        typer.echo("  ⚠ Note: faster-whisper only supports NVIDIA CUDA")
        typer.echo("  Transcription will use CPU mode")

    elif gpu_info.device_type == "mps":
        # Apple Silicon
        typer.secho("✓ Apple Silicon GPU: Detected (MPS)", fg=typer.colors.YELLOW)
        typer.echo("  ⚠ Note: faster-whisper only supports NVIDIA CUDA")
        typer.echo("  Transcription will use CPU mode")

    else:
        # CPU-only
        typer.secho("ℹ CPU Mode: No GPU detected", fg=typer.colors.CYAN)

    # faster-whisper status
    if gpu_info.faster_whisper_available:
        typer.secho("\n✓ faster-whisper: Available", fg=typer.colors.GREEN)
    else:
        typer.secho("\n✗ faster-whisper: Not available", fg=typer.colors.RED)
        typer.echo("  Please reinstall: uvx --reinstall voicepad")
        return

    # Model recommendation
    recommendation = recommend_faster_whisper_model(
        gpu_info.device_type,
        gpu_info.total_memory_gb,
    )

    typer.echo("=" * 70)
    typer.secho("\n📊 Recommended Configuration", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  Model Size: {recommendation.model_size}")
    typer.echo(f"  Compute Type: {recommendation.compute_type}")
    typer.echo(f"  💡 {recommendation.notes}")

    # Overall readiness
    typer.echo("\n" + "=" * 70)

    if gpu_info.device_type == "cuda" and gpu_info.faster_whisper_available:
        typer.secho(
            "✓ System is ready for GPU-accelerated transcription!",
            fg=typer.colors.GREEN,
            bold=True,
        )
        typer.echo("\nExample usage:")
        typer.echo("  uvx voicepad transcribe transcribe audio.wav \\")
        typer.echo(f"    --model {recommendation.model_size} \\")
        typer.echo("    --device cuda")

    elif gpu_info.faster_whisper_available:
        typer.secho(
            "✓ System is ready for CPU transcription",
            fg=typer.colors.YELLOW,
            bold=True,
        )
        if gpu_info.device_type in ("rocm", "mps"):
            typer.echo("\n  Your GPU is detected but not supported by faster-whisper.")
            typer.echo("  CUDA (NVIDIA) is required for GPU acceleration.")

        typer.echo("\nExample usage:")
        typer.echo("  uvx voicepad transcribe transcribe audio.wav \\")
        typer.echo(f"    --model {recommendation.model_size}")

    else:
        typer.secho(
            "✗ System is not ready for transcription",
            fg=typer.colors.RED,
            bold=True,
        )
        typer.echo("\n  Missing dependencies. Try reinstalling:")
        typer.echo("  uvx --reinstall voicepad")

    # Additional model options
    typer.echo("\n💡 Model Options (speed vs accuracy trade-off):")
    typer.echo("  • tiny     - Fastest, lowest accuracy")
    typer.echo("  • base     - Fast, basic accuracy")
    typer.echo("  • small    - Balanced (recommended for most)")
    typer.echo("  • medium   - Slower, better accuracy")
    typer.echo("  • large-v2 - Slow, excellent accuracy")
    typer.echo("  • large-v3 - Slowest, best accuracy")

    typer.echo("=" * 70)


def verify_transcription_ready() -> bool:
    """Verify if the system is ready for transcription (GPU or CPU).

    Returns:
        bool: True if transcription is ready, False otherwise
    """
    gpu_info = check_gpu_capabilities()
    return gpu_info.torch_available and gpu_info.faster_whisper_available
