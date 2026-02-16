"""Model recommendation engine for Whisper transcription."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voicepad_core.diagnostics.models import ModelRecommendation, SystemInfo

logger = logging.getLogger(__name__)


def categorize_model(model_name: str) -> str:
    """Categorize a model name into resource categories.

    Args:
        model_name: Model name from available_models().

    Returns:
        Category: "tiny", "base", "small", "medium", "large", "turbo", or "distil".
    """
    model_lower = model_name.lower()

    # Check for specific patterns
    if "turbo" in model_lower:
        return "turbo"
    if "distil" in model_lower:
        return "distil"
    if "large" in model_lower:
        return "large"
    if "medium" in model_lower:
        return "medium"
    if "small" in model_lower:
        return "small"
    if "base" in model_lower:
        return "base"
    if "tiny" in model_lower:
        return "tiny"

    # Default to tiny for unknown models
    return "tiny"


def estimate_vram_gb(nvidia_output: str | None = None) -> float:
    """Estimate available VRAM from nvidia-smi output.

    Args:
        nvidia_output: Output from nvidia-smi command.

    Returns:
        Estimated VRAM in GB (conservatively assumes 4GB if unknown).
    """
    if not nvidia_output:
        logger.warning("No nvidia-smi output available, assuming 4GB VRAM")
        return 4.0

    try:
        # Look for memory info in nvidia-smi output
        # Pattern: "8192MiB" or "8GB" etc.
        patterns = [
            r"(\d+)MiB",  # Memory in MiB
            r"(\d+)MB",  # Memory in MB
            r"(\d+)GB",  # Memory in GB (less common but possible)
            r"(\d+)\s*GiB",  # Memory in GiB
        ]

        max_memory_mb = 0
        for pattern in patterns:
            matches = re.findall(pattern, nvidia_output)
            for match in matches:
                mem_value = int(match)
                # Convert to MB depending on pattern
                if "GB" in pattern or "GiB" in pattern:
                    mem_value = mem_value * 1024
                max_memory_mb = max(max_memory_mb, mem_value)

        if max_memory_mb > 0:
            vram_gb = max_memory_mb / 1024.0
            logger.info(f"Estimated VRAM: {vram_gb:.1f} GB")
            return round(vram_gb, 1)

    except Exception as e:
        logger.warning(f"Failed to parse VRAM from nvidia-smi: {e}")

    # Default conservative estimate
    logger.warning("Could not determine VRAM, assuming 4GB")
    return 4.0


def get_model_recommendation(system_info: SystemInfo, available_model_list: list[str]) -> ModelRecommendation:
    """Generate model recommendation based on system capabilities.

    Args:
        system_info: Complete system information.
        available_model_list: List of available models from faster-whisper.

    Returns:
        ModelRecommendation with suggested configuration.
    """
    from voicepad_core.diagnostics.models import ModelRecommendation

    gpu_available = system_info.gpu_diagnostics.ctranslate2_cuda.success
    nvidia_output = (
        system_info.gpu_diagnostics.nvidia_smi.output if system_info.gpu_diagnostics.nvidia_smi.success else None
    )
    vram_gb = estimate_vram_gb(nvidia_output) if gpu_available else 0.0
    ram_gb = system_info.ram.available_gb

    logger.info(f"Generating recommendation: GPU={gpu_available}, VRAM={vram_gb}GB, RAM={ram_gb}GB")

    # Helper function to create recommendation
    def recommend(
        model: str, device: str, compute_type: str, reason: str, alternatives: list[str] | None = None
    ) -> ModelRecommendation:
        return ModelRecommendation(
            recommended_model=model,
            recommended_device=device,
            recommended_compute_type=compute_type,
            reason=reason,
            alternative_models=alternatives or [],
            all_available_models=available_model_list,
        )

    if gpu_available:
        # GPU-based recommendations
        if vram_gb >= 6 and "turbo" in available_model_list:
            return recommend(
                "turbo",
                "cuda",
                "float16",
                f"High-end GPU with excellent VRAM ({vram_gb:.1f} GB)",
                ["large-v3", "medium", "small"],
            )
        elif vram_gb >= 5 and "large-v3" in available_model_list:
            return recommend(
                "large-v3",
                "cuda",
                "float16",
                f"High-end GPU with sufficient VRAM ({vram_gb:.1f} GB)",
                ["turbo", "medium", "small"] if "turbo" in available_model_list else ["medium", "small"],
            )
        elif vram_gb >= 3:
            preferred = "medium" if "medium" in available_model_list else "small"
            alts = ["large-v3", "small", "base"] if "large-v3" in available_model_list else ["small", "base"]
            return recommend(
                preferred,
                "cuda",
                "float16",
                f"Mid-range GPU with good VRAM ({vram_gb:.1f} GB)",
                alts,
            )
        elif vram_gb >= 2:
            preferred = "small" if "small" in available_model_list else "base"
            return recommend(
                preferred,
                "cuda",
                "int8",
                f"Consumer GPU with moderate VRAM ({vram_gb:.1f} GB)",
                ["medium", "base", "tiny"],
            )
        else:
            preferred = "tiny" if "tiny" in available_model_list else "base"
            return recommend(
                preferred,
                "cuda",
                "int8",
                f"GPU with limited VRAM ({vram_gb:.1f} GB)",
                ["base", "small"],
            )
    else:
        # CPU-only recommendations
        if ram_gb >= 8:
            # Prefer English-only models for CPU
            if "small.en" in available_model_list:
                preferred = "small.en"
            elif "small" in available_model_list:
                preferred = "small"
            else:
                preferred = "base"

            return recommend(
                preferred,
                "cpu",
                "int8",
                f"CPU with sufficient RAM ({ram_gb:.1f} GB available)",
                ["base", "medium", "tiny"],
            )
        elif ram_gb >= 4:
            preferred = "base" if "base" in available_model_list else "tiny"
            return recommend(
                preferred,
                "cpu",
                "int8",
                f"CPU with moderate RAM ({ram_gb:.1f} GB available)",
                ["small", "tiny"],
            )
        else:
            preferred = "tiny" if "tiny" in available_model_list else "base"
            return recommend(
                preferred,
                "cpu",
                "int8",
                f"CPU with limited RAM ({ram_gb:.1f} GB available)",
                ["base"],
            )
