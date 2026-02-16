"""System resource detection for voicepad."""

from __future__ import annotations

import logging
import platform
from typing import TYPE_CHECKING

import psutil
from faster_whisper import available_models

from voicepad_core.diagnostics.models import CPUInfo, RAMInfo

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def get_available_models() -> list[str]:
    """Get list of all available Whisper models.

    Returns:
        List of model names available in faster-whisper.
    """
    try:
        models = available_models()
        logger.info(f"Retrieved {len(models)} available models from faster-whisper")
        return models
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        # Return a minimal fallback list
        return ["tiny", "base", "small", "medium", "large-v3"]


def get_ram_info() -> RAMInfo:
    """Get system RAM information.

    Returns:
        RAMInfo with total and available RAM in GB.
    """
    try:
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)  # Convert bytes to GB
        available_gb = mem.available / (1024**3)

        logger.info(f"RAM: {available_gb:.1f} GB available / {total_gb:.1f} GB total")

        return RAMInfo(
            total_gb=round(total_gb, 2),
            available_gb=round(available_gb, 2),
        )
    except Exception as e:
        logger.error(f"Failed to get RAM info: {e}")
        # Return conservative defaults
        return RAMInfo(total_gb=0.0, available_gb=0.0)


def get_cpu_info() -> CPUInfo:
    """Get CPU information.

    Returns:
        CPUInfo with core count and model name (if available).
    """
    try:
        cpu_count = psutil.cpu_count(logical=True) or 0

        # Try to get CPU model name
        model_name: str | None = None
        try:
            # This works on most systems
            model_name = platform.processor()
            if not model_name or model_name.strip() == "":
                model_name = None
        except Exception:
            model_name = None

        logger.info(f"CPU: {cpu_count} cores" + (f", {model_name}" if model_name else ""))

        return CPUInfo(
            count=cpu_count,
            model_name=model_name,
        )
    except Exception as e:
        logger.error(f"Failed to get CPU info: {e}")
        return CPUInfo(count=0, model_name=None)
