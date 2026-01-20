"""Utility functions for audio processing."""

import time
from pathlib import Path


def get_timestamp() -> str:
    """Get current timestamp in YYYYMMDD_HHMMSS format."""
    return time.strftime("%Y%m%d_%H%M%S")


def get_recording_path(recordings_dir: Path, prefix: str = "recording") -> Path:
    """
    Generate a unique recording file path with timestamp.

    Args:
        recordings_dir: Directory to save recordings to.
        prefix: Filename prefix (default: "recording").

    Returns:
        Path object for the recording file.
    """
    timestamp = get_timestamp()
    return recordings_dir / f"{prefix}_{timestamp}.wav"
