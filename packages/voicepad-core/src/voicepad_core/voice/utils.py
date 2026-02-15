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


def get_transcript_path(audio_path: Path, markdown_dir: Path) -> Path:
    """
    Generate transcript file path from audio file path.

    Args:
        audio_path: Path to the audio file.
        markdown_dir: Directory where transcripts are saved.

    Returns:
        Path object for the transcript file with .txt extension.
    """
    filename = audio_path.stem + ".txt"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    return markdown_dir / filename
