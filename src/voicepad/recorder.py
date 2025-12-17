import logging
from datetime import datetime
from pathlib import Path

import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)

RECORDINGS_DIR = Path(__file__).parent.parent.parent / "recordings"


def record_audio(
    device_index: int,
    channels: int,
    sample_rate: int,
    duration: float,
) -> str | None:
    """Record audio from specified device.

    Args:
        device_index: Index of the audio device to record from.
        channels: Number of audio channels.
        sample_rate: Sample rate in Hz.
        duration: Duration of recording in seconds.

    Returns:
        Path to the recorded file, or None if recording failed.
    """
    try:
        # Create recordings directory if it doesn't exist
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        filepath = RECORDINGS_DIR / filename

        logger.info(f"Starting recording to {filepath}")

        # Record audio
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            device=device_index,
            blocking=True,
        )

        # Save to file
        sf.write(str(filepath), audio_data, sample_rate)
        logger.info(f"Recording saved to {filepath}")

        return str(filepath)
    except Exception as e:
        logger.error(f"Recording failed: {e}")
        return None


def stop_recording() -> None:
    """Stop any ongoing recording."""
    try:
        sd.stop()
        logger.info("Recording stopped")
    except Exception as e:
        logger.error(f"Failed to stop recording: {e}")
