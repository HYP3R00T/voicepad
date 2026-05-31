"""Recording service for VoicePad TUI."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

from voicepad.tui.workers import RecordingSession

if TYPE_CHECKING:
    from voicepad_core.config import Config

logger = logging.getLogger(__name__)


class RecordingService:
    """Service for managing audio recording operations."""

    def __init__(self, config: Config) -> None:
        """Initialize the recording service.

        Args:
            config: Application configuration
        """
        self.config = config

    def create_session(self) -> RecordingSession:
        """Create a new recording session.

        Returns:
            A new RecordingSession instance
        """
        return RecordingSession(config=self.config)

    def start_session(self, session: RecordingSession) -> None:
        """Start a recording session.

        Args:
            session: The recording session to start

        Raises:
            RuntimeError: If the recorder fails to start
        """
        try:
            session.start()
            logger.info("Recording session started")
        except RuntimeError as e:
            logger.error(f"Failed to start recording: {e}")
            raise

    def stop_session(self, session: RecordingSession) -> np.ndarray:
        """Stop a recording session and return the audio.

        Args:
            session: The recording session to stop

        Returns:
            The recorded audio as a numpy array

        Raises:
            RuntimeError: If the recorder fails to stop
        """
        try:
            audio = session.stop()
            logger.info(f"Recording stopped, captured {len(audio)} samples")
            return audio
        except RuntimeError as e:
            logger.error(f"Failed to stop recording: {e}")
            raise

    def save_audio(self, audio: np.ndarray, prefix: str | None = None) -> Path:
        """Save audio to a WAV file.

        Args:
            audio: The audio data to save
            prefix: Optional prefix for the filename (default: timestamp)

        Returns:
            Path to the saved WAV file

        Raises:
            Exception: If saving fails
        """
        # Create output directory if it doesn't exist
        self.config.recordings_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if prefix is None:
            prefix = time.strftime("%Y%m%d_%H%M%S")
        wav_path = self.config.recordings_path / f"{prefix}.wav"

        # Save the audio file
        try:
            sf.write(wav_path, audio, 16000)
            logger.info(f"Audio saved to {wav_path}")
            return wav_path
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise

    def get_audio_duration(self, audio: np.ndarray) -> float:
        """Calculate the duration of an audio array in seconds.

        Args:
            audio: The audio data

        Returns:
            Duration in seconds
        """
        return len(audio) / 16000
