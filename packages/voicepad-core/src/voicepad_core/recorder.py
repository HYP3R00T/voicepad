"""Audio recording functionality for voicepad."""

from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
import soundfile as sf

if TYPE_CHECKING:
    from voicepad_core.config import Config

logger = logging.getLogger(__name__)


class AudioRecorderError(Exception):
    """Base exception for audio recorder errors."""


class AudioRecorder:
    """Records audio from a configured input device and saves to disk."""

    def __init__(self, config: Config) -> None:
        """Initialize the audio recorder with configuration.

        Args:
            config: Configuration object containing device and path settings.

        Raises:
            AudioRecorderError: If configuration is invalid.
        """
        self.config = config
        self._recording = False
        self._audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._record_thread: threading.Thread | None = None
        self._sample_rate = 16000  # Standard sample rate for voice
        self._channels = 1  # Mono recording
        self._output_file: Path | None = None

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate that configuration is usable for recording.

        Raises:
            AudioRecorderError: If configuration is invalid.
        """
        # Ensure recordings directory exists
        recordings_path = self.config.recordings_path
        if not recordings_path.exists():
            try:
                recordings_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created recordings directory: {recordings_path}")
            except OSError as e:
                msg = f"Failed to create recordings directory: {recordings_path}"
                raise AudioRecorderError(msg) from e

        # Check if directory is writable
        if not recordings_path.is_dir():
            msg = f"Recordings path is not a directory: {recordings_path}"
            raise AudioRecorderError(msg)

    def _generate_filename(self, prefix: str | None = None) -> str:
        """Generate a filename with prefix and timestamp.

        Args:
            prefix: Optional prefix to use instead of configured prefix.

        Returns:
            Filename in format: {prefix}_{timestamp}.wav
        """
        prefix_to_use = prefix if prefix is not None else self.config.recording_prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix_to_use}_{timestamp}.wav"

    def _get_output_path(self, prefix: str | None = None) -> Path:
        """Resolve the full output path for the recording.

        Args:
            prefix: Optional prefix to use for filename.

        Returns:
            Full path to the output file.
        """
        filename = self._generate_filename(prefix)
        return self.config.recordings_path / filename

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        """Callback function for sounddevice input stream.

        Args:
            indata: Audio data array.
            frames: Number of frames.
            time_info: Time information dict.
            status: Status flags.
        """
        if status:
            logger.warning(f"Audio callback status: {status}")

        # Put audio data in queue for processing
        if self._recording:
            self._audio_queue.put(indata.copy())

    def start_recording(self, prefix: str | None = None, duration: float | None = None) -> Path:
        """Start recording audio from the configured input device.

        Args:
            prefix: Optional prefix for the output filename.
            duration: Optional duration in seconds for fixed-length recording.

        Returns:
            Path to the output file that will be created.

        Raises:
            AudioRecorderError: If recording is already in progress or device error.
        """
        if self._recording:
            msg = "Recording is already in progress"
            raise AudioRecorderError(msg)

        # Generate output file path
        self._output_file = self._get_output_path(prefix)
        logger.info(f"Starting recording to: {self._output_file}")

        # Start recording
        self._recording = True
        self._record_thread = threading.Thread(
            target=self._record_worker,
            args=(duration,),
            daemon=True,
        )
        self._record_thread.start()

        return self._output_file

    def _record_worker(self, duration: float | None = None) -> None:
        """Worker thread that handles the actual recording.

        Args:
            duration: Optional duration in seconds for fixed-length recording.
        """
        recorded_frames: list[np.ndarray] = []

        try:
            device_index = self.config.input_device_index

            with sd.InputStream(
                device=device_index,
                channels=self._channels,
                samplerate=self._sample_rate,
                callback=self._audio_callback,
            ):
                logger.info(f"Recording started with device index: {device_index}")

                # Record for specified duration or until stopped
                if duration is not None:
                    import time

                    time.sleep(duration)
                    self._recording = False
                else:
                    # Keep recording until stop_recording is called
                    while self._recording:
                        try:
                            # Process queued audio data
                            data = self._audio_queue.get(timeout=0.1)
                            if data is not None:
                                recorded_frames.append(data)
                        except queue.Empty:
                            continue

        except Exception as e:
            logger.error(f"Error during recording: {e}")
            self._recording = False
            raise AudioRecorderError(f"Recording failed: {e}") from e

        finally:
            # Collect any remaining frames if duration-based recording
            while not self._audio_queue.empty():
                try:
                    data = self._audio_queue.get_nowait()
                    if data is not None:
                        recorded_frames.append(data)
                except queue.Empty:
                    break

            # Save recorded audio to file
            if recorded_frames and self._output_file:
                self._save_recording(recorded_frames)
            else:
                logger.warning("No audio data recorded")

    def _save_recording(self, frames: list[np.ndarray]) -> None:
        """Save recorded audio frames to file.

        Args:
            frames: List of audio data arrays to save.
        """
        if not self._output_file:
            logger.error("No output file specified")
            return

        try:
            # Concatenate all frames
            audio_data = np.concatenate(frames, axis=0)

            # Save to WAV file
            sf.write(
                str(self._output_file),
                audio_data,
                self._sample_rate,
                subtype="PCM_16",
            )

            logger.info(f"Recording saved to: {self._output_file}")
            logger.info(f"Duration: {len(audio_data) / self._sample_rate:.2f} seconds")

        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
            raise AudioRecorderError(f"Failed to save recording: {e}") from e

    def stop_recording(self) -> Path | None:
        """Stop the current recording and save to file.

        Returns:
            Path to the saved recording file, or None if no recording was active.

        Raises:
            AudioRecorderError: If there was an error stopping the recording.
        """
        if not self._recording:
            logger.warning("No recording in progress")
            return None

        logger.info("Stopping recording...")
        self._recording = False

        # Wait for recording thread to finish
        if self._record_thread and self._record_thread.is_alive():
            self._record_thread.join(timeout=5.0)

        output_file = self._output_file
        self._output_file = None

        return output_file

    def is_recording(self) -> bool:
        """Check if recording is currently in progress.

        Returns:
            True if recording is active, False otherwise.
        """
        return self._recording
