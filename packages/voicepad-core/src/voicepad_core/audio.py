"""Audio recording from a microphone input device.

AudioRecorder captures audio and gives it back as a numpy array.
It has no knowledge of transcription, markdown, or file formats beyond WAV.

For long-form recording with VAD chunking, provide an on_chunk callback —
the recorder will invoke it with each completed chunk as it becomes available.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
import soundfile as sf

if TYPE_CHECKING:
    from voicepad_core.chunking import ChunkMetadata
    from voicepad_core.config import Config

logger = logging.getLogger(__name__)

# Audio format constants — fixed for Whisper compatibility
SAMPLE_RATE = 16000
CHANNELS = 1


class AudioRecorderError(Exception):
    """Raised when recording cannot start, continue, or stop cleanly."""


class AudioRecorder:
    """Records audio from a configured input device.

    Usage (simple):
        recorder = AudioRecorder(config)
        recorder.start()
        # ... user speaks ...
        audio = recorder.stop()          # np.ndarray, float32, 16 kHz mono
        result = transcribe_buffer(audio, config)

    Usage (VAD chunking for long recordings):
        def on_chunk(audio: np.ndarray, meta: ChunkMetadata) -> None:
            result = transcribe_buffer(audio, config)
            print(result.text)

        recorder = AudioRecorder(config, on_chunk=on_chunk)
        recorder.start()
        # ... user speaks for a long time ...
        remaining = recorder.stop()      # audio after the last chunk boundary
    """

    def __init__(
        self,
        config: Config,
        on_chunk: Callable[[np.ndarray, ChunkMetadata], None] | None = None,
    ) -> None:
        """Initialise the recorder.

        Args:
            config:   Configuration with device index, paths, and VAD settings.
            on_chunk: Optional callback invoked with each completed VAD chunk.
                      Called from a background thread — must be thread-safe.
                      Only used when config.vad_enabled is True.
        """
        self.config = config
        self._on_chunk = on_chunk

        self._recording = False
        self._frames: list[np.ndarray] = []
        self._frame_lock = threading.Lock()

        self._stream: sd.InputStream | None = None
        self._chunker = None
        self._chunk_thread: threading.Thread | None = None
        self._chunk_queue: queue.Queue[np.ndarray | None] = queue.Queue()

        self._ensure_recordings_dir()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start recording from the configured input device.

        Raises:
            AudioRecorderError: If already recording or device cannot be opened.
        """
        if self._recording:
            raise AudioRecorderError("Recording is already in progress")

        self._frames = []
        self._chunk_queue = queue.Queue()

        if self._on_chunk is not None:
            self._start_chunker()

        try:
            self._stream = sd.InputStream(
                device=self.config.input_device_index,
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            self._stop_chunker()
            raise AudioRecorderError(f"Failed to open audio device: {e}") from e

        self._recording = True
        logger.info(
            f"Recording started — device={self.config.input_device_index}, "
            f"vad={'on' if self._on_chunk is not None else 'off'}"
        )

    def stop(self) -> np.ndarray:
        """Stop recording and return all captured audio as a single array.

        For VAD mode: returns the audio that remained after the last chunk
        boundary (i.e. the tail that was not yet emitted via on_chunk).
        The caller is responsible for transcribing this remainder.

        Returns:
            float32 numpy array at 16 kHz mono. May be empty (length 0) if
            nothing was recorded.

        Raises:
            AudioRecorderError: If no recording was in progress.
        """
        if not self._recording:
            raise AudioRecorderError("No recording in progress")

        self._recording = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning(f"Error closing audio stream: {e}")
            finally:
                self._stream = None

        # Signal chunk worker to drain and exit
        if self._on_chunk is not None:
            self._chunk_queue.put(None)  # sentinel
            self._stop_chunker()

        with self._frame_lock:
            frames = list(self._frames)

        if not frames:
            logger.warning("stop() called but no audio frames were captured")
            return np.array([], dtype=np.float32)

        audio = np.concatenate(frames, axis=0).flatten()
        duration_s = len(audio) / SAMPLE_RATE
        logger.info(f"Recording stopped — {duration_s:.2f}s captured")
        return audio

    def is_recording(self) -> bool:
        """Return True if recording is currently active."""
        return self._recording

    def save_wav(self, audio: np.ndarray, path: Path) -> None:
        """Save a float32 audio array to a WAV file.

        Args:
            audio: float32 numpy array at 16 kHz mono.
            path:  Destination path. Parent directories are created if needed.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")
        logger.debug(f"Saved {len(audio) / SAMPLE_RATE:.2f}s audio to {path}")

    def generate_wav_path(self, prefix: str | None = None) -> Path:
        """Generate a timestamped WAV path under the configured recordings directory.

        Args:
            prefix: Filename prefix. Defaults to config.recording_prefix.

        Returns:
            Path like data/recordings/recording_20260427_143022.wav
        """
        pfx = prefix if prefix is not None else self.config.recording_prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.config.recordings_path / f"{pfx}_{timestamp}.wav"

    # ------------------------------------------------------------------
    # Internal — audio callback
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning(f"Audio stream status: {status}")

        if not self._recording:
            return

        frame = indata.copy()

        with self._frame_lock:
            self._frames.append(frame)

        if self._on_chunk is not None:
            self._chunk_queue.put(frame)

    # ------------------------------------------------------------------
    # Internal — VAD chunk worker
    # ------------------------------------------------------------------

    def _start_chunker(self) -> None:
        """Initialise the RealtimeChunker and start the background worker."""
        from voicepad_core.chunking import RealtimeChunker

        self._chunker = RealtimeChunker()  # uses module-level defaults

        self._chunk_thread = threading.Thread(
            target=self._chunk_worker,
            daemon=True,
            name="VadChunkWorker",
        )
        self._chunk_thread.start()
        logger.debug("VAD chunk worker started")

    def _stop_chunker(self) -> None:
        """Wait for the chunk worker to finish (up to 60 s)."""
        if self._chunk_thread and self._chunk_thread.is_alive():
            self._chunk_thread.join(timeout=60.0)
            if self._chunk_thread.is_alive():
                logger.warning("VAD chunk worker did not finish within timeout")
        self._chunk_thread = None
        self._chunker = None

    def _chunk_worker(self) -> None:
        """Background thread: feed audio frames to the chunker, fire on_chunk."""
        chunker = self._chunker
        if chunker is None:
            return

        while True:
            try:
                frame = self._chunk_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if frame is None:
                # Sentinel — drain remaining buffer and exit
                result = chunker.finalize()
                if result is not None and self._on_chunk is not None:
                    chunk_audio, metadata = result
                    try:
                        self._on_chunk(chunk_audio, metadata)
                    except Exception as e:
                        logger.error(f"on_chunk callback raised: {e}", exc_info=True)
                break

            result = chunker.add_audio(frame)
            if result is not None and self._on_chunk is not None:
                chunk_audio, metadata = result
                try:
                    self._on_chunk(chunk_audio, metadata)
                except Exception as e:
                    logger.error(f"on_chunk callback raised: {e}", exc_info=True)

        logger.debug("VAD chunk worker finished")

    # ------------------------------------------------------------------
    # Internal — setup
    # ------------------------------------------------------------------

    def _ensure_recordings_dir(self) -> None:
        path = self.config.recordings_path
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise AudioRecorderError(f"Cannot create recordings directory '{path}': {e}") from e
        if not path.is_dir():
            raise AudioRecorderError(f"Recordings path is not a directory: {path}")
