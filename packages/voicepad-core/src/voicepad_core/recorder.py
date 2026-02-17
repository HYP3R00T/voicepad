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
    from voicepad_core.chunking import ChunkMetadata
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

        # VAD chunking state (only initialized if VAD is enabled)
        self._chunker = None
        self._chunk_queue: queue.Queue = queue.Queue()
        self._accumulated_chunks: list[np.ndarray] = []  # Accumulate chunks in memory
        self._markdown_file: Path | None = None  # Path to markdown file for live updates
        self._markdown_lock = threading.Lock()  # Thread-safe markdown file access
        self._chunk_worker_thread: threading.Thread | None = None
        self._chunk_worker_running = False

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
            if self._chunker is not None:
                # VAD chunking enabled: pass to chunker
                # Note: Chunker processes in background, we just queue the audio
                self._audio_queue.put(indata.copy())
            else:
                # Normal recording: accumulate all audio
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

        # Initialize chunker if VAD is enabled
        if self.config.vad_enabled:
            from voicepad_core.chunking import RealtimeChunker

            self._chunker = RealtimeChunker(
                min_chunk_duration=self.config.vad_min_chunk_duration,
                vad_threshold=self.config.vad_threshold,
                min_silence_duration_ms=self.config.vad_min_silence_duration_ms,
                speech_pad_ms=self.config.vad_speech_pad_ms,
                sample_rate=self._sample_rate,
            )
            self._accumulated_chunks = []

            # Initialize markdown file for live updates
            # Ensure markdown directory exists
            self.config.markdown_path.mkdir(parents=True, exist_ok=True)

            # Create markdown path with same base name as audio file
            base_name = self._output_file.stem
            self._markdown_file = self.config.markdown_path / f"{base_name}.md"

            with open(self._markdown_file, "w", encoding="utf-8") as f:
                f.write(f"# Transcription: {self._output_file.name}\n\n")
                f.write("**Status:** Recording in progress...\n\n")
                f.write("---\n\n")
            logger.info(f"Markdown file initialized: {self._markdown_file}")

            # Start background chunk processing thread
            self._chunk_worker_running = True
            self._chunk_worker_thread = threading.Thread(
                target=self._chunk_worker,
                daemon=True,
            )
            self._chunk_worker_thread.start()

            logger.info(
                "VAD chunking enabled: min_duration=%.1fs, threshold=%.2f",
                self.config.vad_min_chunk_duration,
                self.config.vad_threshold,
            )

        # Start recording
        self._recording = True
        self._record_thread = threading.Thread(
            target=self._record_worker,
            args=(duration,),
            daemon=True,
        )
        self._record_thread.start()

        return self._output_file

    def _chunk_worker(self) -> None:
        """Background worker thread that processes and transcribes audio chunks.

        This thread continuously processes audio from the queue, feeds it to
        the chunker, and for each completed chunk:
        1. Accumulates the audio in memory
        2. Transcribes the chunk
        3. Appends transcription to the markdown file
        """
        logger.debug("Chunk worker thread started")

        # Type guard: chunker must exist when this worker runs
        if self._chunker is None:
            logger.error("Chunk worker started without chunker initialized")
            return

        chunks_processed = 0

        try:
            while self._chunk_worker_running or not self._audio_queue.empty():
                try:
                    # Get audio data with longer timeout to avoid premature exit
                    data = self._audio_queue.get(timeout=0.5)

                    if data is None:
                        continue

                    # Feed to chunker
                    result = self._chunker.add_audio(data)

                    if result is not None:
                        chunk_audio, chunk_metadata = result

                        # Accumulate in memory (don't save individual file)
                        self._accumulated_chunks.append(chunk_audio)
                        chunks_processed += 1

                        logger.info(
                            f"Chunk {chunk_metadata.index} ready ({chunk_metadata.duration:.2f}s), transcribing..."
                        )

                        # Transcribe with error handling - don't kill worker on failure
                        try:
                            self._transcribe_and_append_chunk(chunk_audio, chunk_metadata)
                            logger.info(f"Chunk {chunk_metadata.index} transcribed successfully")
                        except Exception as transcription_error:
                            logger.error(
                                f"Transcription failed for chunk {chunk_metadata.index}: {transcription_error}"
                            )
                            logger.exception(transcription_error)
                            # Continue processing even if transcription fails

                except queue.Empty:
                    # Queue empty - check if we should keep waiting
                    if self._chunk_worker_running:
                        continue  # Still recording, keep waiting for more data
                    else:
                        # Recording stopped and queue empty, break out
                        logger.debug("Queue empty and recording stopped, exiting main loop")
                        break
                except Exception as e:
                    logger.error(f"Error in chunk worker main loop: {e}")
                    logger.exception(e)
                    # Continue processing - don't let one error kill the worker

            # Drain any remaining items from queue
            logger.info("Draining remaining audio from queue...")
            drained = 0
            while not self._audio_queue.empty():
                try:
                    data = self._audio_queue.get_nowait()
                    if data is not None:
                        result = self._chunker.add_audio(data)
                        drained += 1
                        if result is not None:
                            chunk_audio, chunk_metadata = result
                            self._accumulated_chunks.append(chunk_audio)
                            chunks_processed += 1
                            logger.info(f"Drained chunk {chunk_metadata.index}")
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Error draining queue: {e}")

            if drained > 0:
                logger.info(f"Drained {drained} audio frames from queue")

            # Finalize any remaining audio in buffer
            logger.info("Finalizing remaining audio buffer...")
            try:
                final_result = self._chunker.finalize()

                if final_result is not None:
                    chunk_audio, chunk_metadata = final_result

                    # Accumulate final chunk
                    self._accumulated_chunks.append(chunk_audio)
                    chunks_processed += 1

                    logger.info(
                        f"Final chunk {chunk_metadata.index} ready ({chunk_metadata.duration:.2f}s), transcribing..."
                    )

                    # Transcribe final chunk
                    try:
                        self._transcribe_and_append_chunk(chunk_audio, chunk_metadata)
                        logger.info(f"Final chunk {chunk_metadata.index} transcribed")
                    except Exception as e:
                        logger.error(f"Failed to transcribe final chunk: {e}")
                        logger.exception(e)

                    # Update markdown completion status
                    try:
                        self._update_markdown_completion()
                    except Exception as e:
                        logger.error(f"Failed to update markdown completion: {e}")
            except Exception as e:
                logger.error(f"Error finalizing chunks: {e}")
                logger.exception(e)

        except Exception as e:
            logger.error(f"FATAL error in chunk worker: {e}")
            logger.exception(e)

        finally:
            logger.info(
                f"Chunk worker stopped. Processed {chunks_processed} chunks, "
                f"accumulated {len(self._accumulated_chunks)} total chunks"
            )

    def _transcribe_and_append_chunk(
        self,
        chunk_audio: np.ndarray,
        metadata: ChunkMetadata,
    ) -> None:
        """Transcribe a chunk and append to markdown file.

        Args:
            chunk_audio: Audio data for the chunk.
            metadata: Chunk metadata with timing information.
        """
        from voicepad_core.transcription import transcribe_chunk_to_markdown

        if self._markdown_file is None:
            logger.warning("No markdown file initialized")
            return

        try:
            # Save chunk temporarily (transcription needs a file)
            temp_chunk_path = self.config.recordings_path / f"_temp_chunk_{metadata.index}.wav"
            sf.write(
                str(temp_chunk_path),
                chunk_audio,
                self._sample_rate,
                subtype="PCM_16",
            )

            # Transcribe and get markdown content
            chunk_markdown = transcribe_chunk_to_markdown(
                audio_path=temp_chunk_path,
                chunk_index=metadata.index,
                chunk_start_time=metadata.start_time,
                config=self.config,
            )

            # Thread-safe append to markdown
            with self._markdown_lock, open(self._markdown_file, "a", encoding="utf-8") as f:
                f.write(chunk_markdown)
                f.write("\n")  # Spacing between chunks

            # Clean up temp file
            temp_chunk_path.unlink()

        except Exception as e:
            logger.error(f"Failed to transcribe chunk {metadata.index}: {e}")
            # Don't raise - continue processing other chunks

    def _update_markdown_completion(self) -> None:
        """Update markdown file to indicate recording is complete."""
        if self._markdown_file is None:
            return

        with self._markdown_lock:
            try:
                # Read current content
                content = self._markdown_file.read_text(encoding="utf-8")

                # Replace "in progress" with "complete"
                content = content.replace("**Status:** Recording in progress...", "**Status:** Recording complete")

                # Write back
                self._markdown_file.write_text(content, encoding="utf-8")
                logger.info("Markdown file marked as complete")
            except Exception as e:
                logger.error(f"Failed to update markdown completion status: {e}")

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
                    # If VAD chunking is enabled, chunk worker handles audio
                    # Otherwise, we accumulate frames here
                    if self._chunker is None:
                        while self._recording:
                            try:
                                # Process queued audio data
                                data = self._audio_queue.get(timeout=0.1)
                                if data is not None:
                                    recorded_frames.append(data)
                            except queue.Empty:
                                continue
                    else:
                        # VAD mode: just wait for recording to stop
                        # Chunk worker handles the audio
                        while self._recording:
                            import time

                            time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error during recording: {e}")
            self._recording = False
            raise AudioRecorderError(f"Recording failed: {e}") from e

        finally:
            # Stop chunk worker if running
            if self._chunker is not None:
                self._chunk_worker_running = False
                if self._chunk_worker_thread and self._chunk_worker_thread.is_alive():
                    logger.debug("Signaling chunk worker to stop...")
                    # Don't join here - let stop_recording() handle it
                    # so we don't duplicate the logic

            # For non-VAD mode: collect any remaining frames and save
            if self._chunker is None:
                # Collect any remaining frames
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
            # Note: VAD mode saving is now handled in stop_recording()

    def _save_accumulated_audio(self) -> None:
        """Save all accumulated chunks as single audio file."""
        if not self._output_file or not self._accumulated_chunks:
            return

        try:
            # Concatenate all chunks
            merged_audio = np.concatenate(self._accumulated_chunks, axis=0)

            # Save to output file
            sf.write(
                str(self._output_file),
                merged_audio,
                self._sample_rate,
                subtype="PCM_16",
            )

            total_duration = len(merged_audio) / self._sample_rate
            logger.info(f"Recording saved to: {self._output_file}")
            logger.info(f"Total duration: {total_duration:.2f} seconds")
            logger.info(f"Chunks processed: {len(self._accumulated_chunks)}")

        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
            raise AudioRecorderError(f"Failed to save recording: {e}") from e

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

        # Wait for recording thread to complete
        if self._record_thread and self._record_thread.is_alive():
            logger.info("Waiting for recording thread to finish...")
            self._record_thread.join(timeout=10.0)
            if self._record_thread.is_alive():
                logger.warning("Recording thread did not finish in time")

        # CRITICAL: For VAD mode, save accumulated audio HERE
        # This ensures audio is saved even if recording thread terminates abnormally
        if self._chunker is not None:
            # Stop chunk worker
            self._chunk_worker_running = False
            if self._chunk_worker_thread and self._chunk_worker_thread.is_alive():
                logger.info("Waiting for chunk worker to complete...")
                self._chunk_worker_thread.join(timeout=10.0)
                if self._chunk_worker_thread.is_alive():
                    logger.warning("Chunk worker did not finish in time")

            # Save accumulated chunks
            if self._accumulated_chunks:
                logger.info(f"Saving {len(self._accumulated_chunks)} accumulated chunks...")
                try:
                    self._save_accumulated_audio()
                except Exception as e:
                    logger.error(f"Failed to save accumulated audio: {e}")
                    raise AudioRecorderError(f"Failed to save recording: {e}") from e
            else:
                logger.warning("No chunks were accumulated")
                # Check if there's data in the queue that wasn't processed
                if self._audio_queue.qsize() > 0:
                    logger.warning(f"Audio queue has {self._audio_queue.qsize()} unprocessed items!")

        output_file = self._output_file
        self._output_file = None

        return output_file

    def is_recording(self) -> bool:
        """Check if recording is currently in progress.

        Returns:
            True if recording is active, False otherwise.
        """
        return self._recording
