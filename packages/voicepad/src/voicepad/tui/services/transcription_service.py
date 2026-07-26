from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from voicepad_core import (
    RawAudio,
    TranscriptionError,
    activate_model,
    begin_transcription_session,
    end_transcription_session,
    log_transcription_end,
    log_transcription_start,
    model_is_ready,
    prepare_model,
    transcribe,
)
from voicepad_core.inference import AudioTooShortError

from voicepad.tui.workers import ModelWarmResult

if TYPE_CHECKING:
    from voicepad_core.config import Config
    from voicepad_core.inference import TranscriptionResult

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Service for managing transcription operations."""

    def __init__(self, config: Config) -> None:
        """Initialize the transcription service.

        Args:
            config: Application configuration
        """
        self.config = config

    def warm_model(self) -> ModelWarmResult:
        """Prepare and activate the configured model.

        This operation blocks until complete.

        Returns:
            ModelWarmResult containing device info or error
        """
        try:
            if not model_is_ready(self.config.transcription_model):
                logger.info(f"Model '{self.config.transcription_model}' not cached — downloading")
                prepare_model(self.config.transcription_model)

            runtime = activate_model(
                self.config.transcription_model,
                self.config.transcription_device,
                self.config.transcription_compute_type,
            )

            logger.info(
                "Model activated: backend=%s, device=%s, precision=%s, fallback=%s",
                runtime.backend_id,
                runtime.device,
                runtime.precision,
                runtime.fallback_to_cpu,
            )
            return ModelWarmResult(
                device=runtime.device,
                compute_type=runtime.precision,
                fallback=runtime.fallback_to_cpu,
            )

        except TranscriptionError as e:
            logger.error("Model preparation or activation failed: %s", e)
            return ModelWarmResult(device="cpu", compute_type="int8", fallback=True, error=str(e))
        except Exception as e:
            logger.error("Unexpected model warm failure: %s", e)
            return ModelWarmResult(device="cpu", compute_type="int8", fallback=True, error=str(e))

    def transcribe_audio(self, audio: RawAudio) -> TranscriptionResult | None:
        """Transcribe an audio buffer.

        Args:
            audio: The audio data to transcribe

        Returns:
            TranscriptionResult if successful, None if error

        Raises:
            AudioTooShortError: If the audio is too short to transcribe
            TranscriptionError: If transcription fails
        """
        # Set up per-transcription logging
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        session_logger, log_file = begin_transcription_session(
            logs_path=self.config.logs_path,
            log_level=self.config.log_level,
            session_id=session_id,
        )

        try:
            # Calculate audio duration
            duration_s = audio.duration()

            # Log transcription start
            log_transcription_start(
                session_logger,
                duration_s,
                self.config.transcription_model,
                self.config.transcription_device,
                self.config.transcription_compute_type,
            )

            # We call transcribe safely with expanded args
            result = transcribe(
                audio,
                model_name=self.config.transcription_model,
                device=self.config.transcription_device,
                compute_type=self.config.transcription_compute_type,
            )

            # Log transcription end
            log_transcription_end(
                session_logger,
                success=True,
                latency_ms=result.latency_ms,
                text_length=len(result.text),
            )

            logger.info(f"Transcription complete: {len(result.text)} chars, {result.latency_ms:.0f}ms")
            logger.info(f"Log file: {log_file}")
            return result

        except AudioTooShortError as e:
            log_transcription_end(session_logger, success=False, error=str(e))
            logger.warning(f"Audio too short: {e}")
            raise
        except TranscriptionError as e:
            log_transcription_end(session_logger, success=False, error=str(e))
            logger.error(f"Transcription failed: {e}")
            raise
        except Exception as e:
            log_transcription_end(session_logger, success=False, error=str(e))
            logger.error(f"Unexpected transcription error: {e}")
            raise
        finally:
            end_transcription_session()

    def transcribe_file(self, wav_path: Path) -> TranscriptionResult | None:
        """Transcribe an audio file using the new transcribe_file function.

        Args:
            wav_path: Path to the WAV file to transcribe

        Returns:
            TranscriptionResult if successful, None if error

        Raises:
            FileNotFoundError: If the file doesn't exist
            AudioTooShortError: If the audio is too short to transcribe
            TranscriptionError: If transcription fails
        """
        from voicepad_core import transcribe_file as core_transcribe_file

        # Set up per-transcription logging
        session_id = f"{wav_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_logger, log_file = begin_transcription_session(
            logs_path=self.config.logs_path,
            log_level=self.config.log_level,
            session_id=session_id,
        )

        try:
            session_logger.info(f"Transcribing file: {wav_path}")

            # Use the new transcribe_file function from voicepad-core
            result = core_transcribe_file(
                wav_path,
                model_name=self.config.transcription_model,
                device=self.config.transcription_device,
                compute_type=self.config.transcription_compute_type,
                language=self.config.language,
                local_agreement=self.config.local_agreement_file,
            )

            # Log transcription end
            log_transcription_end(
                session_logger,
                success=True,
                latency_ms=result.latency_ms,
                text_length=len(result.text),
            )

            logger.info(f"File transcription complete: {len(result.text)} chars, {result.latency_ms:.0f}ms")
            logger.info(f"Log file: {log_file}")
            return result

        except (AudioTooShortError, TranscriptionError) as e:
            log_transcription_end(session_logger, success=False, error=str(e))
            raise
        except Exception as e:
            log_transcription_end(session_logger, success=False, error=str(e))
            logger.error(f"Failed to transcribe file {wav_path}: {e}")
            raise
        finally:
            end_transcription_session()
