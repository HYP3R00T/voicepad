"""Transcription module for audio-to-text conversion using faster-whisper."""

from voicepad.transcription.transcriber import TranscriptionResult, transcribe_audio

__all__ = ["transcribe_audio", "TranscriptionResult"]
