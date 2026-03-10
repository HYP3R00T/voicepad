"""Unit tests for transcription module."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

from voicepad_core.config import Config
from voicepad_core.diagnostics.models import (
    CPUInfo,
    CTranslate2Result,
    GPUDiagnosticsReport,
    NvidiaCheckResult,
    RAMInfo,
    WhisperGPUResult,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voicepad_core.transcription import (
    TranscriptionError,
    _model_cache,
    format_transcription_markdown,
    load_model_with_fallback,
    resolve_auto_settings,
    transcribe_audio,
    transcribe_chunk_to_markdown,
)


def _config(device: str = "cpu", compute: str = "int8") -> Config:
    return cast(
        Config,
        SimpleNamespace(
            transcription_model="tiny",
            transcription_device=device,
            transcription_compute_type=compute,
        ),
    )


class TranscriptionTests(unittest.TestCase):
    def setUp(self) -> None:
        _model_cache.clear()

    def test_resolve_auto_settings_direct(self) -> None:
        cfg = _config("cpu", "int8")
        device, compute = resolve_auto_settings(cfg)
        self.assertEqual((device, compute), ("cpu", "int8"))

    def test_resolve_auto_settings_via_recommendation(self) -> None:
        cfg = _config("auto", "auto")
        recommendation = SimpleNamespace(recommended_device="cpu", recommended_compute_type="int8")
        ram = RAMInfo(total_gb=16.0, available_gb=8.0)
        cpu = CPUInfo(count=8, model_name="x")
        gpu = GPUDiagnosticsReport(
            nvidia_smi=NvidiaCheckResult(success=False, output="no"),
            ctranslate2_cuda=CTranslate2Result(success=False, cuda_device_count=0),
            faster_whisper_gpu=WhisperGPUResult(success=False, message="no"),
        )
        with (
            patch("voicepad_core.get_ram_info", return_value=ram),
            patch("voicepad_core.get_cpu_info", return_value=cpu),
            patch("voicepad_core.gpu_diagnostics", return_value=gpu),
            patch("voicepad_core.get_available_models", return_value=["tiny"]),
            patch("voicepad_core.get_model_recommendation", return_value=recommendation),
        ):
            device, compute = resolve_auto_settings(cfg)
        self.assertEqual(device, "cpu")
        self.assertEqual(compute, "int8")

    def test_format_transcription_markdown(self) -> None:
        segments = [SimpleNamespace(start=0.0, end=1.0, text=" hello ")]
        info = SimpleNamespace(language="en", language_probability=0.9, duration=1.0)
        md = format_transcription_markdown(
            audio_path=Path("a.wav"),
            segments=segments,
            info=info,
            config=_config(),
            device="cpu",
            compute_type="int8",
            fallback_info={"fallback_occurred": True, "missing_components": ["cuBLAS"]},
        )
        self.assertIn("# Transcription", md)
        self.assertIn("[0.0 -> 1.0] hello", md)
        self.assertIn("GPU was requested but CUDA libraries not available", md)

    def test_load_model_with_fallback_cpu(self) -> None:
        fake_model = MagicMock()
        with patch("voicepad_core.transcription.WhisperModel", return_value=fake_model):
            model, device, compute, fallback = load_model_with_fallback("tiny", "cpu", "int8")
        self.assertIs(model, fake_model)
        self.assertEqual(device, "cpu")
        self.assertEqual(compute, "int8")
        self.assertFalse(fallback["fallback_occurred"])

    def test_load_model_with_fallback_cuda_import_error(self) -> None:
        fake_model = MagicMock()

        original_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "nvidia":
                raise ImportError("cublas missing")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            patch("voicepad_core.transcription.WhisperModel", return_value=fake_model),
        ):
            _model, device, compute, fallback = load_model_with_fallback("tiny", "cuda", "float16")
        self.assertEqual(device, "cpu")
        self.assertEqual(compute, "int8")
        self.assertTrue(fallback["fallback_occurred"])

    def test_transcribe_audio_success_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            audio = tmp / "input.wav"
            output = tmp / "out.md"
            audio.write_text("x", encoding="utf-8")

            info = SimpleNamespace(language="en", language_probability=0.9, duration=3.0)
            segments = [SimpleNamespace(text="hello world", start=0.0, end=1.0)]
            model = MagicMock()
            model.transcribe.return_value = (iter(segments), info)

            with (
                patch("voicepad_core.transcription.resolve_auto_settings", return_value=("cpu", "int8")),
                patch(
                    "voicepad_core.transcription.load_model_with_fallback",
                    return_value=(model, "cpu", "int8", {"fallback_occurred": False, "missing_components": []}),
                ),
            ):
                stats = transcribe_audio(audio, output, _config())

            self.assertEqual(stats["word_count"], 2)
            self.assertTrue(output.exists())

            with self.assertRaises(TranscriptionError):
                transcribe_audio(tmp / "missing.wav", output, _config())

    def test_transcribe_chunk_to_markdown_cache_and_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            audio = tmp / "chunk.wav"
            audio.write_text("x", encoding="utf-8")

            info = SimpleNamespace(language="en", language_probability=0.8, duration=2.0)
            segments = [SimpleNamespace(start=0.0, end=1.0, text=" chunk ")]
            model = MagicMock()
            model.transcribe.return_value = (iter(segments), info)

            with (
                patch("voicepad_core.transcription.resolve_auto_settings", return_value=("cpu", "int8")),
                patch(
                    "voicepad_core.transcription.load_model_with_fallback",
                    return_value=(model, "cpu", "int8", {}),
                ) as mock_load,
            ):
                md1 = transcribe_chunk_to_markdown(audio, 0, 0.0, _config())
                md2 = transcribe_chunk_to_markdown(audio, 1, 2.0, _config())

            self.assertIn("## Chunk 1", md1)
            self.assertIn("## Chunk 2", md2)
            self.assertEqual(mock_load.call_count, 1)

            with patch("voicepad_core.transcription.resolve_auto_settings", side_effect=RuntimeError("boom")):
                err_md = transcribe_chunk_to_markdown(audio, 2, 4.0, _config())
            self.assertIn("ERROR", err_md)


if __name__ == "__main__":
    unittest.main()
