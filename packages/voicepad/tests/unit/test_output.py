from pathlib import Path

import pytest
from voicepad.output import persist_markdown, render_markdown
from voicepad_core.deployments import PARAKEET_V3_CUDA, PARAKEET_V3_MANIFEST, HuggingFaceSource
from voicepad_core.inference import ActiveDeployment
from voicepad_core.pipeline import TranscriptionResult


def result() -> TranscriptionResult:
    source = PARAKEET_V3_MANIFEST.source
    assert isinstance(source, HuggingFaceSource)
    active = ActiveDeployment(PARAKEET_V3_CUDA, source.revision, "GPU-test", "NVIDIA GPU", 4_000_000_000)
    return TranscriptionResult("hello", (), (), 1.0, 0.1, active, (), (), (), (), True)


def test_markdown_uses_authoritative_result_metadata(tmp_path: Path) -> None:
    content = render_markdown(tmp_path / "recording.wav", result())

    assert "complete: true" in content
    assert PARAKEET_V3_CUDA.id in content
    assert content.endswith("hello\n")


def test_markdown_refuses_to_overwrite_existing_result(tmp_path: Path) -> None:
    audio = tmp_path / "recording.wav"
    destination = tmp_path / "recording.md"
    destination.write_text("existing")

    with pytest.raises(FileExistsError):
        persist_markdown(audio, result(), tmp_path)

    assert destination.read_text() == "existing"
