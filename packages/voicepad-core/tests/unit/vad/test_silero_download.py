from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from voicepad_core.config import Config
from voicepad_core.vad import silero_download


class _FakeResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)
        self.headers = {"Content-Length": str(sum(len(chunk) for chunk in chunks))}

    def read(self, size: int) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def make_config(tmp_path: Path) -> Config:
    return Config(
        recordings_path=tmp_path / "recordings",
        markdown_path=tmp_path / "markdown",
        vad_model_path=tmp_path / "vad",
        vad_download_chunk_size=4,
    )


def test_get_model_path_uses_sherpa_model_filename(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    result = silero_download.get_model_path(config=config)
    assert result == config.vad_model_path / silero_download.MODEL_FILENAME


def test_ensure_model_exists_returns_existing_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    model_path = config.vad_model_path / silero_download.MODEL_FILENAME
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"data")

    with patch("voicepad_core.vad.silero_download._download") as mock_download:
        result = silero_download.ensure_model_exists(config=config, verbose=False)

    assert result == model_path
    mock_download.assert_not_called()


def test_ensure_model_exists_uses_sherpa_download_values(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    with patch("voicepad_core.vad.silero_download._download") as mock_download, patch("builtins.print"):
        result = silero_download.ensure_model_exists(config=config, verbose=False)

    assert result == config.vad_model_path / silero_download.MODEL_FILENAME
    mock_download.assert_called_once_with(
        result,
        download_url=silero_download.MODEL_URL,
        chunk_size=4,
        verbose=False,
    )


@patch("urllib.request.urlopen")
def test_download_writes_chunks(mock_urlopen: Mock, tmp_path: Path) -> None:
    target = tmp_path / "model.onnx"
    mock_urlopen.return_value = _FakeResponse([b"abcd", b"ef"])

    silero_download._download(target, download_url="https://example.com/model.onnx", chunk_size=4, verbose=True)

    assert target.read_bytes() == b"abcdef"


@patch("urllib.request.urlopen", side_effect=RuntimeError("network down"))
def test_download_cleans_partial_file_on_error(mock_urlopen: Mock, tmp_path: Path) -> None:
    target = tmp_path / "model.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"partial")

    with pytest.raises(RuntimeError, match="Failed to download Silero ONNX model"):
        silero_download._download(target, download_url="https://example.com/model.onnx", chunk_size=4, verbose=False)

    assert not target.exists()
