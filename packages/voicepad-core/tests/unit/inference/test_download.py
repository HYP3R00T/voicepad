"""Tests for voicepad_core.inference.download."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from huggingface_hub.utils import HfHubHTTPError
from voicepad_core.config import Config
from voicepad_core.inference import download as download_module
from voicepad_core.inference.errors import ModelNotFoundError

# ============================================================================
# _get_models_dir tests
# ============================================================================


def test_get_models_dir_uses_config_default(tmp_path: Path, monkeypatch) -> None:
    """_get_models_dir returns configured model_cache_path when no override."""
    config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
    monkeypatch.setattr(download_module, "get_config", lambda: config)

    result = download_module._get_models_dir()
    assert result == config.model_cache_path


def test_get_models_dir_uses_override(tmp_path: Path) -> None:
    """_get_models_dir returns override path when provided."""
    override = tmp_path / "override"
    result = download_module._get_models_dir(override)
    assert result == override


# ============================================================================
# _get_cache_roots tests
# ============================================================================


def test_get_cache_roots_returns_hub_and_base(tmp_path: Path) -> None:
    """_get_cache_roots returns both hub and base directories."""
    roots = download_module._get_cache_roots(tmp_path)
    assert len(roots) == 2
    assert roots[0] == tmp_path / "hub"
    assert roots[1] == tmp_path


# ============================================================================
# _get_repo_id tests
# ============================================================================


def test_get_repo_id_uses_faster_whisper_registry() -> None:
    """_get_repo_id uses faster_whisper's _MODELS registry when available."""
    # Mock the faster_whisper.utils module with _MODELS
    mock_utils = MagicMock()
    mock_utils._MODELS = {"turbo": "Systran/faster-distil-whisper-large-v3"}

    with patch.dict("sys.modules", {"faster_whisper.utils": mock_utils}):
        result = download_module._get_repo_id("turbo")
        assert result == "Systran/faster-distil-whisper-large-v3"


def test_get_repo_id_falls_back_to_prefix() -> None:
    """_get_repo_id falls back to HF_REPO_PREFIX for unknown models."""
    result = download_module._get_repo_id("custom-model")
    assert result == "Systran/faster-whisper-custom-model"


def test_get_repo_id_handles_import_error() -> None:
    """_get_repo_id handles ImportError gracefully."""
    # Simulate import failure by making the import raise an exception
    with patch("builtins.__import__", side_effect=ImportError):
        result = download_module._get_repo_id("turbo")
        assert result == "Systran/faster-whisper-turbo"


# ============================================================================
# model_downloaded tests
# ============================================================================


def test_model_downloaded_returns_true_when_model_exists(tmp_path: Path, monkeypatch) -> None:
    """model_downloaded returns True when model.bin exists in snapshots."""
    cache_path = tmp_path / "custom-model-cache"
    config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
    object.__setattr__(config, "model_cache_path", cache_path)

    monkeypatch.setattr(download_module, "get_config", lambda: config)
    monkeypatch.setattr(download_module, "_get_repo_id", lambda model_name: "owner/model")

    snapshot_dir = cache_path / "hub" / "models--owner--model" / "snapshots" / "abc123"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "model.bin").write_bytes(b"binary model placeholder")

    assert download_module.model_downloaded("turbo") is True


def test_model_downloaded_returns_false_when_model_missing(tmp_path: Path, monkeypatch) -> None:
    """model_downloaded returns False when model.bin does not exist."""
    cache_path = tmp_path / "custom-model-cache"
    config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
    object.__setattr__(config, "model_cache_path", cache_path)

    monkeypatch.setattr(download_module, "get_config", lambda: config)
    monkeypatch.setattr(download_module, "_get_repo_id", lambda model_name: "owner/model")

    assert download_module.model_downloaded("turbo") is False


def test_model_downloaded_checks_base_cache_root(tmp_path: Path, monkeypatch) -> None:
    """model_downloaded checks base cache root when hub root doesn't exist."""
    cache_path = tmp_path / "custom-model-cache"
    config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
    object.__setattr__(config, "model_cache_path", cache_path)

    monkeypatch.setattr(download_module, "get_config", lambda: config)
    monkeypatch.setattr(download_module, "_get_repo_id", lambda model_name: "owner/model")

    # Create model in base cache root instead of hub
    snapshot_dir = cache_path / "models--owner--model" / "snapshots" / "abc123"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "model.bin").write_bytes(b"binary model placeholder")

    assert download_module.model_downloaded("turbo") is True


def test_model_downloaded_with_explicit_models_dir(tmp_path: Path, monkeypatch) -> None:
    """model_downloaded accepts explicit models_dir parameter."""
    monkeypatch.setattr(download_module, "_get_repo_id", lambda model_name: "owner/model")

    snapshot_dir = tmp_path / "hub" / "models--owner--model" / "snapshots" / "abc123"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "model.bin").write_bytes(b"binary model placeholder")

    assert download_module.model_downloaded("turbo", models_dir=tmp_path) is True


def test_model_downloaded_ignores_snapshots_without_model_bin(tmp_path: Path, monkeypatch) -> None:
    """model_downloaded returns False if snapshot exists but model.bin is missing."""
    cache_path = tmp_path / "custom-model-cache"
    config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
    object.__setattr__(config, "model_cache_path", cache_path)

    monkeypatch.setattr(download_module, "get_config", lambda: config)
    monkeypatch.setattr(download_module, "_get_repo_id", lambda model_name: "owner/model")

    # Create snapshot directory but no model.bin
    snapshot_dir = cache_path / "hub" / "models--owner--model" / "snapshots" / "abc123"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}")

    assert download_module.model_downloaded("turbo") is False


# ============================================================================
# ensure_model_downloaded tests
# ============================================================================


def test_ensure_model_downloaded_uses_configured_cache_path(tmp_path: Path, monkeypatch) -> None:
    """ensure_model_downloaded passes configured cache path to snapshot_download."""
    cache_path = tmp_path / "custom-model-cache"
    config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
    object.__setattr__(config, "model_cache_path", cache_path)

    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(download_module, "get_config", lambda: config)
    monkeypatch.setattr(download_module, "_get_repo_id", lambda model_name: "owner/model")
    monkeypatch.setattr(download_module, "model_downloaded", lambda model_name, models_dir=None: False)
    monkeypatch.setattr(
        download_module,
        "snapshot_download",
        lambda *, repo_id, cache_dir, ignore_patterns, tqdm_class: calls.append((repo_id, cache_dir)),
    )

    download_module.ensure_model_downloaded("turbo")

    assert calls == [("owner/model", str(cache_path))]


def test_ensure_model_downloaded_skips_if_already_cached(tmp_path: Path, monkeypatch) -> None:
    """ensure_model_downloaded is no-op when model already exists."""
    cache_path = tmp_path / "custom-model-cache"
    download_called = []

    monkeypatch.setattr(download_module, "model_downloaded", lambda model_name, models_dir: True)
    monkeypatch.setattr(
        download_module,
        "snapshot_download",
        lambda **kwargs: download_called.append(True),
    )

    download_module.ensure_model_downloaded("turbo", models_dir=cache_path)

    assert len(download_called) == 0


def test_ensure_model_downloaded_creates_cache_dir(tmp_path: Path, monkeypatch) -> None:
    """ensure_model_downloaded creates cache directory if it doesn't exist."""
    cache_path = tmp_path / "new-cache"

    monkeypatch.setattr(download_module, "_get_repo_id", lambda model_name: "owner/model")
    monkeypatch.setattr(download_module, "model_downloaded", lambda model_name, models_dir: False)
    monkeypatch.setattr(download_module, "snapshot_download", lambda **kwargs: None)

    download_module.ensure_model_downloaded("turbo", models_dir=cache_path)

    assert cache_path.exists()


def test_ensure_model_downloaded_passes_ignore_patterns(tmp_path: Path, monkeypatch) -> None:
    """ensure_model_downloaded passes correct ignore patterns to snapshot_download."""
    captured_kwargs = {}

    def mock_snapshot_download(**kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(download_module, "_get_repo_id", lambda model_name: "owner/model")
    monkeypatch.setattr(download_module, "model_downloaded", lambda model_name, models_dir: False)
    monkeypatch.setattr(download_module, "snapshot_download", mock_snapshot_download)

    download_module.ensure_model_downloaded("turbo", models_dir=tmp_path)

    assert "ignore_patterns" in captured_kwargs
    patterns = captured_kwargs["ignore_patterns"]
    assert "*.msgpack" in patterns
    assert "*.h5" in patterns
    assert "flax_model*" in patterns
    assert "tf_model*" in patterns
    assert "rust_model*" in patterns


def test_ensure_model_downloaded_with_progress_callback(tmp_path: Path, monkeypatch) -> None:
    """ensure_model_downloaded accepts progress callback."""
    progress_calls = []

    def on_progress(downloaded: int, total: int):
        progress_calls.append((downloaded, total))

    monkeypatch.setattr(download_module, "_get_repo_id", lambda model_name: "owner/model")
    monkeypatch.setattr(download_module, "model_downloaded", lambda model_name, models_dir: False)
    monkeypatch.setattr(download_module, "snapshot_download", lambda **kwargs: None)

    download_module.ensure_model_downloaded("turbo", models_dir=tmp_path, on_progress=on_progress)

    # Progress callback should be wrapped in tqdm class
    # We can't easily test the callback itself without mocking tqdm


def test_ensure_model_downloaded_raises_on_hf_http_error(tmp_path: Path, monkeypatch) -> None:
    """ensure_model_downloaded raises ModelNotFoundError on HuggingFace HTTP error."""
    monkeypatch.setattr(download_module, "_get_repo_id", lambda model_name: "owner/model")
    monkeypatch.setattr(download_module, "model_downloaded", lambda model_name, models_dir: False)

    def mock_snapshot_download(**kwargs):
        # Create a mock response object for HfHubHTTPError
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        raise HfHubHTTPError("404 Not Found", response=mock_response)

    monkeypatch.setattr(download_module, "snapshot_download", mock_snapshot_download)

    with pytest.raises(ModelNotFoundError, match="Failed to download model"):
        download_module.ensure_model_downloaded("turbo", models_dir=tmp_path)


def test_ensure_model_downloaded_raises_on_generic_error(tmp_path: Path, monkeypatch) -> None:
    """ensure_model_downloaded raises ModelNotFoundError on generic errors."""
    monkeypatch.setattr(download_module, "_get_repo_id", lambda model_name: "owner/model")
    monkeypatch.setattr(download_module, "model_downloaded", lambda model_name, models_dir: False)

    def mock_snapshot_download(**kwargs):
        raise RuntimeError("Network error")

    monkeypatch.setattr(download_module, "snapshot_download", mock_snapshot_download)

    with pytest.raises(ModelNotFoundError, match="Failed to download model"):
        download_module.ensure_model_downloaded("turbo", models_dir=tmp_path)


# ============================================================================
# _make_progress_tqdm tests
# ============================================================================


def test_make_progress_tqdm_creates_tqdm_subclass() -> None:
    """_make_progress_tqdm returns a tqdm subclass."""
    callback = Mock()
    tqdm_class = download_module._make_progress_tqdm(callback)

    assert tqdm_class is not None
    assert hasattr(tqdm_class, "__init__")
    assert hasattr(tqdm_class, "update")


def test_make_progress_tqdm_calls_callback_on_update() -> None:
    """_make_progress_tqdm tqdm subclass calls callback on update."""
    calls = []

    def callback(downloaded: int, total: int):
        calls.append((downloaded, total))

    tqdm_class = download_module._make_progress_tqdm(callback)

    # Create instance with large total to trigger tracking
    instance = tqdm_class(total=2_000_000)

    # Simulate updates
    instance.update(100_000)
    instance.update(50_000)

    # Callback should have been called
    assert len(calls) >= 1
