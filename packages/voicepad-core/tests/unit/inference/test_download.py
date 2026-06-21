"""Tests for voicepad_core.inference.download."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from huggingface_hub.utils import HfHubHTTPError
from voicepad_core.config import Config
from voicepad_core.inference import download as download_module
from voicepad_core.inference.errors import ModelNotFoundError

REQUIRED_SNAPSHOT_FILES = {
    "model.bin": b"binary model placeholder",
    "tokenizer.json": b"{}",
    "config.json": b'{"model_type": "whisper"}',
}


def _write_snapshot(snapshot_dir: Path, *, include_required: bool = True) -> None:
    snapshot_dir.mkdir(parents=True)
    if include_required:
        for name, content in REQUIRED_SNAPSHOT_FILES.items():
            path = snapshot_dir / name
            if name.endswith(".json"):
                path.write_text(content.decode("utf-8"), encoding="utf-8")
            else:
                path.write_bytes(content)


def test_get_models_dir_uses_config_default(tmp_path: Path, monkeypatch) -> None:
    config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
    monkeypatch.setattr(download_module, "get_config", lambda: config)
    assert download_module._get_models_dir() == config.model_cache_path


def test_get_models_dir_uses_override(tmp_path: Path) -> None:
    override = tmp_path / "override"
    assert download_module._get_models_dir(override) == override


def test_get_cache_roots_returns_hub_and_base(tmp_path: Path) -> None:
    roots = download_module._get_cache_roots(tmp_path)
    assert roots == (tmp_path / "hub", tmp_path)


def test_get_repo_id_uses_voicepad_registry() -> None:
    assert download_module._get_repo_id("turbo") == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"


def test_get_repo_id_falls_back_to_prefix() -> None:
    assert download_module._get_repo_id("custom-model") == "Systran/faster-whisper-custom-model"


def test_model_downloaded_returns_true_when_model_exists(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "custom-model-cache"
    config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
    object.__setattr__(config, "model_cache_path", cache_path)
    monkeypatch.setattr(download_module, "get_config", lambda: config)
    monkeypatch.setattr(download_module, "_get_repo_id", lambda model_name: "owner/model")

    snapshot_dir = cache_path / "hub" / "models--owner--model" / "snapshots" / "abc123"
    _write_snapshot(snapshot_dir)

    assert download_module.model_downloaded("turbo") is True


def test_model_downloaded_returns_false_when_model_missing(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "custom-model-cache"
    config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
    object.__setattr__(config, "model_cache_path", cache_path)
    monkeypatch.setattr(download_module, "get_config", lambda: config)
    monkeypatch.setattr(download_module, "_get_repo_id", lambda model_name: "owner/model")

    assert download_module.model_downloaded("turbo") is False


def test_model_downloaded_checks_base_cache_root(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "custom-model-cache"
    config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
    object.__setattr__(config, "model_cache_path", cache_path)
    monkeypatch.setattr(download_module, "get_config", lambda: config)
    monkeypatch.setattr(download_module, "_get_repo_id", lambda model_name: "owner/model")

    snapshot_dir = cache_path / "models--owner--model" / "snapshots" / "abc123"
    _write_snapshot(snapshot_dir)

    assert download_module.model_downloaded("turbo") is True


def test_model_downloaded_with_explicit_models_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(download_module, "_get_repo_id", lambda model_name: "owner/model")
    snapshot_dir = tmp_path / "hub" / "models--owner--model" / "snapshots" / "abc123"
    _write_snapshot(snapshot_dir)
    assert download_module.model_downloaded("turbo", models_dir=tmp_path) is True


def test_model_downloaded_rejects_incompatible_snapshots(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "custom-model-cache"
    config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
    object.__setattr__(config, "model_cache_path", cache_path)
    monkeypatch.setattr(download_module, "get_config", lambda: config)
    monkeypatch.setattr(download_module, "_get_repo_id", lambda model_name: "owner/model")

    snapshot_dir = cache_path / "hub" / "models--owner--model" / "snapshots" / "abc123"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "model.bin").write_bytes(b"binary model placeholder")

    assert download_module.model_downloaded("turbo") is False


def test_ensure_model_downloaded_uses_configured_cache_path(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "custom-model-cache"
    config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
    object.__setattr__(config, "model_cache_path", cache_path)
    expected_snapshot = cache_path / "models--owner--model" / "snapshots" / "abc123"
    calls: list[tuple[str, str, str | None]] = []

    monkeypatch.setattr(download_module, "get_config", lambda: config)
    monkeypatch.setattr(
        download_module,
        "_find_model_snapshot",
        lambda model_name, models_dir=None: expected_snapshot if expected_snapshot.exists() else None,
    )
    monkeypatch.setattr(
        download_module, "resolve_model_spec", lambda model_name: Mock(repo_id="owner/model", revision=None)
    )

    def mock_snapshot_download(*, repo_id, cache_dir, ignore_patterns, tqdm_class, revision):
        calls.append((repo_id, cache_dir, revision))
        _write_snapshot(expected_snapshot)

    monkeypatch.setattr(download_module, "snapshot_download", mock_snapshot_download)

    snapshot = download_module.ensure_model_downloaded("turbo")

    assert calls == [("owner/model", str(cache_path), None)]
    assert snapshot == expected_snapshot


def test_ensure_model_downloaded_skips_if_already_cached(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "custom-model-cache"
    expected_snapshot = cache_path / "hub" / "models--owner--model" / "snapshots" / "abc123"
    _write_snapshot(expected_snapshot)
    download_called: list[bool] = []

    monkeypatch.setattr(
        download_module,
        "_find_model_snapshot",
        lambda model_name, models_dir=None: expected_snapshot if expected_snapshot.exists() else None,
    )
    monkeypatch.setattr(download_module, "snapshot_download", lambda **kwargs: download_called.append(True))

    result = download_module.ensure_model_downloaded("turbo", models_dir=cache_path)

    assert result == expected_snapshot
    assert len(download_called) == 0


def test_ensure_model_downloaded_creates_cache_dir(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "new-cache"
    expected_snapshot = cache_path / "models--owner--model" / "snapshots" / "abc123"

    monkeypatch.setattr(
        download_module,
        "_find_model_snapshot",
        lambda model_name, models_dir=None: expected_snapshot if expected_snapshot.exists() else None,
    )
    monkeypatch.setattr(
        download_module, "resolve_model_spec", lambda model_name: Mock(repo_id="owner/model", revision=None)
    )

    def mock_snapshot_download(**kwargs):
        _write_snapshot(expected_snapshot)

    monkeypatch.setattr(download_module, "snapshot_download", mock_snapshot_download)

    download_module.ensure_model_downloaded("turbo", models_dir=cache_path)

    assert cache_path.exists()


def test_ensure_model_downloaded_passes_ignore_patterns(tmp_path: Path, monkeypatch) -> None:
    captured_kwargs = {}
    expected_snapshot = tmp_path / "models--owner--model" / "snapshots" / "abc123"

    def mock_snapshot_download(**kwargs):
        captured_kwargs.update(kwargs)
        _write_snapshot(expected_snapshot)

    monkeypatch.setattr(
        download_module,
        "_find_model_snapshot",
        lambda model_name, models_dir=None: expected_snapshot if expected_snapshot.exists() else None,
    )
    monkeypatch.setattr(
        download_module, "resolve_model_spec", lambda model_name: Mock(repo_id="owner/model", revision="main")
    )
    monkeypatch.setattr(download_module, "snapshot_download", mock_snapshot_download)

    download_module.ensure_model_downloaded("turbo", models_dir=tmp_path)

    patterns = captured_kwargs["ignore_patterns"]
    assert "*.msgpack" in patterns
    assert "*.h5" in patterns
    assert "flax_model*" in patterns
    assert "tf_model*" in patterns
    assert "rust_model*" in patterns
    assert captured_kwargs["revision"] == "main"


def test_ensure_model_downloaded_with_progress_callback(tmp_path: Path, monkeypatch) -> None:
    progress_calls = []
    expected_snapshot = tmp_path / "models--owner--model" / "snapshots" / "abc123"

    def on_progress(downloaded: int, total: int):
        progress_calls.append((downloaded, total))

    monkeypatch.setattr(
        download_module,
        "_find_model_snapshot",
        lambda model_name, models_dir=None: expected_snapshot if expected_snapshot.exists() else None,
    )
    monkeypatch.setattr(
        download_module, "resolve_model_spec", lambda model_name: Mock(repo_id="owner/model", revision=None)
    )
    monkeypatch.setattr(download_module, "snapshot_download", lambda **kwargs: _write_snapshot(expected_snapshot))

    download_module.ensure_model_downloaded("turbo", models_dir=tmp_path, on_progress=on_progress)


def test_ensure_model_downloaded_raises_on_hf_http_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(download_module, "_find_model_snapshot", lambda model_name, models_dir=None: None)
    monkeypatch.setattr(
        download_module, "resolve_model_spec", lambda model_name: Mock(repo_id="owner/model", revision=None)
    )

    def mock_snapshot_download(**kwargs):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        raise HfHubHTTPError("404 Not Found", response=mock_response)

    monkeypatch.setattr(download_module, "snapshot_download", mock_snapshot_download)

    with pytest.raises(ModelNotFoundError, match="Failed to download model"):
        download_module.ensure_model_downloaded("turbo", models_dir=tmp_path)


def test_ensure_model_downloaded_raises_on_generic_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(download_module, "_find_model_snapshot", lambda model_name, models_dir=None: None)
    monkeypatch.setattr(
        download_module, "resolve_model_spec", lambda model_name: Mock(repo_id="owner/model", revision=None)
    )

    def mock_snapshot_download(**kwargs):
        raise RuntimeError("Network error")

    monkeypatch.setattr(download_module, "snapshot_download", mock_snapshot_download)

    with pytest.raises(ModelNotFoundError, match="Failed to download model"):
        download_module.ensure_model_downloaded("turbo", models_dir=tmp_path)


def test_ensure_model_downloaded_raises_on_incompatible_download(tmp_path: Path, monkeypatch) -> None:
    expected_snapshot = tmp_path / "models--owner--model" / "snapshots" / "abc123"
    monkeypatch.setattr(
        download_module,
        "_find_model_snapshot",
        lambda model_name, models_dir=None: expected_snapshot if expected_snapshot.exists() else None,
    )
    monkeypatch.setattr(
        download_module, "resolve_model_spec", lambda model_name: Mock(repo_id="owner/model", revision=None)
    )

    def mock_snapshot_download(**kwargs):
        expected_snapshot.mkdir(parents=True)
        (expected_snapshot / "model.bin").write_bytes(b"binary model placeholder")

    monkeypatch.setattr(download_module, "snapshot_download", mock_snapshot_download)

    with pytest.raises(ModelNotFoundError, match="not compatible"):
        download_module.ensure_model_downloaded("turbo", models_dir=tmp_path)


def test_make_progress_tqdm_creates_tqdm_subclass() -> None:
    callback = Mock()
    tqdm_class = download_module._make_progress_tqdm(callback)

    assert tqdm_class is not None
    assert hasattr(tqdm_class, "__init__")
    assert hasattr(tqdm_class, "update")


def test_make_progress_tqdm_calls_callback_on_update() -> None:
    calls = []

    def callback(downloaded: int, total: int):
        calls.append((downloaded, total))

    tqdm_class = download_module._make_progress_tqdm(callback)
    instance = tqdm_class(total=2_000_000)
    instance.update(100_000)
    instance.update(50_000)

    assert len(calls) >= 1
