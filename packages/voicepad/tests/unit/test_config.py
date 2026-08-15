from __future__ import annotations

import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from utilityhub_config.metadata import FieldSource, SettingsMetadata
from voicepad.config import AppConfig, ConfigurationError, config_path, load_config, save_config
from voicepad_core.deployments import PARAKEET_V3_CUDA


def test_config_round_trip_uses_utilityhub_toml(tmp_path: Path) -> None:
    path = tmp_path / "voicepad.toml"
    config = AppConfig(recordings_path=tmp_path / "audio", markdown_path=tmp_path / "text")

    save_config(config, path)

    assert load_config(path) == config
    payload = tomllib.loads(path.read_text())
    assert "schema" not in payload
    assert payload["recordings_path"] == str(tmp_path / "audio")
    assert "input_device_index" not in payload
    assert load_config(path).input_device_index is None


def test_config_path_uses_utilityhub_canonical_path() -> None:
    expected = Path("/tmp/voicepad.toml")
    with patch("voicepad.config.get_config_path", return_value=expected) as get_path:
        assert config_path() == expected

    get_path.assert_called_once_with("voicepad", format="toml")


def test_load_delegates_to_utilityhub_and_logs_source_names(tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "voicepad.toml"
    path.touch()
    expected = AppConfig(theme="nord")
    metadata = SettingsMetadata(
        per_field={"theme": FieldSource("project", str(path), "nord")},
    )
    with (
        caplog.at_level("INFO"),
        patch("voicepad.config.load_settings", return_value=(expected, metadata)) as utility_load,
    ):
        loaded = load_config(path)

    assert loaded == expected
    utility_load.assert_called_once_with(
        AppConfig,
        app_name="voicepad",
        cwd=path.parent,
        config_file=path,
        env_vars=False,
    )
    assert "sources=['project']" in caplog.text


def test_missing_explicit_config_returns_validated_defaults(tmp_path: Path) -> None:
    assert load_config(tmp_path / "missing.toml") == AppConfig()


def test_dotenv_next_to_config_overrides_file_without_reading_repository_dotenv(tmp_path: Path) -> None:
    path = tmp_path / "voicepad.toml"
    path.write_text('theme = "tokyo-night"\n', encoding="utf-8")
    (tmp_path / ".env").write_text("THEME=nord\n", encoding="utf-8")

    assert load_config(path).theme == "nord"


def test_process_environment_does_not_override_desktop_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "voicepad.toml"
    path.write_text('theme = "tokyo-night"\n', encoding="utf-8")
    monkeypatch.setenv("THEME", "nord")

    assert load_config(path).theme == "tokyo-night"


def test_config_expands_user_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOICEPAD_TEST_DATA", str(tmp_path / "data"))
    path = tmp_path / "voicepad.toml"
    path.write_text(
        "\n".join([
            f'deployment_id = "{PARAKEET_V3_CUDA.id}"',
            'recordings_path = "$VOICEPAD_TEST_DATA/recordings"',
            'markdown_path = "$VOICEPAD_TEST_DATA/markdown"',
            'artifact_cache_path = "$VOICEPAD_TEST_DATA/artifacts"',
            'recording_prefix = "recording"',
            "copy_complete_text = true",
            'theme = "tokyo-night"',
        ]),
        encoding="utf-8",
    )

    loaded = load_config(path)

    assert loaded.recordings_path == tmp_path / "data/recordings"
    assert loaded.markdown_path == tmp_path / "data/markdown"
    assert loaded.artifact_cache_path == tmp_path / "data/artifacts"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("deployment_id", '"cpu"', "deployment_id"),
        ("recording_prefix", '"bad/name"', "recording_prefix"),
        ("theme", '"unknown"', "theme"),
        ("unknown_field", '"value"', "unknown_field"),
    ],
)
def test_config_rejects_invalid_or_unknown_fields(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    path = tmp_path / "voicepad.toml"
    path.write_text(f"{field} = {value}\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Could not load") as failure:
        load_config(path)

    assert message in str(failure.value.__cause__)


def test_config_rejects_malformed_toml(tmp_path: Path) -> None:
    path = tmp_path / "voicepad.toml"
    path.write_text("not valid = [", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Could not load"):
        load_config(path)


def test_write_failure_is_reported_as_voicepad_configuration_error(tmp_path: Path) -> None:
    path = tmp_path / "voicepad.toml"
    with (
        patch("voicepad.config.write_config", side_effect=OSError("disk full")),
        pytest.raises(ConfigurationError, match="Could not write") as failure,
    ):
        save_config(AppConfig(), path)

    assert isinstance(failure.value.__cause__, OSError)


def test_app_config_is_frozen_and_rejects_non_curated_deployment() -> None:
    with pytest.raises(ValidationError, match="Unsupported deployment"):
        AppConfig(deployment_id="cpu")

    config = AppConfig()
    field = "theme"
    with pytest.raises(ValidationError, match="frozen"):
        setattr(config, field, "nord")
    assert config.deployment_id == PARAKEET_V3_CUDA.id
