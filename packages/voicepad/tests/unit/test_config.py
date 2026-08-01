import json
from pathlib import Path

import pytest
from voicepad.config import AppConfig, ConfigurationError, load_config, save_config
from voicepad_core.deployments import PARAKEET_V3_CUDA


def test_config_round_trip_uses_schema_one(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = AppConfig(recordings_path=tmp_path / "audio", markdown_path=tmp_path / "text")

    save_config(config, path)

    assert load_config(path) == config
    assert json.loads(path.read_text())["schema"] == 1


def test_config_rejects_obsolete_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"schema": 1, "transcription_model": "turbo"}))

    with pytest.raises(ConfigurationError, match="Obsolete or unknown"):
        load_config(path)


def test_config_rejects_non_curated_deployment() -> None:
    with pytest.raises(ConfigurationError, match="Unsupported deployment"):
        AppConfig(deployment_id="cpu")

    assert AppConfig().deployment_id == PARAKEET_V3_CUDA.id
