import pytest
from pydantic import ValidationError
from voicepad_core.config import VALID_TRANSCRIPTION_MODELS, ConfigError, UnknownTranscriptionModelError
from voicepad_core.config.types import Config
from voicepad_core.models import ModelSpec, register_model


def test_valid_transcription_models_is_populated() -> None:
    assert VALID_TRANSCRIPTION_MODELS
    assert "turbo" in VALID_TRANSCRIPTION_MODELS


def test_config_errors_preserve_hierarchy() -> None:
    assert issubclass(UnknownTranscriptionModelError, ConfigError)


def test_unknown_transcription_model_uses_custom_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Config(transcription_model="definitely-not-a-real-model")

    assert "Unknown transcription model" in str(exc_info.value)


def test_registered_models_appear_in_live_model_view() -> None:
    register_model(ModelSpec("community-demo", "owner/community-demo"), overwrite=True)
    assert "community-demo" in VALID_TRANSCRIPTION_MODELS
