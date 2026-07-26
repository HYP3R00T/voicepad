import pytest
from pydantic import ValidationError
from voicepad_core.config import ConfigError, UnknownTranscriptionModelError
from voicepad_core.config.types import Config
from voicepad_core.models import (
    HuggingFaceArtifact,
    ModelSpec,
    get_model_hint,
    get_model_label,
    list_basic_model_ids,
    list_basic_model_options,
    list_model_ids,
    register_model,
)


def test_valid_transcription_models_is_populated() -> None:
    assert "turbo" in list_model_ids()


def test_config_errors_preserve_hierarchy() -> None:
    assert issubclass(UnknownTranscriptionModelError, ConfigError)


def test_unknown_transcription_model_uses_custom_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Config(transcription_model="definitely-not-a-real-model")

    assert "Unknown transcription model" in str(exc_info.value)


def test_registered_models_appear_in_catalogue() -> None:
    register_model(
        ModelSpec("community-demo", HuggingFaceArtifact("owner/community-demo")),
        overwrite=True,
    )
    assert "community-demo" in list_model_ids()


def test_basic_model_ids_are_curated() -> None:
    assert list_basic_model_ids() == (
        "turbo",
        "small",
        "distil-large-v3.5",
        "parakeet-tdt-0.6b-v3",
    )


def test_basic_model_options_use_friendly_labels() -> None:
    options = dict(list_basic_model_options())
    assert options["Recommended · Turbo"] == "turbo"
    assert options["Lightweight · Small"] == "small"
    assert options["English · Distil Large v3.5"] == "distil-large-v3.5"


def test_basic_model_options_preserve_current_advanced_model() -> None:
    register_model(
        ModelSpec("community-advanced", HuggingFaceArtifact("owner/community-advanced")),
        overwrite=True,
    )

    options = dict(list_basic_model_options(current_model="community-advanced"))

    assert options["Current Advanced · community-advanced"] == "community-advanced"


def test_model_ui_helpers_return_label_and_hint() -> None:
    assert get_model_label("turbo") == "Recommended · Turbo"
    assert "most users" in get_model_hint("turbo")
