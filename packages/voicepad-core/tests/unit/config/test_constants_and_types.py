import pytest
from voicepad_core.config import Config
from voicepad_core.models import MODELS, model_options


def test_default_model_is_supported() -> None:
    """The default configuration selects a curated model."""
    assert Config().transcription_model in MODELS


def test_unknown_model_is_rejected() -> None:
    """Configuration cannot refer to an absent catalogue entry."""
    with pytest.raises(ValueError, match="Unknown transcription model"):
        Config(transcription_model="unknown")


def test_ui_options_follow_catalogue() -> None:
    """Configuration UI options and accepted models cannot drift apart."""
    assert tuple(value for _, value in model_options()) == tuple(MODELS)
