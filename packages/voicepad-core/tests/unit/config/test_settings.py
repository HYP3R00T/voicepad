"""Tests for voicepad_core.config.settings."""

from __future__ import annotations

from pathlib import Path

import pytest
from voicepad_core.config import Config, get_config, get_config_with_metadata


class TestConfigExpandPaths:
    def test_expand_paths_with_absolute_path_string(self) -> None:
        """When recordings_path is an absolute path string, it becomes a Path object."""
        config = Config(recordings_path="/tmp/recordings", markdown_path="/tmp/markdown")
        assert isinstance(config.recordings_path, Path)

    def test_expand_paths_with_relative_path_string(self) -> None:
        """When recordings_path is a relative string, it becomes a Path object."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert isinstance(config.recordings_path, Path)
        assert config.recordings_path == Path("data/recordings")

    def test_expand_paths_with_path_object(self) -> None:
        """When recordings_path is already a Path, it remains a Path."""
        path = Path("data/recordings")
        config = Config(recordings_path=path, markdown_path=Path("data/markdown"))
        assert isinstance(config.recordings_path, Path)
        assert config.recordings_path == path

    def test_markdown_path_is_expanded(self) -> None:
        """The markdown_path field is also validated and expanded."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
        )
        assert isinstance(config.markdown_path, Path)

    def test_config_is_frozen(self) -> None:
        """The Config model is immutable after creation."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        from pydantic import ValidationError

        with pytest.raises(ValidationError):  # FrozenInstanceError from Pydantic
            config.recordings_path = Path("other")

    def test_default_recordings_path(self) -> None:
        """When recordings_path is not provided, the default is used."""
        config = Config(markdown_path="data/markdown")
        assert config.recordings_path == Path("data/recordings")

    def test_default_markdown_path(self) -> None:
        """When markdown_path is not provided, the default is used."""
        config = Config(recordings_path="data/recordings")
        assert config.markdown_path == Path("data/markdown")

    def test_default_input_device_index_is_none(self) -> None:
        """When input_device_index is not provided, it defaults to None."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert config.input_device_index is None

    def test_default_recording_prefix(self) -> None:
        """When recording_prefix is not provided, it defaults to 'recording'."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert config.recording_prefix == "recording"

    def test_default_transcription_model(self) -> None:
        """When transcription_model is not provided, it defaults to 'small'."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert config.transcription_model == "small"

    def test_custom_input_device_index(self) -> None:
        """When input_device_index is provided, it is stored."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            input_device_index=2,
        )
        assert config.input_device_index == 2

    def test_custom_recording_prefix(self) -> None:
        """When recording_prefix is provided, it is stored."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            recording_prefix="meeting",
        )
        assert config.recording_prefix == "meeting"

    def test_custom_transcription_model(self) -> None:
        """When transcription_model is provided, it is stored."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            transcription_model="base",
        )
        assert config.transcription_model == "base"


class TestGetConfig:
    def test_get_config_returns_config_instance(self) -> None:
        """get_config() returns a Config instance."""
        config = get_config()
        assert isinstance(config, Config)

    def test_get_config_has_defaults(self) -> None:
        """get_config() returns a config with default values."""
        config = get_config()
        assert config.recording_prefix == "recording"
        assert isinstance(config.transcription_model, str)


class TestGetConfigWithMetadata:
    def test_get_config_with_metadata_returns_tuple(self) -> None:
        """get_config_with_metadata() returns a tuple of (config, metadata)."""
        result = get_config_with_metadata()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_config_with_metadata_config_is_config_instance(self) -> None:
        """The first element of the tuple is a Config instance."""
        config, _ = get_config_with_metadata()
        assert isinstance(config, Config)

    def test_get_config_with_metadata_second_element_is_dict(self) -> None:
        """The second element of the tuple is metadata (a dict-like object)."""
        config, metadata = get_config_with_metadata()
        assert isinstance(config, Config)
        assert metadata is not None
