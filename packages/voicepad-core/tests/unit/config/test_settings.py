"""Tests for voicepad_core.config.settings."""

from __future__ import annotations

import sys
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
        """When recordings_path is not provided, the default points to ~/.config/voicepad/data/recordings."""
        config = Config(markdown_path="data/markdown")
        assert "voicepad" in str(config.recordings_path)
        assert "recordings" in str(config.recordings_path)

    def test_default_markdown_path(self) -> None:
        """When markdown_path is not provided, the default points to ~/.config/voicepad/data/markdown."""
        config = Config(recordings_path="data/recordings")
        assert "voicepad" in str(config.markdown_path)
        assert "markdown" in str(config.markdown_path)

    def test_default_input_device_index_is_none(self) -> None:
        """When input_device_index is not provided, it defaults to None."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert config.input_device_index is None

    def test_default_recording_prefix(self) -> None:
        """When recording_prefix is not provided, it defaults to 'recording'."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert config.recording_prefix == "recording"

    def test_default_transcription_model(self) -> None:
        """When transcription_model is not provided, it defaults to 'turbo'."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert config.transcription_model == "turbo"

    def test_default_transcription_device(self) -> None:
        """transcription_device defaults to 'auto'."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert config.transcription_device == "auto"

    def test_default_transcription_compute_type(self) -> None:
        """transcription_compute_type defaults to 'auto'."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert config.transcription_compute_type == "auto"

    def test_transcription_device_rejects_invalid(self) -> None:
        """transcription_device only accepts 'auto', 'cuda', 'cpu'."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Config(recordings_path="data/recordings", markdown_path="data/markdown", transcription_device="gpu")  # type: ignore[arg-type]

    def test_transcription_compute_type_rejects_invalid(self) -> None:
        """transcription_compute_type only accepts known values."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Config(
                recordings_path="data/recordings", markdown_path="data/markdown", transcription_compute_type="bfloat16"
            )  # type: ignore[arg-type]

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


# ---------------------------------------------------------------------------
# Comprehensive path expansion tests (bug fix: tilde in defaults)
# ---------------------------------------------------------------------------


class TestDefaultPathsExpandTilde:
    """Tests for the bug fix: default paths with ~ should expand to home directory."""

    def test_default_recordings_path_expands_tilde(self) -> None:
        """When recordings_path uses default, tilde is expanded to home directory."""
        config = Config(markdown_path="data/markdown")
        # Path should NOT contain literal tilde
        assert "~" not in str(config.recordings_path)
        # Path should be absolute
        assert config.recordings_path.is_absolute()
        # Path should contain voicepad config directory
        assert "voicepad" in str(config.recordings_path).lower()

    def test_default_markdown_path_expands_tilde(self) -> None:
        """When markdown_path uses default, tilde is expanded to home directory."""
        config = Config(recordings_path="data/recordings")
        # Path should NOT contain literal tilde
        assert "~" not in str(config.markdown_path)
        # Path should be absolute
        assert config.markdown_path.is_absolute()
        # Path should contain voicepad config directory
        assert "voicepad" in str(config.markdown_path).lower()

    def test_default_model_cache_path_expands_tilde(self) -> None:
        """When model_cache_path uses default, tilde is expanded to home directory."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        # Path should NOT contain literal tilde
        assert "~" not in str(config.model_cache_path)
        # Path should be absolute
        assert config.model_cache_path.is_absolute()
        # Path should contain voicepad config directory
        assert "voicepad" in str(config.model_cache_path).lower()

    def test_all_default_paths_expand_tilde(self) -> None:
        """When all paths use defaults, all tildes are expanded to home directory."""
        config = Config()
        # None should contain literal tilde
        assert "~" not in str(config.recordings_path)
        assert "~" not in str(config.markdown_path)
        assert "~" not in str(config.model_cache_path)
        # All should be absolute
        assert config.recordings_path.is_absolute()
        assert config.markdown_path.is_absolute()
        assert config.model_cache_path.is_absolute()

    def test_default_paths_are_different(self) -> None:
        """Each default path points to a different directory."""
        config = Config()
        assert config.recordings_path != config.markdown_path
        assert config.recordings_path != config.model_cache_path
        assert config.markdown_path != config.model_cache_path

    def test_default_recordings_path_contains_recordings_subdir(self) -> None:
        """Default recordings_path ends with 'recordings' directory."""
        config = Config(markdown_path="data/markdown")
        assert config.recordings_path.name == "recordings"

    def test_default_markdown_path_contains_markdown_subdir(self) -> None:
        """Default markdown_path ends with 'markdown' directory."""
        config = Config(recordings_path="data/recordings")
        assert config.markdown_path.name == "markdown"

    def test_default_model_cache_path_contains_models_subdir(self) -> None:
        """Default model_cache_path ends with 'models' directory."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert config.model_cache_path.name == "models"


class TestProvidedPathsExpandTilde:
    """Tests for tilde expansion when paths are explicitly provided."""

    def test_provided_recordings_path_with_tilde_expands(self) -> None:
        """When recordings_path is provided with tilde, it is expanded."""
        config = Config(recordings_path="~/myrecordings", markdown_path="data/markdown")
        assert "~" not in str(config.recordings_path)
        assert config.recordings_path.is_absolute()

    def test_provided_markdown_path_with_tilde_expands(self) -> None:
        """When markdown_path is provided with tilde, it is expanded."""
        config = Config(recordings_path="data/recordings", markdown_path="~/mymarkdown")
        assert "~" not in str(config.markdown_path)
        assert config.markdown_path.is_absolute()

    def test_provided_model_cache_path_with_tilde_expands(self) -> None:
        """When model_cache_path is provided with tilde, it is expanded."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            model_cache_path="~/.mymodels",
        )
        assert "~" not in str(config.model_cache_path)
        assert config.model_cache_path.is_absolute()

    def test_provided_path_with_tilde_slashfile_expands(self) -> None:
        """Tilde in middle of path like ~/dir/subdir is properly expanded."""
        config = Config(
            recordings_path="~/voicepad/data/recordings",
            markdown_path="data/markdown",
        )
        assert "~" not in str(config.recordings_path)
        assert config.recordings_path.is_absolute()
        assert "voicepad" in str(config.recordings_path)

    def test_multiple_configs_with_tilde_all_expand(self) -> None:
        """Multiple Config instances all properly expand tilde paths."""
        config1 = Config(recordings_path="~/rec1", markdown_path="data/markdown")
        config2 = Config(recordings_path="~/rec2", markdown_path="data/markdown")
        config3 = Config()

        assert "~" not in str(config1.recordings_path)
        assert "~" not in str(config2.recordings_path)
        assert "~" not in str(config3.recordings_path)

        assert config1.recordings_path != config2.recordings_path


class TestPathExpansionWithEnvironmentVariables:
    """Tests for environment variable expansion in paths."""

    def test_provided_path_with_env_var_expands(self, monkeypatch) -> None:
        """When path contains environment variable like $HOME, it is expanded."""
        # Set a test env var
        monkeypatch.setenv("TEST_RECORDINGS_DIR", "/tmp/test_recordings")
        config = Config(
            recordings_path="$TEST_RECORDINGS_DIR",
            markdown_path="data/markdown",
        )
        # Path should be expanded
        # Use as_posix() for cross-platform comparison (Windows vs Unix paths)
        assert config.recordings_path.as_posix() == "/tmp/test_recordings"

    def test_path_with_mixed_tilde_and_env_var(self, monkeypatch) -> None:
        """Path expansion handles both ~ and environment variables."""
        # The expand_path function should handle both
        config = Config(recordings_path="~/data", markdown_path="data/markdown")
        assert "~" not in str(config.recordings_path)
        assert config.recordings_path.is_absolute()


class TestPathObjectsNoLiteralTildeDirectories:
    """Tests ensuring no literal ~ directories are created when using configs."""

    def test_config_recordings_path_does_not_create_literal_tilde_dir(self, tmp_path: Path) -> None:
        """Using config.recordings_path.mkdir() does not create ~/data/recordings literally."""
        config = Config(
            recordings_path=str(tmp_path / "recordings"),
            markdown_path=str(tmp_path / "markdown"),
        )
        config.recordings_path.mkdir(parents=True, exist_ok=True)
        # Check that no literal ~ directory was created
        assert not (tmp_path / "~").exists()
        assert (tmp_path / "recordings").exists()

    def test_config_markdown_path_does_not_create_literal_tilde_dir(self, tmp_path: Path) -> None:
        """Using config.markdown_path.mkdir() does not create ~/data/markdown literally."""
        config = Config(
            recordings_path=str(tmp_path / "recordings"),
            markdown_path=str(tmp_path / "markdown"),
        )
        config.markdown_path.mkdir(parents=True, exist_ok=True)
        # Check that no literal ~ directory was created
        assert not (tmp_path / "~").exists()
        assert (tmp_path / "markdown").exists()

    def test_config_model_cache_path_does_not_create_literal_tilde_dir(self, tmp_path: Path) -> None:
        """Using config.model_cache_path.mkdir() does not create ~/models literally."""
        config = Config(
            recordings_path=str(tmp_path / "recordings"),
            markdown_path=str(tmp_path / "markdown"),
            model_cache_path=str(tmp_path / "models"),
        )
        config.model_cache_path.mkdir(parents=True, exist_ok=True)
        # Check that no literal ~ directory was created
        assert not (tmp_path / "~").exists()
        assert (tmp_path / "models").exists()

    def test_default_paths_can_be_created_without_literal_tilde_dir(self, tmp_path: Path, monkeypatch) -> None:
        """Creating directories with default paths does not create literal ~ dirs."""
        # Change home to temp dir for this test
        import os

        original_home = os.environ.get("HOME")
        monkeypatch.setenv("HOME", str(tmp_path))

        try:
            config = Config(markdown_path=str(tmp_path / "markdown"))  # recordings_path uses default
            # Create the directory
            config.recordings_path.mkdir(parents=True, exist_ok=True)
            # No literal tilde directory should exist anywhere
            assert not (tmp_path / "~").exists()
            # But the recordings path should exist
            assert config.recordings_path.exists()
        finally:
            if original_home:
                monkeypatch.setenv("HOME", original_home)


class TestPathValidationEdgeCases:
    """Tests for edge cases and boundary conditions in path handling."""

    def test_absolute_path_string_remains_absolute(self) -> None:
        """Absolute paths remain absolute after processing."""
        # Use platform-appropriate absolute paths

        if sys.platform == "win32":
            abs_rec = r"C:\recordings"
            abs_md = r"C:\markdown"
        else:
            abs_rec = "/absolute/path/recordings"
            abs_md = "/absolute/path/markdown"
        config = Config(
            recordings_path=abs_rec,
            markdown_path=abs_md,
        )
        assert config.recordings_path.is_absolute()
        assert config.markdown_path.is_absolute()

    def test_relative_path_remains_relative(self) -> None:
        """Relative paths remain relative after processing."""
        config = Config(
            recordings_path="relative/path/recordings",
            markdown_path="relative/path/markdown",
        )
        assert config.recordings_path == Path("relative/path/recordings")
        assert config.markdown_path == Path("relative/path/markdown")

    def test_path_with_parent_directories_expand(self) -> None:
        """Paths with .. are preserved."""
        config = Config(
            recordings_path="../recordings",
            markdown_path="../markdown",
        )
        assert ".." in str(config.recordings_path)
        assert ".." in str(config.markdown_path)

    def test_path_with_dots_in_directory_names(self) -> None:
        """Paths with dots in directory names are preserved."""
        config = Config(
            recordings_path="my.data/recordings",
            markdown_path="my.data/markdown",
        )
        assert "my.data" in str(config.recordings_path)
        assert "my.data" in str(config.markdown_path)

    def test_path_with_spaces(self) -> None:
        """Paths with spaces are preserved."""
        config = Config(
            recordings_path="My Documents/voicepad/recordings",
            markdown_path="My Documents/voicepad/markdown",
        )
        assert "My Documents" in str(config.recordings_path)
        assert "My Documents" in str(config.markdown_path)

    def test_path_as_path_object_input(self) -> None:
        """Path objects passed to config are handled correctly."""
        rec_path = Path("data/recordings")
        md_path = Path("data/markdown")
        config = Config(recordings_path=rec_path, markdown_path=md_path)
        assert config.recordings_path == rec_path
        assert config.markdown_path == md_path
