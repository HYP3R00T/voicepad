"""Tests for voicepad_core.config.settings."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from voicepad_core.config import Config as ConfigModel
from voicepad_core.config import get_config, get_config_with_metadata


def make_config(*args: Any, **kwargs: Any) -> ConfigModel:
    return ConfigModel(*args, **kwargs)


Config = make_config


class TestProperNouns:
    def test_default_proper_nouns_are_empty(self) -> None:
        """Proper-noun biasing is opt-in."""
        assert Config().proper_nouns == ()

    def test_proper_nouns_preserve_user_spelling_and_order(self) -> None:
        """Configured terms reach inference without rewriting their spelling or order."""
        config = Config(proper_nouns=["VoicePad", "HYP3R00T"])

        assert config.proper_nouns == ("VoicePad", "HYP3R00T")

    def test_blank_proper_noun_is_rejected(self) -> None:
        """Empty vocabulary entries are rejected before backend translation."""
        with pytest.raises(ValueError, match="proper_nouns"):
            Config(proper_nouns=["VoicePad", " "])


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

    def test_default_vad_model_path(self) -> None:
        """When vad_model_path is not provided, the default points to ~/.config/voicepad/models/vad."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert "voicepad" in str(config.vad_model_path)
        assert "vad" in str(config.vad_model_path)

    def test_vad_model_path_is_expanded(self) -> None:
        """The vad_model_path field is validated and expanded."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            vad_model_path="data/vad",
        )
        assert isinstance(config.vad_model_path, Path)
        assert config.vad_model_path == Path("data/vad")

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
        assert isinstance(config, ConfigModel)

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
        assert isinstance(config, ConfigModel)

    def test_get_config_with_metadata_second_element_is_dict(self) -> None:
        """The second element of the tuple is metadata (a dict-like object)."""
        config, metadata = get_config_with_metadata()
        assert isinstance(config, ConfigModel)
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


# ---------------------------------------------------------------------------
# Model validation tests
# ---------------------------------------------------------------------------


class TestTranscriptionModelValidation:
    """Tests for transcription_model field validation."""

    def test_valid_transcription_model_turbo(self) -> None:
        """Valid model 'turbo' is accepted."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            transcription_model="turbo",
        )
        assert config.transcription_model == "turbo"

    def test_valid_transcription_model_base(self) -> None:
        """Valid model 'base' is accepted."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            transcription_model="base",
        )
        assert config.transcription_model == "base"

    def test_valid_transcription_model_large_v3(self) -> None:
        """Valid model 'large-v3' is accepted."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            transcription_model="large-v3",
        )
        assert config.transcription_model == "large-v3"

    def test_invalid_transcription_model_raises_error(self) -> None:
        """Invalid model name raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Unknown transcription model"):
            Config(
                recordings_path="data/recordings",
                markdown_path="data/markdown",
                transcription_model="invalid-model",
            )

    def test_transcription_model_case_sensitive(self) -> None:
        """Model names are case-sensitive."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Config(
                recordings_path="data/recordings",
                markdown_path="data/markdown",
                transcription_model="TURBO",  # Should be lowercase
            )


class TestLogLevelValidation:
    """Tests for log_level field validation."""

    def test_valid_log_level_debug(self) -> None:
        """Valid log level 'DEBUG' is accepted."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            log_level="DEBUG",
        )
        assert config.log_level == "DEBUG"

    def test_valid_log_level_info(self) -> None:
        """Valid log level 'INFO' is accepted."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            log_level="INFO",
        )
        assert config.log_level == "INFO"

    def test_valid_log_level_warning(self) -> None:
        """Valid log level 'WARNING' is accepted."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            log_level="WARNING",
        )
        assert config.log_level == "WARNING"

    def test_valid_log_level_error(self) -> None:
        """Valid log level 'ERROR' is accepted."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            log_level="ERROR",
        )
        assert config.log_level == "ERROR"

    def test_invalid_log_level_raises_error(self) -> None:
        """Invalid log level raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Config(
                recordings_path="data/recordings",
                markdown_path="data/markdown",
                log_level="TRACE",  # type: ignore[arg-type]
            )

    def test_default_log_level_is_info(self) -> None:
        """Default log level is INFO."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert config.log_level == "INFO"


class TestGlobalHotkeyField:
    """Tests for global_hotkey field."""

    def test_default_global_hotkey(self) -> None:
        """Default global hotkey is '<ctrl>+<alt>+v'."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert config.global_hotkey == "<ctrl>+<alt>+v"

    def test_custom_global_hotkey(self) -> None:
        """Custom global hotkey is accepted."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            global_hotkey="<ctrl>+<shift>+space",
        )
        assert config.global_hotkey == "<ctrl>+<shift>+space"

    def test_empty_global_hotkey_disables_feature(self) -> None:
        """Empty string disables global hotkey."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            global_hotkey="",
        )
        assert config.global_hotkey == ""


class TestLanguageField:
    """Tests for language field."""

    def test_default_language_is_english(self) -> None:
        """Default language is 'en'."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert config.language == "en"

    def test_custom_language(self) -> None:
        """Custom language is accepted."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            language="es",
        )
        assert config.language == "es"

    def test_language_accepts_any_string(self) -> None:
        """Language field accepts any string value."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            language="zh-CN",
        )
        assert config.language == "zh-CN"


class TestStreamingParameters:
    """Tests for streaming-related parameters."""

    def test_default_silence_threshold_ms(self) -> None:
        """Default silence threshold is 1000ms."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert config.silence_threshold_ms == 1000

    def test_custom_silence_threshold_ms(self) -> None:
        """Custom silence threshold is accepted."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            silence_threshold_ms=500,
        )
        assert config.silence_threshold_ms == 500

    def test_default_min_chunk_s(self) -> None:
        """Default min_chunk_s is 15.0."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert config.min_chunk_s == 15.0

    def test_custom_min_chunk_s(self) -> None:
        """Custom min_chunk_s is accepted."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            min_chunk_s=10.0,
        )
        assert config.min_chunk_s == 10.0

    def test_default_max_chunk_s(self) -> None:
        """Default max_chunk_s is 29.0."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert config.max_chunk_s == 29.0

    def test_custom_max_chunk_s(self) -> None:
        """Custom max_chunk_s is accepted."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            max_chunk_s=25.0,
        )
        assert config.max_chunk_s == 25.0

    def test_default_overlap_s(self) -> None:
        """Default overlap_s is 0.5."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert config.overlap_s == 0.5

    def test_custom_overlap_s(self) -> None:
        """Custom overlap_s is accepted."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            overlap_s=1.0,
        )
        assert config.overlap_s == 1.0


class TestLocalAgreementFields:
    """Tests for local_agreement_mic and local_agreement_file fields."""

    def test_default_local_agreement_mic_is_false(self) -> None:
        """Default local_agreement_mic is False."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert config.local_agreement_mic is False

    def test_custom_local_agreement_mic_true(self) -> None:
        """local_agreement_mic can be set to True."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            local_agreement_mic=True,
        )
        assert config.local_agreement_mic is True

    def test_default_local_agreement_file_is_true(self) -> None:
        """Default local_agreement_file is True."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert config.local_agreement_file is True

    def test_custom_local_agreement_file_false(self) -> None:
        """local_agreement_file can be set to False."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            local_agreement_file=False,
        )
        assert config.local_agreement_file is False


class TestLogsPathField:
    """Tests for logs_path field."""

    def test_default_logs_path_expands_tilde(self) -> None:
        """Default logs_path expands tilde."""
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        assert "~" not in str(config.logs_path)
        assert config.logs_path.is_absolute()
        assert "voicepad" in str(config.logs_path).lower()

    def test_custom_logs_path(self) -> None:
        """Custom logs_path is accepted."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            logs_path="/tmp/logs",
        )
        assert config.logs_path == Path("/tmp/logs")

    def test_logs_path_with_tilde_expands(self) -> None:
        """logs_path with tilde is expanded."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            logs_path="~/mylogs",
        )
        assert "~" not in str(config.logs_path)
        assert config.logs_path.is_absolute()


class TestAllPathFieldsExpanded:
    """Tests ensuring all path fields are properly expanded."""

    def test_all_path_fields_expanded_in_defaults(self) -> None:
        """All path fields with defaults are expanded."""
        config = Config()
        # All paths should be absolute
        assert config.recordings_path.is_absolute()
        assert config.markdown_path.is_absolute()
        assert config.model_cache_path.is_absolute()
        assert config.logs_path.is_absolute()
        # None should contain literal tilde
        assert "~" not in str(config.recordings_path)
        assert "~" not in str(config.markdown_path)
        assert "~" not in str(config.model_cache_path)
        assert "~" not in str(config.logs_path)

    def test_all_path_fields_with_custom_values(self) -> None:
        """All path fields accept custom values."""
        config = Config(
            recordings_path="/custom/recordings",
            markdown_path="/custom/markdown",
            model_cache_path="/custom/models",
            logs_path="/custom/logs",
        )
        assert config.recordings_path == Path("/custom/recordings")
        assert config.markdown_path == Path("/custom/markdown")
        assert config.model_cache_path == Path("/custom/models")
        assert config.logs_path == Path("/custom/logs")


class TestConfigImmutability:
    """Tests for config immutability (frozen model)."""

    def test_cannot_modify_recordings_path(self) -> None:
        """Cannot modify recordings_path after creation."""
        from pydantic import ValidationError

        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        with pytest.raises(ValidationError):
            config.recordings_path = Path("other")  # type: ignore[misc]

    def test_cannot_modify_transcription_model(self) -> None:
        """Cannot modify transcription_model after creation."""
        from pydantic import ValidationError

        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        with pytest.raises(ValidationError):
            config.transcription_model = "base"  # type: ignore[misc]

    def test_cannot_modify_log_level(self) -> None:
        """Cannot modify log_level after creation."""
        from pydantic import ValidationError

        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        with pytest.raises(ValidationError):
            config.log_level = "DEBUG"  # type: ignore[misc]


class TestGetConfigWithCwd:
    """Tests for get_config with custom cwd parameter."""

    def test_get_config_with_cwd_parameter(self, tmp_path: Path) -> None:
        """get_config accepts cwd parameter."""
        config = get_config(cwd=tmp_path)
        assert isinstance(config, ConfigModel)

    def test_get_config_with_metadata_with_cwd_parameter(self, tmp_path: Path) -> None:
        """get_config_with_metadata accepts cwd parameter."""
        config, metadata = get_config_with_metadata(cwd=tmp_path)
        assert isinstance(config, ConfigModel)
        assert metadata is not None


class TestGetConfigWithAppName:
    """Tests for get_config with custom app_name parameter."""

    def test_get_config_with_app_name_parameter(self) -> None:
        """get_config accepts app_name parameter."""
        config = get_config(app_name="voicepad")
        assert isinstance(config, ConfigModel)

    def test_get_config_with_metadata_with_app_name_parameter(self) -> None:
        """get_config_with_metadata accepts app_name parameter."""
        config, metadata = get_config_with_metadata(app_name="voicepad")
        assert isinstance(config, ConfigModel)
        assert metadata is not None


class TestComputeTypeValidation:
    """Tests for transcription_compute_type validation."""

    def test_valid_compute_type_auto(self) -> None:
        """Valid compute type 'auto' is accepted."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            transcription_compute_type="auto",
        )
        assert config.transcription_compute_type == "auto"

    def test_valid_compute_type_float16(self) -> None:
        """Valid compute type 'float16' is accepted."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            transcription_compute_type="float16",
        )
        assert config.transcription_compute_type == "float16"

    def test_valid_compute_type_int8(self) -> None:
        """Valid compute type 'int8' is accepted."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            transcription_compute_type="int8",
        )
        assert config.transcription_compute_type == "int8"

    def test_valid_compute_type_float32(self) -> None:
        """Valid compute type 'float32' is accepted."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            transcription_compute_type="float32",
        )
        assert config.transcription_compute_type == "float32"

    def test_valid_compute_type_int8_float16(self) -> None:
        """Valid compute type 'int8_float16' is accepted."""
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            transcription_compute_type="int8_float16",
        )
        assert config.transcription_compute_type == "int8_float16"


class TestAdvancedRuntimeParameters:
    """Tests for newly configurable runtime tuning parameters."""

    def test_default_runtime_tuning_fields(self) -> None:
        config = Config(recordings_path="data/recordings", markdown_path="data/markdown")

        assert config.initial_prompt
        assert config.no_speech_threshold == 0.6
        assert config.hallucination_silence_threshold == 2.0
        assert config.hallucination_max_repetitions == 3
        assert config.stream_poll_interval_s == 0.3
        assert config.stream_context_chars == 200
        assert config.vad_threshold == 0.5
        assert config.vad_min_speech_duration_ms == 250
        assert config.vad_speech_pad_ms == 30
        assert config.vad_model_filename == "silero_vad_v6.onnx"
        assert config.vad_download_chunk_size == 8192
        assert config.model_warmup_enabled is True
        assert config.model_warmup_duration_s == 0.5

    def test_custom_runtime_tuning_fields(self) -> None:
        config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            initial_prompt="Test prompt",
            no_speech_threshold=0.4,
            hallucination_silence_threshold=1.2,
            hallucination_max_repetitions=2,
            min_audio_duration_s=0.75,
            trim_trailing_silence_rms_threshold=0.02,
            trim_trailing_silence_frame_ms=40,
            stream_poll_interval_s=0.1,
            stream_context_chars=120,
            dedup_prev_tail_words=25,
            dedup_full_duplicate_threshold=0.9,
            dedup_min_overlap_words_for_partial=4,
            dedup_partial_lead_words=6,
            vad_threshold=0.6,
            vad_min_speech_duration_ms=300,
            vad_speech_pad_ms=45,
            vad_model_filename="custom_vad.onnx",
            vad_model_url="https://example.com/custom_vad.onnx",
            vad_download_chunk_size=4096,
            model_warmup_enabled=False,
            model_warmup_duration_s=0.25,
            model_warmup_language="fr",
            model_warmup_beam_size=2,
            model_warmup_vad_filter=True,
        )

        assert config.initial_prompt == "Test prompt"
        assert config.no_speech_threshold == 0.4
        assert config.hallucination_silence_threshold == 1.2
        assert config.hallucination_max_repetitions == 2
        assert config.min_audio_duration_s == 0.75
        assert config.trim_trailing_silence_rms_threshold == 0.02
        assert config.trim_trailing_silence_frame_ms == 40
        assert config.stream_poll_interval_s == 0.1
        assert config.stream_context_chars == 120
        assert config.dedup_prev_tail_words == 25
        assert config.dedup_full_duplicate_threshold == 0.9
        assert config.dedup_min_overlap_words_for_partial == 4
        assert config.dedup_partial_lead_words == 6
        assert config.vad_threshold == 0.6
        assert config.vad_min_speech_duration_ms == 300
        assert config.vad_speech_pad_ms == 45
        assert config.vad_model_filename == "custom_vad.onnx"
        assert config.vad_model_url == "https://example.com/custom_vad.onnx"
        assert config.vad_download_chunk_size == 4096
        assert config.model_warmup_enabled is False
        assert config.model_warmup_duration_s == 0.25
        assert config.model_warmup_language == "fr"
        assert config.model_warmup_beam_size == 2
        assert config.model_warmup_vad_filter is True
