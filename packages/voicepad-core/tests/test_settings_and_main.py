"""Tests for configuration helpers and core CLI entrypoint."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voicepad_core.config.settings import Config, get_config, get_config_with_metadata
from voicepad_core.main import main


class SettingsAndMainTests(unittest.TestCase):
    def test_expand_paths_handles_str_input(self) -> None:
        with patch("voicepad_core.config.settings.expand_path", return_value=Path("expanded/path")) as mock_expand:
            result = Config.expand_paths("~/recordings")

        self.assertEqual(result, Path("expanded/path"))
        mock_expand.assert_called_once_with("~/recordings")

    def test_expand_paths_handles_path_input(self) -> None:
        with patch("voicepad_core.config.settings.expand_path", return_value=Path("expanded/path")) as mock_expand:
            result = Config.expand_paths(Path("~/markdown"))

        self.assertEqual(result, Path("expanded/path"))
        mock_expand.assert_called_once_with(str(Path("~/markdown")))

    def test_get_config_returns_loaded_settings(self) -> None:
        expected_settings = MagicMock(spec=Config)

        with patch(
            "voicepad_core.config.settings.load_settings", return_value=(expected_settings, {"source": "test"})
        ) as mock_load:
            result = get_config(cwd=Path("."), app_name="voicepad")

        self.assertIs(result, expected_settings)
        mock_load.assert_called_once_with(Config, app_name="voicepad", cwd=Path("."))

    def test_get_config_with_metadata_returns_tuple(self) -> None:
        expected_settings = MagicMock(spec=Config)
        expected_metadata = {"source": "test"}

        with patch(
            "voicepad_core.config.settings.load_settings",
            return_value=(expected_settings, expected_metadata),
        ) as mock_load:
            result_settings, result_metadata = get_config_with_metadata(cwd=Path("."), app_name="voicepad")

        self.assertIs(result_settings, expected_settings)
        self.assertEqual(result_metadata, expected_metadata)
        mock_load.assert_called_once_with(Config, app_name="voicepad", cwd=Path("."))

    def test_main_prints_gpu_diagnostics_report(self) -> None:
        report = MagicMock()
        report.model_dump_json.return_value = '{"ok": true}'

        with (
            patch("voicepad_core.main.gpu_diagnostics", return_value=report) as mock_diag,
            patch("builtins.print") as mock_print,
        ):
            main()

        mock_diag.assert_called_once_with()
        report.model_dump_json.assert_called_once_with(indent=2)
        mock_print.assert_called_once_with('{"ok": true}')


if __name__ == "__main__":
    unittest.main()
