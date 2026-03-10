"""Tests for package entrypoints and CLI exports."""

from __future__ import annotations

import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voicepad-core" / "src"))

from voicepad import main as package_main
from voicepad.cli import __all__ as cli_all


class EntrypointTests(unittest.TestCase):
    def test_voicepad_main_calls_typer_app(self) -> None:
        with patch("voicepad.main.app") as mock_app:
            package_main()

        mock_app.assert_called_once_with()

    def test_voicepad_dunder_main_invokes_main(self) -> None:
        with patch("voicepad.main.main") as mock_main:
            runpy.run_module("voicepad.__main__", run_name="__main__")

        mock_main.assert_called_once_with()

    def test_cli_dunder_all_exports_expected_names(self) -> None:
        self.assertEqual(cli_all, ["config_app", "record_app"])


if __name__ == "__main__":
    unittest.main()
