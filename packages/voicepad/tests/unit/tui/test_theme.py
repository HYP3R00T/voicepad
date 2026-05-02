"""Tests for theme.py module."""

from __future__ import annotations

from textual.theme import Theme
from voicepad.tui.theme import CATPPUCCIN_MOCHA_BLUE, MD_PLACEHOLDER, THEME_NAME


class TestThemeConstants:
    """Test theme constant values."""

    def test_theme_name_is_string(self) -> None:
        assert isinstance(THEME_NAME, str)
        assert THEME_NAME == "catppuccin-mocha-blue"

    def test_catppuccin_mocha_blue_is_theme(self) -> None:
        assert isinstance(CATPPUCCIN_MOCHA_BLUE, Theme)

    def test_theme_has_correct_name(self) -> None:
        assert CATPPUCCIN_MOCHA_BLUE.name == THEME_NAME

    def test_theme_has_primary_color(self) -> None:
        assert CATPPUCCIN_MOCHA_BLUE.primary == "#89b4fa"

    def test_theme_has_secondary_color(self) -> None:
        assert CATPPUCCIN_MOCHA_BLUE.secondary == "#74c7ec"

    def test_theme_has_warning_color(self) -> None:
        assert CATPPUCCIN_MOCHA_BLUE.warning == "#FAE3B0"

    def test_theme_has_error_color(self) -> None:
        assert CATPPUCCIN_MOCHA_BLUE.error == "#F28FAD"

    def test_theme_has_success_color(self) -> None:
        assert CATPPUCCIN_MOCHA_BLUE.success == "#ABE9B3"

    def test_theme_has_accent_color(self) -> None:
        assert CATPPUCCIN_MOCHA_BLUE.accent == "#fab387"

    def test_theme_has_foreground_color(self) -> None:
        assert CATPPUCCIN_MOCHA_BLUE.foreground == "#cdd6f4"

    def test_theme_has_background_color(self) -> None:
        assert CATPPUCCIN_MOCHA_BLUE.background == "#181825"

    def test_theme_has_surface_color(self) -> None:
        assert CATPPUCCIN_MOCHA_BLUE.surface == "#313244"

    def test_theme_has_panel_color(self) -> None:
        assert CATPPUCCIN_MOCHA_BLUE.panel == "#45475a"

    def test_theme_has_variables(self) -> None:
        assert isinstance(CATPPUCCIN_MOCHA_BLUE.variables, dict)
        assert len(CATPPUCCIN_MOCHA_BLUE.variables) > 0

    def test_theme_variables_has_input_cursor_foreground(self) -> None:
        assert CATPPUCCIN_MOCHA_BLUE.variables["input-cursor-foreground"] == "#11111b"

    def test_theme_variables_has_input_cursor_background(self) -> None:
        assert CATPPUCCIN_MOCHA_BLUE.variables["input-cursor-background"] == "#f5e0dc"

    def test_theme_variables_has_border(self) -> None:
        assert CATPPUCCIN_MOCHA_BLUE.variables["border"] == "#89b4fa"

    def test_theme_variables_has_border_blurred(self) -> None:
        assert CATPPUCCIN_MOCHA_BLUE.variables["border-blurred"] == "#585b70"

    def test_theme_variables_has_footer_background(self) -> None:
        assert CATPPUCCIN_MOCHA_BLUE.variables["footer-background"] == "#45475a"

    def test_theme_variables_has_button_color_foreground(self) -> None:
        assert CATPPUCCIN_MOCHA_BLUE.variables["button-color-foreground"] == "#181825"

    def test_md_placeholder_is_string(self) -> None:
        assert isinstance(MD_PLACEHOLDER, str)

    def test_md_placeholder_contains_voicepad_header(self) -> None:
        assert "# voicepad" in MD_PLACEHOLDER

    def test_md_placeholder_contains_instructions(self) -> None:
        assert "Select a recording" in MD_PLACEHOLDER
