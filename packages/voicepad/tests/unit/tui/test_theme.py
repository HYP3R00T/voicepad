"""Tests for theme.py module."""

from __future__ import annotations

from voicepad.tui.theme import MD_PLACEHOLDER, get_available_themes


class TestGetAvailableThemes:
    """Test get_available_themes function."""

    def test_returns_list(self) -> None:
        themes = get_available_themes()
        assert isinstance(themes, list)

    def test_list_is_not_empty(self) -> None:
        assert len(get_available_themes()) > 0

    def test_all_entries_are_strings(self) -> None:
        for theme in get_available_themes():
            assert isinstance(theme, str)

    def test_tokyo_night_is_included(self) -> None:
        assert "tokyo-night" in get_available_themes()

    def test_catppuccin_mocha_is_included(self) -> None:
        assert "catppuccin-mocha" in get_available_themes()

    def test_list_is_sorted_alphabetically(self) -> None:
        themes = get_available_themes()
        assert themes == sorted(themes)

    def test_no_duplicates(self) -> None:
        themes = get_available_themes()
        assert len(themes) == len(set(themes))


class TestMdPlaceholder:
    """Test MD_PLACEHOLDER constant."""

    def test_md_placeholder_is_string(self) -> None:
        assert isinstance(MD_PLACEHOLDER, str)

    def test_md_placeholder_contains_voicepad_header(self) -> None:
        assert "# voicepad" in MD_PLACEHOLDER

    def test_md_placeholder_contains_instructions(self) -> None:
        assert "Select a recording" in MD_PLACEHOLDER
