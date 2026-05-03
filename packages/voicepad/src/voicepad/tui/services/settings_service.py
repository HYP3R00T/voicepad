"""Settings service for VoicePad TUI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from utilityhub_config import get_config_path, write_config

if TYPE_CHECKING:
    from voicepad_core.config import Config

logger = logging.getLogger(__name__)


class SettingsService:
    """Service for managing application settings."""

    def __init__(self, config: Config) -> None:
        """Initialize the settings service.

        Args:
            config: Application configuration
        """
        self.config = config

    def get_config_path(self) -> Path:
        """Get the path to the global config file.

        Returns:
            Path to the config file
        """
        return get_config_path("voicepad", format="yaml")

    def config_exists(self) -> bool:
        """Check if a config file exists.

        Returns:
            True if config file exists, False otherwise
        """
        return self.get_config_path().exists()

    def save_config(self, config: Config) -> None:
        """Save configuration to disk.

        Args:
            config: The configuration to save

        Raises:
            Exception: If saving fails
        """
        try:
            config_path = self.get_config_path()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            write_config(config, "voicepad", path=config_path, format="yaml")
            logger.info(f"Config saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise

    def update_field(self, field_name: str, value: Any) -> Config:
        """Update a single configuration field.

        Args:
            field_name: Name of the field to update
            value: New value for the field

        Returns:
            Updated Config object

        Raises:
            Exception: If update or save fails
        """
        from voicepad_core.config import Config as _Config

        # Get current config as dict
        raw = self.config.model_dump(mode="json")

        # Update the field
        raw[field_name] = value

        # Create new config and save
        try:
            new_config = _Config(**raw)
            self.save_config(new_config)
            return new_config
        except Exception as e:
            logger.error(f"Failed to update field {field_name}: {e}")
            raise

    def update_fields(self, updates: dict[str, Any]) -> Config:
        """Update multiple configuration fields.

        Args:
            updates: Dictionary of field names to new values

        Returns:
            Updated Config object

        Raises:
            Exception: If update or save fails
        """
        from voicepad_core.config import Config as _Config

        # Get current config as dict
        raw = self.config.model_dump(mode="json")

        # Update all fields
        for field_name, value in updates.items():
            raw[field_name] = value

        # Create new config and save
        try:
            new_config = _Config(**raw)
            self.save_config(new_config)
            logger.info(f"Updated {len(updates)} config fields")
            return new_config
        except Exception as e:
            logger.error(f"Failed to update fields: {e}")
            raise

    def validate_field(self, field_name: str, value: Any) -> tuple[bool, str | None]:
        """Validate a field value without saving.

        Args:
            field_name: Name of the field to validate
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        from voicepad_core.config import Config as _Config

        # Get current config as dict
        raw = self.config.model_dump(mode="json")

        # Update the field
        raw[field_name] = value

        # Try to create config to validate
        try:
            _Config(**raw)
            return True, None
        except Exception as e:
            return False, str(e)
