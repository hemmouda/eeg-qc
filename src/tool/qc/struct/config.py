from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rule import Rule


class Config:

    config_file: Path
    """File path of the config file from which the values were extracted."""

    min_duration_h: int | float
    """Shortest a file can be"""
    max_duration_h: int | float
    """Longest a file can be"""

    channels_rules: dict[str, list["Rule"]]
    """Maps a channel name to the list of rules it has."""
