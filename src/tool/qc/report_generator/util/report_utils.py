from pathlib import Path
from datetime import datetime


def get_unique_dated_filename(dir_location: Path, file_name: str) -> Path:
    """
    Prefixes file_name with the current datetime (YYYY_MM_DD_HH_MM_SS)
    and ensures the result doesn't already exist in dir_location,
    appending an incrementing number if needed. Creates the file.

    Args:
        dir_location: Directory to check for existing files.
        file_name: Original file name with extension.
    Returns:
        Full path to a unique, timestamp-prefixed file name.
    """

    dir_location = Path(dir_location)

    original = Path(file_name)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    candidate = dir_location / f"{timestamp}_{original.stem}{original.suffix}"

    counter = 1
    while candidate.exists():
        candidate = (
            dir_location / f"{timestamp}_{original.stem}_{counter}{original.suffix}"
        )
        counter += 1

    candidate.touch()
    return candidate
