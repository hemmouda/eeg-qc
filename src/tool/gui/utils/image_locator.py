import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...qc.struct.recording import RecordingQuality

_RES_DIR = Path(__file__).resolve().parent.parent / "res"


def get_image_path(image_file_name: str) -> str:
    """Takes file name without the extension."""

    image_path = _RES_DIR / f"{image_file_name}.png"
    assert image_path.is_file(), f"Image not found: {image_path}"
    return str(image_path.resolve())


def get_traffic_light_image_path_for(recording_quality: "RecordingQuality") -> str:
    """Returns one of the 3 images of the quality at random."""

    color = recording_quality.name.lower()
    index = random.randint(1, 3)
    return get_image_path(f"{color}_{index}")
