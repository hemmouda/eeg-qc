from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent / "config.yaml"


def get_default_config_file_path() -> Path:
    return _PATH
