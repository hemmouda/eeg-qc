import traceback
from typing import Callable, Optional
from pathlib import Path

from .struct.config import Config
from .struct.model_config import ModelsConfig
from .struct.recording import Recording, RecordingQuality

from .module.edf_file_reader import read_edf_file
from .module.epoch_cutter import prepare_epochs
from .module.features_extractor import extract_features
from .module.artifact_predictor import predict
from .module.quality_engine import determine_quality as quality_control_engine

from .consts import MODELS_CONFIG


def determine_quality(
    hst_edf_file: Path | str,
    config: Config,
    models_config: Optional[ModelsConfig] = None,
) -> Recording:
    """Determines the quality of a single HST file.

    If no models_config is passed, the default one defined in consts is used."""

    if models_config is None:
        models_config = MODELS_CONFIG

    recording = Recording()

    try:

        # Parse if not Path object already
        hst_edf_file = Path(hst_edf_file)

        # Make sure file exists
        if not hst_edf_file.is_file():
            recording.quality = RecordingQuality.BLACK
            recording.quality_justification = f"File does not exist `{hst_edf_file}`."
            return recording

        recording.file_path = hst_edf_file

        # Start by reading and checking the EDF file and its structure
        if not read_edf_file(recording, config):
            return recording

        # Then prepare the epochs
        prepare_epochs(recording, models_config)

        # Then extract the features
        extract_features(recording)

        # Then predict the artifacts
        predict(recording, models_config)

        # And, finally, determine the quality
        quality_control_engine(recording, config, models_config)

    except Exception:
        # No exception should escape normally. If it does we share the stack to have something to work from
        recording.quality = RecordingQuality.BLACK
        recording.quality_justification = (
            f"Something went wrong: {traceback.format_exc()}"
        )

    return recording


def determine_qualities(
    hst_edf_files: list[Path | str],
    config: Config,
    *,
    progress_callback: Optional[Callable[[Recording], None]] = None,
    models_config: Optional[ModelsConfig] = None,
) -> list[Recording]:
    """The main entry point for the QC API. Determines the quality of the given HST files.

    Args:
        progress_callback: Called with each Recording once its quality has been
            determined. Calls it on the same thread; so it better not crash or hold
            up the show.
        models_config: If not passed, the default one defined in consts is used.

    Returns:
        List of evaluated Recordings.
    """

    recordings = []

    for file in hst_edf_files:
        recording = determine_quality(file, config, models_config)
        recordings.append(recording)

        if progress_callback is not None:
            progress_callback(recording)

    return recordings
