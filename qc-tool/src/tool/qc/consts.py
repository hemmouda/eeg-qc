from pathlib import Path
from .struct.model_config import ModelsConfig, ModelConfig

NEEDED_CHANNELS = [
    "EEG",
    "EOGl",
    "EOGr",
    "EMG",
]
"""Channels that need to exist for the study."""

CRITICAL_CHANNELS = {"EEG"}
"""Channels that are critical for the study.
The quality of a recording cannot be higher than the lowest quality of a critical channel.
See more in struct.recording.RecordingQuality."""

assert CRITICAL_CHANNELS.issubset(
    set(NEEDED_CHANNELS)
), f"Critical channels should be from the needed channels"

SPECTROGRAM_CHANNELS = CRITICAL_CHANNELS
"""Channels for which we want spectrogram data"""

assert SPECTROGRAM_CHANNELS.issubset(
    set(NEEDED_CHANNELS)
), f"Channels for which spectrogram data is needed should be from the needed channels"

COMMON_FS = 256.0
"""The expected sampling frequency of HST recordings / the one we resample to if different."""

PHYSICAL_MIN = -800
"""Min physical value that the channels can reach according to HST documentation."""
PHYSICAL_MAX = 800
"""Max physical value that the channels can reach according to HST documentation."""

REPORT_EPOCHS_AGGREGATED = 3
"""How many epochs should be regrouped. With epoch_duration_s == 10 this is 30 second segments.
Don't go below 3 for the HTML report to work though."""

# region Default models config

MODELS_CONFIG = ModelsConfig()
MODELS_CONFIG.epoch_duration_s = 10

_MODELS_DIR = Path(__file__).resolve().parent / "res" / "model"

# Model trained on train + val + test
_ARTF_MODEL = ModelConfig(MODELS_CONFIG, _MODELS_DIR / "dt_artf.joblib")
_ARTF_MODEL.model_name = "dt_artf"
_ARTF_MODEL.artifact_name = "ARTF"
_ARTF_MODEL.target_channels = {"EEG"}
_ARTF_MODEL.features_names = [
    "mad_diff_uv_z",
    "ptp_uv_z",
    "power_0_5_z",
    "flatline_max_s",
    "max_diff_uv_z",
]

# Model just for testing. Not production valid
_MUSC_MODEL = ModelConfig(MODELS_CONFIG, _MODELS_DIR / "dt_musc.joblib")
_MUSC_MODEL.model_name = "dt_musc"
_MUSC_MODEL.artifact_name = "MUSC"
_MUSC_MODEL.target_channels = {"EEG"}
_MUSC_MODEL.features_names = [
    "flatline_max_s",
    "mad_diff_uv_z",
    "ptp_uv_z",
    "power_10_15_z",
    "power_0_5_z",
]

# endregion
