import mne

from ..struct.config import Config
from ..struct.recording import Recording, RecordingQuality, Remark, RemarkType
from ..struct.channel import Channel
from ..consts import NEEDED_CHANNELS, COMMON_FS

# region Consts and utils

_MNE_MUTED_VERBOSITY_LEVEL = 100  # 20 is info, 50 is critical


_HST_CHANNELS = {
    "Fz",
    "EOGl",
    "EOGr",
    "EMG",
    "AUDIO",
    "EEG",
    "REM",
    "Temp.",
    "EMG1",
    "X",
    "Y",
    "Z",
    "Fp1",
    "Fp2",
    "ADC1Count",
    "VBat",
    "Light",
    "Lead off",
    "Activity",
    "ADC2Count",
    "ADC3Count",
    "EXG2",
    "EMGz",
    "EXG3",
    "Time rem.",
    "EXG",
    "EMG2",
    "RSSI",
    "ActDevice",
    "Hub Accu",
}
"""Set of channels that we expect an HST EDF file to have."""

_EXPECTED_FS = COMMON_FS


def _parse_MNE_sex(value: int | None) -> str:
    """MNE: Subject sex (0=unknown, 1=male, 2=female)"""

    if value is None or value == 0:
        return "UNKNOWN"
    elif value == 1:
        return "Male"
    elif value == 2:
        return "Female"
    else:
        return f"UNKNOWN ({value})"


def _format_duration(total_seconds: int | float) -> str:
    total_seconds = int(total_seconds)
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


# endregion


def read_edf_file(recording: Recording, config: Config) -> bool:
    """Reads and validates an HST EDF recording file and then extracts the needed channels.

    Returns:
        True if the file was successfully read and QC can continue, False otherwise
    """

    # First we read the file just to extract / check basic data
    try:
        raw = mne.io.read_raw_edf(
            recording.file_path, preload=False, verbose=_MNE_MUTED_VERBOSITY_LEVEL
        )
    except Exception as e:
        # We are forced to use MNE because it's tolerant towards faulty EDF file
        # formats compared to pyEDFlib which is strict about the EDF file format.
        # If a file is so corrupt that MNE can't read it then halt.
        recording.quality = RecordingQuality.BLACK
        recording.quality_justification = f"Corrupt EDF file! {e}"
        return False

    ch_names = set(raw.ch_names)

    # Make sure needed channels exist
    NEEDED_CHANNELS_SET = set(NEEDED_CHANNELS)
    if not NEEDED_CHANNELS_SET.issubset(ch_names):
        recording.quality = RecordingQuality.BLACK
        recording.quality_justification = f"The following required channels are missing: {NEEDED_CHANNELS_SET - ch_names}"
        return False

    # Make sure duration is respected
    if raw.duration < config.min_duration_h * 3600.0:
        recording.quality = RecordingQuality.BLACK
        recording.quality_justification = f"Recording is too short! ({raw.duration}s)"
        return False

    if raw.duration > config.max_duration_h * 3600.0:
        recording.quality = RecordingQuality.BLACK
        recording.quality_justification = (
            f"Recording is too long! ({(raw.duration/3600.0):.2f}h)"
        )
        return False

    # Then add remarks if:
    # A channel is missing
    missing_channels = _HST_CHANNELS - ch_names
    if missing_channels:
        recording.remarks.append(
            Remark(RemarkType.MISSING_CHANNELS, str(missing_channels))
        )

    # An unexpected channel exists
    extra_channels = ch_names - _HST_CHANNELS
    if extra_channels:
        recording.remarks.append(Remark(RemarkType.EXTRA_CHANNELS, str(extra_channels)))

    # A different sampling frequency
    if raw.info["sfreq"] != _EXPECTED_FS:
        recording.remarks.append(
            Remark(RemarkType.DIFFERENT_FS, str(raw.info["sfreq"]))
        )
    # F*ck MNE because it automatically resamples all channels to highest
    # frequency when reading. And so frequency for MNE is at the level of recording
    # not channels.

    # Store basic info
    subject_info = raw.info["subject_info"] or {}
    recording.subject_identifier = subject_info.get("his_id", "UNKNOWN")
    recording.last_name = subject_info.get("last_name", "UNKNOWN")
    recording.first_name = subject_info.get("first_name", "UNKNOWN")
    recording.birthday = subject_info.get("birthday", None)
    recording.sex = _parse_MNE_sex(subject_info.get("sex", None))
    recording.recording_date = raw.info["meas_date"]
    recording.read_duration = raw.duration
    recording.formatted_duration = _format_duration(recording.read_duration)
    recording.original_fs = raw.info["sfreq"]

    # Now we re-read with only channels of interest and save
    raw = mne.io.read_raw_edf(
        recording.file_path,
        include=NEEDED_CHANNELS,
        preload=True,
        verbose=_MNE_MUTED_VERBOSITY_LEVEL,
    )

    # And f*ck MNE once again because no where in documentation does it state
    # what order will the data be returned in. So forced to read one by one
    for channel_name in NEEDED_CHANNELS:
        channel = Channel()
        channel.recording = recording
        recording.channels.append(channel)

        channel.name = channel_name
        index = raw.ch_names.index(channel_name)
        channel.signal = raw.get_data(picks=[index], units="uV")[0]

    return True
