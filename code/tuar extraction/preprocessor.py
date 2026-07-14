import pyedflib
import numpy as np
from scipy.signal import resample_poly, sosfiltfilt, butter, iirnotch, filtfilt
from structure import Recording, Channel, Epoch

EXPECTED_CHANNELS = {
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "C3-CZ",
    "C4-T4",
    "FP1-F3",
    "C3-P3",
    "FP2-F4",
    "C4-P4",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "A1-T3",
    "T3-C3",
    "CZ-C4",
    "T4-A2",
    "F3-C3",
    "P3-O1",
    "F4-C4",
    "P4-O2",
}

# "Configuration"

RESAMPLING_FREQ: float = 256
DO_PREPROCESS: bool = True
EPOCH_DURATION_S: float = 10
# EPOCH_DURATION_S: float = 9
# EPOCH_DURATION_S: float = 8
# EPOCH_DURATION_S: float = 7
# EPOCH_DURATION_S: float = 6
# EPOCH_DURATION_S: float = 5
# EPOCH_DURATION_S: float = 4
# EPOCH_DURATION_S: float = 3
# EPOCH_DURATION_S: float = 2
# EPOCH_DURATION_S: float = 1

# I find it acceptable to put this here since
# it's "only needed" here and it's "not" modified


def butterworth_filter(
    signal: np.ndarray,
    fs: float,
    high_pass: float = 0.1,
    low_pass: float = 124.9,  # Was 127.9 | Making it 124.9 because some recordings are 250 Hz
    order: int = 5,
) -> np.ndarray:
    """
    Returns the filtered version of the given signal.
    """

    sos = butter(
        order,
        [high_pass, low_pass],
        btype="band",
        fs=fs,
        output="sos",
    )
    signal = sosfiltfilt(sos, signal)
    return signal


def notch_filter(
    signal: np.ndarray, fs: float, undesired_f: float = 60, quality: float = 15
) -> np.ndarray:
    """
    Returns the filtered version of the given signal.
    """

    b, a = iirnotch(undesired_f, quality, fs)
    signal = filtfilt(b, a, signal)
    return signal


def resample(
    signal: np.ndarray, fs: float, desired_fs: float = RESAMPLING_FREQ
) -> np.ndarray:
    if desired_fs == fs:
        return signal
    signal = resample_poly(signal, desired_fs, fs)
    return signal


def cut_epochs(
    signal: np.ndarray, fs: float, EPOCH_LENGTH_S: float = EPOCH_DURATION_S
) -> tuple[int, int, np.ndarray]:
    """
    Cuts from middle when epochs dont perfectly divide the signal.
    Returns the indices in the sample where the cutting started and ended (inclusive) and the cut-up signal
    """

    n_samples = len(signal)
    epoch_length = int(fs * EPOCH_LENGTH_S)

    n_epochs = n_samples // epoch_length
    assert n_epochs > 0, f"Signal too short!"
    usable = n_epochs * epoch_length

    start = (n_samples - usable) // 2
    end = start + usable

    signal = signal[start:end]
    return (start, (end - 1), signal.reshape(n_epochs, epoch_length))


def preprocess(recording: Recording) -> None:
    """
    Reads the channels, preprocesses them, resamples, and then segments the epochs.
    """

    # Read the file
    with pyedflib.EdfReader(str(recording.edf_file_path)) as f:

        # Make sure we have the expected channels
        labels = f.getSignalLabels()
        assert set(labels).issubset(
            EXPECTED_CHANNELS
        ), f"Channels don't match {labels} in {recording.edf_file_path}"

        # Go over the signals (channels)
        for label in labels:
            index = labels.index(label)

            # Make sure it's in uV
            assert (
                f.getPhysicalDimension(index) == "uV"
            ), f"Channel isn't in the expected microVolts {label} in {recording.edf_file_path}"
            # Yes, one can simply turn into uV, but in TUAR we know it's uV in all.

            # Store the signal and its basic info
            channel = Channel(recording, label)  # Adds itself to recording
            channel.signal = f.readSignal(index)  # Reads in physical dimension
            channel.fs = f.getSampleFrequency(index)
            header = f.getSignalHeader(index)
            channel.physical_min = header["physical_min"]
            channel.physical_max = header["physical_max"]

            # Preprocess
            if DO_PREPROCESS:
                channel.signal = butterworth_filter(channel.signal, channel.fs)
                channel.signal = notch_filter(channel.signal, channel.fs)

            channel.signal = resample(channel.signal, channel.fs)

            # Update info after resampling
            channel.fs = RESAMPLING_FREQ
            channel.n_samples = len(channel.signal)
            channel.duration_s = channel.n_samples / channel.fs

            # Segment and free signal
            start_index, end_index, signals = cut_epochs(channel.signal, channel.fs)
            channel.signal = None

            # Store epoch info
            for signal in signals:
                epoch = Epoch(channel)  # Adds itself to channel
                epoch.signal = signal
                epoch.n_samples = len(epoch.signal)
                epoch.duration_s = epoch.n_samples / channel.fs
                assert epoch.duration_s == EPOCH_DURATION_S, f"??? UNREACHABLE"
                epoch.start_sample_index = start_index
                epoch.start_time = epoch.start_sample_index / channel.fs
                epoch.end_sample_index = epoch.start_sample_index + epoch.n_samples - 1
                epoch.end_time = epoch.start_time + epoch.duration_s

                start_index += epoch.n_samples  # For next iter

            assert start_index - 1 == end_index, f"??? UNREACHABLE"
