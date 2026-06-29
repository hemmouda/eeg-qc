import numpy as np
from scipy.signal import resample_poly, spectrogram

from ..struct.recording import Recording
from ..struct.epoch import Epoch
from ..struct.model_config import ModelsConfig
from ..consts import COMMON_FS, SPECTROGRAM_CHANNELS

# region Utilities


def _resample(signal: np.ndarray, signal_fs: float, desired_fs: float) -> np.ndarray:
    """Resamples the signal to the desired sampling frequency."""
    if desired_fs == signal_fs:
        return signal

    return resample_poly(signal, desired_fs, signal_fs)


def _cut_epochs(
    signal: np.ndarray, fs: float, epoch_duration_s: float
) -> tuple[int, int, np.ndarray]:
    """
    Cuts the signal into epoch of duration epoch_duration_s.
    Cuts are centered around the middle sample of the signal when epochs dont perfectly divide the signal.

    Returns the indices in the sample where the cutting started and ended (inclusive) and the cut-up signal.
    """

    n_samples = len(signal)
    epoch_length = int(fs * epoch_duration_s)  # Length sample count wise

    n_epochs = n_samples // epoch_length
    assert n_epochs > 0, f"Signal too short!"
    usable = n_epochs * epoch_length

    # To cut from the center
    start = (n_samples - usable) // 2
    end = start + usable

    signal = signal[start:end]
    return (start, (end - 1), signal.reshape(n_epochs, epoch_length))


# endregion


def prepare_epochs(recording: Recording, models_config: ModelsConfig) -> None:
    """Prepares the epochs for feature extraction.

    Resamples the signal and extract spectrogram data if needed. And then cuts
    the epochs"""

    for channel in recording.channels:

        # First, resample signal if needed
        channel.signal = _resample(
            channel.signal, channel.recording.original_fs, COMMON_FS
        )

        # Set values
        channel.fs = COMMON_FS
        channel.n_samples = len(channel.signal)
        channel.duration_s = channel.n_samples / channel.fs

        # Get spectrogram data if wanted
        if channel.name in SPECTROGRAM_CHANNELS:
            channel.spectrogram_data = spectrogram(
                channel.signal, channel.fs, nperseg=256, noverlap=128
            )
        else:
            channel.spectrogram_data = None

        # Cut and free signal
        start_index, end_index, signals = _cut_epochs(
            channel.signal, channel.fs, models_config.epoch_duration_s
        )
        del channel.signal

        # Then save epochs
        for signal in signals:
            epoch = Epoch()
            epoch.channel = channel
            channel.epochs.append(epoch)
            epoch.order = len(channel.epochs)

            epoch.signal = signal
            epoch.n_samples = len(epoch.signal)
            epoch.duration_s = epoch.n_samples / channel.fs
            assert (
                epoch.duration_s == models_config.epoch_duration_s
            ), f"UNREACHABLE"  # Maybe reachable if fs is not a whole number

            epoch.first_sample_index = start_index
            epoch.start_time = epoch.first_sample_index / channel.fs

            epoch.last_sample_index = epoch.first_sample_index + epoch.n_samples - 1
            epoch.end_time = epoch.start_time + epoch.duration_s

            start_index += epoch.n_samples  # For next iter

        assert start_index - 1 == end_index, f"UNREACHABLE"
