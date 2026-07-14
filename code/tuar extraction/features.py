from collections import defaultdict
import numpy as np
from structure import Recording, Epoch
from scipy.signal import welch

_NP_EPS = np.finfo(float).eps


# region Methods to compute the features
def flatline_max_s(signal: np.ndarray, fs: float, eps: float = 0.2) -> float:
    """
    Longest duration where consecutive samples satisfy:
        |Δx| < eps

    Parameters
    ----------
    signal : np.ndarray
        1D signal array
    fs : float
        Sampling frequency in Hz
    eps : float
        Threshold for |Δx|

    Returns
    -------
    float
        Longest flatline duration in seconds
    """

    # Consecutive absolute differences
    dx = np.abs(np.diff(signal))

    # True where "flat"
    flat = dx < eps

    max_run = 0
    current_run = 0

    for v in flat:
        if v:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    # Run length in samples -> seconds
    # Plus 1 because N diffs correspond to N+1 samples
    max_samples = max_run + 1 if max_run > 0 else 0

    return max_samples / fs


def clip_frac(
    signal: np.ndarray, physical_min: float, physical_max: float, eps: float = 1.0
) -> float:
    """
    Fraction of samples near saturation limits.

    A sample is considered clipped if:
        x <= physical_min + eps
    or
        x >= physical_max - eps

    Parameters
    ----------
    signal : np.ndarray
        1D signal array
    physical_min : float
        Minimum physical but preferably ADC limit
    physical_max : float
        Maximum physical but preferably ADC limit
    eps : float
        Tolerance margin near limits in same unit as signal

    Returns
    -------
    float
        Fraction of clipped samples in [0, 1]
    """

    clipped = (signal <= physical_min + eps) | (signal >= physical_max - eps)

    return np.mean(clipped)


def max_diff_uv(signal: np.ndarray) -> float:
    """
    Largest absolute sample-to-sample difference:
        max_t |x[t] - x[t-1]|

    Parameters
    ----------
    signal : np.ndarray
        1D signal array

    Returns
    -------
    float
        Maximum absolute first difference
    """

    return np.max(np.abs(np.diff(signal)))


def mad_diff_uv(signal: np.ndarray) -> float:
    """
    MAD (median absolute deviation) of the differences.

    Returns
    -------
    float
        MAD of difference-ed signal
    """

    x = np.abs(np.diff(signal))

    med = np.median(x)
    mad = np.median(np.abs(x - med))

    return mad


def rms_uv(signal: np.ndarray) -> float:
    """
    Root mean square amplitude of a signal.

    Parameters
    ----------
    signal : np.ndarray
        1D signal array

    Returns
    -------
    float
        RMS amplitude
    """

    return np.sqrt(np.mean(signal**2))


def std_uv(signal: np.ndarray) -> float:
    """
    Standard deviation of a signal.

    Parameters
    ----------
    signal : np.ndarray
        1D signal array

    Returns
    -------
    float
        Standard deviation
    """

    return np.std(signal)


def ptp_uv(signal: np.ndarray) -> float:
    """
    Peak-to-peak amplitude:
        max(x) - min(x)

    Parameters
    ----------
    signal : np.ndarray
        1D signal array

    Returns
    -------
    float
        Peak-to-peak amplitude
    """

    return np.max(signal) - np.min(signal)


def compute_psd(signal: np.ndarray, fs: float):
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), 256))
    return freqs, psd


def sum_power_bin(psd_result, lower_f: float, upper_f: float) -> float:
    """Sums the PSD over the specified frequency bin"""

    assert lower_f < upper_f, f"Come fix this."

    freqs, psd = psd_result

    # Inclusive on both sides should be okay with the ratio features, I think
    mask = (freqs >= lower_f) & (freqs <= upper_f)

    power = np.sum(psd[mask])
    return power


def line_ratio(
    psd_result, line_band=(59, 61), lower_band=(55, 58), upper_band=(62, 65)
) -> float:

    line_power = sum_power_bin(psd_result, line_band[0], line_band[1])
    lower_power = sum_power_bin(psd_result, lower_band[0], lower_band[1])
    upper_power = sum_power_bin(psd_result, upper_band[0], upper_band[1])

    denominator = lower_power + upper_power

    # This happens if signal is constant. In which case numerator will most luckily also be 0
    if denominator == 0:
        denominator = _NP_EPS

    return line_power / denominator


def hf_ratio(psd_result, hf_band=(20, 35), lf_band=(1, 20)) -> float:
    hf_power = sum_power_bin(psd_result, hf_band[0], hf_band[1])
    lf_power = sum_power_bin(psd_result, lf_band[0], lf_band[1])

    if lf_power == 0:
        lf_power = _NP_EPS

    return hf_power / lf_power


def lf_ratio(psd_result, lf_band=(0.3, 1), hf_band=(1, 20)) -> float:
    lf_power = sum_power_bin(psd_result, lf_band[0], lf_band[1])
    hf_power = sum_power_bin(psd_result, hf_band[0], hf_band[1])

    if hf_power == 0:
        hf_power = _NP_EPS

    return lf_power / hf_power


def compute_normalize_stats(xs: list | np.ndarray) -> tuple:
    """
    Computes median and MAD once for normalization.
    """
    xs = np.asarray(xs)

    med = np.median(xs)
    mad = np.median(np.abs(xs - med))
    return (med, mad)


def robust_normalize(value: float, stats: tuple, eps=1e-8) -> float:
    """
    Z_robust = (value - median) / (1.4826 * MAD + eps)

    stats = (med, mad)
    """
    med, mad = stats
    z_robust = (value - med) / ((1.4826 * mad) + eps)
    return z_robust


# endregion


# region Power bins utility methods
def get_power_bins():
    for i in range(Epoch.POWER_BINS_RANGE):
        lower = i * Epoch.POWER_BINS_DELTA
        upper = (i + 1) * Epoch.POWER_BINS_DELTA
        name = f"power_{lower}_{upper}"
        z_name = f"power_{lower}_{upper}_z"
        yield name, z_name, i, lower, upper


# endregion


def compute_features(recording: Recording) -> None:
    """Computes the features and their normalized version."""

    # Collect the values that need to be normalized
    max_diff_uvs = []
    mad_diff_uvs = []
    rms_uvs = []
    std_uvs = []
    ptp_uvs = []
    line_ratios = []
    hf_ratios = []
    lf_ratios = []

    # Do see structure.Epoch documentation to understand how this is intended to work
    power_bins: defaultdict[int, list] = defaultdict(list)

    # Now go over the epochs from the different channels
    # and simply compute the different features
    for ch in recording.channels:
        for ep in ch.epochs:

            # Simple features
            ep.flatline_max_s = flatline_max_s(ep.signal, ch.fs)
            ep.clip_frac = clip_frac(ep.signal, ch.physical_min, ch.physical_max)

            # Features that need to be normalized
            ep.max_diff_uv = max_diff_uv(ep.signal)
            max_diff_uvs.append(ep.max_diff_uv)

            ep.mad_diff_uv = mad_diff_uv(ep.signal)
            mad_diff_uvs.append(ep.mad_diff_uv)

            ep.rms_uv = rms_uv(ep.signal)
            rms_uvs.append(ep.rms_uv)

            ep.std_uv = std_uv(ep.signal)
            std_uvs.append(ep.std_uv)

            ep.ptp_uv = ptp_uv(ep.signal)
            ptp_uvs.append(ep.ptp_uv)

            psd_result = compute_psd(ep.signal, ch.fs)

            ep.line_ratio = line_ratio(psd_result)
            line_ratios.append(ep.line_ratio)

            ep.hf_ratio = hf_ratio(psd_result)
            hf_ratios.append(ep.hf_ratio)

            ep.lf_ratio = lf_ratio(psd_result)
            lf_ratios.append(ep.lf_ratio)

            # Power bins
            for name, _, i, lower, upper in get_power_bins():
                value = sum_power_bin(psd_result, lower, upper)
                setattr(ep, name, value)
                power_bins[i].append(value)

            # Free the signal when done
            ep.signal = None

    # Compute the normalization stats
    max_diff_uv_stats = compute_normalize_stats(max_diff_uvs)
    mad_diff_uv_stats = compute_normalize_stats(mad_diff_uvs)
    rms_uv_stats = compute_normalize_stats(rms_uvs)
    std_uv_stats = compute_normalize_stats(std_uvs)
    ptp_uv_stats = compute_normalize_stats(ptp_uvs)
    line_ratio_stats = compute_normalize_stats(line_ratios)
    hf_ratio_stats = compute_normalize_stats(hf_ratios)
    lf_ratio_stats = compute_normalize_stats(lf_ratios)

    # Power bins stats
    power_bins_stats: dict[int, tuple] = {}
    for _, _, i, _, _ in get_power_bins():
        power_bins_stats[i] = compute_normalize_stats(power_bins[i])

    # Normalize the features
    for ch in recording.channels:
        for ep in ch.epochs:
            ep.max_diff_uv_z = robust_normalize(ep.max_diff_uv, max_diff_uv_stats)
            ep.mad_diff_uv_z = robust_normalize(ep.mad_diff_uv, mad_diff_uv_stats)
            ep.rms_uv_z = robust_normalize(ep.rms_uv, rms_uv_stats)
            ep.std_uv_z = robust_normalize(ep.std_uv, std_uv_stats)
            ep.ptp_uv_z = robust_normalize(ep.ptp_uv, ptp_uv_stats)
            ep.line_ratio_z = robust_normalize(ep.line_ratio, line_ratio_stats)
            ep.hf_ratio_z = robust_normalize(ep.hf_ratio, hf_ratio_stats)
            ep.lf_ratio_z = robust_normalize(ep.lf_ratio, lf_ratio_stats)

            # Normalized power bins
            for name, z_name, i, _, _ in get_power_bins():
                value = getattr(ep, name)
                z_value = robust_normalize(value, power_bins_stats[i])
                setattr(ep, z_name, z_value)
