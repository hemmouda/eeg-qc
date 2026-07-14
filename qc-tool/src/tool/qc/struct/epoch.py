import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .channel import Channel
    from .rule import Rule, RuleType


class RuleResult:
    """See Channel.ChannelQuality documentation. This is supposed to hold
    the result of the rule `x_of_epochs_that_are_y`. For example
    epoch's ptp is greater than 150"""

    rule: "Rule"
    result: bool


class Epoch:

    channel: "Channel"

    signal: np.ndarray
    """Deleted once no longer needed."""

    order: int
    """Order of this epoch in the list of epochs in channel (index + 1)"""
    n_samples: int
    """Same across all epochs"""
    duration_s: float | int
    """Same across all epochs. Determined from n_samples and channel.fs"""

    first_sample_index: int
    """In the original signal, where does this epoch start"""
    start_time: float
    """From 0.0 in the original signal (from the start of the signal), when does this epoch start. In seconds"""

    last_sample_index: int
    """In the original signal, where does this epoch end. Calculated as first_sample_index + n_samples - 1"""
    end_time: float
    """ From 0.0 in the original signal (from the start of the signal), when does this epoch end. Calculated as start_time + duration_s"""

    rule_results: dict["RuleType", RuleResult]
    """Dict always exists, but only populated if channel has quality."""

    # region Features

    # - z suffix means robust scaled/normalized version
    # - uv suffix denotes microvolt
    # - How each feature is computed can be found the features_extractor.py file
    # - Not all features may be used / needed

    flatline_max_s: float
    clip_frac: float

    max_diff_uv: float
    max_diff_uv_z: float
    mad_diff_uv: float
    mad_diff_uv_z: float
    mad_uv: float
    mad_uv_z: float  # Doesn't make sense to normalize this one, but meh
    rms_uv: float
    rms_uv_z: float
    std_uv: float
    std_uv_z: float
    ptp_uv: float
    ptp_uv_z: float

    line_ratio: float
    line_ratio_z: float
    hf_ratio: float
    hf_ratio_z: float
    lf_ratio: float
    lf_ratio_z: float

    POWER_BINS_DELTA: int = 5
    POWER_BINS_RANGE: int = int((120 / POWER_BINS_DELTA))

    power_0_5: float
    power_0_5_z: float
    power_5_10: float
    power_5_10_z: float
    # ... all the way to
    power_115_120: float
    power_115_120_z: float

    # These power bins are added "dynamically" when computing the features
    # POWER_BINS_RANGE allows for their "index"-ing:
    # for i in 0 .. POWER_BINS_RANGE
    #   first_bin == [i * POWER_BINS_DELTA, (i + 1) * POWER_BINS_DELTA]

    # endregion

    # region Predictions
    predictions: dict[str, bool]
    """Maps model name to result of prediction as a bool (predicted as having an artifact or not).
    Dict always exists, but only populated if channel has predictions."""
    # endregion

    def __init__(self):
        self.rule_results = {}
        self.predictions = {}
