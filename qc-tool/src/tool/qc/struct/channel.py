from __future__ import annotations

import numpy as np
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .recording import Recording
    from .rule import Rule
    from .epoch import Epoch


class ChannelQuality(Enum):
    """Channel quality is based on the threshold rules assigned to it. Such that:

    A Channel FOO is considered RED if
        %_of_epochs_that_are_1 >= x11% (for example: %_of_epochs_that_have_a_ptp_greater_than_150 >= 50%)
            or
        %_of_epochs_that_are_2 >= x21%
            or …

    Otherwise YELLOW if
        %_of_epochs_that_are_1 >= x12%
            or
        %_of_epochs_that_are_2 >= x22%
            or …

    Otherwise GREEN.

    Let it be stated clearly that determining the quality of a channel goes in that order.
    """

    RED = auto()
    YELLOW = auto()
    GREEN = auto()


class Channel:

    recording: "Recording"

    name: str
    signal: np.ndarray
    """Deleted once no longer needed."""
    fs: float | int
    """Sampling frequency after resampling"""
    n_samples: int
    """After resampling"""
    duration_s: float | int
    """Determined from fs and n_samples"""

    spectrogram_data: tuple | None
    """Spectrogram data if needed. Tuple is result of scipy.signal.spectrogram"""

    has_predictions: bool
    """Was this channel used for predictions by a model?"""

    quality: ChannelQuality | None
    """Quality of this channel if it has at least one rule otherwise None"""
    rule_that_gave_quality: "Rule" | None
    """The rule that entailed this quality or None if GREEN"""

    epochs: list["Epoch"]

    def __init__(self):
        self.epochs = []
