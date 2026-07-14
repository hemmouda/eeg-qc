from enum import Enum, auto
from typing import Any


class RuleType(Enum):

    RANDOM_RULE = auto()
    """A test/debug rule that passes based on probability specified"""

    CLIPPED = auto()
    """Highlights saturated signals. 
    Passes if Epoch.clipped_frac >= rule.value"""

    FLAT = auto()
    """Highlights constant/dead signals.
    Passes if Epoch.flatline_max_s >= rule.value"""

    HIGH_RMS = auto()
    """Highlights powerful signals.
    Passes if Epoch.rms_uv >= rule.value"""

    LOW_RMS = auto()
    """Highlights weak signals.
    Passes if Epoch.rms_uv <= rule.value"""  # PS: EMG normally has low RMS

    HIGH_DIFF = auto()
    """Highlights abrupt jumps in signals.
    Passes if Epoch.max_diff_uv >= rule.value"""

    HIGH_MAD_DIFF = auto()
    """Highlights unstable signals.
    Passes if Epoch.mad_diff_uv >= rule.value"""

    HIGH_PTP = auto()
    """Highlights high varying signals.
    Passes if Epoch.ptp_uv >= rule.value"""

    HIGH_MAD = auto()
    """Highlights high varying signals.
    Passes if Epoch.mad_uv >= rule.value"""

    HAS_ARTF = auto()
    """Highlights signals contaminated with artifacts.
    Passes if a model predicts the Epoch as having an artifact of type rule.value"""

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return str(self)


class Rule:
    rule_type: RuleType
    value: Any

    red_threshold: float
    """Percentage threshold after which the channel is considered RED."""
    yellow_threshold: float
    """Percentage threshold after which the channel is considered YELLOW. (Must be less than red_threshold)"""
    # green_threshold is then 0% -- yellow_threshold

    def __str__(self):
        return f"{self.rule_type.name} ({self.value}): RED @ {self.red_threshold}% | YELLOW @ {self.yellow_threshold}%"

    def __repr__(self):
        return str(self)
