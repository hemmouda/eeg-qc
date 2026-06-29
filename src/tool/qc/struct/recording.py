from datetime import datetime, date
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .channel import Channel


class RecordingQuality(Enum):
    """The quality of a non-BLACK recording is based firstly on its critical-channels' quality
    and secondly on its "normal"-channels' quality.

    To make it easier to follow the decisions based on which the quality of a non-BLACK recording
    is determined, we distinguish between two disjunct scenarios:
        - 1. At least one critical channel has a rule assigned to it (scenario A).
        - 2. No critical channel has a rule assigned to it (scenario B).

    Also know the following:
        - At least one channel (critical or normal) has to have at least one rule assigned
        to it (guaranteed by the config reader).
        - At least one rule needs to be assigned to a channel for its quality to be determined.
        - The quality is determined successively. I.e., if not RED, then we check YELLOW, if
        not YELLOW then GREEN.
    """

    BLACK = auto()
    """A BLACK recording indicates severe technical or recording failures that
    that halt any further action on the recording let alone a study or analysis.

    A recording is BLACK if any of these happened or are true:
        - The given EDF file doesn't exist.
        - The EDF file is corrupt and does not follow the EDF file format.
        - One of the channels needed for the study are missing.
        - The recording's duration is too short or too long.
        - An exception occurred while trying to determine the quality.
    """

    RED = auto()
    """A RED recording indicates a recording that contains severely degraded or unusable channels.

    A recording is RED if it's not BLACK and if any of these are true (logical OR given scenario A or B):
        - 1. For scenario A:
            - Any of the critical channels are RED.

        - 2. For scenario B:
            - The majority of the normal channels are RED.
            - The majority cannot be determined and RED is part of the majority. (If 2G 2Y 1R => Y)
    """

    YELLOW = auto()
    """A YELLOW recording indicates a recording that contains partially degraded or suspicious channels.

    A recording is YELLOW if it's not RED and if any of these are true:
        - 1. For scenario A:
            - Any of the critical channels are YELLOW.
            - Any of the normal channels are RED.

        - 2. For scenario B:
            - The majority of the normal channels are YELLOW.
            - The majority cannot be determined. (This branch would only be reached if majority is between Y and G)
    """

    GREEN = auto()
    """A GREEN recording indicates a recording that is usable and that contains no severely degraded channels.

    A recording is GREEN simply if it's not YELLOW (for either scenarios A or B).
    """

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return str(self)


class RemarkType(Enum):
    """The different remarks that aren't critical, but that the user should be informed of"""

    MISSING_CHANNELS = auto()
    EXTRA_CHANNELS = auto()
    DIFFERENT_FS = auto()


class Remark:
    """A way to hold additional info about the different remarks that we can have"""

    def __init__(self, remark_type: RemarkType, details: str | None = None):
        self.remark_type = remark_type
        self.details = details

    def repr_nicely(self) -> str:
        # Eh, a little behavior but makes my life easier.

        if self.remark_type is RemarkType.MISSING_CHANNELS:
            return f"This EDF recording does not contain all channels an HST recording is supposed to have. Namely, these non-essential channels are missing: {self.details}."
        elif self.remark_type is RemarkType.EXTRA_CHANNELS:
            return f"This EDF recording contain additional unexpected channels from an HST recording. Namely: {self.details}."
        elif self.remark_type is RemarkType.DIFFERENT_FS:
            return f"This HST recording is recorded at a different sampling rate than expected. Namely, at {self.details} Hz."
        else:
            assert False, f"You forgot this value: {self.remark_type}"

    def title(self) -> str:
        # Same as repr_nicely

        if self.remark_type is RemarkType.MISSING_CHANNELS:
            return "Missing expected channels"
        elif self.remark_type is RemarkType.EXTRA_CHANNELS:
            return "Unexpected additional channels"
        elif self.remark_type is RemarkType.DIFFERENT_FS:
            return "Unexpected sampling rate"
        else:
            assert False, f"You forgot this value: {self.remark_type}"

    def __str__(self):
        return f"{self.remark_type}: {self.details}"

    def __repr__(self):
        return str(self)


class Recording:

    file_path: Path

    quality: RecordingQuality
    quality_justification: str

    remarks: list[Remark]

    # Basic recording and patient info
    subject_identifier: str
    last_name: str
    first_name: str
    birthday: date | None
    sex: str
    recording_date: datetime
    """When did the recording start"""
    read_duration: float | int
    """Duration that MNE reported"""
    formatted_duration: str
    """In the form xh ym zs"""
    original_fs: float | int
    """Original sampling frequency. We resample to common fs if different."""

    channels: list["Channel"]

    def __init__(self):
        self.file_path = Path("UNKNOWN_FILE")  # Temporary initial value

        self.remarks = []
        self.channels = []
