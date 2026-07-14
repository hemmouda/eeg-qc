import os
import numpy as np
from pathlib import Path


class Patient:
    """
    Holds a single TUAR patient with its info and recordings
    """

    def __init__(self, patient_dir: str | Path):
        """Automatically reads and instantiates (but does not load) the different Recordings this Patient has"""

        self.patient_dir = Path(patient_dir)

        # Read the info file and store its data
        info_file = [f for f in os.listdir(self.patient_dir) if f.endswith("_info.txt")]
        assert len(info_file) == 1, f"Couldn't find the info file for {patient_dir}"
        info_path = os.path.join(self.patient_dir, info_file[0])

        info = {}
        with open(info_path, "r") as f:
            for line in f:
                if ":" in line:
                    key, val = line.strip().split(":", 1)
                    info[key.strip().lower()] = val.strip()

        self.code_name: str = info.get("code name")
        self.human_name: str = info.get("human name")
        self.name: str = self.human_name
        self.gender: str = info.get("gender")
        self.age = float(info.get("age"))

        # Store the recordings (EDF files)
        self.recordings: list["Recording"] = []
        edf_files = [f for f in os.listdir(self.patient_dir) if f.endswith(".edf")]
        assert len(edf_files) != 0, f"This Patient `{patient_dir}` has no EDF files?!"
        for edf_file in edf_files:
            edf_file_path = os.path.join(self.patient_dir, edf_file)
            self.recordings.append(Recording(self, edf_file_path))


class Recording:
    """
    Holds a single TUAR recording (edf file) with its channels
    """

    def __init__(self, patient: Patient, edf_file_path: str | Path):
        """Since this is normally only instantiated here, unlike the others, it does not
        automatically add itself to the Patient. Rather the Patient does so."""

        # Save Patient, EDF file path, and CSV file path
        self.patient = patient
        self.edf_file_path = Path(edf_file_path)
        self.csv_file_path = self.edf_file_path.with_suffix(".csv")

        # Where the channels are going to be stored along side their info
        self.channels: list["Channel"] = []


class Channel:
    """Holds a single channel alongside its info and epochs"""

    recording: Recording
    name: str  # Name of channel / label
    fs: float  # Sampling frequency. Initial one, then updated to hold the new one after resampling
    signal: np.ndarray | None  # Cleared after segmenting into the epochs
    physical_min: float
    physical_max: float
    n_samples: int  # After resampling
    duration_s: float  # After resampling

    def __init__(self, recording: Recording, label: str):
        """Automatically adds self to recording"""
        self.recording = recording
        self.recording.channels.append(self)
        self.name = label

        # Where the epochs are going to be stored along side their info
        self.epochs: list["Epoch"] = []


class Epoch:
    """Holds a single epoch alongside its info, features, and labels (artifacts)"""

    # Info
    channel: Channel
    channel_name: str  # Duplicated channel name to make saving to CSV file easier
    index: int  # Index of this epoch in the list of epochs in the channel
    signal: np.ndarray | None  # Cleared after features are extracted
    n_samples: int  # Same across all epochs
    duration_s: float  # Same across all epochs
    start_sample_index: int  # In the original signal, where does this epoch start
    start_time: float  # From 0.0 in the original signal, when does this epoch start
    end_sample_index: int  # In the original signal, where does this epoch end. Equals start_sample_index + n_samples - 1
    end_time: float  # From 0.0 in the original signal, when does this epoch end. Equals start_time + duration_s

    # Features (z suffix means robust scaled / normalized version)
    # How each feature is computed can be found the features.py file
    flatline_max_s: float
    clip_frac: float

    max_diff_uv: float
    max_diff_uv_z: float
    mad_diff_uv: float
    mad_diff_uv_z: float
    rms_uv: float
    rms_uv_z: float
    std_uv: float
    std_uv_z: float
    ptp_uv: float
    ptp_uv_z: float

    line_ratio: float
    line_ratio_z: float
    hf_ratio: float  # Proxy for muscle activity
    hf_ratio_z: float
    lf_ratio: float  # Proxy for electrode drift
    lf_ratio_z: float

    POWER_BINS_DELTA: int = 5
    POWER_BINS_RANGE: int = int((120 / POWER_BINS_DELTA))
    power_0_5: float
    power_0_5_z: float
    power_5_10: float
    power_5_10_z: float
    # And so on until
    power_115_120: float
    power_115_120_z: float
    # These are added "dynamically" when computing the features
    # The POWER_BINS_RANGE is to "index" them:
    # for i in range 0 .. POWER_BINS_RANGE
    #   first bin = i * POWER_BINS_DELTA <-> (i + 1) * POWER_BINS_DELTA

    def __init__(self, channel: Channel):
        """Automatically adds self to the channel"""
        self.channel = channel
        self.channel_name = channel.name
        self.channel.epochs.append(self)
        self.index = len(self.channel.epochs) - 1

        # Where the labels (artifacts) are stored
        self.labels: set[int] = set()
