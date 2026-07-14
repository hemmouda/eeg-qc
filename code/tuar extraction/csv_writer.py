from pathlib import Path
from datetime import datetime
from structure import Patient, Epoch
from preprocessor import RESAMPLING_FREQ, DO_PREPROCESS, EPOCH_DURATION_S
from labels import LABELS, ADD_LABEL_CONDITION


def _write_line(f, line: str | None = None) -> None:
    """Write a line to file with newline. If line is None, write empty line."""

    if line is None:
        f.write("\n")
    else:
        f.write(line + "\n")


def _get_epoch_values(epoch: Epoch, features: list[str]) -> str:
    """Extract feature values from epoch and return as CSV string.
    Handles list/set/tuple attributes (for epoch.labels) by converting to string representation.
    """

    values: list[str] = []
    for feature in features:
        attr = getattr(epoch, feature)

        # Handle iterable attributes (list, set, tuple)
        if isinstance(attr, (list, set, tuple)):
            values.append(
                f'"{",".join(map(str, attr))}"'
            )  # => "label_1,label_2,..." or "" if nothing

        else:
            values.append(str(attr))

    return ",".join(values)


# Define which values to save
FEATURES = [
    "channel_name",
    "index",
    "start_time",
    "end_time",
    "flatline_max_s",
    "clip_frac",
    "max_diff_uv_z",
    "mad_diff_uv_z",
    "rms_uv_z",
    "std_uv_z",
    "ptp_uv_z",
    "hf_ratio_z",
    "lf_ratio_z",
    "power_0_5_z",
    "power_5_10_z",
    "power_10_15_z",
    "power_15_20_z",
    "power_20_25_z",
    "power_25_30_z",
    "power_30_35_z",
    "power_35_40_z",
    "power_40_45_z",
    "power_45_50_z",
    "power_50_55_z",
    "power_55_60_z",
    "power_60_65_z",
    "power_65_70_z",
    "power_70_75_z",
    "power_75_80_z",
    "power_80_85_z",
    "power_85_90_z",
    "power_90_95_z",
    "power_95_100_z",
    "power_100_105_z",
    "power_105_110_z",
    "power_110_115_z",
    "power_115_120_z",
    "labels",
]

# For debugging
# FEATURES = [
#     "channel_name",
#     "index",
#     "start_time",
#     "end_time",
#     "labels",
# ]


def save_patient(patient: Patient, output_dir: str | Path) -> None:
    """Takes a TUAR patient and saves its recordings in a single CSV file
    with the patient's name as the file name in the output_dir.

    The CSV file structure is as such:
        # Comments with some general info about the patient

        feature_1, feature_2, feature_3, ...

        # Comments with info about recording #1 (EDF file)
        # from which the following values came

        value_1, value_2, value_3, ...
        value_1, value_2, value_3, ...

        # Comments with info about recording #2 (EDF file)
        # from which the following values came

        value_1, value_2, value_3, ...
        value_1, value_2, value_3, ...

        ...

    Yes, this structure of having comments between the values breaks
    the convention, but:
    - Most patients only have 1 recording.
    - We do not care if this value came from this recording or that recording, we
    only care that it comes from the same patient.
    - pandas (2.3.3) does handle such comments / can parse the file fine, so we gucci.
    """

    # Create output file path
    output_dir = Path(output_dir)
    csv_file = output_dir / f"{patient.name}.csv"

    with open(csv_file, "w") as f:
        # Write patient info comments
        _write_line(f, f"# Patient: {patient.name}")
        _write_line(f, f"# Code Name: {patient.code_name}")
        _write_line(f, f"# Gender: {patient.gender}")
        _write_line(f, f"# Age: {patient.age}")
        _write_line(f, f"# Number of recordings: {len(patient.recordings)}")
        _write_line(f)

        # Write column headers
        _write_line(f, ",".join(FEATURES))
        _write_line(f)

        # Iterate through recordings
        for recording in patient.recordings:
            _write_line(
                f,
                f"# Recording `{recording.edf_file_path.name}` + `{recording.csv_file_path.name}` values:",
            )
            _write_line(f)

            # Iterate through epochs and write features and labels
            for ch in recording.channels:
                for ep in ch.epochs:
                    _write_line(f, _get_epoch_values(ep, FEATURES))


def save_config(output_dir: str | Path) -> None:
    """Logs/saves the used "config" """

    # Create output file path
    output_dir = Path(output_dir)
    config_file = output_dir / "00_config_info.txt"

    with open(config_file, "w") as f:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        _write_line(f, f"{timestamp}")
        _write_line(f)
        _write_line(
            f, f"Resampling freq: {RESAMPLING_FREQ}, Did preprocess: {DO_PREPROCESS}"
        )
        _write_line(f)
        _write_line(f, f"Epoch duration: {EPOCH_DURATION_S}s")
        _write_line(f)
        _write_line(f, f"Add label condition: {ADD_LABEL_CONDITION}")
        _write_line(f)
        _write_line(f, f"Labels code: {LABELS}")
        _write_line(f)
        _write_line(f, f"Saved features: {FEATURES}")
