from pathlib import Path
import pandas as pd

from ...struct.recording import Recording, RecordingQuality
from ...struct.epoch import Epoch


from ..util.report_utils import get_unique_dated_filename


def generate_csv_report(recording: Recording, location: Path) -> None:
    """Generates and saves the CSV report for the given recording in the given location."""

    # Special report for BLACK recording
    if recording.quality is RecordingQuality.BLACK:
        _generate_black_quality_report(recording, location)
        return

    # Generate epoch data as DF
    rows = []
    for channel in recording.channels:
        for epoch in channel.epochs:
            rows.append(_get_row(epoch))

    df = pd.DataFrame(rows)

    # Get file
    file_path = get_unique_dated_filename(location, f"{recording.file_path.name}.csv")

    # Dump
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        _write_report_header(f, recording)
        df.to_csv(f, index=False)


def _generate_black_quality_report(recording: Recording, location: Path) -> None:

    file_path = get_unique_dated_filename(location, f"{recording.file_path.name}.csv")

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        _write_comment(f, "NO_REPORT")
        _write_line(f)
        _write_comment(f, f"File: {recording.file_path.resolve()}")
        _write_comment(f, f"Quality: {recording.quality.name}")
        _write_comment(f, f"Quality justification: {recording.quality_justification}")
        _write_line(f)


def _write_report_header(f, recording: Recording) -> None:
    _write_comment(f, f"File: {recording.file_path.resolve()}")
    _write_comment(f, f"Patient ID: {recording.subject_identifier}")
    _write_comment(f, f"Patient name: {recording.last_name}, {recording.first_name}")
    _write_comment(f, f"Patient gender: {recording.sex}")
    _write_comment(f, f"Recording date: {recording.recording_date}")
    _write_comment(f, f"Recording duration: {recording.formatted_duration}")
    _write_comment(f, f"Quality: {recording.quality.name}")
    _write_comment(f, f"Quality justification: {recording.quality_justification}")

    _write_line(f)
    if len(recording.remarks) == 0:
        _write_comment(f, f"Remarks: NO_REMARKS")
    else:
        _write_comment(f, f"Remarks:")
        for remark in recording.remarks:
            _write_comment(f, f"    - {remark.repr_nicely()}")

    _write_line(f)
    for channel in recording.channels:
        if channel.quality is None:
            _write_comment(f, f"{channel.name} channel rules: NO_RULE")
        else:
            _write_comment(f, f"{channel.name} channel rules:")
            for rule_result in channel.epochs[0].rule_results.values():
                _write_comment(f, f"    - {rule_result.rule}")

    _write_line(f)


def _get_row(epoch: Epoch) -> dict:
    return {
        "channel_name": epoch.channel.name,
        "channel_quality": _get_channel_quality(epoch),
        "epoch_order": epoch.order,
        "start_time_offset": epoch.start_time,
        "end_time_offset": epoch.end_time,
        "rules_results": _get_epoch_rules_results(epoch),
        "models_predictions": _get_models_predictions(epoch),
        "flatline_max_s": epoch.flatline_max_s,
        "clip_frac": epoch.clip_frac,
        "max_diff_uv": epoch.max_diff_uv,
        "max_diff_uv_z": epoch.max_diff_uv_z,
        "mad_diff_uv": epoch.mad_diff_uv,
        "mad_diff_uv_z": epoch.mad_diff_uv_z,
        "mad_uv": epoch.mad_uv,
        "mad_uv_z": epoch.mad_uv_z,
        "rms_uv": epoch.rms_uv,
        "rms_uv_z": epoch.rms_uv_z,
        "std_uv": epoch.std_uv,
        "std_uv_z": epoch.std_uv_z,
        "ptp_uv": epoch.ptp_uv,
        "ptp_uv_z": epoch.ptp_uv_z,
        "line_ratio": epoch.line_ratio,
        "line_ratio_z": epoch.line_ratio_z,
        "hf_ratio": epoch.hf_ratio,
        "hf_ratio_z": epoch.hf_ratio_z,
        "lf_ratio": epoch.lf_ratio,
        "lf_ratio_z": epoch.lf_ratio_z,
        "power_0_5": epoch.power_0_5,
        "power_0_5_z": epoch.power_0_5_z,
        "power_5_10": epoch.power_5_10,
        "power_5_10_z": epoch.power_5_10_z,
        "power_10_15": epoch.power_10_15,
        "power_10_15_z": epoch.power_10_15_z,
        "power_105_110": epoch.power_105_110,
        "power_105_110_z": epoch.power_105_110_z,
    }


def _get_channel_quality(epoch: Epoch) -> str:
    return (
        epoch.channel.quality.name if epoch.channel.quality is not None else "NO_RULE"
    )


def _get_epoch_rules_results(epoch: Epoch) -> str:
    if epoch.channel.quality is None:
        return "NO_RULE"
    return " | ".join(
        [
            f"{rule_type.name} : {rule_result.result}"
            for rule_type, rule_result in epoch.rule_results.items()
        ]
    )


def _get_models_predictions(epoch: Epoch) -> str:
    if not epoch.channel.has_predictions:
        return "NO_PREDICTIONS"
    return " | ".join(
        f"{model_name} : {prediction}"
        for model_name, prediction in epoch.predictions.items()
    )


def _write_line(f, line: str | None = None) -> None:
    """Write a line to file with newline. If line is None, write empty line."""

    if line is None:
        f.write("\n")
    else:
        f.write(line + "\n")


def _write_comment(f, comment: str) -> None:
    _write_line(f, f"# {comment}")
