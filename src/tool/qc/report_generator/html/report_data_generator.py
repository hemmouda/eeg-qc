from __future__ import annotations

import base64
import io
from datetime import timedelta
from typing import Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ...struct.recording import Recording
from ...struct.channel import Channel, ChannelQuality
from ...struct.epoch import Epoch
from ...consts import REPORT_EPOCHS_AGGREGATED

# ── Public entry point ──────────────────────────────────────────────────────


def build_report_data(recording: Recording) -> dict[str, Any]:
    """Builds the full JSON-serializable dict describing `recording`,
    ready to be injected into the HTML template.

    Assumes `recording.quality` is not BLACK (caller's responsibility).
    """
    return {
        "filename": recording.file_path.name,
        "quality": recording.quality.name.lower(),
        "quality_up": recording.quality.name,
        "quality_justification": recording.quality_justification,
        "patient_id": recording.subject_identifier,
        "full_name": _format_full_name(recording),
        "sex": recording.sex,
        "recording_date": recording.recording_date.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": recording.formatted_duration,
        "remarks": [_build_remark(r) for r in recording.remarks],
        "channels": [_build_channel(ch, recording) for ch in recording.channels],
    }


# ── Remarks ──────────────────────────────────────────────────────────────────


def _build_remark(remark) -> dict[str, Any]:
    """Builds the dict for a single Remark"""

    return {
        "title": remark.title(),
        "description": remark.repr_nicely(),
    }


# ── Channels ─────────────────────────────────────────────────────────────────


def _build_channel(channel: Channel, recording: Recording) -> dict[str, Any]:
    """Builds the dict for a single Channel: its traffic-light quality,
    the rule/hover explanation, the spectrogram image (if available),
    and its epochs grouped into REPORT_EPOCHS_AGGREGATED-sized chunks,
    themselves bucketed into wall-clock hour rows.
    """

    quality_str, hover_text = _channel_quality_and_hover(channel)

    epoch_duration_s = channel.epochs[0].duration_s if channel.epochs else 0

    return {
        "name": channel.name,
        "quality": quality_str,
        "quality_hover": hover_text,
        "n_epochs": len(channel.epochs),
        "epoch_duration_s": epoch_duration_s,
        "spectrogram_b64": _render_spectrogram_b64(channel),
        "hour_rows": _build_hour_rows(channel, recording),
    }


def _channel_quality_and_hover(channel: Channel) -> tuple[str, str]:
    """Determines the channel's traffic-light state ('red'/'yellow'/'green'/
    'none') and the text to show on hover, per the spec:
        - None quality  -> muted light, "no rule was assigned to this channel"
        - RED/YELLOW    -> the rule_that_gave_quality's rule_type name
        - GREEN         -> "no threshold was surpassed"
    """

    if channel.quality is None:
        return "none", "No rule was assigned to this channel."

    if channel.quality in (ChannelQuality.RED, ChannelQuality.YELLOW):
        rule = channel.rule_that_gave_quality
        rule_name = rule.rule_type.name if rule is not None else "Unknown rule"
        return channel.quality.name.lower(), rule_name

    # GREEN
    return "green", "No rule threshold was surpassed."


def _render_spectrogram_b64(channel: Channel) -> str | None:
    """Renders channel.spectrogram_data (the (f, t, Sxx) tuple from
    scipy.signal.spectrogram) to a base64-encoded PNG, or returns None
    if the channel has no spectrogram data.

    Sxx is converted to dB (10*log10) before plotting, since raw linear
    power values span orders of magnitude and otherwise render as a
    mostly-black image under a linear color scale.
    """
    if channel.spectrogram_data is None:
        return None

    f, t, Sxx = channel.spectrogram_data

    # Avoid log(0); clip to a small positive floor before converting to dB.
    Sxx_db = 10 * np.log10(np.clip(Sxx, a_min=1e-12, a_max=None))

    fig, ax = plt.subplots(figsize=(7, 1.8), dpi=110)
    ax.pcolormesh(t, f, Sxx_db, shading="gouraud")
    ax.set_ylabel("Hz")
    ax.set_xlabel("Time (s)")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── Epoch grouping, hour bucketing & traffic light ──────────────────────────


def _build_hour_rows(channel: Channel, recording: Recording) -> list[dict[str, Any]]:
    """Splits channel.epochs into consecutive groups of size
    REPORT_EPOCHS_AGGREGATED (last group may be smaller), then buckets
    those groups into rows by their wall-clock hour, derived from
    `recording.recording_date + group.start_time`.

    The first and last hour rows may be partial, since the recording
    doesn't necessarily start or end exactly on the hour.
    """
    epochs = channel.epochs
    groups = []

    for start in range(0, len(epochs), REPORT_EPOCHS_AGGREGATED):
        chunk = epochs[start : start + REPORT_EPOCHS_AGGREGATED]
        groups.append(_build_single_epoch_group(chunk, channel, recording))

    return _bucket_groups_by_hour(groups)


def _build_single_epoch_group(
    chunk: list[Epoch], channel: Channel, recording: Recording
) -> dict[str, Any]:
    """Builds a single epoch-group dict: wall-clock start/end time, the
    group's traffic-light quality (see `_group_traffic_light`), and the
    list of per-epoch modal entries.
    """

    quality = _group_traffic_light(chunk, channel)

    wall_start = recording.recording_date + timedelta(seconds=chunk[0].start_time)
    wall_end = recording.recording_date + timedelta(seconds=chunk[-1].end_time)

    return {
        "wall_start": wall_start,
        "start_time": wall_start.strftime("%H:%M:%S"),
        "end_time": wall_end.strftime("%H:%M:%S"),
        "quality": quality,
        "epochs": [_build_epoch_modal_entry(ep, channel) for ep in chunk],
    }


def _bucket_groups_by_hour(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Buckets epoch groups into rows keyed by their wall-clock hour
    (e.g. all groups starting between 22:00:00 and 22:59:59 land in the
    same '22:00' row). Order is preserved; a new row starts whenever the
    hour changes. The pre-computed `wall_start` datetime is dropped from
    the output after being used to derive the row label.
    """

    rows = []
    current_hour_label = None
    current_row = None

    for group in groups:
        hour_label = group["wall_start"].strftime("%H:00")
        # Drop the internal-only datetime before sending to the frontend.
        group_out = {k: v for k, v in group.items() if k != "wall_start"}

        if hour_label != current_hour_label:
            current_hour_label = hour_label
            current_row = {"hour_label": hour_label, "groups": []}
            rows.append(current_row)

        current_row["groups"].append(group_out)

    return rows


def _group_traffic_light(chunk: list[Epoch], channel: Channel) -> str:
    """Computes the group's traffic light using the 3-interval rule:

        n = number of epochs in the group
        count = number of epochs in the group where ANY rule_result is True

        - count in [0, n/3)      -> 'green'
        - count in [n/3, 2n/3)   -> 'yellow'
        - count in [2n/3, n]     -> 'red'

    If the channel has no quality (channel.quality is None), rule_results
    is never populated, so the group is always 'none' (muted), matching
    the per-channel "no rule assigned" state.

    This applies identically to the small timeline dot and the modal's
    header traffic light -- both are the same computed value, just
    rendered at different sizes.
    """

    if channel.quality is None:
        return "none"

    n = len(chunk)
    if n == 0:
        return "green"

    count = sum(1 for ep in chunk if any(rr.result for rr in ep.rule_results.values()))

    if count >= (2 * n) / 3:
        return "red"
    elif count >= n / 3:
        return "yellow"
    else:
        return "green"


def _build_epoch_modal_entry(epoch: Epoch, channel: Channel) -> dict[str, Any]:
    """Builds a single epoch's row data for display inside the group's modal.

    Includes: order, start/end time, median amplitude (mad_uv), peak-to-peak
    (ptp_uv), flat duration, clip fraction, the model-prediction list (or an
    explanatory placeholder), and the satisfied-rules list (or an
    explanatory placeholder).
    """

    return {
        "order": epoch.order,
        "start_time": epoch.start_time,
        "end_time": epoch.end_time,
        "mad_uv": epoch.mad_uv,
        "ptp_uv": epoch.ptp_uv,
        "flatline_max_s": epoch.flatline_max_s,
        "clip_frac": epoch.clip_frac,
        "predictions": _build_predictions_display(epoch, channel),
        "satisfied_rules": _build_satisfied_rules_display(epoch, channel),
    }


def _build_predictions_display(epoch: Epoch, channel: Channel) -> str | list[str]:
    """Determines what to show in the modal's "model predictions" column:
    - If the channel never had any model run on it: an explanatory string.
    - If predictions exist but none are True: an explanatory string.
    - Otherwise: the list of model names that predicted an artifact.
    """

    if not channel.has_predictions:
        return "No models are assigned to this channel."

    positive_models = [
        name for name, predicted in epoch.predictions.items() if predicted
    ]

    if not positive_models:
        return "None."

    return positive_models


def _build_satisfied_rules_display(epoch: Epoch, channel: Channel) -> str | list[str]:
    """Determines what to show in the modal's "rules satisfied" column:
    - If the channel has no quality (no rules assigned at all): an
      explanatory string.
    - Otherwise: the list of RuleType names whose RuleResult.result is True
      (an empty list if the channel has rules but none fired for this epoch).
    """

    if channel.quality is None:
        return "No rules were assigned to this channel."

    return [
        rule_type.name
        for rule_type, rule_result in epoch.rule_results.items()
        if rule_result.result
    ]


# ── Small formatting helpers ────────────────────────────────────────────────


def _format_full_name(recording: Recording) -> str:

    return f"{recording.last_name} , {recording.first_name}"
