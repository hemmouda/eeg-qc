from structure import Recording
import pandas as pd

LABELS = {
    "musc": 1,
    "eyem": 2,
    "elec": 3,
    "eyem_musc": 4,
    "musc_elec": 5,
    "chew": 6,
    "eyem_elec": 7,
    "eyem_chew": 8,
    "shiv": 9,
    "chew_musc": 10,
    "elpp": 11,
    "chew_elec": 12,
    "eyem_shiv": 13,
    "shiv_elec": 14,
}


def _any_overlap(
    ep_start: float, ep_end: float, artf_start: float, artf_end: float
) -> bool:
    return True


def _artf_covers_x_percent_ep(
    ep_start: float,
    ep_end: float,
    artf_start: float,
    artf_end: float,
    percentage: float = 0.5,
) -> bool:
    """
    Check if artifact covers at least x percent of the epoch.
    """

    ep_length = ep_end - ep_start
    overlap_start = max(ep_start, artf_start)
    overlap_end = min(ep_end, artf_end)
    assert overlap_end >= overlap_start, f"You changed something."
    overlap_length = overlap_end - overlap_start
    assert overlap_length <= ep_length, f"You changed something."

    return (overlap_length / ep_length) >= percentage


# To log it when saving the CSV files
# ADD_LABEL_CONDITION = "Artifact covers 50% of epoch"
ADD_LABEL_CONDITION = "Any overlap"


def _add_label(
    ep_start: float, ep_end: float, artf_start: float, artf_end: float
) -> bool:
    if ADD_LABEL_CONDITION == "Artifact covers 50% of epoch":
        return _artf_covers_x_percent_ep(ep_start, ep_end, artf_start, artf_end)
    elif ADD_LABEL_CONDITION == "Any overlap":
        return _any_overlap(ep_start, ep_end, artf_start, artf_end)
    else:
        assert False, "Come add the other"


def add_labels(recording: Recording) -> None:
    """
    Maps the artifacts to the epochs.
    The rule by which that is determined is _add_label().
    """

    # Load CSV file
    df = pd.read_csv(recording.csv_file_path, comment="#")

    # For every channel extract its row in the CSV file
    for ch in recording.channels:
        ch_rows = df[df["channel"] == ch.name]

        # Then iterate over the epochs and get the artifacts that have any overlap
        for ep in ch.epochs:
            ep_rows = ch_rows[
                (ch_rows["start_time"] <= ep.end_time)
                & (ch_rows["stop_time"] >= ep.start_time)
            ]

            # Then add artifacts according to rule
            for _, row in ep_rows.iterrows():
                if _add_label(
                    ep.start_time, ep.end_time, row["start_time"], row["stop_time"]
                ):
                    label_int = LABELS[row["label"]]
                    ep.labels.add(label_int)
