import yaml
from pathlib import Path

DATA_SPLIT_PATH = Path(
    "/Users/telos_matter/Desktop/uni/fha_semesters/bachelor/code/17 - better 09/core/data_split.yaml"
)

DATA_SPLIT: dict[str, list[str]] = None
with open(DATA_SPLIT_PATH, "r") as f:
    DATA_SPLIT = yaml.safe_load(f)

LABELS: dict[str, int] = {
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

MUSC_TARGET: set[int] = {
    LABELS["musc"],
    LABELS["eyem_musc"],
    LABELS["musc_elec"],
    LABELS["chew_musc"],
}

EYEM_TARGET: set[int] = {
    LABELS["eyem"],
    LABELS["eyem_musc"],
    LABELS["eyem_elec"],
    LABELS["eyem_chew"],
    LABELS["eyem_shiv"],
}

ELEC_TARGET: set[int] = {
    LABELS["elec"],
    LABELS["musc_elec"],
    LABELS["eyem_elec"],
    LABELS["chew_elec"],
    LABELS["shiv_elec"],
}

ARTF_TARGET: set[int] = set(LABELS.values())  # For artifact vs. non-artifact

TARGETS: dict[str, set[int]] = {
    "artf": ARTF_TARGET,
    "musc": MUSC_TARGET,
    "eyem": EYEM_TARGET,
    "elec": ELEC_TARGET,
}
