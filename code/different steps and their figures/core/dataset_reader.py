from pathlib import Path
import pandas as pd
from .consts import DATA_SPLIT


def _read_csv_file(filepath: Path) -> pd.DataFrame:
    needed_cols = lambda col: col not in [
        "channel_name",
        "index",
        "start_time",
        "end_time",
    ]

    return pd.read_csv(
        filepath,
        comment="#",
        decimal=".",
        converters={"labels": str},
        usecols=needed_cols,
    )


def _get_concatenated_df(dataset_path: Path, names_list: list[str]) -> pd.DataFrame:
    dfs = []
    for name in names_list:
        filepath = dataset_path / f"{name}.csv"
        df = _read_csv_file(filepath)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def _add_target_column(df: pd.DataFrame, target_values: set[int]) -> None:

    def check_label(labels: str):
        """
        Checks the labels column and see if it contains any of the desired targets,
        if so, it returns 1, otherwise 0.
        """

        if len(labels) == 0:
            return 0

        # Convert labels string to list of ints and check if any match target
        labels_list = [int(x.strip()) for x in labels.split(",")]
        has_target = any((label in target_values) for label in labels_list)
        return 1 if has_target else 0

    df["target"] = df["labels"].apply(check_label)


def get_splits(
    dataset_path: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Only reads the dataset and returns the train, val, and test splits.
    They can then be prepared using prepare_splits_for"""

    dataset_path = Path(dataset_path)

    train = _get_concatenated_df(dataset_path, DATA_SPLIT["train"])
    val = _get_concatenated_df(dataset_path, DATA_SPLIT["val"])
    test = _get_concatenated_df(dataset_path, DATA_SPLIT["test"])

    return (train, val, test)


def prepare_splits_for(
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], target: set[int]
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Takes the splits and prepares them by adding a target column (containing 0 or 1) for the desired target.

    Returns X_train, X_val, X_test, y_train, y_val, y_test.

    The passed splits are copied. This is to save time and not have to re-read the same
    CSVs just for a different target.
    """

    assert len(splits) == 3, f"?"

    copies = [split.copy() for split in splits]

    # Add target column
    for split in copies:
        _add_target_column(split, target)

    # Get ys
    ys = [split["target"].copy() for split in copies]

    # Get Xs
    Xs = [split.drop(columns=["labels", "target"]).copy() for split in copies]

    return (*Xs, *ys)
