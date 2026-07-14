import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from typing import Any


from .dataset_reader import get_splits, prepare_splits_for
from .consts import TARGETS


def create_fresh_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        l2_regularization=20.0,
        learning_rate=0.15,
        max_depth=15,
        max_features=1.0,
        max_iter=200,
        max_leaf_nodes=150,
        min_samples_leaf=10,
        random_state=42,
    )


def report_perf_for_target(
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    target: set[int],
    eval_on_test: bool,
) -> dict:
    """Takes the splits, prepares them for the specified target and reports the performance.

    If eval_on_test is False, the model is trained on the train split and performance is done
    on val split. Otherwise, the model is trained on (train + val) and performance is done on
    test split.

    Returns dict result of classification_report.

    Class 0 is non-target, class 1 is target. If target is musc, the class 1 is musc artifacts,
    and class 0 is anything that is not musc (e.g. clean epochs as well as elec epochs). If target
    is artifact then it is artifact vs non-artifact.
    """

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits_for(splits, target)

    if eval_on_test:
        X_train = pd.concat([X_train, X_val], ignore_index=True)
        y_train = pd.concat([y_train, y_val], ignore_index=True)

        X_eval = X_test
        y_eval = y_test
    else:
        # X_train = X_train
        # y_train = y_train

        X_eval = X_val
        y_eval = y_val

    model = create_fresh_model()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)

    return classification_report(y_eval, y_pred, output_dict=True)


def report_perf_on_dataset(dataset_path: str, eval_on_test: bool) -> dict[str, Any]:
    """
    Does same thing as report_perf_for_target but for each of the different targets.
    """

    splits = get_splits(dataset_path)

    results: dict[str, Any] = {}

    for target_name, target_set in TARGETS.items():
        print(
            f"Extracting results ({eval_on_test=}) for `{target_name}` from {dataset_path}"
        )

        results[target_name] = report_perf_for_target(splits, target_set, eval_on_test)

    return results
