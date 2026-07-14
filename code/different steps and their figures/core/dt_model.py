import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _09_feature_importance import OUTPUT as FEATURE_IMPORTANCE


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from typing import Any


from .dataset_reader import get_splits, prepare_splits_for
from .consts import TARGETS


def _get_top_k_features(target_name: str, k: int = 5) -> list[str]:
    importances = FEATURE_IMPORTANCE[target_name]
    sorted_features = sorted(
        importances.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    return [f for f, _ in sorted_features[:k]]


def create_fresh_model(depth) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=depth,
        random_state=42,
    )


def report_perf_for_target(
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    target_name: str,
    target: set[int],
    use_all_features: bool,
    tree_depth: int,
    eval_on_test: bool,
) -> dict:
    """Same as hgb's method. But you can specify whether we are using full feature set or a reduced one."""

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

    if use_all_features:
        pass
    else:
        top_features = _get_top_k_features(target_name)
        X_train = X_train[top_features]
        X_eval = X_eval[top_features]

    model = create_fresh_model(tree_depth)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)

    return classification_report(y_eval, y_pred, output_dict=True)


def report_perf_on_dataset(
    dataset_path: str, tree_depth: int, use_all_features: bool, eval_on_test: bool
) -> dict[str, Any]:
    """
    Does same thing as report_perf_for_target but for each of the different targets.
    """

    splits = get_splits(dataset_path)

    results: dict[str, Any] = {}

    for target_name, target_set in TARGETS.items():
        print(
            f"Extracting results ({eval_on_test=}) with ({tree_depth=}, {use_all_features=}) for `{target_name}` from {dataset_path}"
        )

        results[target_name] = report_perf_for_target(
            splits, target_name, target_set, use_all_features, tree_depth, eval_on_test
        )

    return results
