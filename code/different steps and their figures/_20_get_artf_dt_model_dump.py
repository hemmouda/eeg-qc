"""
Train and dump final artf Decision Tree model.

Workflow:
1. Train on train + val and print tree.
2. Train on train + val + test, print tree, and dump model.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from core.dataset_reader import get_splits, prepare_splits_for
from core.dt_cm_model import _get_top_k_features
from core.consts import TARGETS

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATASET = "/Users/telos_matter/Desktop/uni/fha_semesters/bachelor/code/06 - tuar extraction/output/1-10s any overlap no preprocessing/10s"

TARGET_NAME = "artf"

TREE_DEPTH = 1
USE_ALL_FEATURES = False

MODEL_OUTPUT = Path("_20_artf_dt_model.joblib")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def create_model():
    return DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=TREE_DEPTH,
        random_state=42,
    )


def reduce_features(
    target_name: str,
    X: pd.DataFrame,
) -> pd.DataFrame:

    if USE_ALL_FEATURES:
        return X

    top_features = _get_top_k_features(target_name)

    return X[top_features]


def print_tree(model, features):
    print(
        export_text(
            model,
            feature_names=list(features),
        )
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():

    splits = get_splits(DATASET)

    target = TARGETS[TARGET_NAME]

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits_for(splits, target)

    # =========================================================================
    # 1) Train on TRAIN + VAL
    # =========================================================================

    X_train_val = pd.concat([X_train, X_val], ignore_index=True)

    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    X_train_val = reduce_features(TARGET_NAME, X_train_val)

    model_train_val = create_model()

    model_train_val.fit(X_train_val, y_train_val)

    print("\n================================")
    print("Decision tree: TRAIN + VAL")
    print("================================\n")

    print_tree(model_train_val, X_train_val.columns)

    # =========================================================================
    # 2) Train on TRAIN + VAL + TEST
    # =========================================================================

    X_full = pd.concat([X_train, X_val, X_test], ignore_index=True)

    y_full = pd.concat([y_train, y_val, y_test], ignore_index=True)

    X_full = reduce_features(TARGET_NAME, X_full)

    final_model = create_model()

    final_model.fit(X_full, y_full)

    print("\n================================")
    print("Decision tree: TRAIN + VAL + TEST")
    print("================================\n")

    print_tree(final_model, X_full.columns)

    # =========================================================================
    # Dump final model
    # =========================================================================

    joblib.dump(
        final_model,
        MODEL_OUTPUT,
    )

    print(f"\nSaved model: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()


# ================================
# Decision tree: TRAIN + VAL
# ================================

# |--- ptp_uv_z <= 0.82
# |   |--- class: 0
# |--- ptp_uv_z >  0.82
# |   |--- class: 1


# ================================
# Decision tree: TRAIN + VAL + TEST
# ================================

# |--- ptp_uv_z <= 0.82
# |   |--- class: 0
# |--- ptp_uv_z >  0.82
# |   |--- class: 1
