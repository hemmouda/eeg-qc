"""I wanna see perf of reduced DT on different depths.
Once again, path of the any overlap, no pp, 10s dataset"""

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report

import os

from core.dataset_reader import get_splits, prepare_splits_for
from core.consts import TARGETS
from _09_feature_importance import OUTPUT as FEATURE_IMPORTANCE

OUTPUT_DIR = "_10_reduced_dt_hyperparams_features_results"

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


def create_tree(max_depth: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=max_depth,
        random_state=42,
    )


# -----------------------------------------------------------------------------
# Top-k feature selection
# -----------------------------------------------------------------------------


def get_top_k_features(target_name: str, k: int = 5) -> list[str]:
    importances = FEATURE_IMPORTANCE[target_name]
    sorted_features = sorted(
        importances.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    return [f for f, _ in sorted_features[:k]]


# for target_name, _ in TARGETS.items():
#     print(f"{target_name} top features:")
#     for feature in get_top_k_features(target_name):
#         print(f"    {feature}")
# Output of this ^^^
# artf top features:
#     mad_diff_uv_z
#     ptp_uv_z
#     power_0_5_z
#     flatline_max_s
#     max_diff_uv_z
# musc top features:
#     mad_diff_uv_z
#     flatline_max_s
#     ptp_uv_z
#     max_diff_uv_z
#     rms_uv_z
# eyem top features:
#     power_0_5_z
#     mad_diff_uv_z
#     power_10_15_z
#     rms_uv_z
#     lf_ratio_z
# elec top features:
#     max_diff_uv_z
#     ptp_uv_z
#     mad_diff_uv_z
#     rms_uv_z
#     hf_ratio_z

# -----------------------------------------------------------------------------
# Feature info
# -----------------------------------------------------------------------------


def record_feature_info(
    target_outputs: dict,
    target_name: str,
    model: DecisionTreeClassifier,
    depth: int,
    feature_names: list[str],
):
    lines = [f"\n{'='*40}\nDepth {depth}\n{'='*40}"]
    lines.append(export_text(model, feature_names=feature_names))
    lines.append("Features used:")
    for name, imp in zip(feature_names, model.feature_importances_):
        if imp > 0:
            lines.append(f"  {name}: {imp:.4f}")
    target_outputs[target_name].append("\n".join(lines))


# -----------------------------------------------------------------------------
# Per-target evaluation over depths (restricted features)
# -----------------------------------------------------------------------------


def report_perf_for_target_depths_top_features(
    splits,
    target: set[int],
    target_name: str,
    target_outputs: dict,
) -> dict[int, dict]:

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits_for(splits, target)

    top_features = get_top_k_features(target_name, k=5)

    # restrict feature space
    X_train = X_train[top_features]
    X_val = X_val[top_features]

    results = {}
    DEPTHS = range(1, 8)
    for depth in DEPTHS:
        model = create_tree(max_depth=depth)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        results[depth] = classification_report(y_val, y_pred, output_dict=True)
        record_feature_info(target_outputs, target_name, model, depth, top_features)

    return results


# -----------------------------------------------------------------------------
# Dataset-level wrapper
# -----------------------------------------------------------------------------


def report_perf_on_dataset_depths_top_features(
    dataset_path: str,
) -> dict[str, dict[int, dict]]:
    splits = get_splits(dataset_path)
    target_outputs = {name: [] for name in TARGETS}
    all_results = {}

    for target_name, target_set in TARGETS.items():
        print(f"Processing target={target_name} (top-5 features) from {dataset_path}")
        all_results[target_name] = report_perf_for_target_depths_top_features(
            splits,
            target_set,
            target_name,
            target_outputs,
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for target_name, lines in target_outputs.items():
        with open(os.path.join(OUTPUT_DIR, f"{target_name}.txt"), "w") as f:
            f.write("\n".join(lines))
        print(f"Saved {target_name}.txt")

    return all_results


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

DATASET = "/Users/telos_matter/Desktop/uni/fha_semesters/bachelor/code/06 - tuar extraction/output/1-10s any overlap no preprocessing/10s"

# print(report_perf_on_dataset_depths_top_features(DATASET))

# Use 11 for plot and check output files if you wanna see how features were used
OUTPUT = {
    "artf": {
        1: {
            "0": {
                "precision": 0.8407667100380705,
                "recall": 0.8196638517283095,
                "f1-score": 0.8300811799339856,
                "support": 85141.0,
            },
            "1": {
                "precision": 0.5585901563937442,
                "recall": 0.595154225503109,
                "f1-score": 0.5762928030134508,
                "support": 32647.0,
            },
            "accuracy": 0.7574370903657418,
            "macro avg": {
                "precision": 0.6996784332159074,
                "recall": 0.7074090386157093,
                "f1-score": 0.7031869914737181,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.7625565532578694,
                "recall": 0.7574370903657418,
                "f1-score": 0.7597393018027268,
                "support": 117788.0,
            },
        },
        2: {
            "0": {
                "precision": 0.862265642647391,
                "recall": 0.7173512173923257,
                "f1-score": 0.7831611881543601,
                "support": 85141.0,
            },
            "1": {
                "precision": 0.4874989351733538,
                "recall": 0.7011670291297822,
                "f1-score": 0.5751290780498223,
                "support": 32647.0,
            },
            "accuracy": 0.7128654871463986,
            "macro avg": {
                "precision": 0.6748822889103724,
                "recall": 0.7092591232610539,
                "f1-score": 0.6791451331020912,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.758392508721143,
                "recall": 0.7128654871463986,
                "f1-score": 0.7255014579731629,
                "support": 117788.0,
            },
        },
        3: {
            "0": {
                "precision": 0.862265642647391,
                "recall": 0.7173512173923257,
                "f1-score": 0.7831611881543601,
                "support": 85141.0,
            },
            "1": {
                "precision": 0.4874989351733538,
                "recall": 0.7011670291297822,
                "f1-score": 0.5751290780498223,
                "support": 32647.0,
            },
            "accuracy": 0.7128654871463986,
            "macro avg": {
                "precision": 0.6748822889103724,
                "recall": 0.7092591232610539,
                "f1-score": 0.6791451331020912,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.758392508721143,
                "recall": 0.7128654871463986,
                "f1-score": 0.7255014579731629,
                "support": 117788.0,
            },
        },
        4: {
            "0": {
                "precision": 0.86891955264392,
                "recall": 0.7181616377538437,
                "f1-score": 0.7863802970870041,
                "support": 85141.0,
            },
            "1": {
                "precision": 0.49395811805394463,
                "recall": 0.7174625539865838,
                "f1-score": 0.5850922988534459,
                "support": 32647.0,
            },
            "accuracy": 0.7179678744863653,
            "macro avg": {
                "precision": 0.6814388353489322,
                "recall": 0.7178120958702137,
                "f1-score": 0.6857362979702251,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.7649924466988414,
                "recall": 0.7179678744863653,
                "f1-score": 0.7305898152184693,
                "support": 117788.0,
            },
        },
        5: {
            "0": {
                "precision": 0.8647153219076963,
                "recall": 0.7911229607357207,
                "f1-score": 0.8262837655487132,
                "support": 85141.0,
            },
            "1": {
                "precision": 0.5542075050760785,
                "recall": 0.6772138328177167,
                "f1-score": 0.609567135373587,
                "support": 32647.0,
            },
            "accuracy": 0.7595510578327164,
            "macro avg": {
                "precision": 0.7094614134918874,
                "recall": 0.7341683967767187,
                "f1-score": 0.7179254504611501,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.778652661058528,
                "recall": 0.7595510578327164,
                "f1-score": 0.7662169690556295,
                "support": 117788.0,
            },
        },
        6: {
            "0": {
                "precision": 0.8670502685653201,
                "recall": 0.7716493816140285,
                "f1-score": 0.8165728277216401,
                "support": 85141.0,
            },
            "1": {
                "precision": 0.5372605022015947,
                "recall": 0.6914264710386866,
                "f1-score": 0.6046717205539632,
                "support": 32647.0,
            },
            "accuracy": 0.7494142017862601,
            "macro avg": {
                "precision": 0.7021553853834575,
                "recall": 0.7315379263263575,
                "f1-score": 0.7106222741378017,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.775643278867927,
                "recall": 0.7494142017862601,
                "f1-score": 0.7578407374772761,
                "support": 117788.0,
            },
        },
        7: {
            "0": {
                "precision": 0.8674708201584885,
                "recall": 0.7830070118979106,
                "f1-score": 0.8230776828485358,
                "support": 85141.0,
            },
            "1": {
                "precision": 0.5486967779759142,
                "recall": 0.6880264649125494,
                "f1-score": 0.610513155033703,
                "support": 32647.0,
            },
            "accuracy": 0.7566814955683092,
            "macro avg": {
                "precision": 0.7080837990672013,
                "recall": 0.7355167384052299,
                "f1-score": 0.7167954189411194,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.779117030679641,
                "recall": 0.7566814955683092,
                "f1-score": 0.7641617139928727,
                "support": 117788.0,
            },
        },
    },
    "musc": {
        1: {
            "0": {
                "precision": 0.9427183118145217,
                "recall": 0.6628466561892955,
                "f1-score": 0.7783897795074309,
                "support": 102443.0,
            },
            "1": {
                "precision": 0.24518117050570393,
                "recall": 0.7311176278918214,
                "f1-score": 0.36721601230708806,
                "support": 15345.0,
            },
            "accuracy": 0.6717407545760179,
            "macro avg": {
                "precision": 0.5939497411601128,
                "recall": 0.6969821420405584,
                "f1-score": 0.5728028959072595,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.851845664062766,
                "recall": 0.6717407545760179,
                "f1-score": 0.7248235294845996,
                "support": 117788.0,
            },
        },
        2: {
            "0": {
                "precision": 0.9526780935472501,
                "recall": 0.7123766387161641,
                "f1-score": 0.8151871586072853,
                "support": 102443.0,
            },
            "1": {
                "precision": 0.28456962486342113,
                "recall": 0.7637666992505702,
                "f1-score": 0.41464709004068634,
                "support": 15345.0,
            },
            "accuracy": 0.7190715522803681,
            "macro avg": {
                "precision": 0.6186238592053356,
                "recall": 0.7380716689833671,
                "f1-score": 0.6149171243239858,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8656393081705278,
                "recall": 0.7190715522803681,
                "f1-score": 0.7630062288678003,
                "support": 117788.0,
            },
        },
        3: {
            "0": {
                "precision": 0.9563297086857296,
                "recall": 0.6953818220864285,
                "f1-score": 0.8052426624767567,
                "support": 102443.0,
            },
            "1": {
                "precision": 0.27927386946279276,
                "recall": 0.7880091234929945,
                "f1-score": 0.41239363606909607,
                "support": 15345.0,
            },
            "accuracy": 0.7074489761266003,
            "macro avg": {
                "precision": 0.6178017890742612,
                "recall": 0.7416954727897115,
                "f1-score": 0.6088181492729263,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.86812529182768,
                "recall": 0.7074489761266003,
                "f1-score": 0.7540636942437825,
                "support": 117788.0,
            },
        },
        4: {
            "0": {
                "precision": 0.9562287288639187,
                "recall": 0.6911453198363968,
                "f1-score": 0.8023593805776063,
                "support": 102443.0,
            },
            "1": {
                "precision": 0.27670080468178493,
                "recall": 0.7887911371782339,
                "f1-score": 0.40968708219804023,
                "support": 15345.0,
            },
            "accuracy": 0.7038662682106837,
            "macro avg": {
                "precision": 0.6164647667728518,
                "recall": 0.7399682285073154,
                "f1-score": 0.6060232313878232,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8677022576056,
                "recall": 0.7038662682106837,
                "f1-score": 0.7512034358410079,
                "support": 117788.0,
            },
        },
        5: {
            "0": {
                "precision": 0.9584469205727444,
                "recall": 0.706988276407368,
                "f1-score": 0.8137341370380149,
                "support": 102443.0,
            },
            "1": {
                "precision": 0.2890673108805836,
                "recall": 0.7953730856956663,
                "f1-score": 0.4240276547327462,
                "support": 15345.0,
            },
            "accuracy": 0.7185027337249974,
            "macro avg": {
                "precision": 0.6237571157266639,
                "recall": 0.7511806810515171,
                "f1-score": 0.6188808958853805,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8712425354849068,
                "recall": 0.7185027337249974,
                "f1-score": 0.762964568228167,
                "support": 117788.0,
            },
        },
        6: {
            "0": {
                "precision": 0.9579751417241104,
                "recall": 0.7027127280536494,
                "f1-score": 0.8107258895539701,
                "support": 102443.0,
            },
            "1": {
                "precision": 0.28579803949158106,
                "recall": 0.7942000651678071,
                "f1-score": 0.42033559246037905,
                "support": 15345.0,
            },
            "accuracy": 0.7146313716168031,
            "macro avg": {
                "precision": 0.6218865906078457,
                "recall": 0.7484563966107283,
                "f1-score": 0.6155307410071745,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.870406309298412,
                "recall": 0.7146313716168031,
                "f1-score": 0.7598672357955129,
                "support": 117788.0,
            },
        },
        7: {
            "0": {
                "precision": 0.9553102602429957,
                "recall": 0.7299181007975166,
                "f1-score": 0.8275415569180371,
                "support": 102443.0,
            },
            "1": {
                "precision": 0.2998101986587372,
                "recall": 0.7720430107526882,
                "f1-score": 0.4318993802406125,
                "support": 15345.0,
            },
            "accuracy": 0.7354059836316094,
            "macro avg": {
                "precision": 0.6275602294508664,
                "recall": 0.7509805557751024,
                "f1-score": 0.6297204685793247,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8699140531165444,
                "recall": 0.7354059836316094,
                "f1-score": 0.7759987070427096,
                "support": 117788.0,
            },
        },
    },
    "eyem": {
        1: {
            "0": {
                "precision": 0.9462660185594344,
                "recall": 0.817156703745397,
                "f1-score": 0.8769849801885923,
                "support": 104822.0,
            },
            "1": {
                "precision": 0.297124834971395,
                "recall": 0.624865031621163,
                "f1-score": 0.40274394790475715,
                "support": 12966.0,
            },
            "accuracy": 0.7959894046931776,
            "macro avg": {
                "precision": 0.6216954267654147,
                "recall": 0.72101086768328,
                "f1-score": 0.6398644640466747,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8748091249335768,
                "recall": 0.7959894046931776,
                "f1-score": 0.8247809422170485,
                "support": 117788.0,
            },
        },
        2: {
            "0": {
                "precision": 0.9462660185594344,
                "recall": 0.817156703745397,
                "f1-score": 0.8769849801885923,
                "support": 104822.0,
            },
            "1": {
                "precision": 0.297124834971395,
                "recall": 0.624865031621163,
                "f1-score": 0.40274394790475715,
                "support": 12966.0,
            },
            "accuracy": 0.7959894046931776,
            "macro avg": {
                "precision": 0.6216954267654147,
                "recall": 0.72101086768328,
                "f1-score": 0.6398644640466747,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8748091249335768,
                "recall": 0.7959894046931776,
                "f1-score": 0.8247809422170485,
                "support": 117788.0,
            },
        },
        3: {
            "0": {
                "precision": 0.9446105343262489,
                "recall": 0.8289290416134018,
                "f1-score": 0.8829970478692323,
                "support": 104822.0,
            },
            "1": {
                "precision": 0.3050420493741038,
                "recall": 0.6070492056146846,
                "f1-score": 0.40604606773453017,
                "support": 12966.0,
            },
            "accuracy": 0.8045047033653683,
            "macro avg": {
                "precision": 0.6248262918501764,
                "recall": 0.7179891236140432,
                "f1-score": 0.6445215578018813,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8742073949921104,
                "recall": 0.8045047033653683,
                "f1-score": 0.8304947012089057,
                "support": 117788.0,
            },
        },
        4: {
            "0": {
                "precision": 0.9446960667461264,
                "recall": 0.8317528763045925,
                "f1-score": 0.8846341166443443,
                "support": 104822.0,
            },
            "1": {
                "precision": 0.30833790885559653,
                "recall": 0.6063550825235231,
                "f1-score": 0.40879783693843597,
                "support": 12966.0,
            },
            "accuracy": 0.806941284341359,
            "macro avg": {
                "precision": 0.6265169878008614,
                "recall": 0.7190539794140578,
                "f1-score": 0.6467159767913901,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8746463174065621,
                "recall": 0.806941284341359,
                "f1-score": 0.8322544752320884,
                "support": 117788.0,
            },
        },
        5: {
            "0": {
                "precision": 0.9450664607578181,
                "recall": 0.8173379634046288,
                "f1-score": 0.8765737145546535,
                "support": 104822.0,
            },
            "1": {
                "precision": 0.29432794014668484,
                "recall": 0.6159185562239704,
                "f1-score": 0.39831417242325245,
                "support": 12966.0,
            },
            "accuracy": 0.7951658912622678,
            "macro avg": {
                "precision": 0.6196972004522514,
                "recall": 0.7166282598142997,
                "f1-score": 0.637443943488953,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8734337336697959,
                "recall": 0.7951658912622678,
                "f1-score": 0.8239273225344499,
                "support": 117788.0,
            },
        },
        6: {
            "0": {
                "precision": 0.9464251922927079,
                "recall": 0.8111274350804221,
                "f1-score": 0.8735686508201521,
                "support": 104822.0,
            },
            "1": {
                "precision": 0.2916890272262173,
                "recall": 0.6287983958044115,
                "f1-score": 0.39851406505853315,
                "support": 12966.0,
            },
            "accuracy": 0.7910568139369036,
            "macro avg": {
                "precision": 0.6190571097594626,
                "recall": 0.7199629154424168,
                "f1-score": 0.6360413579393427,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8743524080001475,
                "recall": 0.7910568139369036,
                "f1-score": 0.8212750575934639,
                "support": 117788.0,
            },
        },
        7: {
            "0": {
                "precision": 0.9448298536906132,
                "recall": 0.8224513937913797,
                "f1-score": 0.8794034692222557,
                "support": 104822.0,
            },
            "1": {
                "precision": 0.2988358512602193,
                "recall": 0.6117538176770014,
                "f1-score": 0.4015287655977119,
                "support": 12966.0,
            },
            "accuracy": 0.7992579889292627,
            "macro avg": {
                "precision": 0.6218328524754162,
                "recall": 0.7171026057341905,
                "f1-score": 0.6404661174099837,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8737193990134604,
                "recall": 0.7992579889292627,
                "f1-score": 0.8267994398882332,
                "support": 117788.0,
            },
        },
    },
    "elec": {
        1: {
            "0": {
                "precision": 0.9630900770712909,
                "recall": 0.7389970523281064,
                "f1-score": 0.8362917688394393,
                "support": 108221.0,
            },
            "1": {
                "precision": 0.18711868308967422,
                "recall": 0.6796278875300512,
                "f1-score": 0.29344465756515853,
                "support": 9567.0,
            },
            "accuracy": 0.734174958399837,
            "macro avg": {
                "precision": 0.5751043800804826,
                "recall": 0.7093124699290788,
                "f1-score": 0.564868213202299,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.9000639765668071,
                "recall": 0.734174958399837,
                "f1-score": 0.7922005344729416,
                "support": 117788.0,
            },
        },
        2: {
            "0": {
                "precision": 0.9630900770712909,
                "recall": 0.7389970523281064,
                "f1-score": 0.8362917688394393,
                "support": 108221.0,
            },
            "1": {
                "precision": 0.18711868308967422,
                "recall": 0.6796278875300512,
                "f1-score": 0.29344465756515853,
                "support": 9567.0,
            },
            "accuracy": 0.734174958399837,
            "macro avg": {
                "precision": 0.5751043800804826,
                "recall": 0.7093124699290788,
                "f1-score": 0.564868213202299,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.9000639765668071,
                "recall": 0.734174958399837,
                "f1-score": 0.7922005344729416,
                "support": 117788.0,
            },
        },
        3: {
            "0": {
                "precision": 0.9687583195782972,
                "recall": 0.6724850075308859,
                "f1-score": 0.7938803894297636,
                "support": 108221.0,
            },
            "1": {
                "precision": 0.16922932683292705,
                "recall": 0.754677537368036,
                "f1-score": 0.27646416878865043,
                "support": 9567.0,
            },
            "accuracy": 0.6791608652833905,
            "macro avg": {
                "precision": 0.5689938232056121,
                "recall": 0.713581272449461,
                "f1-score": 0.5351722791092071,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.9038188191742241,
                "recall": 0.6791608652833905,
                "f1-score": 0.7518547078418808,
                "support": 117788.0,
            },
        },
        4: {
            "0": {
                "precision": 0.9685028175567388,
                "recall": 0.6924256844789828,
                "f1-score": 0.8075196801603509,
                "support": 108221.0,
            },
            "1": {
                "precision": 0.17641528107680127,
                "recall": 0.7452701996446117,
                "f1-score": 0.28529700098033334,
                "support": 9567.0,
            },
            "accuracy": 0.6967178320372194,
            "macro avg": {
                "precision": 0.57245904931677,
                "recall": 0.7188479420617973,
                "f1-score": 0.5464083405703422,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.904167728570564,
                "recall": 0.6967178320372194,
                "f1-score": 0.7651036074558715,
                "support": 117788.0,
            },
        },
        5: {
            "0": {
                "precision": 0.970213849287169,
                "recall": 0.6690845584498387,
                "f1-score": 0.791991381054727,
                "support": 108221.0,
            },
            "1": {
                "precision": 0.17017332468254703,
                "recall": 0.7676387582314205,
                "f1-score": 0.2785880924833564,
                "support": 9567.0,
            },
            "accuracy": 0.6770893469623391,
            "macro avg": {
                "precision": 0.570193586984858,
                "recall": 0.7183616583406296,
                "f1-score": 0.5352897367690417,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.905232801142261,
                "recall": 0.6770893469623391,
                "f1-score": 0.7502916386211829,
                "support": 117788.0,
            },
        },
        6: {
            "0": {
                "precision": 0.9657509581500631,
                "recall": 0.6938671791981224,
                "f1-score": 0.8075386476676972,
                "support": 108221.0,
            },
            "1": {
                "precision": 0.17245341459759206,
                "recall": 0.7216473293613463,
                "f1-score": 0.2783814842442693,
                "support": 9567.0,
            },
            "accuracy": 0.6961235439942949,
            "macro avg": {
                "precision": 0.5691021863738276,
                "recall": 0.7077572542797343,
                "f1-score": 0.5429600659559832,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.9013175897325122,
                "recall": 0.6961235439942949,
                "f1-score": 0.7645593409261622,
                "support": 117788.0,
            },
        },
        7: {
            "0": {
                "precision": 0.9657763261512244,
                "recall": 0.6912706406335184,
                "f1-score": 0.8057862366842236,
                "support": 108221.0,
            },
            "1": {
                "precision": 0.17149800381878147,
                "recall": 0.7229016410578029,
                "f1-score": 0.27722772277227725,
                "support": 9567.0,
            },
            "accuracy": 0.6938397799436276,
            "macro avg": {
                "precision": 0.5686371649850029,
                "recall": 0.7070861408456606,
                "f1-score": 0.5415069797282503,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.9012632967275609,
                "recall": 0.6938397799436276,
                "f1-score": 0.7628555535705313,
                "support": 117788.0,
            },
        },
    },
}
