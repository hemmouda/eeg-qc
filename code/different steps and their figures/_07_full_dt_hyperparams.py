"""Determine the full_DT depth using the val set
Specify path for the any overlap, no pp, 10s dataset"""

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report

from core.dataset_reader import get_splits, prepare_splits_for
from core.consts import TARGETS

import os

OUTPUT_DIR = "_07_full_dt_hyperparams_features_results"

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
# Per-target evaluation over depths
# -----------------------------------------------------------------------------


def report_perf_for_target_depths(
    splits,
    target: set[int],
    target_name: str,
    target_outputs: dict,
) -> dict[int, dict]:

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits_for(splits, target)
    feature_names = X_train.columns.tolist()

    results = {}
    DEPTHS = range(1, 8)
    for depth in DEPTHS:
        model = create_tree(max_depth=depth)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        results[depth] = classification_report(y_val, y_pred, output_dict=True)
        record_feature_info(target_outputs, target_name, model, depth, feature_names)

    return results


# -----------------------------------------------------------------------------
# Dataset-level wrapper
# -----------------------------------------------------------------------------


def report_perf_on_dataset_depths(dataset_path: str) -> dict[str, dict[int, dict]]:
    splits = get_splits(dataset_path)
    target_outputs = {name: [] for name in TARGETS}
    all_results = {}

    for target_name, target_set in TARGETS.items():
        print(f"Processing target={target_name} from {dataset_path}")
        all_results[target_name] = report_perf_for_target_depths(
            splits, target_set, target_name, target_outputs
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

# print(report_perf_on_dataset_depths(DATASET))

# Use 08 for plots and check output files if you wanna see how features were used
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
                "precision": 0.8595216717002694,
                "recall": 0.7420631658073079,
                "f1-score": 0.7964852786374782,
                "support": 85141.0,
            },
            "1": {
                "precision": 0.504064857052527,
                "recall": 0.6837075382117805,
                "f1-score": 0.5803013167986065,
                "support": 32647.0,
            },
            "accuracy": 0.7258888851156314,
            "macro avg": {
                "precision": 0.6817932643763982,
                "recall": 0.7128853520095442,
                "f1-score": 0.6883932977180424,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.7610006115939356,
                "recall": 0.7258888851156314,
                "f1-score": 0.7365661204706561,
                "support": 117788.0,
            },
        },
        4: {
            "0": {
                "precision": 0.8682124539019117,
                "recall": 0.7216969497656828,
                "f1-score": 0.7882037533512064,
                "support": 85141.0,
            },
            "1": {
                "precision": 0.4960119110922046,
                "recall": 0.714307593347015,
                "f1-score": 0.5854736260701464,
                "support": 32647.0,
            },
            "accuracy": 0.7196488606649234,
            "macro avg": {
                "precision": 0.6821121824970582,
                "recall": 0.7180022715563489,
                "f1-score": 0.6868386897106764,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.7650505773006576,
                "recall": 0.7196488606649234,
                "f1-score": 0.7320135602471146,
                "support": 117788.0,
            },
        },
        5: {
            "0": {
                "precision": 0.8636042221107982,
                "recall": 0.7889500945490422,
                "f1-score": 0.8245909085329176,
                "support": 85141.0,
            },
            "1": {
                "precision": 0.5508536006198915,
                "recall": 0.6750390541244219,
                "f1-score": 0.6066562061276737,
                "support": 32647.0,
            },
            "accuracy": 0.7573776615614494,
            "macro avg": {
                "precision": 0.7072289113653449,
                "recall": 0.731994574336732,
                "f1-score": 0.7156235573302956,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.7769199288057618,
                "recall": 0.7573776615614494,
                "f1-score": 0.7641865020617661,
                "support": 117788.0,
            },
        },
        6: {
            "0": {
                "precision": 0.8615482963625903,
                "recall": 0.7914635721920109,
                "f1-score": 0.82502020127819,
                "support": 85141.0,
            },
            "1": {
                "precision": 0.5513355065322316,
                "recall": 0.6683003032437896,
                "f1-score": 0.6042093602880089,
                "support": 32647.0,
            },
            "accuracy": 0.7573267225863416,
            "macro avg": {
                "precision": 0.706441901447411,
                "recall": 0.7298819377179002,
                "f1-score": 0.7146147807830994,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.7755674073960426,
                "recall": 0.7573267225863416,
                "f1-score": 0.7638186227998522,
                "support": 117788.0,
            },
        },
        7: {
            "0": {
                "precision": 0.8639633412692318,
                "recall": 0.7927790371266488,
                "f1-score": 0.8268419215150643,
                "support": 85141.0,
            },
            "1": {
                "precision": 0.5551661540013111,
                "recall": 0.6744570710938218,
                "f1-score": 0.6090251559280311,
                "support": 32647.0,
            },
            "accuracy": 0.7599840391211329,
            "macro avg": {
                "precision": 0.7095647476352714,
                "recall": 0.7336180541102353,
                "f1-score": 0.7179335387215477,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.7783748112599286,
                "recall": 0.7599840391211329,
                "f1-score": 0.7664702032914772,
                "support": 117788.0,
            },
        },
    },
    "musc": {
        1: {
            "0": {
                "precision": 0.9283556762260309,
                "recall": 0.8010503401891784,
                "f1-score": 0.860017396954485,
                "support": 102443.0,
            },
            "1": {
                "precision": 0.3066036131051611,
                "recall": 0.5872922776148582,
                "f1-score": 0.40287898430864144,
                "support": 15345.0,
            },
            "accuracy": 0.7732027031616124,
            "macro avg": {
                "precision": 0.617479644665596,
                "recall": 0.6941713089020183,
                "f1-score": 0.6314481906315632,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8473560378198287,
                "recall": 0.7732027031616124,
                "f1-score": 0.8004630370701974,
                "support": 117788.0,
            },
        },
        2: {
            "0": {
                "precision": 0.9485907700318948,
                "recall": 0.7112833478129301,
                "f1-score": 0.8129734795657656,
                "support": 102443.0,
            },
            "1": {
                "precision": 0.27813438117784883,
                "recall": 0.7426523297491039,
                "f1-score": 0.4047018715153237,
                "support": 15345.0,
            },
            "accuracy": 0.7153699867558665,
            "macro avg": {
                "precision": 0.6133625756048718,
                "recall": 0.726967838781017,
                "f1-score": 0.6088376755405447,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8612461060001994,
                "recall": 0.7153699867558665,
                "f1-score": 0.7597853124729036,
                "support": 117788.0,
            },
        },
        3: {
            "0": {
                "precision": 0.943963094792057,
                "recall": 0.7869937428618842,
                "f1-score": 0.858361147931073,
                "support": 102443.0,
            },
            "1": {
                "precision": 0.3260963557751699,
                "recall": 0.6881068752036494,
                "f1-score": 0.44249345206914614,
                "support": 15345.0,
            },
            "accuracy": 0.7741111148843686,
            "macro avg": {
                "precision": 0.6350297252836135,
                "recall": 0.7375503090327669,
                "f1-score": 0.6504273000001095,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8634696225350008,
                "recall": 0.7741111148843686,
                "f1-score": 0.8041833896449889,
                "support": 117788.0,
            },
        },
        4: {
            "0": {
                "precision": 0.9454464117536259,
                "recall": 0.7839481467743038,
                "f1-score": 0.8571565796986984,
                "support": 102443.0,
            },
            "1": {
                "precision": 0.3261174034831324,
                "recall": 0.6980123818833496,
                "f1-score": 0.4445412853555791,
                "support": 15345.0,
            },
            "accuracy": 0.7727527422148266,
            "macro avg": {
                "precision": 0.6357819076183792,
                "recall": 0.7409802643288267,
                "f1-score": 0.6508489325271387,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.86476244028021,
                "recall": 0.7727527422148266,
                "f1-score": 0.8034025326676327,
                "support": 117788.0,
            },
        },
        5: {
            "0": {
                "precision": 0.9465448384364032,
                "recall": 0.7634782269164315,
                "f1-score": 0.8452124296899062,
                "support": 102443.0,
            },
            "1": {
                "precision": 0.31082541669036917,
                "recall": 0.7121537960247638,
                "f1-score": 0.4327663703146348,
                "support": 15345.0,
            },
            "accuracy": 0.7567918633477094,
            "macro avg": {
                "precision": 0.6286851275633862,
                "recall": 0.7378160114705976,
                "f1-score": 0.6389894000022704,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8637255824282115,
                "recall": 0.7567918633477094,
                "f1-score": 0.7914804299860863,
                "support": 117788.0,
            },
        },
        6: {
            "0": {
                "precision": 0.9461241217798595,
                "recall": 0.7887215329500308,
                "f1-score": 0.8602822569912107,
                "support": 102443.0,
            },
            "1": {
                "precision": 0.3317278004199086,
                "recall": 0.7001629195177582,
                "f1-score": 0.450170741415792,
                "support": 15345.0,
            },
            "accuracy": 0.777184433049207,
            "macro avg": {
                "precision": 0.6389259610998841,
                "recall": 0.7444422262338946,
                "f1-score": 0.6552264992035014,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8660827631417262,
                "recall": 0.777184433049207,
                "f1-score": 0.8068543933166021,
                "support": 117788.0,
            },
        },
        7: {
            "0": {
                "precision": 0.9464291918057276,
                "recall": 0.7977802290053981,
                "f1-score": 0.8657704612385856,
                "support": 102443.0,
            },
            "1": {
                "precision": 0.34098934308891365,
                "recall": 0.6985337243401759,
                "f1-score": 0.4582727661393758,
                "support": 15345.0,
            },
            "accuracy": 0.784850748802934,
            "macro avg": {
                "precision": 0.6437092674473206,
                "recall": 0.748156976672787,
                "f1-score": 0.6620216136889807,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8675546504385296,
                "recall": 0.784850748802934,
                "f1-score": 0.8126831167612418,
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
                "precision": 0.9441316680687677,
                "recall": 0.835110949991414,
                "f1-score": 0.8862812594917485,
                "support": 104822.0,
            },
            "1": {
                "precision": 0.31057040287195853,
                "recall": 0.6004935986426038,
                "f1-score": 0.409401619518351,
                "support": 12966.0,
            },
            "accuracy": 0.8092844771963188,
            "macro avg": {
                "precision": 0.6273510354703631,
                "recall": 0.7178022743170089,
                "f1-score": 0.6478414395050498,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8743897982302287,
                "recall": 0.8092844771963188,
                "f1-score": 0.8337867658939705,
                "support": 117788.0,
            },
        },
        4: {
            "0": {
                "precision": 0.9459512600085203,
                "recall": 0.8261529068325352,
                "f1-score": 0.8820027601097933,
                "support": 104822.0,
            },
            "1": {
                "precision": 0.30555237986357225,
                "recall": 0.618386549436989,
                "f1-score": 0.40900859540388196,
                "support": 12966.0,
            },
            "accuracy": 0.8032821679627806,
            "macro avg": {
                "precision": 0.6257518199360462,
                "recall": 0.7222697281347621,
                "f1-score": 0.6455056777568376,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.875456711497981,
                "recall": 0.8032821679627806,
                "f1-score": 0.8299359762304775,
                "support": 117788.0,
            },
        },
        5: {
            "0": {
                "precision": 0.9456262221405961,
                "recall": 0.8350346301348953,
                "f1-score": 0.886896167388606,
                "support": 104822.0,
            },
            "1": {
                "precision": 0.31448959365708623,
                "recall": 0.6118309424649082,
                "f1-score": 0.41543819224424605,
                "support": 12966.0,
            },
            "accuracy": 0.8104645634529833,
            "macro avg": {
                "precision": 0.6300579078988412,
                "recall": 0.7234327862999017,
                "f1-score": 0.651167179816426,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8761512541904044,
                "recall": 0.8104645634529833,
                "f1-score": 0.8349984859123795,
                "support": 117788.0,
            },
        },
        6: {
            "0": {
                "precision": 0.9448796085668912,
                "recall": 0.8400812806471923,
                "f1-score": 0.8894039935763416,
                "support": 104822.0,
            },
            "1": {
                "precision": 0.31835556278464544,
                "recall": 0.6038099645225976,
                "f1-score": 0.41690185845891686,
                "support": 12966.0,
            },
            "accuracy": 0.814072740856454,
            "macro avg": {
                "precision": 0.6316175856757683,
                "recall": 0.721945622584895,
                "f1-score": 0.6531529260176292,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8759123896854042,
                "recall": 0.814072740856454,
                "f1-score": 0.8373913718837029,
                "support": 117788.0,
            },
        },
        7: {
            "0": {
                "precision": 0.9449936380542233,
                "recall": 0.8289767415237259,
                "f1-score": 0.8831914623300724,
                "support": 104822.0,
            },
            "1": {
                "precision": 0.306096380878653,
                "recall": 0.6099028227672374,
                "f1-score": 0.4076183603515373,
                "support": 12966.0,
            },
            "accuracy": 0.804861276191123,
            "macro avg": {
                "precision": 0.6255450094664381,
                "recall": 0.7194397821454817,
                "f1-score": 0.6454049113408049,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8746643868865455,
                "recall": 0.804861276191123,
                "f1-score": 0.8308407912918199,
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
                "precision": 0.9663143336955448,
                "recall": 0.6327237781946202,
                "f1-score": 0.7647223059827342,
                "support": 108221.0,
            },
            "1": {
                "precision": 0.15300360133824878,
                "recall": 0.7504964983798474,
                "f1-score": 0.25418628526923215,
                "support": 9567.0,
            },
            "accuracy": 0.6422895371345129,
            "macro avg": {
                "precision": 0.5596589675168968,
                "recall": 0.6916101382872338,
                "f1-score": 0.5094542956259831,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.9002554501381175,
                "recall": 0.6422895371345129,
                "f1-score": 0.7232554493405798,
                "support": 117788.0,
            },
        },
        3: {
            "0": {
                "precision": 0.9663143336955448,
                "recall": 0.6327237781946202,
                "f1-score": 0.7647223059827342,
                "support": 108221.0,
            },
            "1": {
                "precision": 0.15300360133824878,
                "recall": 0.7504964983798474,
                "f1-score": 0.25418628526923215,
                "support": 9567.0,
            },
            "accuracy": 0.6422895371345129,
            "macro avg": {
                "precision": 0.5596589675168968,
                "recall": 0.6916101382872338,
                "f1-score": 0.5094542956259831,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.9002554501381175,
                "recall": 0.6422895371345129,
                "f1-score": 0.7232554493405798,
                "support": 117788.0,
            },
        },
        4: {
            "0": {
                "precision": 0.9650044598059985,
                "recall": 0.6398111272303897,
                "f1-score": 0.769459863423958,
                "support": 108221.0,
            },
            "1": {
                "precision": 0.15327135285428795,
                "recall": 0.7375352775164629,
                "f1-score": 0.2537992554358578,
                "support": 9567.0,
            },
            "accuracy": 0.6477484973002343,
            "macro avg": {
                "precision": 0.5591379063301432,
                "recall": 0.6886732023734263,
                "f1-score": 0.5116295594299078,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8990737144481775,
                "recall": 0.6477484973002343,
                "f1-score": 0.7275767765507438,
                "support": 117788.0,
            },
        },
        5: {
            "0": {
                "precision": 0.9660727606587479,
                "recall": 0.6222729414808587,
                "f1-score": 0.7569648567687112,
                "support": 108221.0,
            },
            "1": {
                "precision": 0.14979201331114808,
                "recall": 0.7527960698233511,
                "f1-score": 0.24986556108730723,
                "support": 9567.0,
            },
            "accuracy": 0.6328743165687506,
            "macro avg": {
                "precision": 0.557932386984948,
                "recall": 0.6875345056521049,
                "f1-score": 0.5034152089280092,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8997726459622213,
                "recall": 0.6328743165687506,
                "f1-score": 0.715777138480057,
                "support": 117788.0,
            },
        },
        6: {
            "0": {
                "precision": 0.9622862912578968,
                "recall": 0.6516757376109997,
                "f1-score": 0.7770921712302352,
                "support": 108221.0,
            },
            "1": {
                "precision": 0.15287983999640442,
                "recall": 0.7110902059161701,
                "f1-score": 0.25165538416010064,
                "support": 9567.0,
            },
            "accuracy": 0.6565015111895949,
            "macro avg": {
                "precision": 0.5575830656271505,
                "recall": 0.6813829717635849,
                "f1-score": 0.5143737776951679,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8965445219841277,
                "recall": 0.6565015111895949,
                "f1-score": 0.7344150416253521,
                "support": 117788.0,
            },
        },
        7: {
            "0": {
                "precision": 0.9612023456680193,
                "recall": 0.6982286247585958,
                "f1-score": 0.8088784696575569,
                "support": 108221.0,
            },
            "1": {
                "precision": 0.1663560944479898,
                "recall": 0.681195777150622,
                "f1-score": 0.26740798490008616,
                "support": 9567.0,
            },
            "accuracy": 0.6968451794749889,
            "macro avg": {
                "precision": 0.5637792200580045,
                "recall": 0.6897122009546088,
                "f1-score": 0.5381432272788216,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8966431878130424,
                "recall": 0.6968451794749889,
                "f1-score": 0.7648990479195639,
                "support": 117788.0,
            },
        },
    },
}
