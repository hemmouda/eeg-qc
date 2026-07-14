"""Evaluates the DT depth = 6 model with full features on each of the targets.
Trains on train + val set and evaluates on test set.
Using the any overlap, no pp, 10s dataset."""

from core.dt_model import report_perf_on_dataset

DATASET = "/Users/telos_matter/Desktop/uni/fha_semesters/bachelor/code/06 - tuar extraction/output/1-10s any overlap no preprocessing/10s"

# print(report_perf_on_dataset(DATASET, 6, True, True))

# Use 13 for plot of this + hgb
OUTPUT = {
    "artf": {
        "0": {
            "precision": 0.8860020220064979,
            "recall": 0.8127588690797767,
            "f1-score": 0.8478014830538323,
            "support": 111060.0,
        },
        "1": {
            "precision": 0.5563354739604446,
            "recall": 0.6918546033430618,
            "f1-score": 0.6167382126512222,
            "support": 37690.0,
        },
        "accuracy": 0.7821243697478991,
        "macro avg": {
            "precision": 0.7211687479834712,
            "recall": 0.7523067362114193,
            "f1-score": 0.7322698478525272,
            "support": 148750.0,
        },
        "weighted avg": {
            "precision": 0.8024717215301567,
            "recall": 0.7821243697478991,
            "f1-score": 0.789255098775013,
            "support": 148750.0,
        },
    },
    "musc": {
        "0": {
            "precision": 0.9273753411064252,
            "recall": 0.8381592223058567,
            "f1-score": 0.8805131440588854,
            "support": 124882.0,
        },
        "1": {
            "precision": 0.43673708265982947,
            "recall": 0.6565694653929948,
            "f1-score": 0.5245523012552301,
            "support": 23868.0,
        },
        "accuracy": 0.8090218487394958,
        "macro avg": {
            "precision": 0.6820562118831273,
            "recall": 0.7473643438494257,
            "f1-score": 0.7025327226570577,
            "support": 148750.0,
        },
        "weighted avg": {
            "precision": 0.84864892797968,
            "recall": 0.8090218487394958,
            "f1-score": 0.8233966842535902,
            "support": 148750.0,
        },
    },
    "eyem": {
        "0": {
            "precision": 0.9830995299922566,
            "recall": 0.7921887583710013,
            "f1-score": 0.8773790725986284,
            "support": 137827.0,
        },
        "1": {
            "precision": 0.24002334960730207,
            "recall": 0.8281607616955049,
            "f1-score": 0.37217913640945466,
            "support": 10923.0,
        },
        "accuracy": 0.7948302521008404,
        "macro avg": {
            "precision": 0.6115614397997793,
            "recall": 0.8101747600332532,
            "f1-score": 0.6247791045040415,
            "support": 148750.0,
        },
        "weighted avg": {
            "precision": 0.9285340098689299,
            "recall": 0.7948302521008404,
            "f1-score": 0.8402812648474058,
            "support": 148750.0,
        },
    },
    "elec": {
        "0": {
            "precision": 0.9522516819210919,
            "recall": 0.6485836646398503,
            "f1-score": 0.7716157527963865,
            "support": 136832.0,
        },
        "1": {
            "precision": 0.13443018378845428,
            "recall": 0.6266152038932706,
            "f1-score": 0.2213691808332469,
            "support": 11918.0,
        },
        "accuracy": 0.6468235294117647,
        "macro avg": {
            "precision": 0.5433409328547731,
            "recall": 0.6375994342665605,
            "f1-score": 0.4964924668148167,
            "support": 148750.0,
        },
        "weighted avg": {
            "precision": 0.8867269987967573,
            "recall": 0.6468235294117647,
            "f1-score": 0.727529442580207,
            "support": 148750.0,
        },
    },
}
