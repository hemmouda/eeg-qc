"""Returns the confusion matrix of  the DT depth = 1 model with reduced features on each of the targets.
Trains on train + val set and evaluates the CM on test set.
Using the any overlap, no pp, 10s dataset.
"""

from core.dt_cm_model import report_perf_on_dataset

DATASET = "/Users/telos_matter/Desktop/uni/fha_semesters/bachelor/code/06 - tuar extraction/output/1-10s any overlap no preprocessing/10s"

# results = report_perf_on_dataset(DATASET, 1, False, True)


# def matrix_to_dict(cm):
#     return {
#         "tn": float(cm[0, 0]),
#         "fp": float(cm[0, 1]),
#         "fn": float(cm[1, 0]),
#         "tp": float(cm[1, 1]),
#     }


# results = {target: matrix_to_dict(cm) for target, cm in results.items()}

# print(results)

# Use 19 for plots
OUTPUT = {
    "artf": {
        "tn": 0.8205204394021249,
        "fp": 0.179479560597875,
        "fn": 0.4273812682409127,
        "tp": 0.5726187317590873,
    },
    "musc": {
        "tn": 0.7349738152816259,
        "fp": 0.26502618471837414,
        "fn": 0.38461538461538464,
        "tp": 0.6153846153846154,
    },
    "eyem": {
        "tn": 0.8366575489563003,
        "fp": 0.1633424510436997,
        "fn": 0.18309988098507737,
        "tp": 0.8169001190149227,
    },
    "elec": {
        "tn": 0.7424798292797007,
        "fp": 0.25752017072029937,
        "fn": 0.48078536667226046,
        "tp": 0.5192146333277395,
    },
}
