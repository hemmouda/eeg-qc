"""Evaluates the DT depth = 1 model with reduced features on each of the targets.
Trains on train + val set and evaluates on test set.
Using the any overlap, no pp, 10s dataset."""

from core.dt_model import report_perf_on_dataset

DATASET = "/Users/telos_matter/Desktop/uni/fha_semesters/bachelor/code/06 - tuar extraction/output/1-10s any overlap no preprocessing/10s"

# print(report_perf_on_dataset(DATASET, 1, False, True))

# Use 17 for plot of this + full DT with depth = 6
OUTPUT = {
    "artf": {
        "0": {
            "precision": 0.8497878491164266,
            "recall": 0.8205204394021249,
            "f1-score": 0.8348977301358254,
            "support": 111060.0,
        },
        "1": {
            "precision": 0.5198602914609177,
            "recall": 0.5726187317590873,
            "f1-score": 0.544965595606338,
            "support": 37690.0,
        },
        "accuracy": 0.75770756302521,
        "macro avg": {
            "precision": 0.6848240702886721,
            "recall": 0.6965695855806061,
            "f1-score": 0.6899316628710817,
            "support": 148750.0,
        },
        "weighted avg": {
            "precision": 0.7661914145077803,
            "recall": 0.75770756302521,
            "f1-score": 0.761435261897732,
            "support": 148750.0,
        },
    },
    "musc": {
        "0": {
            "precision": 0.9090774030604665,
            "recall": 0.7349738152816259,
            "f1-score": 0.8128069002466272,
            "support": 124882.0,
        },
        "1": {
            "precision": 0.30737679188029715,
            "recall": 0.6153846153846154,
            "f1-score": 0.4099758558608851,
            "support": 23868.0,
        },
        "accuracy": 0.7157848739495798,
        "macro avg": {
            "precision": 0.6082270974703818,
            "recall": 0.6751792153331206,
            "f1-score": 0.6113913780537561,
            "support": 148750.0,
        },
        "weighted avg": {
            "precision": 0.8125302421350998,
            "recall": 0.7157848739495798,
            "f1-score": 0.7481697818103321,
            "support": 148750.0,
        },
    },
    "eyem": {
        "0": {
            "precision": 0.9829517363656511,
            "recall": 0.8366575489563003,
            "f1-score": 0.9039237127705857,
            "support": 137827.0,
        },
        "1": {
            "precision": 0.2838465453620053,
            "recall": 0.8169001190149227,
            "f1-score": 0.42130361906560587,
            "support": 10923.0,
        },
        "accuracy": 0.8352067226890756,
        "macro avg": {
            "precision": 0.6333991408638282,
            "recall": 0.8267788339856115,
            "f1-score": 0.6626136659180958,
            "support": 148750.0,
        },
        "weighted avg": {
            "precision": 0.9316150909785399,
            "recall": 0.8352067226890756,
            "f1-score": 0.8684839864946899,
            "support": 148750.0,
        },
    },
    "elec": {
        "0": {
            "precision": 0.9466107617051013,
            "recall": 0.7424798292797007,
            "f1-score": 0.8322104219825768,
            "support": 136832.0,
        },
        "1": {
            "precision": 0.14937839468919734,
            "recall": 0.5192146333277395,
            "f1-score": 0.23200794855932363,
            "support": 11918.0,
        },
        "accuracy": 0.7245915966386555,
        "macro avg": {
            "precision": 0.5479945781971494,
            "recall": 0.6308472313037201,
            "f1-score": 0.5321091852709502,
            "support": 148750.0,
        },
        "weighted avg": {
            "precision": 0.8827357005279884,
            "recall": 0.7245915966386555,
            "f1-score": 0.7841215945657141,
            "support": 148750.0,
        },
    },
}
