"""Evaluates the DT depth = 6 model with reduced features on each of the targets.
Trains on train + val set and evaluates on test set.
Using the any overlap, no pp, 10s dataset."""

from core.dt_model import report_perf_on_dataset

DATASET = "/Users/telos_matter/Desktop/uni/fha_semesters/bachelor/code/06 - tuar extraction/output/1-10s any overlap no preprocessing/10s"

# print(report_perf_on_dataset(DATASET, 6, False, True))

# Use 15 for plot of this + full DT
OUTPUT = {
    "artf": {
        "0": {
            "precision": 0.89083054960121,
            "recall": 0.7371690977849811,
            "f1-score": 0.8067480279656883,
            "support": 111060.0,
        },
        "1": {
            "precision": 0.4865164388622091,
            "recall": 0.73380206951446,
            "f1-score": 0.5851042448988227,
            "support": 37690.0,
        },
        "accuracy": 0.7363159663865546,
        "macro avg": {
            "precision": 0.6886734942317095,
            "recall": 0.7354855836497205,
            "f1-score": 0.6959261364322555,
            "support": 148750.0,
        },
        "weighted avg": {
            "precision": 0.7883861876936272,
            "recall": 0.7363159663865546,
            "f1-score": 0.7505883359738217,
            "support": 148750.0,
        },
    },
    "musc": {
        "0": {
            "precision": 0.9370851904474591,
            "recall": 0.7710718918659214,
            "f1-score": 0.846011245826744,
            "support": 124882.0,
        },
        "1": {
            "precision": 0.37839189424247693,
            "recall": 0.7291352438411262,
            "f1-score": 0.4982250214715145,
            "support": 23868.0,
        },
        "accuracy": 0.7643428571428571,
        "macro avg": {
            "precision": 0.657738542344968,
            "recall": 0.7501035678535237,
            "f1-score": 0.6721181336491293,
            "support": 148750.0,
        },
        "weighted avg": {
            "precision": 0.8474388604049682,
            "recall": 0.7643428571428571,
            "f1-score": 0.7902064619416307,
            "support": 148750.0,
        },
    },
    "eyem": {
        "0": {
            "precision": 0.9829703309895387,
            "recall": 0.8317238276970405,
            "f1-score": 0.9010442253200077,
            "support": 137827.0,
        },
        "1": {
            "precision": 0.2781512605042017,
            "recall": 0.8181818181818182,
            "f1-score": 0.4151627064316076,
            "support": 10923.0,
        },
        "accuracy": 0.8307294117647058,
        "macro avg": {
            "precision": 0.6305607957468702,
            "recall": 0.8249528229394294,
            "f1-score": 0.6581034658758076,
            "support": 148750.0,
        },
        "weighted avg": {
            "precision": 0.9312141043884541,
            "recall": 0.8307294117647058,
            "f1-score": 0.8653650062892985,
            "support": 148750.0,
        },
    },
    "elec": {
        "0": {
            "precision": 0.9512219381919095,
            "recall": 0.698908150140318,
            "f1-score": 0.805774974828221,
            "support": 136832.0,
        },
        "1": {
            "precision": 0.14547943500715574,
            "recall": 0.5885215640208089,
            "f1-score": 0.23329064875022867,
            "support": 11918.0,
        },
        "accuracy": 0.6900638655462185,
        "macro avg": {
            "precision": 0.5483506865995326,
            "recall": 0.6437148570805635,
            "f1-score": 0.5195328117892248,
            "support": 148750.0,
        },
        "weighted avg": {
            "precision": 0.8866650363232983,
            "recall": 0.6900638655462185,
            "f1-score": 0.7599069533277335,
            "support": 148750.0,
        },
    },
}
