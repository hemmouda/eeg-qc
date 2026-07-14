"""Evaluates the HGB model with full features on each of the targets.
Trains on train + val set and evaluates on test set.
Using the any overlap, no pp, 10s dataset."""

from core.hgb_model import report_perf_on_dataset

DATASET = "/Users/telos_matter/Desktop/uni/fha_semesters/bachelor/code/06 - tuar extraction/output/1-10s any overlap no preprocessing/10s"

# print(report_perf_on_dataset(DATASET, True))

# Use 06 for plots
OUTPUT = {
    "artf": {
        "0": {
            "precision": 0.9128783800976717,
            "recall": 0.8398793444984693,
            "f1-score": 0.8748587265931654,
            "support": 111060.0,
        },
        "1": {
            "precision": 0.618152927787679,
            "recall": 0.7638100291854604,
            "f1-score": 0.6833054437996225,
            "support": 37690.0,
        },
        "accuracy": 0.8206050420168067,
        "macro avg": {
            "precision": 0.7655156539426754,
            "recall": 0.8018446868419649,
            "f1-score": 0.779082085196394,
            "support": 148750.0,
        },
        "weighted avg": {
            "precision": 0.8382013898619499,
            "recall": 0.8206050420168067,
            "f1-score": 0.826323309931057,
            "support": 148750.0,
        },
    },
    "musc": {
        "0": {
            "precision": 0.949955417817291,
            "recall": 0.8787175093288064,
            "f1-score": 0.9129488891384739,
            "support": 124882.0,
        },
        "1": {
            "precision": 0.5442481870429994,
            "recall": 0.7577928607340372,
            "f1-score": 0.6335090453757377,
            "support": 23868.0,
        },
        "accuracy": 0.8593142857142857,
        "macro avg": {
            "precision": 0.7471018024301452,
            "recall": 0.8182551850314218,
            "f1-score": 0.7732289672571058,
            "support": 148750.0,
        },
        "weighted avg": {
            "precision": 0.8848567947307646,
            "recall": 0.8593142857142857,
            "f1-score": 0.8681107702078589,
            "support": 148750.0,
        },
    },
    "eyem": {
        "0": {
            "precision": 0.9825174536972164,
            "recall": 0.864852314858482,
            "f1-score": 0.9199376418108571,
            "support": 137827.0,
        },
        "1": {
            "precision": 0.32090123591818875,
            "recall": 0.8058225762153255,
            "f1-score": 0.4590112640801001,
            "support": 10923.0,
        },
        "accuracy": 0.8605176470588235,
        "macro avg": {
            "precision": 0.6517093448077026,
            "recall": 0.8353374455369038,
            "f1-score": 0.6894744529454786,
            "support": 148750.0,
        },
        "weighted avg": {
            "precision": 0.9339336960716682,
            "recall": 0.8605176470588235,
            "f1-score": 0.8860909270279794,
            "support": 148750.0,
        },
    },
    "elec": {
        "0": {
            "precision": 0.9485470290251917,
            "recall": 0.8101248246024322,
            "f1-score": 0.8738884334006054,
            "support": 136832.0,
        },
        "1": {
            "precision": 0.18519099291224989,
            "recall": 0.49546903842926665,
            "f1-score": 0.2696100812711168,
            "support": 11918.0,
        },
        "accuracy": 0.7849142857142857,
        "macro avg": {
            "precision": 0.5668690109687208,
            "recall": 0.6527969315158494,
            "f1-score": 0.5717492573358611,
            "support": 148750.0,
        },
        "weighted avg": {
            "precision": 0.8873861736410301,
            "recall": 0.7849142857142857,
            "f1-score": 0.8254730424716693,
            "support": 148750.0,
        },
    },
}
