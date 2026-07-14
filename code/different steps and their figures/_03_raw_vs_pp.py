"""
This file compares using raw data (+ resampling to 256 Hz if needed) vs preprocessing the data.

Like the other, specify the path for the two datasets. First raw 10s any overlap that you should normally already have.
And a new preprocessing 10s any overlap one.


"""

RAW = "/Users/telos_matter/Desktop/uni/fha_semesters/bachelor/code/06 - tuar extraction/output/1-10s any overlap no preprocessing/10s"
PP = "/Users/telos_matter/Desktop/uni/fha_semesters/bachelor/code/06 - tuar extraction/output/10s any overlap preprocessing"

from core.hgb_model import report_perf_on_dataset
from core.consts import TARGETS


def compare():

    raw_perf = report_perf_on_dataset(RAW, False)
    pp_perf = report_perf_on_dataset(PP, False)

    # target -> {raw, pp}
    results = {
        target: {"raw": raw_perf[target], "pp": pp_perf[target]} for target in TARGETS
    }

    return results


# print(compare())

# Use 04 for plots
OUTPUT = {
    "artf": {
        "raw": {
            "0": {
                "precision": 0.8878785852777964,
                "recall": 0.7798827826781457,
                "f1-score": 0.8303840526243388,
                "support": 85141.0,
            },
            "1": {
                "precision": 0.5641931958235472,
                "recall": 0.7431616993904494,
                "f1-score": 0.6414276272306676,
                "support": 32647.0,
            },
            "accuracy": 0.769704893537542,
            "macro avg": {
                "precision": 0.7260358905506719,
                "recall": 0.7615222410342976,
                "f1-score": 0.7359058399275031,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.7981635301829405,
                "recall": 0.769704893537542,
                "f1-score": 0.7780114813961391,
                "support": 117788.0,
            },
        },
        "pp": {
            "0": {
                "precision": 0.8866677306945442,
                "recall": 0.7829952666752799,
                "f1-score": 0.8316128911537046,
                "support": 85141.0,
            },
            "1": {
                "precision": 0.5663114407774283,
                "recall": 0.7389959261187858,
                "f1-score": 0.6412311127058167,
                "support": 32647.0,
            },
            "accuracy": 0.7708000815023601,
            "macro avg": {
                "precision": 0.7264895857359863,
                "recall": 0.7609955963970328,
                "f1-score": 0.7364220019297607,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.7978753936404802,
                "recall": 0.7708000815023601,
                "f1-score": 0.7788452584492849,
                "support": 117788.0,
            },
        },
    },
    "musc": {
        "raw": {
            "0": {
                "precision": 0.9557101846491902,
                "recall": 0.7896879240162822,
                "f1-score": 0.864803035971992,
                "support": 102443.0,
            },
            "1": {
                "precision": 0.34989891674964546,
                "recall": 0.7556858911697621,
                "f1-score": 0.4783236398135544,
                "support": 15345.0,
            },
            "accuracy": 0.7852582606037967,
            "macro avg": {
                "precision": 0.6528045506994178,
                "recall": 0.7726869075930222,
                "f1-score": 0.6715633378927732,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8767872561172639,
                "recall": 0.7852582606037967,
                "f1-score": 0.8144538804209068,
                "support": 117788.0,
            },
        },
        "pp": {
            "0": {
                "precision": 0.9570736101070196,
                "recall": 0.7882920258094746,
                "f1-score": 0.8645219997858902,
                "support": 102443.0,
            },
            "1": {
                "precision": 0.3508724671515369,
                "recall": 0.7639622026718801,
                "f1-score": 0.48088440397079335,
                "support": 15345.0,
            },
            "accuracy": 0.7851224233368425,
            "macro avg": {
                "precision": 0.6539730386292782,
                "recall": 0.7761271142406774,
                "f1-score": 0.6727032018783418,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8780998900451128,
                "recall": 0.7851224233368425,
                "f1-score": 0.8145430638350067,
                "support": 117788.0,
            },
        },
    },
    "eyem": {
        "raw": {
            "0": {
                "precision": 0.9410509945674446,
                "recall": 0.8543721737803133,
                "f1-score": 0.8956192590592482,
                "support": 104822.0,
            },
            "1": {
                "precision": 0.32518456301666593,
                "recall": 0.5673299398426654,
                "f1-score": 0.41340939106977265,
                "support": 12966.0,
            },
            "accuracy": 0.8227748157707067,
            "macro avg": {
                "precision": 0.6331177787920552,
                "recall": 0.7108510568114894,
                "f1-score": 0.6545143250645105,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8732569565373618,
                "recall": 0.8227748157707067,
                "f1-score": 0.8425380186243011,
                "support": 117788.0,
            },
        },
        "pp": {
            "0": {
                "precision": 0.9426587676729481,
                "recall": 0.8561466104443723,
                "f1-score": 0.8973223213214414,
                "support": 104822.0,
            },
            "1": {
                "precision": 0.3323740370140795,
                "recall": 0.5789757828165972,
                "f1-score": 0.42231098109810983,
                "support": 12966.0,
            },
            "accuracy": 0.8256358882059293,
            "macro avg": {
                "precision": 0.6375164023435138,
                "recall": 0.7175611966304848,
                "f1-score": 0.6598166512097756,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8754791583942195,
                "recall": 0.8256358882059293,
                "f1-score": 0.845033488525777,
                "support": 117788.0,
            },
        },
    },
    "elec": {
        "raw": {
            "0": {
                "precision": 0.9568933236414899,
                "recall": 0.8028386357546132,
                "f1-score": 0.8731226666800658,
                "support": 108221.0,
            },
            "1": {
                "precision": 0.2094479436828455,
                "recall": 0.5908853350057489,
                "f1-score": 0.3092704543589463,
                "support": 9567.0,
            },
            "accuracy": 0.7856233232587361,
            "macro avg": {
                "precision": 0.5831706336621677,
                "recall": 0.6968619853801811,
                "f1-score": 0.591196560519506,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8961841686336423,
                "recall": 0.7856233232587361,
                "f1-score": 0.827325351883345,
                "support": 117788.0,
            },
        },
        "pp": {
            "0": {
                "precision": 0.9573850686530657,
                "recall": 0.7957050849650252,
                "f1-score": 0.8690895511843606,
                "support": 108221.0,
            },
            "1": {
                "precision": 0.20594045181912868,
                "recall": 0.5993519389568308,
                "f1-score": 0.30654905105586744,
                "support": 9567.0,
            },
            "accuracy": 0.779756851292152,
            "macro avg": {
                "precision": 0.5816627602360972,
                "recall": 0.697528511960928,
                "f1-score": 0.587819301120114,
                "support": 117788.0,
            },
            "weighted avg": {
                "precision": 0.8963510868446448,
                "recall": 0.779756851292152,
                "f1-score": 0.8233987765321948,
                "support": 117788.0,
            },
        },
    },
}
