"""
You should use the TUAR feature extractor and extract the features for different epochs lengths
(from 1s to 10s), both for 50% overlap and any overlap. No preprocessing/

Group them by the later, that is a folder with the 50% overlap different duration epochs, and
another folder with the any overlap different epoch durations.

Inside each of those folders, name each folder according to its epoch duration.

Sorry, this is how it ATM. Will make the entire pipeline automatic when I have the time.
Can't promise it'll be soon. I have A LOT going on...

Anyhow, this file "returns" the results of 50% overlap and any overlap for the different epoch durations.
"""

_ANY_OVERLAP_DIR = "/Users/telos_matter/Desktop/uni/fha_semesters/bachelor/code/06 - tuar extraction/output/1-10s any overlap no preprocessing"
_50_OVERLAP_DIR = "/Users/telos_matter/Desktop/uni/fha_semesters/bachelor/code/06 - tuar extraction/output/1-10s 50% overlap no preprocessing"

from pathlib import Path
from core.hgb_model import report_perf_on_dataset
from core.consts import TARGETS


def get_epoch_duration_perf(folder_with_epochs: str):
    # Should contain the different datasets each with with a different epoch.
    # Folders should be called: 1s, 2s, 3s, ..., 10s
    parent_folder = Path(folder_with_epochs)

    # Collect results per epoch folder
    epoch_results = (
        {}
    )  # Maps folder name (epoch duration) to target name to perf report.
    for folder in parent_folder.iterdir():
        if folder.is_dir() and folder.name.endswith("s"):
            epoch_results[folder.name] = report_perf_on_dataset(folder, False)

    # Restructure: target -> {epoch_duration -> perf}, sorted by epoch duration
    results = {}
    for target_name in TARGETS:
        results[target_name] = {
            epoch: epoch_results[epoch][target_name]
            for epoch in sorted(epoch_results, key=lambda e: int(e[:-1]))
        }

    return results


# results = {}
# results["ANY_OVERLAP"] = get_epoch_duration_perf(_ANY_OVERLAP_DIR)
# results["50_OVERLAP"] = get_epoch_duration_perf(_50_OVERLAP_DIR)

# print(results)


# This is the output
# Use file 02 if you want to get the graphs
OUTPUT = {
    "ANY_OVERLAP": {
        "artf": {
            "1s": {
                "0": {
                    "precision": 0.9358510691488475,
                    "recall": 0.7625339103711155,
                    "f1-score": 0.8403491362538537,
                    "support": 957303.0,
                },
                "1": {
                    "precision": 0.4346688617378697,
                    "recall": 0.7774382514244539,
                    "f1-score": 0.5575880153636097,
                    "support": 224823.0,
                },
                "accuracy": 0.7653684970976021,
                "macro avg": {
                    "precision": 0.6852599654433587,
                    "recall": 0.7699860808977848,
                    "f1-score": 0.6989685758087316,
                    "support": 1182126.0,
                },
                "weighted avg": {
                    "precision": 0.8405335755679956,
                    "recall": 0.7653684970976021,
                    "f1-score": 0.786572124766155,
                    "support": 1182126.0,
                },
            },
            "2s": {
                "0": {
                    "precision": 0.9309791699502254,
                    "recall": 0.7783929808750951,
                    "f1-score": 0.8478758132104739,
                    "support": 470486.0,
                },
                "1": {
                    "precision": 0.4716205669805297,
                    "recall": 0.7741444424110336,
                    "f1-score": 0.5861497764061221,
                    "support": 120214.0,
                },
                "accuracy": 0.7775283561875741,
                "macro avg": {
                    "precision": 0.7012998684653775,
                    "recall": 0.7762687116430644,
                    "f1-score": 0.717012794808298,
                    "support": 590700.0,
                },
                "weighted avg": {
                    "precision": 0.8374946006300984,
                    "recall": 0.7775283561875741,
                    "f1-score": 0.7946116625614162,
                    "support": 590700.0,
                },
            },
            "3s": {
                "0": {
                    "precision": 0.9248921447673929,
                    "recall": 0.7820158544526261,
                    "f1-score": 0.8474742882420161,
                    "support": 308683.0,
                },
                "1": {
                    "precision": 0.49298491492984914,
                    "recall": 0.7694551270742923,
                    "f1-score": 0.6009469879628737,
                    "support": 85029.0,
                },
                "accuracy": 0.7793031454464177,
                "macro avg": {
                    "precision": 0.708938529848621,
                    "recall": 0.7757354907634593,
                    "f1-score": 0.7242106381024449,
                    "support": 393712.0,
                },
                "weighted avg": {
                    "precision": 0.8316142161143254,
                    "recall": 0.7793031454464177,
                    "f1-score": 0.7942324012397525,
                    "support": 393712.0,
                },
            },
            "4s": {
                "0": {
                    "precision": 0.9190188016815263,
                    "recall": 0.7859099288912299,
                    "f1-score": 0.8472682618663979,
                    "support": 227820.0,
                },
                "1": {
                    "precision": 0.5137527789685665,
                    "recall": 0.7656068934779379,
                    "f1-score": 0.6148898978027288,
                    "support": 67310.0,
                },
                "accuracy": 0.7812794361806662,
                "macro avg": {
                    "precision": 0.7163857903250463,
                    "recall": 0.7757584111845839,
                    "f1-score": 0.7310790798345634,
                    "support": 295130.0,
                },
                "weighted avg": {
                    "precision": 0.826590190598921,
                    "recall": 0.7812794361806662,
                    "f1-score": 0.7942699638786448,
                    "support": 295130.0,
                },
            },
            "5s": {
                "0": {
                    "precision": 0.9137528835437132,
                    "recall": 0.7819037743904805,
                    "f1-score": 0.842702235927224,
                    "support": 180347.0,
                },
                "1": {
                    "precision": 0.5190388848129127,
                    "recall": 0.7612855784923866,
                    "f1-score": 0.6172448141227451,
                    "support": 55757.0,
                },
                "accuracy": 0.777034696574391,
                "macro avg": {
                    "precision": 0.7163958841783129,
                    "recall": 0.7715946764414336,
                    "f1-score": 0.7299735250249846,
                    "support": 236104.0,
                },
                "weighted avg": {
                    "precision": 0.8205394334232864,
                    "recall": 0.777034696574391,
                    "f1-score": 0.7894594722825915,
                    "support": 236104.0,
                },
            },
            "6s": {
                "0": {
                    "precision": 0.9096987089761736,
                    "recall": 0.783316275687468,
                    "f1-score": 0.8417903103846252,
                    "support": 148516.0,
                },
                "1": {
                    "precision": 0.531633410470244,
                    "recall": 0.7597969881021716,
                    "f1-score": 0.6255597893565098,
                    "support": 48076.0,
                },
                "accuracy": 0.7775647025311304,
                "macro avg": {
                    "precision": 0.7206660597232089,
                    "recall": 0.7715566318948197,
                    "f1-score": 0.7336750498705675,
                    "support": 196592.0,
                },
                "weighted avg": {
                    "precision": 0.8172439433144422,
                    "recall": 0.7775647025311304,
                    "f1-score": 0.7889117673668641,
                    "support": 196592.0,
                },
            },
            "7s": {
                "0": {
                    "precision": 0.9019289639656254,
                    "recall": 0.7801376582152285,
                    "f1-score": 0.8366241152949729,
                    "support": 125383.0,
                },
                "1": {
                    "precision": 0.5397214986976557,
                    "recall": 0.7524266194920974,
                    "f1-score": 0.6285669839479646,
                    "support": 42961.0,
                },
                "accuracy": 0.7730658651332984,
                "macro avg": {
                    "precision": 0.7208252313316406,
                    "recall": 0.7662821388536629,
                    "f1-score": 0.7325955496214688,
                    "support": 168344.0,
                },
                "weighted avg": {
                    "precision": 0.8094944553678897,
                    "recall": 0.7730658651332984,
                    "f1-score": 0.7835284158949419,
                    "support": 168344.0,
                },
            },
            "8s": {
                "0": {
                    "precision": 0.8982655558940634,
                    "recall": 0.785523330056683,
                    "f1-score": 0.838119976475201,
                    "support": 108851.0,
                },
                "1": {
                    "precision": 0.5522868923194937,
                    "recall": 0.7483564171192475,
                    "f1-score": 0.635543099262921,
                    "support": 38483.0,
                },
                "accuracy": 0.7758154940475382,
                "macro avg": {
                    "precision": 0.7252762241067785,
                    "recall": 0.7669398735879652,
                    "f1-score": 0.736831537869061,
                    "support": 147334.0,
                },
                "weighted avg": {
                    "precision": 0.8078974337339364,
                    "recall": 0.7758154940475382,
                    "f1-score": 0.785207777215287,
                    "support": 147334.0,
                },
            },
            "9s": {
                "0": {
                    "precision": 0.8925200143386307,
                    "recall": 0.7819910174939018,
                    "f1-score": 0.8336076871139284,
                    "support": 95519.0,
                },
                "1": {
                    "precision": 0.5582894960122179,
                    "recall": 0.7452923686818632,
                    "f1-score": 0.6383778605124971,
                    "support": 35315.0,
                },
                "accuracy": 0.7720852377822278,
                "macro avg": {
                    "precision": 0.7254047551754244,
                    "recall": 0.7636416930878824,
                    "f1-score": 0.7359927738132128,
                    "support": 130834.0,
                },
                "weighted avg": {
                    "precision": 0.80230378037271,
                    "recall": 0.7720852377822278,
                    "f1-score": 0.7809108244755504,
                    "support": 130834.0,
                },
            },
            "10s": {
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
        },
        "musc": {
            "1s": {
                "0": {
                    "precision": 0.9770552304871355,
                    "recall": 0.777209110038601,
                    "f1-score": 0.8657489166289374,
                    "support": 1076395.0,
                },
                "1": {
                    "precision": 0.26414868547021136,
                    "recall": 0.8141888377107944,
                    "f1-score": 0.3988860752455245,
                    "support": 105731.0,
                },
                "accuracy": 0.7805166285150652,
                "macro avg": {
                    "precision": 0.6206019579786735,
                    "recall": 0.7956989738746978,
                    "f1-score": 0.632317495937231,
                    "support": 1182126.0,
                },
                "weighted avg": {
                    "precision": 0.9132918736950639,
                    "recall": 0.7805166285150652,
                    "f1-score": 0.8239920522318175,
                    "support": 1182126.0,
                },
            },
            "2s": {
                "0": {
                    "precision": 0.9741216484642352,
                    "recall": 0.7840584615327079,
                    "f1-score": 0.8688169280244008,
                    "support": 534779.0,
                },
                "1": {
                    "precision": 0.2794281899128308,
                    "recall": 0.8008082831136782,
                    "f1-score": 0.4142952299892684,
                    "support": 55921.0,
                },
                "accuracy": 0.7856441510072795,
                "macro avg": {
                    "precision": 0.626774919188533,
                    "recall": 0.7924333723231931,
                    "f1-score": 0.6415560790068346,
                    "support": 590700.0,
                },
                "weighted avg": {
                    "precision": 0.9083556879163207,
                    "recall": 0.7856441510072795,
                    "f1-score": 0.8257877966957693,
                    "support": 590700.0,
                },
            },
            "3s": {
                "0": {
                    "precision": 0.9709846863622468,
                    "recall": 0.7891911733562988,
                    "f1-score": 0.8706999811527361,
                    "support": 354155.0,
                },
                "1": {
                    "precision": 0.29476498148567976,
                    "recall": 0.788861642692823,
                    "f1-score": 0.4291677268069949,
                    "support": 39557.0,
                },
                "accuracy": 0.7891580647783151,
                "macro avg": {
                    "precision": 0.6328748339239633,
                    "recall": 0.7890264080245609,
                    "f1-score": 0.6499338539798655,
                    "support": 393712.0,
                },
                "weighted avg": {
                    "precision": 0.9030435952453837,
                    "recall": 0.7891580647783151,
                    "f1-score": 0.8263383884526038,
                    "support": 393712.0,
                },
            },
            "4s": {
                "0": {
                    "precision": 0.9687040635100231,
                    "recall": 0.7910437588040169,
                    "f1-score": 0.8709058125840308,
                    "support": 264084.0,
                },
                "1": {
                    "precision": 0.30570339334918656,
                    "recall": 0.7826128969915609,
                    "f1-score": 0.43966523410992986,
                    "support": 31046.0,
                },
                "accuracy": 0.7901568800189747,
                "macro avg": {
                    "precision": 0.6372037284296048,
                    "recall": 0.7868283278977889,
                    "f1-score": 0.6552855233469803,
                    "support": 295130.0,
                },
                "weighted avg": {
                    "precision": 0.8989601580927042,
                    "recall": 0.7901568800189747,
                    "f1-score": 0.8255417526805748,
                    "support": 295130.0,
                },
            },
            "5s": {
                "0": {
                    "precision": 0.9658570292131439,
                    "recall": 0.7912557856330779,
                    "f1-score": 0.8698814965118347,
                    "support": 210219.0,
                },
                "1": {
                    "precision": 0.3131309969164306,
                    "recall": 0.7728414139463009,
                    "f1-score": 0.4456846232678341,
                    "support": 25885.0,
                },
                "accuracy": 0.7892369464303866,
                "macro avg": {
                    "precision": 0.6394940130647873,
                    "recall": 0.7820485997896893,
                    "f1-score": 0.6577830598898344,
                    "support": 236104.0,
                },
                "weighted avg": {
                    "precision": 0.8942961350902132,
                    "recall": 0.7892369464303866,
                    "f1-score": 0.8233751431085846,
                    "support": 236104.0,
                },
            },
            "6s": {
                "0": {
                    "precision": 0.9652562586999252,
                    "recall": 0.7911411273677934,
                    "f1-score": 0.8695685051278416,
                    "support": 174424.0,
                },
                "1": {
                    "precision": 0.3207286830377953,
                    "recall": 0.7759382894261999,
                    "f1-score": 0.4538582303196612,
                    "support": 22168.0,
                },
                "accuracy": 0.789426833238382,
                "macro avg": {
                    "precision": 0.6429924708688602,
                    "recall": 0.7835397083969966,
                    "f1-score": 0.6617133677237514,
                    "support": 196592.0,
                },
                "weighted avg": {
                    "precision": 0.8925783913539594,
                    "recall": 0.789426833238382,
                    "f1-score": 0.8226924096003139,
                    "support": 196592.0,
                },
            },
            "7s": {
                "0": {
                    "precision": 0.9609838081574094,
                    "recall": 0.790877808514945,
                    "f1-score": 0.8676721505635028,
                    "support": 148210.0,
                },
                "1": {
                    "precision": 0.33157928788630336,
                    "recall": 0.7636336545147512,
                    "f1-score": 0.4623851555568922,
                    "support": 20134.0,
                },
                "accuracy": 0.7876193983747565,
                "macro avg": {
                    "precision": 0.6462815480218563,
                    "recall": 0.7772557315148481,
                    "f1-score": 0.6650286530601974,
                    "support": 168344.0,
                },
                "weighted avg": {
                    "precision": 0.8857068121781144,
                    "recall": 0.7876193983747565,
                    "f1-score": 0.8191996872891175,
                    "support": 168344.0,
                },
            },
            "8s": {
                "0": {
                    "precision": 0.9607406572346542,
                    "recall": 0.7904451263342582,
                    "f1-score": 0.8673126258878405,
                    "support": 129379.0,
                },
                "1": {
                    "precision": 0.33692036783408336,
                    "recall": 0.7672514619883041,
                    "f1-score": 0.4682290161956392,
                    "support": 17955.0,
                },
                "accuracy": 0.787618608060597,
                "macro avg": {
                    "precision": 0.6488305125343687,
                    "recall": 0.7788482941612811,
                    "f1-score": 0.6677708210417399,
                    "support": 147334.0,
                },
                "weighted avg": {
                    "precision": 0.8847181960499497,
                    "recall": 0.787618608060597,
                    "f1-score": 0.8186779169135137,
                    "support": 147334.0,
                },
            },
            "9s": {
                "0": {
                    "precision": 0.9579251271056012,
                    "recall": 0.7902022589965852,
                    "f1-score": 0.866017665996555,
                    "support": 114210.0,
                },
                "1": {
                    "precision": 0.3457032850004096,
                    "recall": 0.7615495668912415,
                    "f1-score": 0.47553760916517984,
                    "support": 16624.0,
                },
                "accuracy": 0.7865615971383585,
                "macro avg": {
                    "precision": 0.6518142060530054,
                    "recall": 0.7758759129439134,
                    "f1-score": 0.6707776375808674,
                    "support": 130834.0,
                },
                "weighted avg": {
                    "precision": 0.8801351344190158,
                    "recall": 0.7865615971383585,
                    "f1-score": 0.8164025776803315,
                    "support": 130834.0,
                },
            },
            "10s": {
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
        },
        "eyem": {
            "1s": {
                "0": {
                    "precision": 0.9645975356450542,
                    "recall": 0.7864918028005199,
                    "f1-score": 0.8664869782827441,
                    "support": 1097082.0,
                },
                "1": {
                    "precision": 0.18558335535374046,
                    "recall": 0.6276280513616481,
                    "f1-score": 0.28646258211326264,
                    "support": 85044.0,
                },
                "accuracy": 0.7750628951566922,
                "macro avg": {
                    "precision": 0.5750904454993974,
                    "recall": 0.707059927081084,
                    "f1-score": 0.5764747801980034,
                    "support": 1182126.0,
                },
                "weighted avg": {
                    "precision": 0.9085540327116153,
                    "recall": 0.7750628951566922,
                    "f1-score": 0.8247591127693916,
                    "support": 1182126.0,
                },
            },
            "2s": {
                "0": {
                    "precision": 0.9611446172403395,
                    "recall": 0.8195902608900089,
                    "f1-score": 0.8847412124865174,
                    "support": 544444.0,
                },
                "1": {
                    "precision": 0.2231651376146789,
                    "recall": 0.6100181598062954,
                    "f1-score": 0.3267823226942141,
                    "support": 46256.0,
                },
                "accuracy": 0.8031792788217369,
                "macro avg": {
                    "precision": 0.5921548774275092,
                    "recall": 0.7148042103481521,
                    "f1-score": 0.6057617675903657,
                    "support": 590700.0,
                },
                "weighted avg": {
                    "precision": 0.9033555892911868,
                    "recall": 0.8031792788217369,
                    "f1-score": 0.8410490736576147,
                    "support": 590700.0,
                },
            },
            "3s": {
                "0": {
                    "precision": 0.9574713601600113,
                    "recall": 0.8287302014465533,
                    "f1-score": 0.8884612467486886,
                    "support": 360443.0,
                },
                "1": {
                    "precision": 0.24470844446619522,
                    "recall": 0.6011902972737383,
                    "f1-score": 0.3478344043198873,
                    "support": 33269.0,
                },
                "accuracy": 0.809502885357825,
                "macro avg": {
                    "precision": 0.6010899023131033,
                    "recall": 0.7149602493601458,
                    "f1-score": 0.618147825534288,
                    "support": 393712.0,
                },
                "weighted avg": {
                    "precision": 0.897242285500825,
                    "recall": 0.809502885357825,
                    "f1-score": 0.8427778171840226,
                    "support": 393712.0,
                },
            },
            "4s": {
                "0": {
                    "precision": 0.9537059733409968,
                    "recall": 0.8403616701175279,
                    "f1-score": 0.8934534296743541,
                    "support": 268532.0,
                },
                "1": {
                    "precision": 0.2673639595296691,
                    "recall": 0.5881645236483947,
                    "f1-score": 0.36761837621901067,
                    "support": 26598.0,
                },
                "accuracy": 0.8176329075322739,
                "macro avg": {
                    "precision": 0.610534966435333,
                    "recall": 0.7142630968829613,
                    "f1-score": 0.6305359029466824,
                    "support": 295130.0,
                },
                "weighted avg": {
                    "precision": 0.8918507743325813,
                    "recall": 0.8176329075322739,
                    "f1-score": 0.8460635989156877,
                    "support": 295130.0,
                },
            },
            "5s": {
                "0": {
                    "precision": 0.9515821799627474,
                    "recall": 0.8422230005837712,
                    "f1-score": 0.8935690537654655,
                    "support": 214125.0,
                },
                "1": {
                    "precision": 0.27481915555841757,
                    "recall": 0.5825105782792666,
                    "f1-score": 0.3734503981565207,
                    "support": 21979.0,
                },
                "accuracy": 0.8180462846881036,
                "macro avg": {
                    "precision": 0.6132006677605825,
                    "recall": 0.7123667894315189,
                    "f1-score": 0.6335097259609931,
                    "support": 236104.0,
                },
                "weighted avg": {
                    "precision": 0.888582084609078,
                    "recall": 0.8180462846881036,
                    "f1-score": 0.8451510348770561,
                    "support": 236104.0,
                },
            },
            "6s": {
                "0": {
                    "precision": 0.9499441016150528,
                    "recall": 0.8472553968182432,
                    "f1-score": 0.8956660264713326,
                    "support": 177512.0,
                },
                "1": {
                    "precision": 0.29148919490971803,
                    "recall": 0.584643605870021,
                    "f1-score": 0.3890216045615442,
                    "support": 19080.0,
                },
                "accuracy": 0.8217679254496623,
                "macro avg": {
                    "precision": 0.6207166482623854,
                    "recall": 0.7159495013441322,
                    "f1-score": 0.6423438155164385,
                    "support": 196592.0,
                },
                "weighted avg": {
                    "precision": 0.8860385529663906,
                    "recall": 0.8217679254496623,
                    "f1-score": 0.8464942617502923,
                    "support": 196592.0,
                },
            },
            "7s": {
                "0": {
                    "precision": 0.9472780756901092,
                    "recall": 0.8531391838056038,
                    "f1-score": 0.8977474972191324,
                    "support": 151361.0,
                },
                "1": {
                    "precision": 0.3058860265417642,
                    "recall": 0.5768121062238709,
                    "f1-score": 0.3997714658831211,
                    "support": 16983.0,
                },
                "accuracy": 0.8252625576201112,
                "macro avg": {
                    "precision": 0.6265820511159368,
                    "recall": 0.7149756450147373,
                    "f1-score": 0.6487594815511267,
                    "support": 168344.0,
                },
                "weighted avg": {
                    "precision": 0.8825727035313964,
                    "recall": 0.8252625576201112,
                    "f1-score": 0.8475103225043846,
                    "support": 168344.0,
                },
            },
            "8s": {
                "0": {
                    "precision": 0.9462084611183464,
                    "recall": 0.8524313440518364,
                    "f1-score": 0.8968752364379209,
                    "support": 132108.0,
                },
                "1": {
                    "precision": 0.3115929234789364,
                    "recall": 0.5795350059109418,
                    "f1-score": 0.4052818922953267,
                    "support": 15226.0,
                },
                "accuracy": 0.8242293021298546,
                "macro avg": {
                    "precision": 0.6289006922986414,
                    "recall": 0.7159831749813891,
                    "f1-score": 0.6510785643666238,
                    "support": 147334.0,
                },
                "weighted avg": {
                    "precision": 0.8806251186712694,
                    "recall": 0.8242293021298546,
                    "f1-score": 0.8460722971441045,
                    "support": 147334.0,
                },
            },
            "9s": {
                "0": {
                    "precision": 0.9424628170656036,
                    "recall": 0.8586969556800466,
                    "f1-score": 0.8986320526381842,
                    "support": 116742.0,
                },
                "1": {
                    "precision": 0.32581330717672063,
                    "recall": 0.5657110417258019,
                    "f1-score": 0.4134854771784232,
                    "support": 14092.0,
                },
                "accuracy": 0.8271397343198251,
                "macro avg": {
                    "precision": 0.6341380621211621,
                    "recall": 0.7122039987029243,
                    "f1-score": 0.6560587649083037,
                    "support": 130834.0,
                },
                "weighted avg": {
                    "precision": 0.8760441117340069,
                    "recall": 0.8271397343198251,
                    "f1-score": 0.846377397568562,
                    "support": 130834.0,
                },
            },
            "10s": {
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
        },
        "elec": {
            "1s": {
                "0": {
                    "precision": 0.9803075005513417,
                    "recall": 0.7911586877674195,
                    "f1-score": 0.8756348846610886,
                    "support": 1129312.0,
                },
                "1": {
                    "precision": 0.12879322382006037,
                    "recall": 0.6601658651115234,
                    "f1-score": 0.21553687945673777,
                    "support": 52814.0,
                },
                "accuracy": 0.7853063040657257,
                "macro avg": {
                    "precision": 0.5545503621857011,
                    "recall": 0.7256622764394715,
                    "f1-score": 0.5455858820589132,
                    "support": 1182126.0,
                },
                "weighted avg": {
                    "precision": 0.9422642843364155,
                    "recall": 0.7853063040657257,
                    "f1-score": 0.8461435985825635,
                    "support": 1182126.0,
                },
            },
            "2s": {
                "0": {
                    "precision": 0.9774063050267361,
                    "recall": 0.8076494966171031,
                    "f1-score": 0.8844560596013574,
                    "support": 561501.0,
                },
                "1": {
                    "precision": 0.14769454155191325,
                    "recall": 0.6409808555087503,
                    "f1-score": 0.2400718317085685,
                    "support": 29199.0,
                },
                "accuracy": 0.7994108684611478,
                "macro avg": {
                    "precision": 0.5625504232893247,
                    "recall": 0.7243151760629267,
                    "f1-score": 0.562263945654963,
                    "support": 590700.0,
                },
                "weighted avg": {
                    "precision": 0.9363926707255656,
                    "recall": 0.7994108684611478,
                    "f1-score": 0.8526033846898261,
                    "support": 590700.0,
                },
            },
            "3s": {
                "0": {
                    "precision": 0.975764689190845,
                    "recall": 0.8119747560936342,
                    "f1-score": 0.8863666320059753,
                    "support": 372684.0,
                },
                "1": {
                    "precision": 0.16165386547986504,
                    "recall": 0.6425718090165493,
                    "f1-score": 0.25832106601410904,
                    "support": 21028.0,
                },
                "accuracy": 0.80292701263868,
                "macro avg": {
                    "precision": 0.568709277335355,
                    "recall": 0.7272732825550918,
                    "f1-score": 0.5723438490100422,
                    "support": 393712.0,
                },
                "weighted avg": {
                    "precision": 0.9322833566406701,
                    "recall": 0.80292701263868,
                    "f1-score": 0.8528229702388029,
                    "support": 393712.0,
                },
            },
            "4s": {
                "0": {
                    "precision": 0.9719856768874312,
                    "recall": 0.8135235874852488,
                    "f1-score": 0.88572295940851,
                    "support": 277944.0,
                },
                "1": {
                    "precision": 0.17070673130770092,
                    "recall": 0.6207959967415339,
                    "f1-score": 0.2677793813139236,
                    "support": 17186.0,
                },
                "accuracy": 0.8023006810558059,
                "macro avg": {
                    "precision": 0.5713462040975661,
                    "recall": 0.7171597921133913,
                    "f1-score": 0.5767511703612168,
                    "support": 295130.0,
                },
                "weighted avg": {
                    "precision": 0.9253256289128665,
                    "recall": 0.8023006810558059,
                    "f1-score": 0.8497388902419273,
                    "support": 295130.0,
                },
            },
            "5s": {
                "0": {
                    "precision": 0.9694237722895527,
                    "recall": 0.8120717207009561,
                    "f1-score": 0.8837986219432811,
                    "support": 221526.0,
                },
                "1": {
                    "precision": 0.17619471653309587,
                    "recall": 0.610783372204692,
                    "f1-score": 0.2734937723649655,
                    "support": 14578.0,
                },
                "accuracy": 0.7996433774946634,
                "macro avg": {
                    "precision": 0.5728092444113243,
                    "recall": 0.7114275464528241,
                    "f1-score": 0.5786461971541232,
                    "support": 236104.0,
                },
                "weighted avg": {
                    "precision": 0.9204466555324557,
                    "recall": 0.7996433774946634,
                    "f1-score": 0.8461159732073312,
                    "support": 236104.0,
                },
            },
            "6s": {
                "0": {
                    "precision": 0.9672509254892333,
                    "recall": 0.8149563437629281,
                    "f1-score": 0.8845966362473449,
                    "support": 183708.0,
                },
                "1": {
                    "precision": 0.1869214762371738,
                    "recall": 0.6065662837628066,
                    "f1-score": 0.285776973287258,
                    "support": 12884.0,
                },
                "accuracy": 0.8012991372995849,
                "macro avg": {
                    "precision": 0.5770862008632036,
                    "recall": 0.7107613137628674,
                    "f1-score": 0.5851868047673014,
                    "support": 196592.0,
                },
                "weighted avg": {
                    "precision": 0.9161106724567419,
                    "recall": 0.8012991372995849,
                    "f1-score": 0.8453519440036231,
                    "support": 196592.0,
                },
            },
            "7s": {
                "0": {
                    "precision": 0.9633873374547005,
                    "recall": 0.8085300789112169,
                    "f1-score": 0.8791918097084276,
                    "support": 156505.0,
                },
                "1": {
                    "precision": 0.1900205427613796,
                    "recall": 0.5938001520398682,
                    "f1-score": 0.28790826251663765,
                    "support": 11839.0,
                },
                "accuracy": 0.7934289312360405,
                "macro avg": {
                    "precision": 0.57670394010804,
                    "recall": 0.7011651154755425,
                    "f1-score": 0.5835500361125326,
                    "support": 168344.0,
                },
                "weighted avg": {
                    "precision": 0.9089993611539459,
                    "recall": 0.7934289312360405,
                    "f1-score": 0.837609062980278,
                    "support": 168344.0,
                },
            },
            "8s": {
                "0": {
                    "precision": 0.9624337524680453,
                    "recall": 0.8138519896307796,
                    "f1-score": 0.8819286286770832,
                    "support": 136558.0,
                },
                "1": {
                    "precision": 0.2020842488542909,
                    "recall": 0.5974387527839644,
                    "f1-score": 0.30201247830370126,
                    "support": 10776.0,
                },
                "accuracy": 0.79802353835503,
                "macro avg": {
                    "precision": 0.5822590006611681,
                    "recall": 0.705645371207372,
                    "f1-score": 0.5919705534903923,
                    "support": 147334.0,
                },
                "weighted avg": {
                    "precision": 0.9068218349816415,
                    "recall": 0.79802353835503,
                    "f1-score": 0.8395135959187005,
                    "support": 147334.0,
                },
            },
            "9s": {
                "0": {
                    "precision": 0.9595304111851808,
                    "recall": 0.8081184478561136,
                    "f1-score": 0.8773396623334592,
                    "support": 120762.0,
                },
                "1": {
                    "precision": 0.20447679209008515,
                    "recall": 0.5913423351866561,
                    "f1-score": 0.30387755102040814,
                    "support": 10072.0,
                },
                "accuracy": 0.7914303621382821,
                "macro avg": {
                    "precision": 0.582003601637633,
                    "recall": 0.6997303915213848,
                    "f1-score": 0.5906086066769336,
                    "support": 130834.0,
                },
                "weighted avg": {
                    "precision": 0.9014040827726443,
                    "recall": 0.7914303621382821,
                    "f1-score": 0.8331928015392845,
                    "support": 130834.0,
                },
            },
            "10s": {
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
        },
    },
    "50_OVERLAP": {
        "artf": {
            "1s": {
                "0": {
                    "precision": 0.946237207485262,
                    "recall": 0.7700852717378526,
                    "f1-score": 0.8491217074597658,
                    "support": 978167.0,
                },
                "1": {
                    "precision": 0.4174534716555931,
                    "recall": 0.7901588064267818,
                    "f1-score": 0.5462921218818536,
                    "support": 203959.0,
                },
                "accuracy": 0.7735486741684051,
                "macro avg": {
                    "precision": 0.6818453395704276,
                    "recall": 0.7801220390823171,
                    "f1-score": 0.6977069146708097,
                    "support": 1182126.0,
                },
                "weighted avg": {
                    "precision": 0.8550031072488375,
                    "recall": 0.7735486741684051,
                    "f1-score": 0.7968727767663496,
                    "support": 1182126.0,
                },
            },
            "2s": {
                "0": {
                    "precision": 0.9473690669926338,
                    "recall": 0.7875662153007338,
                    "f1-score": 0.8601080146322276,
                    "support": 490068.0,
                },
                "1": {
                    "precision": 0.43203107524945855,
                    "recall": 0.786926623737976,
                    "f1-score": 0.5578155102155821,
                    "support": 100632.0,
                },
                "accuracy": 0.7874572541052988,
                "macro avg": {
                    "precision": 0.6897000711210461,
                    "recall": 0.7872464195193549,
                    "f1-score": 0.7089617624239049,
                    "support": 590700.0,
                },
                "weighted avg": {
                    "precision": 0.8595757831174023,
                    "recall": 0.7874572541052988,
                    "f1-score": 0.8086092854897595,
                    "support": 590700.0,
                },
            },
            "3s": {
                "0": {
                    "precision": 0.9486963036050358,
                    "recall": 0.7918824771073165,
                    "f1-score": 0.8632254825189242,
                    "support": 327834.0,
                },
                "1": {
                    "precision": 0.4317506059116993,
                    "recall": 0.786893955493488,
                    "f1-score": 0.5575734760278577,
                    "support": 65878.0,
                },
                "accuracy": 0.7910477709594831,
                "macro avg": {
                    "precision": 0.6902234547583675,
                    "recall": 0.7893882163004022,
                    "f1-score": 0.710399479273391,
                    "support": 393712.0,
                },
                "weighted avg": {
                    "precision": 0.8621981814430452,
                    "recall": 0.7910477709594831,
                    "f1-score": 0.812082152156582,
                    "support": 393712.0,
                },
            },
            "4s": {
                "0": {
                    "precision": 0.9495388401247844,
                    "recall": 0.7943839258214592,
                    "f1-score": 0.8650594008720366,
                    "support": 246756.0,
                },
                "1": {
                    "precision": 0.4279545403296728,
                    "recall": 0.7846570471741018,
                    "f1-score": 0.5538418886975808,
                    "support": 48374.0,
                },
                "accuracy": 0.7927896181343814,
                "macro avg": {
                    "precision": 0.6887466902272286,
                    "recall": 0.7895204864977805,
                    "f1-score": 0.7094506447848087,
                    "support": 295130.0,
                },
                "weighted avg": {
                    "precision": 0.8640472976916576,
                    "recall": 0.7927896181343814,
                    "f1-score": 0.8140485380863925,
                    "support": 295130.0,
                },
            },
            "5s": {
                "0": {
                    "precision": 0.9502203863384556,
                    "recall": 0.795652787721722,
                    "f1-score": 0.8660944111787402,
                    "support": 198334.0,
                },
                "1": {
                    "precision": 0.42127884395704823,
                    "recall": 0.7811225840614244,
                    "f1-score": 0.5473553366356839,
                    "support": 37770.0,
                },
                "accuracy": 0.7933283637718971,
                "macro avg": {
                    "precision": 0.6857496151477519,
                    "recall": 0.7883876858915733,
                    "f1-score": 0.7067248739072121,
                    "support": 236104.0,
                },
                "weighted avg": {
                    "precision": 0.8656046150861865,
                    "recall": 0.7933283637718971,
                    "f1-score": 0.8151051232145751,
                    "support": 236104.0,
                },
            },
            "6s": {
                "0": {
                    "precision": 0.9488580867566792,
                    "recall": 0.7975359020638783,
                    "f1-score": 0.8666410844246784,
                    "support": 165659.0,
                },
                "1": {
                    "precision": 0.4151904031245641,
                    "recall": 0.7697927779394175,
                    "f1-score": 0.5394347850710767,
                    "support": 30933.0,
                },
                "accuracy": 0.7931706274924717,
                "macro avg": {
                    "precision": 0.6820242449406216,
                    "recall": 0.7836643400016479,
                    "f1-score": 0.7030379347478775,
                    "support": 196592.0,
                },
                "weighted avg": {
                    "precision": 0.8648875159410193,
                    "recall": 0.7931706274924717,
                    "f1-score": 0.8151564235132224,
                    "support": 196592.0,
                },
            },
            "7s": {
                "0": {
                    "precision": 0.9505968916585329,
                    "recall": 0.8005690799873538,
                    "f1-score": 0.8691563122125352,
                    "support": 142335.0,
                },
                "1": {
                    "precision": 0.4143956429352423,
                    "recall": 0.7723095851436041,
                    "f1-score": 0.5393786418195,
                    "support": 26009.0,
                },
                "accuracy": 0.7962030128783919,
                "macro avg": {
                    "precision": 0.6824962672968876,
                    "recall": 0.786439332565479,
                    "f1-score": 0.7042674770160176,
                    "support": 168344.0,
                },
                "weighted avg": {
                    "precision": 0.8677542701332985,
                    "recall": 0.7962030128783919,
                    "f1-score": 0.8182059520615796,
                    "support": 168344.0,
                },
            },
            "8s": {
                "0": {
                    "precision": 0.949231441545857,
                    "recall": 0.7999583680266444,
                    "f1-score": 0.8682255415656526,
                    "support": 124904.0,
                },
                "1": {
                    "precision": 0.40611332953032897,
                    "recall": 0.7617476593847525,
                    "f1-score": 0.5297820222628755,
                    "support": 22430.0,
                },
                "accuracy": 0.794141202980982,
                "macro avg": {
                    "precision": 0.6776723855380931,
                    "recall": 0.7808530137056985,
                    "f1-score": 0.699003781914264,
                    "support": 147334.0,
                },
                "weighted avg": {
                    "precision": 0.8665476126095064,
                    "recall": 0.794141202980982,
                    "f1-score": 0.816701194585585,
                    "support": 147334.0,
                },
            },
            "9s": {
                "0": {
                    "precision": 0.9495753715498938,
                    "recall": 0.8032218670306382,
                    "f1-score": 0.8702885719289369,
                    "support": 111364.0,
                },
                "1": {
                    "precision": 0.4018125238849157,
                    "recall": 0.7560349255264509,
                    "f1-score": 0.5247397690004277,
                    "support": 19470.0,
                },
                "accuracy": 0.7961997645871868,
                "macro avg": {
                    "precision": 0.6756939477174048,
                    "recall": 0.7796283962785446,
                    "f1-score": 0.6975141704646823,
                    "support": 130834.0,
                },
                "weighted avg": {
                    "precision": 0.8680603017359529,
                    "recall": 0.7961997645871868,
                    "f1-score": 0.818865889804886,
                    "support": 130834.0,
                },
            },
            "10s": {
                "0": {
                    "precision": 0.9503342660781399,
                    "recall": 0.8076597777092309,
                    "f1-score": 0.8732074659781615,
                    "support": 100499.0,
                },
                "1": {
                    "precision": 0.4029712450196127,
                    "recall": 0.7546416796807218,
                    "f1-score": 0.5253896025450006,
                    "support": 17289.0,
                },
                "accuracy": 0.7998777464597412,
                "macro avg": {
                    "precision": 0.6766527555488763,
                    "recall": 0.7811507286949764,
                    "f1-score": 0.699298534261581,
                    "support": 117788.0,
                },
                "weighted avg": {
                    "precision": 0.8699919623538142,
                    "recall": 0.7998777464597412,
                    "f1-score": 0.8221545315459959,
                    "support": 117788.0,
                },
            },
        },
        "musc": {
            "1s": {
                "0": {
                    "precision": 0.9802183747369587,
                    "recall": 0.7832916861527506,
                    "f1-score": 0.8707598789181384,
                    "support": 1084107.0,
                },
                "1": {
                    "precision": 0.2561040096004965,
                    "recall": 0.8251665493424745,
                    "f1-score": 0.390889144492021,
                    "support": 98019.0,
                },
                "accuracy": 0.7867638475086413,
                "macro avg": {
                    "precision": 0.6181611921687276,
                    "recall": 0.8042291177476125,
                    "f1-score": 0.6308245117050797,
                    "support": 1182126.0,
                },
                "weighted avg": {
                    "precision": 0.9201765805827732,
                    "recall": 0.7867638475086413,
                    "f1-score": 0.8309701699381197,
                    "support": 1182126.0,
                },
            },
            "2s": {
                "0": {
                    "precision": 0.978679940936144,
                    "recall": 0.7972011201422684,
                    "f1-score": 0.8786677917509633,
                    "support": 542074.0,
                },
                "1": {
                    "precision": 0.2629136941479376,
                    "recall": 0.8063998683831695,
                    "f1-score": 0.39654143702280426,
                    "support": 48626.0,
                },
                "accuracy": 0.7979583544946673,
                "macro avg": {
                    "precision": 0.6207968175420407,
                    "recall": 0.8018004942627189,
                    "f1-score": 0.6376046143868838,
                    "support": 590700.0,
                },
                "weighted avg": {
                    "precision": 0.9197585772721464,
                    "recall": 0.7979583544946673,
                    "f1-score": 0.8389794962963985,
                    "support": 590700.0,
                },
            },
            "3s": {
                "0": {
                    "precision": 0.9785685919484138,
                    "recall": 0.8031233772690612,
                    "f1-score": 0.8822078406742944,
                    "support": 362044.0,
                },
                "1": {
                    "precision": 0.26196442253929464,
                    "recall": 0.7989137299482127,
                    "f1-score": 0.39455421611590225,
                    "support": 31668.0,
                },
                "accuracy": 0.8027847766895598,
                "macro avg": {
                    "precision": 0.6202665072438542,
                    "recall": 0.8010185536086369,
                    "f1-score": 0.6383810283950984,
                    "support": 393712.0,
                },
                "weighted avg": {
                    "precision": 0.9209289445999765,
                    "recall": 0.8027847766895598,
                    "f1-score": 0.8429836997222402,
                    "support": 393712.0,
                },
            },
            "4s": {
                "0": {
                    "precision": 0.9781250557011212,
                    "recall": 0.8077595375041859,
                    "f1-score": 0.8848162368614647,
                    "support": 271743.0,
                },
                "1": {
                    "precision": 0.2612913261121638,
                    "recall": 0.7900970624706033,
                    "f1-score": 0.3927102704425907,
                    "support": 23387.0,
                },
                "accuracy": 0.8063599091925592,
                "macro avg": {
                    "precision": 0.6197081909066425,
                    "recall": 0.7989282999873946,
                    "f1-score": 0.6387632536520277,
                    "support": 295130.0,
                },
                "weighted avg": {
                    "precision": 0.9213209678960964,
                    "recall": 0.8063599091925592,
                    "f1-score": 0.8458202614044179,
                    "support": 295130.0,
                },
            },
            "5s": {
                "0": {
                    "precision": 0.9769202332805491,
                    "recall": 0.8130952708970508,
                    "f1-score": 0.8875109962681611,
                    "support": 217758.0,
                },
                "1": {
                    "precision": 0.2581521243825529,
                    "recall": 0.7719938951270031,
                    "f1-score": 0.3869196410277425,
                    "support": 18346.0,
                },
                "accuracy": 0.8099015688001897,
                "macro avg": {
                    "precision": 0.617536178831551,
                    "recall": 0.792544583012027,
                    "f1-score": 0.6372153186479518,
                    "support": 236104.0,
                },
                "weighted avg": {
                    "precision": 0.9210697617686617,
                    "recall": 0.8099015688001897,
                    "f1-score": 0.8486135231070087,
                    "support": 236104.0,
                },
            },
            "6s": {
                "0": {
                    "precision": 0.977662132340832,
                    "recall": 0.8172577092511013,
                    "f1-score": 0.8902925870937748,
                    "support": 181600.0,
                },
                "1": {
                    "precision": 0.25902605666822964,
                    "recall": 0.7738127001067235,
                    "f1-score": 0.38812961073286606,
                    "support": 14992.0,
                },
                "accuracy": 0.8139446162610889,
                "macro avg": {
                    "precision": 0.6183440945045309,
                    "recall": 0.7955352046789124,
                    "f1-score": 0.6392110989133204,
                    "support": 196592.0,
                },
                "weighted avg": {
                    "precision": 0.9228593323973773,
                    "recall": 0.8139446162610889,
                    "f1-score": 0.8519979090722747,
                    "support": 196592.0,
                },
            },
            "7s": {
                "0": {
                    "precision": 0.9781262172256039,
                    "recall": 0.8206973450646912,
                    "f1-score": 0.8925228759991916,
                    "support": 156049.0,
                },
                "1": {
                    "precision": 0.2520916308037743,
                    "recall": 0.767059780398536,
                    "f1-score": 0.3794712911922102,
                    "support": 12295.0,
                },
                "accuracy": 0.8167799268165186,
                "macro avg": {
                    "precision": 0.6151089240146891,
                    "recall": 0.7938785627316136,
                    "f1-score": 0.6359970835957008,
                    "support": 168344.0,
                },
                "weighted avg": {
                    "precision": 0.9251002986300115,
                    "recall": 0.8167799268165186,
                    "f1-score": 0.8550521658152717,
                    "support": 168344.0,
                },
            },
            "8s": {
                "0": {
                    "precision": 0.9771481758555298,
                    "recall": 0.8225882042795003,
                    "f1-score": 0.8932314445204281,
                    "support": 136558.0,
                },
                "1": {
                    "precision": 0.2516987892265876,
                    "recall": 0.7562175204157386,
                    "f1-score": 0.37768817204301075,
                    "support": 10776.0,
                },
                "accuracy": 0.8177338564078895,
                "macro avg": {
                    "precision": 0.6144234825410587,
                    "recall": 0.7894028623476195,
                    "f1-score": 0.6354598082817194,
                    "support": 147334.0,
                },
                "weighted avg": {
                    "precision": 0.9240888508503479,
                    "recall": 0.8177338564078895,
                    "f1-score": 0.8555246402239545,
                    "support": 147334.0,
                },
            },
            "9s": {
                "0": {
                    "precision": 0.9761969664518391,
                    "recall": 0.8255299880624049,
                    "f1-score": 0.8945638161680413,
                    "support": 121465.0,
                },
                "1": {
                    "precision": 0.2462654716175843,
                    "recall": 0.739032981107909,
                    "f1-score": 0.3694277711084434,
                    "support": 9369.0,
                },
                "accuracy": 0.8193359524282678,
                "macro avg": {
                    "precision": 0.6112312190347117,
                    "recall": 0.7822814845851569,
                    "f1-score": 0.6319957936382423,
                    "support": 130834.0,
                },
                "weighted avg": {
                    "precision": 0.9239266989747144,
                    "recall": 0.8193359524282678,
                    "f1-score": 0.8569589152541858,
                    "support": 130834.0,
                },
            },
            "10s": {
                "0": {
                    "precision": 0.9770136079440971,
                    "recall": 0.8261696775668878,
                    "f1-score": 0.8952822754508825,
                    "support": 109325.0,
                },
                "1": {
                    "precision": 0.25009865046168417,
                    "recall": 0.7489070069715231,
                    "f1-score": 0.3749741162549919,
                    "support": 8463.0,
                },
                "accuracy": 0.820618399157809,
                "macro avg": {
                    "precision": 0.6135561292028906,
                    "recall": 0.7875383422692055,
                    "f1-score": 0.6351281958529371,
                    "support": 117788.0,
                },
                "weighted avg": {
                    "precision": 0.9247851866688087,
                    "recall": 0.820618399157809,
                    "f1-score": 0.8578984337074551,
                    "support": 117788.0,
                },
            },
        },
        "eyem": {
            "1s": {
                "0": {
                    "precision": 0.9688975772378832,
                    "recall": 0.8024165461314754,
                    "f1-score": 0.877833510414865,
                    "support": 1106869.0,
                },
                "1": {
                    "precision": 0.17610427772231535,
                    "recall": 0.6211515207887638,
                    "f1-score": 0.2744098948641335,
                    "support": 75257.0,
                },
                "accuracy": 0.790876776248894,
                "macro avg": {
                    "precision": 0.5725009274800993,
                    "recall": 0.7117840334601195,
                    "f1-score": 0.5761217026394992,
                    "support": 1182126.0,
                },
                "weighted avg": {
                    "precision": 0.9184264385084727,
                    "recall": 0.790876776248894,
                    "f1-score": 0.8394181037361342,
                    "support": 1182126.0,
                },
            },
            "2s": {
                "0": {
                    "precision": 0.9687521398735477,
                    "recall": 0.8433475076863048,
                    "f1-score": 0.9017105696458629,
                    "support": 553582.0,
                },
                "1": {
                    "precision": 0.20278730269629247,
                    "recall": 0.5942938735923272,
                    "f1-score": 0.30239141312021495,
                    "support": 37118.0,
                },
                "accuracy": 0.8276976468596581,
                "macro avg": {
                    "precision": 0.5857697212849201,
                    "recall": 0.718820690639316,
                    "f1-score": 0.602050991383039,
                    "support": 590700.0,
                },
                "weighted avg": {
                    "precision": 0.9206209686760779,
                    "recall": 0.8276976468596581,
                    "f1-score": 0.8640509650209789,
                    "support": 590700.0,
                },
            },
            "3s": {
                "0": {
                    "precision": 0.969284162738335,
                    "recall": 0.8588951574479144,
                    "f1-score": 0.9107569092096481,
                    "support": 369392.0,
                },
                "1": {
                    "precision": 0.21488499600837488,
                    "recall": 0.5865953947368421,
                    "f1-score": 0.31454431203077976,
                    "support": 24320.0,
                },
                "accuracy": 0.8420749177063437,
                "macro avg": {
                    "precision": 0.5920845793733549,
                    "recall": 0.7227452760923783,
                    "f1-score": 0.612650610620214,
                    "support": 393712.0,
                },
                "weighted avg": {
                    "precision": 0.922684141060376,
                    "recall": 0.8420749177063437,
                    "f1-score": 0.8739282365672343,
                    "support": 393712.0,
                },
            },
            "4s": {
                "0": {
                    "precision": 0.969824892447756,
                    "recall": 0.8667459416608111,
                    "f1-score": 0.9153927243527535,
                    "support": 277515.0,
                },
                "1": {
                    "precision": 0.21504531850310968,
                    "recall": 0.5751348282713596,
                    "f1-score": 0.31304267218737447,
                    "support": 17615.0,
                },
                "accuracy": 0.8493409683868126,
                "macro avg": {
                    "precision": 0.5924351054754329,
                    "recall": 0.7209403849660854,
                    "f1-score": 0.614217698270064,
                    "support": 295130.0,
                },
                "weighted avg": {
                    "precision": 0.9247754491684046,
                    "recall": 0.8493409683868126,
                    "f1-score": 0.8794411227910921,
                    "support": 295130.0,
                },
            },
            "5s": {
                "0": {
                    "precision": 0.9700836778303676,
                    "recall": 0.8735057566861628,
                    "f1-score": 0.9192650610674924,
                    "support": 222437.0,
                },
                "1": {
                    "precision": 0.21431363788674188,
                    "recall": 0.5615716689836833,
                    "f1-score": 0.3102326239414701,
                    "support": 13667.0,
                },
                "accuracy": 0.855449293531664,
                "macro avg": {
                    "precision": 0.5921986578585547,
                    "recall": 0.7175387128349231,
                    "f1-score": 0.6147488425044813,
                    "support": 236104.0,
                },
                "weighted avg": {
                    "precision": 0.9263355450756937,
                    "recall": 0.855449293531664,
                    "f1-score": 0.8840109090065306,
                    "support": 236104.0,
                },
            },
            "6s": {
                "0": {
                    "precision": 0.9692839054084581,
                    "recall": 0.8798823592898386,
                    "f1-score": 0.9224219863773166,
                    "support": 185310.0,
                },
                "1": {
                    "precision": 0.21551420314372313,
                    "recall": 0.5420138273355788,
                    "f1-score": 0.30840225943110755,
                    "support": 11282.0,
                },
                "accuracy": 0.8604927972654025,
                "macro avg": {
                    "precision": 0.5923990542760906,
                    "recall": 0.7109480933127087,
                    "f1-score": 0.6154121229042121,
                    "support": 196592.0,
                },
                "weighted avg": {
                    "precision": 0.9260266529213237,
                    "recall": 0.8604927972654025,
                    "f1-score": 0.8871846900508785,
                    "support": 196592.0,
                },
            },
            "7s": {
                "0": {
                    "precision": 0.9720235312179429,
                    "recall": 0.8760432455550035,
                    "f1-score": 0.9215409959708631,
                    "support": 158999.0,
                },
                "1": {
                    "precision": 0.2130564983030545,
                    "recall": 0.5710005350454789,
                    "f1-score": 0.31032276824658334,
                    "support": 9345.0,
                },
                "accuracy": 0.8591099177873878,
                "macro avg": {
                    "precision": 0.5925400147604987,
                    "recall": 0.7235218903002412,
                    "f1-score": 0.6159318821087232,
                    "support": 168344.0,
                },
                "weighted avg": {
                    "precision": 0.9298922588079396,
                    "recall": 0.8591099177873878,
                    "f1-score": 0.8876114568243334,
                    "support": 168344.0,
                },
            },
            "8s": {
                "0": {
                    "precision": 0.9718050897612941,
                    "recall": 0.8837476318962053,
                    "f1-score": 0.925686925213378,
                    "support": 139352.0,
                },
                "1": {
                    "precision": 0.213935659178029,
                    "recall": 0.5523678276121273,
                    "f1-score": 0.30841873316778007,
                    "support": 7982.0,
                },
                "accuracy": 0.865794724910747,
                "macro avg": {
                    "precision": 0.5928703744696615,
                    "recall": 0.7180577297541664,
                    "f1-score": 0.617052829190579,
                    "support": 147334.0,
                },
                "weighted avg": {
                    "precision": 0.9307465846306684,
                    "recall": 0.865794724910747,
                    "f1-score": 0.8922456644798883,
                    "support": 147334.0,
                },
            },
            "9s": {
                "0": {
                    "precision": 0.9725359930921459,
                    "recall": 0.8907692556007489,
                    "f1-score": 0.929858554543693,
                    "support": 123912.0,
                },
                "1": {
                    "precision": 0.21943483275663206,
                    "recall": 0.5496966194741404,
                    "f1-score": 0.3136592201797049,
                    "support": 6922.0,
                },
                "accuracy": 0.8727242154180106,
                "macro avg": {
                    "precision": 0.595985412924389,
                    "recall": 0.7202329375374447,
                    "f1-score": 0.6217588873616989,
                    "support": 130834.0,
                },
                "weighted avg": {
                    "precision": 0.9326918682328399,
                    "recall": 0.8727242154180106,
                    "f1-score": 0.8972574585558953,
                    "support": 130834.0,
                },
            },
            "10s": {
                "0": {
                    "precision": 0.9718293246311568,
                    "recall": 0.8860281927430123,
                    "f1-score": 0.9269474888199282,
                    "support": 111589.0,
                },
                "1": {
                    "precision": 0.20765061366893028,
                    "recall": 0.5376673657041459,
                    "f1-score": 0.29959550561797754,
                    "support": 6199.0,
                },
                "accuracy": 0.86769450198662,
                "macro avg": {
                    "precision": 0.5897399691500436,
                    "recall": 0.7118477792235791,
                    "f1-score": 0.6132714972189528,
                    "support": 117788.0,
                },
                "weighted avg": {
                    "precision": 0.9316117826977268,
                    "recall": 0.86769450198662,
                    "f1-score": 0.8939309256397325,
                    "support": 117788.0,
                },
            },
        },
        "elec": {
            "1s": {
                "0": {
                    "precision": 0.9835913968271803,
                    "recall": 0.8013272953716295,
                    "f1-score": 0.8831535433128153,
                    "support": 1136145.0,
                },
                "1": {
                    "precision": 0.12004413014494336,
                    "recall": 0.6696896544224789,
                    "f1-score": 0.2035934478255839,
                    "support": 45981.0,
                },
                "accuracy": 0.7962070033143676,
                "macro avg": {
                    "precision": 0.5518177634860618,
                    "recall": 0.7355084748970542,
                    "f1-score": 0.5433734955691996,
                    "support": 1182126.0,
                },
                "weighted avg": {
                    "precision": 0.9500021120391664,
                    "recall": 0.7962070033143676,
                    "f1-score": 0.8567207833950075,
                    "support": 1182126.0,
                },
            },
            "2s": {
                "0": {
                    "precision": 0.9827791271252222,
                    "recall": 0.8289686974734453,
                    "f1-score": 0.8993449550271193,
                    "support": 568165.0,
                },
                "1": {
                    "precision": 0.12814025265575654,
                    "recall": 0.6337696915908587,
                    "f1-score": 0.21317849706323558,
                    "support": 22535.0,
                },
                "accuracy": 0.8215219231420349,
                "macro avg": {
                    "precision": 0.5554596898904893,
                    "recall": 0.731369194532152,
                    "f1-score": 0.5562617260451774,
                    "support": 590700.0,
                },
                "weighted avg": {
                    "precision": 0.9501749506631104,
                    "recall": 0.8215219231420349,
                    "f1-score": 0.8731679427870378,
                    "support": 590700.0,
                },
            },
            "3s": {
                "0": {
                    "precision": 0.9821976126384673,
                    "recall": 0.8448617722615173,
                    "f1-score": 0.908368083252195,
                    "support": 378759.0,
                },
                "1": {
                    "precision": 0.13477537437603992,
                    "recall": 0.6121179696381996,
                    "f1-score": 0.22091086814857722,
                    "support": 14953.0,
                },
                "accuracy": 0.8360222700857479,
                "macro avg": {
                    "precision": 0.5584864935072535,
                    "recall": 0.7284898709498584,
                    "f1-score": 0.5646394757003861,
                    "support": 393712.0,
                },
                "weighted avg": {
                    "precision": 0.9500129072478821,
                    "recall": 0.8360222700857479,
                    "f1-score": 0.8822587755921685,
                    "support": 393712.0,
                },
            },
            "4s": {
                "0": {
                    "precision": 0.9823379424128551,
                    "recall": 0.8530433650031506,
                    "f1-score": 0.9131365352586084,
                    "support": 284077.0,
                },
                "1": {
                    "precision": 0.13822430485312637,
                    "recall": 0.6058083778159775,
                    "f1-score": 0.22509076240419523,
                    "support": 11053.0,
                },
                "accuracy": 0.8437840951445126,
                "macro avg": {
                    "precision": 0.5602811236329908,
                    "recall": 0.729425871409564,
                    "f1-score": 0.5691136488314018,
                    "support": 295130.0,
                },
                "weighted avg": {
                    "precision": 0.9507247955421619,
                    "recall": 0.8437840951445126,
                    "f1-score": 0.8873683316623633,
                    "support": 295130.0,
                },
            },
            "5s": {
                "0": {
                    "precision": 0.9824762783143944,
                    "recall": 0.8616399755581834,
                    "f1-score": 0.9180992083938357,
                    "support": 227479.0,
                },
                "1": {
                    "precision": 0.14012512635576319,
                    "recall": 0.5946666666666667,
                    "f1-score": 0.22680640311311576,
                    "support": 8625.0,
                },
                "accuracy": 0.8518873038999762,
                "macro avg": {
                    "precision": 0.5613007023350788,
                    "recall": 0.728153321112425,
                    "f1-score": 0.5724528057534758,
                    "support": 236104.0,
                },
                "weighted avg": {
                    "precision": 0.9517047594682791,
                    "recall": 0.8518873038999762,
                    "f1-score": 0.8928459282903805,
                    "support": 236104.0,
                },
            },
            "6s": {
                "0": {
                    "precision": 0.9816605871938976,
                    "recall": 0.8723510965166985,
                    "f1-score": 0.9237834876336151,
                    "support": 189418.0,
                },
                "1": {
                    "precision": 0.14459067430835634,
                    "recall": 0.5696961248954558,
                    "f1-score": 0.23064334085778782,
                    "support": 7174.0,
                },
                "accuracy": 0.8613066655815089,
                "macro avg": {
                    "precision": 0.563125630751127,
                    "recall": 0.7210236107060772,
                    "f1-score": 0.5772134142457015,
                    "support": 196592.0,
                },
                "weighted avg": {
                    "precision": 0.951114382083614,
                    "recall": 0.8613066655815089,
                    "f1-score": 0.8984895417305784,
                    "support": 196592.0,
                },
            },
            "7s": {
                "0": {
                    "precision": 0.9824388563554943,
                    "recall": 0.8782093523097483,
                    "f1-score": 0.927404748200971,
                    "support": 162377.0,
                },
                "1": {
                    "precision": 0.14736569802535138,
                    "recall": 0.5728171610524552,
                    "f1-score": 0.23442268783649395,
                    "support": 5967.0,
                },
                "accuracy": 0.8673846409732453,
                "macro avg": {
                    "precision": 0.5649022771904229,
                    "recall": 0.7255132566811018,
                    "f1-score": 0.5809137180187325,
                    "support": 168344.0,
                },
                "weighted avg": {
                    "precision": 0.952839455511057,
                    "recall": 0.8673846409732453,
                    "f1-score": 0.9028418059268486,
                    "support": 168344.0,
                },
            },
            "8s": {
                "0": {
                    "precision": 0.981165040937047,
                    "recall": 0.881771838165918,
                    "f1-score": 0.9288169786061552,
                    "support": 142022.0,
                },
                "1": {
                    "precision": 0.14762170668561855,
                    "recall": 0.5474397590361446,
                    "f1-score": 0.2325376834192955,
                    "support": 5312.0,
                },
                "accuracy": 0.869717784082425,
                "macro avg": {
                    "precision": 0.5643933738113328,
                    "recall": 0.7146057986010312,
                    "f1-score": 0.5806773310127253,
                    "support": 147334.0,
                },
                "weighted avg": {
                    "precision": 0.9511123566174495,
                    "recall": 0.869717784082425,
                    "f1-score": 0.9037132305504954,
                    "support": 147334.0,
                },
            },
            "9s": {
                "0": {
                    "precision": 0.981309632727748,
                    "recall": 0.8919901470809539,
                    "f1-score": 0.9345205002033009,
                    "support": 126257.0,
                },
                "1": {
                    "precision": 0.15134731470533325,
                    "recall": 0.5313524142451387,
                    "f1-score": 0.23559042913881623,
                    "support": 4577.0,
                },
                "accuracy": 0.8793738630631182,
                "macro avg": {
                    "precision": 0.5663284737165406,
                    "recall": 0.7116712806630463,
                    "f1-score": 0.5850554646710586,
                    "support": 130834.0,
                },
                "weighted avg": {
                    "precision": 0.952274844143828,
                    "recall": 0.8793738630631182,
                    "f1-score": 0.9100696469444985,
                    "support": 130834.0,
                },
            },
            "10s": {
                "0": {
                    "precision": 0.9825175164961177,
                    "recall": 0.8888498158193183,
                    "f1-score": 0.9333394876528964,
                    "support": 113747.0,
                },
                "1": {
                    "precision": 0.15062143097077596,
                    "recall": 0.5548131650581539,
                    "f1-score": 0.2369227517700518,
                    "support": 4041.0,
                },
                "accuracy": 0.8773898869154753,
                "macro avg": {
                    "precision": 0.5665694737334468,
                    "recall": 0.7218314904387362,
                    "f1-score": 0.5851311197114741,
                    "support": 117788.0,
                },
                "weighted avg": {
                    "precision": 0.9539773249519204,
                    "recall": 0.8773898869154753,
                    "f1-score": 0.909447240312738,
                    "support": 117788.0,
                },
            },
        },
    },
}
