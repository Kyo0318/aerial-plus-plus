"""
This script runs rule quality experiments for Aerial+ and the baselines
Check out the config.py for the parameters of each algorithm before running
"""
import csv
import time
import warnings
import config

from datetime import datetime
from niapy.algorithms.basic import SineCosineAlgorithm, GreyWolfOptimizer, BatAlgorithm, FishSchoolSearch

from ucimlrepo import fetch_ucirepo

from src.algorithm.classic_arm import ClassicARM
from src.algorithm.aerial_plus.aerial_plus import AerialPlus
from src.algorithm.arm_ae.armae import ARMAE
from src.algorithm.optimization_arm import OptimizationARM
from src.util.ucimlrepo import *

# todo: resolve the warnings
warnings.filterwarnings("ignore")


def get_datasets():
    print("LOADING: Loading the datasets ...")
    congress_voting_records = fetch_ucirepo(id=105)
    # breast_cancer = discretize_numerical_features(fetch_ucirepo(id=14))  # low accuracy
    # mushroom = fetch_ucirepo(id=73)
    # chess_king_rook_vs_king_pawn = fetch_ucirepo(id=22)  # low accuracy
    # spambase = discretize_numerical_features(fetch_ucirepo(id=94))

    datasets = [congress_voting_records]
    print("LOADED: Following dataset(s) are loaded:", ", ".join([dataset.metadata.name for dataset in datasets]), "\n")
    return datasets


def print_parameters():
    print("----------------------------------------------------")
    print("Algorithm parameters:\n")
    print("Aerial+: Antecedent similarity:", config.ANTECEDENT_SIMILARITY)
    print("Aerial+: Consequent similarity:", config.CONSEQUENT_SIMILARITY)
    print("Exhaustive: Minimum support:", config.MIN_SUPPORT)
    print("Exhaustive: Minimum confidence:", config.MIN_CONFIDENCE)
    print("Optimization-based: Population size:", config.POPULATION_SIZE)
    print("Optimization-based: Maximum evaluations:", config.SIMILARITY_THRESHOLD)
    print("ARM-AE: Similarity threshold:", config.SIMILARITY_THRESHOLD)
    print("Generic: Number of antecedents:", config.MAX_ANTECEDENT)
    print("Generic: Number of bins for discretization:", config.NUM_BINS)
    print("Generic DL: Epochs", config.EPOCHS)
    print("Generic DL: Learning rate", config.LEARNING_RATE)
    print("Generic DL: Batch size", config.BATCH_SIZE)
    print("----------------------------------------------------\n")


def print_stats(stats, algorithm):
    print("COMPLETED: Rule mining with", algorithm, "algorithm completed.")
    print("STATS: Printing rule mining statistics ...")
    print("Rule Count, Rule Mining Time (s), Support, Confidence, Data Coverage")
    print(", ".join(f"{num:.2f}" for num in stats), "\n")


def save_results(result_list):
    timestamp = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    for dataset, results in result_list.items():
        print("\nRESULTS: Rule quality evaluation results for the dataset", dataset)
        with open(dataset + "_" + timestamp + '.csv', 'w+', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Dataset: " + dataset])
            columns = ["Algorithm", "Rule Count", "Rule Mining Time (s)", "Support",
                       "Confidence", "Data Coverage", "Zhang"]
            writer.writerow(columns)
            print(columns)
            for algorithm in results:
                if len(results[algorithm]['stats']) > 0:
                    alg_stats = [round(float_value, 2) for float_value in
                                 list(pd.DataFrame(results[algorithm]["stats"]).mean())]
                    row = [algorithm] + alg_stats
                    print(row)
                else:
                    row = [algorithm, "No rules found!"]
                writer.writerow(row)
        print("\nSAVED: The results are saved into '", dataset + "_" + timestamp + ".csv' file.")


if __name__ == "__main__":
    print_parameters()

    dataset_list = get_datasets()
    results = {}

    fpgrowth = ClassicARM(min_support=config.MIN_SUPPORT, min_confidence=config.MIN_CONFIDENCE,
                          algorithm="fpgrowth")
    hmine = ClassicARM(min_support=config.MIN_SUPPORT, min_confidence=config.MIN_CONFIDENCE, algorithm="hmine")

    aerial_plus = AerialPlus(max_antecedents=config.MAX_ANTECEDENT, ant_similarity=config.ANTECEDENT_SIMILARITY,
                             cons_similarity=config.CONSEQUENT_SIMILARITY)
    sc = OptimizationARM(SineCosineAlgorithm(config.POPULATION_SIZE), max_evals=config.MAX_EVALS)
    gwo = OptimizationARM(GreyWolfOptimizer(config.POPULATION_SIZE), max_evals=config.MAX_EVALS)
    bat = OptimizationARM(BatAlgorithm(config.POPULATION_SIZE), max_evals=config.MAX_EVALS)
    fss = OptimizationARM(FishSchoolSearch(config.POPULATION_SIZE), max_evals=config.MAX_EVALS)

    for dataset in dataset_list:
        print("MINING: Mining rules from dataset:", dataset.metadata.name, "...")
        classical_arm_input = prepare_classic_arm_input(dataset.data.features)
        optimization_based_arm_input = prepare_opt_arm_input(dataset)

        results[dataset.metadata.name] = {}
        results[dataset.metadata.name]["bat"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["gwo"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["sc"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["fss"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["fpgrowth"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["hmine"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["arm-ae"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["aerial_plus"] = {"stats": [], "rules": None}

        # exhaustive ARM
        fpgrowth_stats, fpgrowth_rules = fpgrowth.mine_rules(classical_arm_input)
        if fpgrowth_stats:
            results[dataset.metadata.name]["fpgrowth"]["stats"].append(fpgrowth_stats)
            results[dataset.metadata.name]["fpgrowth"]["rules"] = fpgrowth_rules
            print_stats(fpgrowth_stats, "FP-Growth")
        hmine_stats, hmine_rules = hmine.mine_rules(classical_arm_input)
        if hmine_stats:
            results[dataset.metadata.name]["hmine"]["stats"].append(hmine_stats)
            results[dataset.metadata.name]["hmine"]["rules"] = hmine_rules
            print_stats(hmine_stats, "HMine")

        # optimization-based ARM
        sc_stats, sc_rules = sc.learn_rules(optimization_based_arm_input)
        results[dataset.metadata.name]["sc"]["stats"].append(sc_stats)
        results[dataset.metadata.name]["sc"]["rules"] = sc_rules
        print_stats(sc_stats, "Sine Cosine Algorithm")

        gwo_stats, gwo_rules = gwo.learn_rules(optimization_based_arm_input)
        results[dataset.metadata.name]["gwo"]["stats"].append(gwo_stats)
        results[dataset.metadata.name]["gwo"]["rules"] = gwo_rules
        print_stats(gwo_stats, "Grey Wolf Optimizer")

        bat_stats, bat_rules = bat.learn_rules(optimization_based_arm_input)
        results[dataset.metadata.name]["bat"]["stats"].append(bat_stats)
        results[dataset.metadata.name]["bat"]["rules"] = bat_rules
        print_stats(bat_stats, "Bat Algorithm")

        fss_stats, fss_rules = fss.learn_rules(optimization_based_arm_input)
        results[dataset.metadata.name]["fss"]["stats"].append(fss_stats)
        results[dataset.metadata.name]["fss"]["rules"] = fss_rules
        print_stats(fss_stats, "Fish School Search Algorithm")

        # aerial_plus+ (2025)
        print("[計測] Aerial+ 実行開始...")
        aerial_plus_overall_start = time.time()
        
        # Phase 1: データ前処理
        create_vectors_start = time.time()
        aerial_plus.create_input_vectors(dataset.data.features)
        create_vectors_time = time.time() - create_vectors_start
        print(f"[計測] Phase 1 - create_input_vectors: {create_vectors_time:.4f}秒")
        
        # Phase 2: Autoencoder訓練
        train_start = time.time()
        aerial_plus_training_time = aerial_plus.train(lr=config.LEARNING_RATE, epochs=config.EPOCHS,
                                                      batch_size=config.BATCH_SIZE)
        actual_train_time = time.time() - train_start
        print(f"[計測] Phase 2 - train: {actual_train_time:.4f}秒 (内部計測: {aerial_plus_training_time:.4f}秒)")
        
        # Phase 3: ルール生成
        generate_start = time.time()
        aerial_plus_association_rules, ae_exec_time = aerial_plus.generate_rules()
        actual_generate_time = time.time() - generate_start
        print(f"[計測] Phase 3 - generate_rules: {actual_generate_time:.4f}秒 (内部計測: {ae_exec_time:.4f}秒) | 生成ルール数: {len(aerial_plus_association_rules)}")
        
        # Phase 4: 統計計算
        stats_start = time.time()
        aerial_plus_stats, aerial_plus_rules = aerial_plus.calculate_stats(aerial_plus_association_rules,
                                                                           classical_arm_input,
                                                                           aerial_plus_training_time + ae_exec_time)
        actual_stats_time = time.time() - stats_start
        print(f"[計測] Phase 4 - calculate_stats: {actual_stats_time:.4f}秒 (calculate_stats内で詳細計測)")
        
        # 全体時間
        aerial_plus_overall_time = time.time() - aerial_plus_overall_start
        print(f"[計測] ★ Aerial+ 全体実行時間: {aerial_plus_overall_time:.4f}秒")
        print(f"[計測] 内訳 - Phase1: {create_vectors_time:.4f}s | Phase2: {actual_train_time:.4f}s | Phase3: {actual_generate_time:.4f}s | Phase4: {actual_stats_time:.4f}s")
        print(f"[計測] 合計: {create_vectors_time + actual_train_time + actual_generate_time + actual_stats_time:.4f}秒")
        
        if aerial_plus_stats:
            results[dataset.metadata.name]["aerial_plus"]["stats"].append(aerial_plus_stats)
            results[dataset.metadata.name]["aerial_plus"]["rules"] = aerial_plus_association_rules

        # ARM-AE from Berteloot et al. (2024)
        one_hot_encoded = one_hot_encoding(classical_arm_input)
        arm_ae = ARMAE(len(one_hot_encoded.loc[0]), maxEpoch=config.EPOCHS, batchSize=config.BATCH_SIZE,
                       learningRate=config.LEARNING_RATE, likeness=config.SIMILARITY_THRESHOLD)
        dataLoader = arm_ae.dataPreprocessing(one_hot_encoded)
        arm_ae_training_time = arm_ae.train(dataLoader)
        # numberOfRules per consequent is adjusted to approximate aerial_plus+
        arm_ae.generateRules(one_hot_encoded,
                             numberOfRules=max(int(len(aerial_plus_association_rules) / one_hot_encoded.shape[1]), 2),
                             nbAntecedent=config.MAX_ANTECEDENT)
        arm_ae_stats, arm_ae_rules = arm_ae.reformat_rules(classical_arm_input, list(one_hot_encoded.columns))
        if arm_ae_stats:
            results[dataset.metadata.name]["arm-ae"]["stats"].append(arm_ae_stats)
            results[dataset.metadata.name]["arm-ae"]["rules"] = arm_ae_rules

    save_results(results)
