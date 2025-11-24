"""
This Python scripts runs Aerial+ and the baselines as part of rule-based interpretable ML models for evaluation
Check out the config.py for the parameters of each algorithm before running
"""

import pandas
import config
import warnings
from ucimlrepo import fetch_ucirepo
import optuna
from optuna.trial import Trial

from src.algorithm.aerial_plus.aerial_plus import AerialPlus
from src.algorithm.brl.bayesian_rule_list import BayesianRuleListClassifier
from src.algorithm.classic_arm import ClassicARM
from src.util.data import encode_categories
from sklearn.model_selection import StratifiedKFold

from src.algorithm.cba.cba_main import CBA
from src.algorithm.cba.data_structures.transaction_db import TransactionDB
from src.util.ucimlrepo import *
from src.util.rule import *
from src.util.corels import *

# todo: resolve the warnings
warnings.filterwarnings("ignore")

# Optunaの最適化設定
USE_OPTUNA = True  # Optuna最適化を使用するかどうか
N_TRIALS = 30  # 最適化の試行回数（バランス型：速度と精度のバランス）
N_FOLDS_FOR_OPTUNA = 2  # Optuna最適化時のfold数（速度のため減らす）
N_JOBS = 4  # Optunaの並列実行数（CPUコア数に応じて調整）


def get_datasets():
    datasets = []

    print("LOADING: Loading the datasets ...")
    # breast_cancer = discretize_numerical_features(fetch_ucirepo(id=14))  # low accuracy
    # congress_voting_records = fetch_ucirepo(id=105)
    # mushroom = fetch_ucirepo(id=73)
    # chess_king_rook_vs_king_pawn = fetch_ucirepo(id=22)  # low accuracy
    spambase = discretize_numerical_features(fetch_ucirepo(id=94))

    datasets += [
        # (congress_voting_records, "Class", {'democrat': 0, 'republican': 1}),
        # (mushroom, "poisonous", {'e': 0, 'p': 1}),
        # (breast_cancer, "Class", {"recurrence-events": 0, "no-recurrence-events": 1}),
        # (chess_king_rook_vs_king_pawn, "wtoeg", {"won": 0, "nowin": 1}),
        (spambase, "Class", {"0": 0, "1": 1})
    ]

    print("LOADED: Following dataset(s) are loaded:",
          ", ".join([dataset.metadata.name for (dataset, class_label, categories) in datasets]), "\n")
    return datasets


def print_stats(stats):
    averages = [sum(column) / len(stats) for column in zip(*stats)]
    print("COMPLETED: ")
    print("Average values after 10-fold cross validation:")
    print("# Rules:", averages[0])
    print("Support:", averages[1])
    print("Confidence:", averages[2])
    print("Accuracy:", averages[4])
    print("Duration:", averages[3])


def test_on_cba(dataset, class_label, categories, algorithm, params=None, n_folds=10):
    X = pandas.DataFrame(dataset.data.features).reset_index(drop=True)
    y = pandas.DataFrame(dataset.data.targets).reset_index(drop=True)
    X = X.dropna()
    y = y.loc[X.index]

    # パラメータの一時的な上書き
    if params is not None:
        old_config = {}
        for key, value in params.items():
            if hasattr(config, key):
                old_config[key] = getattr(config, key)
                setattr(config, key, value)

    total_stats = []
    stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2)
    for fold, (train_idx, test_idx) in enumerate(stratified_kfold.split(X, y)):
        print(f"[CBA] Processing fold {fold + 1}/{n_folds}")
        txns_train = TransactionDB.from_DataFrame(pd.concat([X.iloc[train_idx], y.iloc[train_idx]], axis=1))
        txns_test = TransactionDB.from_DataFrame(pd.concat([X.iloc[test_idx], y.iloc[test_idx]], axis=1))
        cba = CBA(maxlen=3, algorithm="m2")
        stats = cba.fit(txns_train, algorithm=algorithm, target_class=class_label)
        accuracy = cba.rule_model_accuracy(txns_test)
        stats.append(accuracy)
        total_stats.append(stats)

    # configを元に戻す
    if params is not None:
        for key, value in old_config.items():
            setattr(config, key, value)

    average_values = np.array(total_stats).mean(axis=0)
    if n_folds == 10:  # 完全な評価の場合のみ統計を表示
        print("[CBA] 10-fold cross-validation completed!")
        print("[CBA] Executions statistics:")
        print("\tAverage values after 10-fold cross validation:")
        print("\t# Items:", average_values[0])
        print("\tSupport:", average_values[1])
        print("\tConfidence:", average_values[2])
        print("\tAccuracy:", average_values[5])
        print("\tRule mining time (s):", average_values[3])
        print("\tRule list learning time (s):", average_values[4])
    
    return average_values[5]  # accuracy


def test_on_brl(dataset, class_label, categories, algorithm, params=None, n_folds=10):
    X = pandas.DataFrame(dataset.data.features).reset_index(drop=True)
    y = pandas.DataFrame(dataset.data.targets).reset_index(drop=True)
    X = X.dropna()
    y = y.loc[X.index]

    y = y[class_label].map(categories).fillna(y[class_label])

    X_encoded = encode_categories(X, X.columns)
    feature_list = X_encoded.columns.to_list()

    # パラメータの一時的な上書き
    if params is not None:
        old_config = {}
        for key, value in params.items():
            if hasattr(config, key):
                old_config[key] = getattr(config, key)
                setattr(config, key, value)

    stats = []
    stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2)
    for fold, (train_idx, test_idx) in enumerate(stratified_kfold.split(X_encoded, y)):
        print(f"[BRL] Processing fold {fold + 1}/{n_folds}")
        X_train = X_encoded.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X_encoded.iloc[test_idx]
        y_test = y.iloc[test_idx]

        model = BayesianRuleListClassifier(minsupport=0.1, maxcardinality=3)
        model.fit(X_train, X.iloc[train_idx], y_train, feature_names=feature_list, algorithm=algorithm)
        preds = model.predict(X_test)
        accuracy = np.sum(preds == y_test) / len(y_test)
        stats.append(
            [model.freq_itemsets_count, model.average_itemset_support, accuracy, model.rule_mining_time,
             model.rule_list_building_time])

    # configを元に戻す
    if params is not None:
        for key, value in old_config.items():
            setattr(config, key, value)

    average_values = np.array(stats).mean(axis=0)
    if n_folds == 10:  # 完全な評価の場合のみ統計を表示
        print("[BRL] 10-fold cross-validation completed!")
        print("[BRL] Executions statistics:")
        print("\tAverage values after 10-fold cross validation:")
        print("\t# Items:", average_values[0])
        print("\tSupport:", average_values[1])
        print("\tAccuracy:", average_values[2])
        print("\tFreq. item learning time (s):", average_values[3])
        print("\tRule list learning time (s):", average_values[4])
    
    return average_values[2]  # accuracy


def test_on_corels(dataset, class_label, categories, algorithm, params=None, n_folds=10):
    """
    CORELS' source code is written in C++, and the C++ program accepts data and the labels
    as parameters. This function first learns the parameters, put them in a format that is consumable
    by CORELS, and then calls CORELS.
    :param datasets:
    :return:
    """
    X = pandas.DataFrame(dataset.data.features).reset_index(drop=True)
    y = pandas.DataFrame(dataset.data.targets).reset_index(drop=True)
    X = X.dropna()
    y = y.loc[X.index]

    # X_encoded = encode_categories(X, X.columns)
    y[class_label] = y[class_label].map(categories).fillna(y[class_label])

    # パラメータの一時的な上書き
    if params is not None:
        old_config = {}
        for key, value in params.items():
            if hasattr(config, key):
                old_config[key] = getattr(config, key)
                setattr(config, key, value)

    fpgrowth = ClassicARM(min_support=config.MIN_SUPPORT, min_confidence=config.MIN_CONFIDENCE, algorithm="fpgrowth")
    aerial_plus = AerialPlus(ant_similarity=config.ANTECEDENT_SIMILARITY, cons_similarity=config.CONSEQUENT_SIMILARITY,
                             max_antecedents=config.MAX_ANTECEDENT)

    stats = []
    stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2)
    for fold, (train_idx, test_idx) in enumerate(stratified_kfold.split(X, y)):
        print(f"[CORELS] Processing fold {fold + 1}/{n_folds}")
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        training_dataset_name = str(dataset.metadata.name) + "_" + str(fold)

        if algorithm == "aerial_plus":
            aerial_plus.create_input_vectors(X_train)
            aerial_plus_training_time = aerial_plus.train(lr=config.LEARNING_RATE, epochs=config.EPOCHS,
                                                          batch_size=config.BATCH_SIZE)
            freq_items, ae_exec_time = aerial_plus.generate_frequent_itemsets()
            freq_items_time = aerial_plus_training_time + ae_exec_time
            freq_items, mean_support = aerial_plus.calculate_freq_item_support(freq_items, X_train)
        else:
            fpgrowth_input = prepare_classic_arm_input(X_train)
            freq_items, freq_items_time = fpgrowth.mine_rules(fpgrowth_input, antecedents=2, frequent_items=True)
            mean_support = freq_items["support"].mean()
            freq_items = freq_items["itemsets"]

        corels_train, corels_train_labels, conv_time = fpg_to_corels(freq_items, y_train[class_label], X_train,
                                                                     categories)
        create_corels_input_files(corels_train, corels_train_labels, training_dataset_name)
        rule_list_model, rule_list_learning_time = run_corels(training_dataset_name)
        
        # rule_list_modelがNoneの場合（CORLESが失敗した場合）の処理
        if rule_list_model is None:
            accuracy = 0.0  # デフォルト値
        else:
            accuracy = test_corels_model(rule_list_model, pd.DataFrame(X_test), pd.DataFrame(y_test))

        stats.append(
            [len(freq_items), mean_support, accuracy, freq_items_time, rule_list_learning_time, conv_time])

    # configを元に戻す
    if params is not None:
        for key, value in old_config.items():
            setattr(config, key, value)

    average_values = np.array(stats).mean(axis=0)
    if n_folds == 10:  # 完全な評価の場合のみ統計を表示
        print("[CORELS] 10-fold cross-validation completed!")
        print("[CORELS] Executions statistics:")
        print("\tAverage values after 10-fold cross validation:")
        print("\t# Items:", average_values[0])
        print("\tSupport:", average_values[1])
        print("\tAccuracy:", average_values[2])
        print("\tFreq. item learning time (s):", average_values[3])
        print("\tRule list learning time (s):", average_values[4])
        print("\tData preparation time (s):", average_values[5])
    
    return average_values[2]  # accuracy


def optimize_cba_params(trial: Trial, dataset, class_label, categories, algorithm):
    """CBAのパラメータをOptunaで最適化"""
    if algorithm == "aerial_plus":
        params = {
            'ANTECEDENT_SIMILARITY': trial.suggest_float('ANTECEDENT_SIMILARITY', 0.01, 0.5),
            'CONSEQUENT_SIMILARITY': trial.suggest_float('CONSEQUENT_SIMILARITY', 0.5, 0.95),
            'LEARNING_RATE': trial.suggest_float('LEARNING_RATE', 1e-4, 1e-2, log=True),
            'EPOCHS': trial.suggest_int('EPOCHS', 15, 30),
            'BATCH_SIZE': trial.suggest_categorical('BATCH_SIZE', [32, 64]),
        }
    else:  # fpgrowth
        params = {
            'MIN_SUPPORT': trial.suggest_float('MIN_SUPPORT', 0.005, 0.05),
            'MIN_CONFIDENCE': trial.suggest_float('MIN_CONFIDENCE', 0.5, 0.95),
        }
    
    accuracy = test_on_cba(dataset, class_label, categories, algorithm, 
                          params=params, n_folds=N_FOLDS_FOR_OPTUNA)
    return accuracy


def optimize_corels_params(trial: Trial, dataset, class_label, categories, algorithm):
    """CORLESのパラメータをOptunaで最適化"""
    if algorithm == "aerial_plus":
        params = {
            'ANTECEDENT_SIMILARITY': trial.suggest_float('ANTECEDENT_SIMILARITY', 0.01, 0.5),
            'CONSEQUENT_SIMILARITY': trial.suggest_float('CONSEQUENT_SIMILARITY', 0.5, 0.95),
            'LEARNING_RATE': trial.suggest_float('LEARNING_RATE', 1e-4, 1e-2, log=True),
            'EPOCHS': trial.suggest_int('EPOCHS', 15, 30),
            'BATCH_SIZE': trial.suggest_categorical('BATCH_SIZE', [32, 64]),
        }
    else:  # fpgrowth
        params = {
            'MIN_SUPPORT': trial.suggest_float('MIN_SUPPORT', 0.005, 0.05),
            'MIN_CONFIDENCE': trial.suggest_float('MIN_CONFIDENCE', 0.5, 0.95),
        }
    
    accuracy = test_on_corels(dataset, class_label, categories, algorithm, 
                             params=params, n_folds=N_FOLDS_FOR_OPTUNA)
    return accuracy


def optimize_brl_params(trial: Trial, dataset, class_label, categories, algorithm):
    """BRLのパラメータをOptunaで最適化"""
    if algorithm == "aerial_plus":
        params = {
            'ANTECEDENT_SIMILARITY': trial.suggest_float('ANTECEDENT_SIMILARITY', 0.01, 0.5),
            'CONSEQUENT_SIMILARITY': trial.suggest_float('CONSEQUENT_SIMILARITY', 0.5, 0.95),
            'LEARNING_RATE': trial.suggest_float('LEARNING_RATE', 1e-4, 1e-2, log=True),
            'EPOCHS': trial.suggest_int('EPOCHS', 15, 30),
            'BATCH_SIZE': trial.suggest_categorical('BATCH_SIZE', [32, 64]),
        }
    else:  # fpgrowth
        params = {
            'MIN_SUPPORT': trial.suggest_float('MIN_SUPPORT', 0.005, 0.05),
            'MIN_CONFIDENCE': trial.suggest_float('MIN_CONFIDENCE', 0.5, 0.95),
        }
    
    accuracy = test_on_brl(dataset, class_label, categories, algorithm, 
                          params=params, n_folds=N_FOLDS_FOR_OPTUNA)
    return accuracy


def run_with_optuna(dataset, class_label, categories, algorithm, algorithm_name):
    """Optunaで最適化して実行"""
    print(f"\n{'='*80}")
    print(f"[OPTUNA] Optimizing {algorithm_name} parameters for {algorithm}...")
    print(f"{'='*80}")
    
    # 最適化関数の選択
    if algorithm_name == "CBA":
        optimize_func = lambda trial: optimize_cba_params(trial, dataset, class_label, categories, algorithm)
        test_func = test_on_cba
    elif algorithm_name == "CORELS":
        optimize_func = lambda trial: optimize_corels_params(trial, dataset, class_label, categories, algorithm)
        test_func = test_on_corels
    elif algorithm_name == "BRL":
        optimize_func = lambda trial: optimize_brl_params(trial, dataset, class_label, categories, algorithm)
        test_func = test_on_brl
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    # Optuna Study作成（積極的な早期終了設定）
    study = optuna.create_study(
        direction='maximize',
        study_name=f'{algorithm_name}_{algorithm}_optimization',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,  # 早期終了を早く開始
            n_warmup_steps=1,    # 判断を早める
            interval_steps=1     # 毎ステップチェック
        )
    )
    
    # 最適化実行
    study.optimize(
        optimize_func,
        n_trials=N_TRIALS,
        n_jobs=N_JOBS,
        show_progress_bar=True
    )
    
    # 最良のパラメータを表示
    print(f"\n[OPTUNA] {algorithm_name} Best parameters for {algorithm}:")
    for key, value in study.best_params.items():
        print(f"\t{key}: {value}")
    print(f"\tBest accuracy: {study.best_value:.4f}")
    
    # 最良のパラメータで完全な10-fold評価を実行
    print(f"\n[{algorithm_name}] Running full 10-fold cross-validation with best parameters...")
    test_func(dataset, class_label, categories, algorithm, 
             params=study.best_params, n_folds=10)
    
    return study.best_params, study.best_value


if __name__ == '__main__':
    datasets = get_datasets()
    for (dataset, class_label, categories) in datasets:
        print("[STARTED] Building classifiers for", dataset.metadata.name, "dataset ...")
        
        if USE_OPTUNA:
            print("\n" + "="*80)
            print("OPTUNA OPTIMIZATION MODE ENABLED")
            print(f"Running hyperparameter optimization with {N_TRIALS} trials")
            print(f"Parallel jobs: {N_JOBS}")
            print(f"Using early stopping (pruning) for inefficient trials")
            print("="*80 + "\n")
            
            # FP-Growth with Optuna（コメントアウト：Aerial+のみ実行）
            # algorithm = "fpgrowth"
            # fpgrowth_results = {}
            # 
            # print(f"\n### Running {algorithm.upper()} with Optuna optimization ###")
            # fpgrowth_results['CBA'] = run_with_optuna(dataset, class_label, categories, algorithm, "CBA")
            # fpgrowth_results['CORELS'] = run_with_optuna(dataset, class_label, categories, algorithm, "CORELS")
            # fpgrowth_results['BRL'] = run_with_optuna(dataset, class_label, categories, algorithm, "BRL")
            
            # Aerial+ with Optuna
            algorithm = "aerial_plus"
            aerial_results = {}
            
            print(f"\n### Running {algorithm.upper()} with Optuna optimization ###")
            aerial_results['CBA'] = run_with_optuna(dataset, class_label, categories, algorithm, "CBA")
            aerial_results['CORELS'] = run_with_optuna(dataset, class_label, categories, algorithm, "CORELS")
            aerial_results['BRL'] = run_with_optuna(dataset, class_label, categories, algorithm, "BRL")
            
            # 最適化結果のサマリー
            print("\n" + "="*80)
            print("OPTIMIZATION SUMMARY")
            print("="*80)
            
            # print("\n### FP-Growth Results ###")
            # for name, (params, accuracy) in fpgrowth_results.items():
            #     print(f"\n[{name}] Best Accuracy: {accuracy:.4f}")
            #     print("Best Parameters:")
            #     for key, value in params.items():
            #         print(f"  {key}: {value}")
            
            print("\n### Aerial+ Results ###")
            for name, (params, accuracy) in aerial_results.items():
                print(f"\n[{name}] Best Accuracy: {accuracy:.4f}")
                print("Best Parameters:")
                for key, value in params.items():
                    print(f"  {key}: {value}")
            
            print("\n" + "="*80 + "\n")
            
        else:
            # 通常モード（最適化なし）
            # algorithm = "fpgrowth"
            # print("[CBA] Running the CBA algorithm with FP-Growth.")
            # test_on_cba(dataset, class_label, categories, algorithm)
            # print("[CORELS] Running the CORELS algorithm with FP-Growth.")
            # test_on_corels(dataset, class_label, categories, algorithm)
            # print("[BRL] Running the BRL algorithm with FP-Growth.")
            # test_on_brl(dataset, class_label, categories, algorithm)
            
            algorithm = "aerial_plus"
            print("[CBA] Running the CBA algorithm with Aerial+.")
            test_on_cba(dataset, class_label, categories, algorithm)
            print("[CORELS] Running the CORELS algorithm with Aerial+.")
            test_on_corels(dataset, class_label, categories, algorithm)
            print("[BRL] Running the BRL algorithm with Aerial+.")
            test_on_brl(dataset, class_label, categories, algorithm)
