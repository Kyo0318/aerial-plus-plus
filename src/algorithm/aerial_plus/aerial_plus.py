import concurrent
import time
import torch
import os
import json
import matplotlib.pyplot as plt

from itertools import combinations
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src.algorithm.aerial_plus.autoencoder import AutoEncoder
from src.util.ucimlrepo import *
from src.util.rule_quality import *
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed

import numpy as np


class AerialPlus:
    """
    Neurosymbolic association rule mining from tabular data
    """

    def __init__(self, noise_factor=0.5, cons_similarity=0.8, ant_similarity=0.5, max_antecedents=2):
        """
        @param cons_similarity: consequent similarity threshold
        @param ant_similarity: antecedent similarity threshold
        @param noise_factor: amount of noise introduced for the one-hot encoded input of denoising Autoencoder
        @param max_antecedents: maximum number of antecedents that the learned rules will have
        """
        self.noise_factor = noise_factor
        self.cons_similarity = cons_similarity
        self.ant_similarity = ant_similarity
        self.max_antecedents = max_antecedents

        self.model = None
        self.input_vectors = None
        self.softmax = nn.Softmax(dim=0)
        
        # 訓練履歴
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epochs_completed': 0,
            'early_stopped': False,
            'best_epoch': 0
        }

    def create_input_vectors(self, transactions):
        """
        Create input vectors for training the Autoencoder in a one-hot encoded form.
        :param transactions: pandas DataFrame of transactions
        :return:
        """
        columns = transactions.columns.tolist()

        # Get input vectors in the form of one-hot encoded vectors
        unique_values, value_count = get_unique_values_per_column(transactions)
        feature_value_indices = []
        vector_tracker = []
        start = 0

        # Track what each value in the input vector corresponds to
        # Track where do values for each feature start and end in the input feature
        for feature, values in unique_values.items():
            end = start + len(values)
            feature_value_indices.append({'feature': feature, 'start': start, 'end': end})
            vector_tracker.extend([f"{feature}__{value}" for value in values])
            start = end

        # Map tracker entries to indices for fast lookup
        tracker_index_map = {key: idx for idx, key in enumerate(vector_tracker)}

        # Preallocate vector list
        vector_list = np.zeros((len(transactions), value_count), dtype=int)

        # Function to process each transaction
        def process_transaction(transaction_idx, transaction):
            transaction_vector = np.zeros(value_count, dtype=int)
            for col_idx, value in enumerate(transaction):
                key = f"{columns[col_idx]}__{value}"
                transaction_vector[tracker_index_map[key]] = 1
            return transaction_idx, transaction_vector

        # Parallelize transaction processing
        # NOTE: Preparing the input data for each of the algorithms is not included in the execution time calculation
        # Therefore, we preprocess data in parallel where possible for each of the algorithm
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(process_transaction, transaction_idx, transaction)
                for transaction_idx, transaction in enumerate(transactions.itertuples(index=False))
            ]

            for future in concurrent.futures.as_completed(futures):
                transaction_idx, transaction_vector = future.result()
                vector_list[transaction_idx] = transaction_vector

        self.input_vectors = {
            "vector_list": vector_list.tolist(),
            "vector_tracker_list": vector_tracker,
            "feature_value_indices": feature_value_indices,
        }

    def generate_rules(self, target_class=None):
        """
        generate rules using Aerial+ algorithm
        @param target_class: if given a target class, generate rules with the target class on the right hand side only
        """
        association_rules = []
        input_vector_size = len(self.input_vectors['vector_tracker_list'])

        start = time.time()

        # Precompute feature combinations
        feature_value_indices = self.input_vectors['feature_value_indices']
        target_range = range(input_vector_size)

        # If target_class is specified, narrow the target range and features
        # this is to do "constraint-based rule mining"
        if target_class:
            for feature in feature_value_indices:
                if feature["feature"] == target_class:
                    target_range = range(feature["start"], feature["end"])
                    break

        low_support_antecedents = np.array([])

        # Initialize input vectors
        unmarked_features = self.initialize_input_vectors(input_vector_size, feature_value_indices)

        # Precompute target indices for softmax to speed things up
        feature_value_indices = [(cat['start'], cat['end']) for cat in feature_value_indices]
        softmax_ranges = feature_value_indices

        for r in range(1, self.max_antecedents + 1):
            if r == 2:
                softmax_ranges = [
                    (start, end) for (start, end) in softmax_ranges
                    if not all(idx in low_support_antecedents for idx in range(start, end))
                ]

            feature_combinations = list(combinations(softmax_ranges, r))  # Generate combinations

            # Vectorized model evaluation batch
            batch_vectors = []
            batch_candidate_antecedent_list = []

            for category_list in feature_combinations:
                test_vectors, candidate_antecedent_list = self.mark_features(unmarked_features, list(category_list),
                                                                             low_support_antecedents)
                if len(test_vectors) > 0:
                    batch_vectors.extend(test_vectors)
                    batch_candidate_antecedent_list.extend(candidate_antecedent_list)

            if batch_vectors:
                batch_vectors = torch.tensor(np.array(batch_vectors), dtype=torch.float32)
                # Perform a single model evaluation for the batch
                implications_batch = self.model(batch_vectors, feature_value_indices).detach().numpy()
                for test_vector, implication_probabilities, candidate_antecedents \
                        in zip(batch_vectors, implications_batch, batch_candidate_antecedent_list):
                    if len(candidate_antecedents) == 0:
                        continue

                    # Identify low-support antecedents
                    if any(implication_probabilities[ant] <= self.ant_similarity for ant in candidate_antecedents):
                        if r == 1:
                            low_support_antecedents = np.append(low_support_antecedents, candidate_antecedents)
                        continue

                    # Identify high-support consequents
                    consequent_list = [
                        prob_index for prob_index in target_range
                        if prob_index not in candidate_antecedents and
                           implication_probabilities[prob_index] >= self.cons_similarity
                    ]

                    if consequent_list:
                        new_rule = self.get_rule(candidate_antecedents, consequent_list)
                        for consequent in new_rule['consequents']:
                            association_rules.append({'antecedents': new_rule['antecedents'], 'consequent': consequent})

        execution_time = time.time() - start
        return association_rules, execution_time

    def generate_frequent_itemsets(self):
        """
        Generate frequent itemsets using the Aerial+ algorithm.
        """
        frequent_itemsets = []
        input_vector_size = len(self.input_vectors['vector_tracker_list'])

        # Timing variables
        start_time = time.time()

        low_support_antecedents = np.array([])

        # Create a copy of the feature_value_indices
        feature_value_indices = self.input_vectors['feature_value_indices'][:]

        # Initialize input vectors once
        unmarked_features = self.initialize_input_vectors(
            input_vector_size,
            self.input_vectors["feature_value_indices"]
        )

        # Precompute target indices for softmax
        feature_value_indices = [(cat['start'], cat['end']) for cat in feature_value_indices]
        softmax_ranges = feature_value_indices

        # Iteratively process combinations of increasing size
        for r in range(1, self.max_antecedents + 1):
            softmax_ranges = [
                (start, end) for (start, end) in softmax_ranges
                if not all(idx in low_support_antecedents for idx in range(start, end))
            ]

            feature_combinations = list(combinations(softmax_ranges, r))  # Generate combinations

            # Vectorized model evaluation batch
            batch_vectors = []
            batch_candidate_antecedent_list = []

            for category_list in feature_combinations:
                test_vectors, candidate_antecedent_list = self.mark_features(unmarked_features, list(category_list),
                                                                             low_support_antecedents)
                if len(test_vectors) > 0:
                    batch_vectors.extend(test_vectors)
                    batch_candidate_antecedent_list.extend(candidate_antecedent_list)
            if batch_vectors:
                batch_vectors = torch.tensor(np.array(batch_vectors), dtype=torch.float32)
                # Perform a single model evaluation for the batch
                implications_batch = self.model(batch_vectors, feature_value_indices).detach().numpy()
                for test_vector, implication_probabilities, candidate_antecedents \
                        in zip(batch_vectors, implications_batch, batch_candidate_antecedent_list):
                    if len(candidate_antecedents) == 0:
                        continue

                    # Identify low-support antecedents
                    if any(implication_probabilities[ant] <= self.ant_similarity for ant in candidate_antecedents):
                        if r == 1:
                            low_support_antecedents = np.append(low_support_antecedents, candidate_antecedents)
                        continue

                    # Add to frequent itemsets
                    frequent_itemsets.append(
                        [self.input_vectors['vector_tracker_list'][idx] for idx in candidate_antecedents]
                    )
        execution_time = time.time() - start_time
        return frequent_itemsets, execution_time

    @staticmethod
    def mark_features(unmarked_test_vector, features, low_support_antecedents):
        """
        Create a list of test vectors by marking the given features in the unmarked test vector.
        This optimized version processes features in bulk using NumPy operations.
        """
        input_vector_size = len(unmarked_test_vector)

        # Compute valid feature ranges excluding low_support_antecedents
        feature_ranges = [
            np.setdiff1d(np.arange(start, end), low_support_antecedents)
            for (start, end) in features
        ]

        # Create all combinations of feature indices
        combinations = np.array(np.meshgrid(*feature_ranges)).T.reshape(-1, len(features))

        # Initialize test_vectors and candidate_antecedents
        n_combinations = combinations.shape[0]
        test_vectors = np.tile(unmarked_test_vector, (n_combinations, 1))
        candidate_antecedents = [[] for _ in range(n_combinations)]

        # Vectorized marking of test_vectors
        for i, (start, end) in enumerate(features):
            # Get the feature range
            valid_indices = combinations[:, i]

            # Ensure indices are within bounds
            valid_indices = valid_indices[(valid_indices >= 0) & (valid_indices < input_vector_size)]

            # Mark test_vectors based on valid indices for the current feature
            for j, idx in enumerate(valid_indices):
                test_vectors[j, start:end] = 0  # Set feature range to 0
                test_vectors[j, idx] = 1  # Mark the valid index with 1
                candidate_antecedents[j].append(idx)  # Append the index to the j-th test vector's antecedents

        # Convert lists of candidate_antecedents to numpy arrays
        candidate_antecedents = [np.array(lst) for lst in candidate_antecedents]
        return test_vectors, candidate_antecedents

    @staticmethod
    def initialize_input_vectors(input_vector_size, categories):
        """
        Initialize the input vectors with equal probabilities for each feature range.
        """
        vector_with_unmarked_features = np.zeros(input_vector_size)
        for category in categories:
            vector_with_unmarked_features[category['start']:category['end']] = 1 / (
                    category['end'] - category['start'])
        return vector_with_unmarked_features

    def calculate_basic_stats(self, rules, transactions):
        """
        Calculate support and confidence in parallel.
        :param rules: List of rules to process.
        :param transactions: List of transactions.
        :return: Updated rules with support and confidence.
        """
        num_transactions = len(transactions)

        def process_rule(rule):
            ant_count = 0
            cons_count = 0
            co_occurrence_count = 0

            for index in range(len(self.input_vectors['vector_list'])):
                encoded_transaction = self.input_vectors['vector_list'][index]
                antecedent_match = all(
                    encoded_transaction[self.input_vectors['vector_tracker_list'].index(antecedent)] == 1
                    for antecedent in rule['antecedents']
                )
                if antecedent_match:
                    ant_count += 1
                if encoded_transaction[self.input_vectors['vector_tracker_list'].index(rule['consequent'])] == 1:
                    cons_count += 1
                    if antecedent_match:
                        co_occurrence_count += 1

            support_body = ant_count / num_transactions
            rule['support'] = co_occurrence_count / num_transactions
            rule['confidence'] = rule['support'] / support_body if support_body != 0 else 0

            return rule

        # Use ThreadPoolExecutor to process rules in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            rules = list(executor.map(process_rule, rules))

        return rules if rules else None

    @staticmethod
    def calculate_freq_item_support(freq_items, transactions):
        num_rows = len(transactions)
        support_values = {}

        for item in freq_items:
            conditions = {pair.split("__")[0]: pair.split("__")[1] for pair in item}

            mask = pd.Series(True, index=transactions.index)
            for feature, value in conditions.items():
                mask &= (transactions[feature] == value)

            support_values[tuple(item)] = mask.sum() / num_rows

        average_support = sum(support_values.values()) / len(support_values) if support_values else 0
        return support_values, average_support

    def calculate_stats(self, rules, transactions, exec_time):
        """
        Calculate rule quality stats for the given set of rules based on the input transactions.
        """
        num_transactions = len(transactions)
        vector_list = np.array(self.input_vectors['vector_list'])
        vector_tracker_list = self.input_vectors['vector_tracker_list']

        dataset_coverage = np.zeros(num_transactions, dtype=bool)

        def process_rule(rule):
            antecedents_indices = [vector_tracker_list.index(ant) for ant in rule['antecedents']]
            consequent_index = vector_tracker_list.index(rule['consequent'])

            # Find transactions where all antecedents are present
            antecedent_matches = np.all(vector_list[:, antecedents_indices] == 1, axis=1)
            co_occurrence_matches = antecedent_matches & (vector_list[:, consequent_index] == 1)

            antecedents_occurrence_count = np.sum(antecedent_matches)
            co_occurrence_count = np.sum(co_occurrence_matches)

            support_body = antecedents_occurrence_count / num_transactions if num_transactions else 0
            rule_support = co_occurrence_count / num_transactions if num_transactions else 0
            rule_confidence = rule_support / support_body if support_body != 0 else 0

            rule['support'] = rule_support
            rule['confidence'] = rule_confidence

            return antecedent_matches, rule

        # Parallel processing of rules
        results = Parallel(n_jobs=10)(delayed(process_rule)(rule) for rule in rules)

        # Aggregate dataset coverage and collect updated rules
        updated_rules = []
        for antecedent_matches, rule in results:
            dataset_coverage |= antecedent_matches
            updated_rules.append(rule)

        if not updated_rules:
            return None

        stats = calculate_average_rule_quality(updated_rules)
        stats["coverage"] = np.sum(dataset_coverage) / num_transactions

        return [len(updated_rules), exec_time, stats['support'], stats["confidence"], stats["coverage"]], updated_rules

    def get_rule(self, antecedents, consequents):
        rule = {'antecedents': [], 'consequents': []}
        for antecedent in antecedents:
            rule['antecedents'].append(self.input_vectors['vector_tracker_list'][antecedent])

        for consequent in consequents:
            rule['consequents'].append(self.input_vectors['vector_tracker_list'][consequent])

        return rule

    def train(self, lr=5e-3, epochs=1, batch_size=2, 
              use_early_stopping=True, early_stopping_patience=10, early_stopping_min_delta=1e-4,
              use_lr_scheduler=True, lr_scheduler_type="ReduceLROnPlateau", 
              lr_scheduler_factor=0.5, lr_scheduler_patience=5, lr_scheduler_step_size=30,
              lr_scheduler_min_lr=1e-6, use_adaptive_batch_size=False,
              adaptive_batch_size_start=2, adaptive_batch_size_max=128,
              adaptive_batch_size_increase_interval=10, save_training_history=True,
              training_history_path="training_history"):
        """
        train the autoencoder
        """
        # pretrain categorical attributes from the knowledge graph, to create a numerical representation for them
        self.model = AutoEncoder(len(self.input_vectors['vector_list'][0]))

        # if not self.model.load("test"):
        training_time = self.train_ae_model(
            lr=lr, epochs=epochs, batch_size=batch_size,
            use_early_stopping=use_early_stopping,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            use_lr_scheduler=use_lr_scheduler,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_step_size=lr_scheduler_step_size,
            lr_scheduler_min_lr=lr_scheduler_min_lr,
            use_adaptive_batch_size=use_adaptive_batch_size,
            adaptive_batch_size_start=adaptive_batch_size_start,
            adaptive_batch_size_max=adaptive_batch_size_max,
            adaptive_batch_size_increase_interval=adaptive_batch_size_increase_interval,
            save_training_history=save_training_history,
            training_history_path=training_history_path
        )
        # self.model.save("test")
        return training_time

    def train_ae_model(self, loss_function=torch.nn.BCELoss(), lr=5e-3, epochs=1, batch_size=2,
                      use_early_stopping=True, early_stopping_patience=10, early_stopping_min_delta=1e-4,
                      use_lr_scheduler=True, lr_scheduler_type="ReduceLROnPlateau",
                      lr_scheduler_factor=0.5, lr_scheduler_patience=5, lr_scheduler_step_size=30,
                      lr_scheduler_min_lr=1e-6, use_adaptive_batch_size=False,
                      adaptive_batch_size_start=2, adaptive_batch_size_max=128,
                      adaptive_batch_size_increase_interval=10, save_training_history=True,
                      training_history_path="training_history"):
        """
        Train the autoencoder model with advanced training strategies:
        - Early Stopping
        - Learning Rate Scheduling
        - Adaptive Batch Size
        - Training History Tracking
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=2e-8)

        vectors_tensor = torch.tensor(self.input_vectors["vector_list"], dtype=torch.float32)
        feature_value_indices = self.input_vectors["feature_value_indices"]
        
        # 訓練データと検証データに分割（80/20）
        dataset_size = len(vectors_tensor)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        train_tensor = vectors_tensor[:train_size]
        val_tensor = vectors_tensor[train_size:]
        
        # 適応的バッチサイズの初期化
        current_batch_size = adaptive_batch_size_start if use_adaptive_batch_size else batch_size
        
        train_dataset = TensorDataset(train_tensor)
        val_dataset = TensorDataset(val_tensor)
        
        softmax_ranges = [(cat['start'], cat['end']) for cat in feature_value_indices]
        
        # 学習率スケジューラの設定
        scheduler = None
        if use_lr_scheduler:
            if lr_scheduler_type == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=lr_scheduler_factor, 
                    patience=lr_scheduler_patience, min_lr=lr_scheduler_min_lr, verbose=True
                )
            elif lr_scheduler_type == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_factor
                )
            elif lr_scheduler_type == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=lr_scheduler_min_lr
                )
        
        # Early Stopping用の変数
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # 訓練履歴の初期化
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epochs_completed': 0,
            'early_stopped': False,
            'best_epoch': 0,
            'batch_sizes': []
        }
        
        training_start_time = time.time()
        
        for epoch in range(epochs):
            # 適応的バッチサイズの更新
            if use_adaptive_batch_size and epoch > 0 and epoch % adaptive_batch_size_increase_interval == 0:
                current_batch_size = min(current_batch_size * 2, adaptive_batch_size_max)
                print(f"Batch size increased to {current_batch_size}")
            
            # DataLoaderの作成（バッチサイズが変わる可能性があるため毎回作成）
            train_dataloader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False)
            
            # 訓練フェーズ
            self.model.train()
            train_loss_epoch = 0.0
            train_batches = 0
            
            for batch_index, (batch,) in enumerate(train_dataloader):
                noisy_batch = (batch + torch.randn_like(batch) * self.noise_factor).clamp(0, 1)
                
                # Forward pass
                reconstructed_batch = self.model(noisy_batch, softmax_ranges)
                
                # Compute loss for the entire batch
                total_loss = sum(
                    loss_function(
                        reconstructed_batch[:, start:end],
                        batch[:, start:end]
                    )
                    for (start, end) in softmax_ranges
                )
                
                # Backpropagation and optimization step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                train_loss_epoch += total_loss.item()
                train_batches += 1
            
            train_loss_avg = train_loss_epoch / train_batches if train_batches > 0 else 0
            
            # 検証フェーズ
            self.model.eval()
            val_loss_epoch = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_index, (batch,) in enumerate(val_dataloader):
                    reconstructed_batch = self.model(batch, softmax_ranges)
                    
                    total_loss = sum(
                        loss_function(
                            reconstructed_batch[:, start:end],
                            batch[:, start:end]
                        )
                        for (start, end) in softmax_ranges
                    )
                    
                    val_loss_epoch += total_loss.item()
                    val_batches += 1
            
            val_loss_avg = val_loss_epoch / val_batches if val_batches > 0 else 0
            
            # 訓練履歴を記録
            current_lr = optimizer.param_groups[0]['lr']
            self.training_history['train_loss'].append(train_loss_avg)
            self.training_history['val_loss'].append(val_loss_avg)
            self.training_history['learning_rates'].append(current_lr)
            self.training_history['batch_sizes'].append(current_batch_size)
            self.training_history['epochs_completed'] = epoch + 1
            
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss_avg:.6f}, Val Loss: {val_loss_avg:.6f}, LR: {current_lr:.6e}")
            
            # 学習率スケジューラの更新
            if scheduler is not None:
                if lr_scheduler_type == "ReduceLROnPlateau":
                    scheduler.step(val_loss_avg)
                else:
                    scheduler.step()
            
            # Early Stoppingのチェック
            if use_early_stopping:
                if val_loss_avg < best_val_loss - early_stopping_min_delta:
                    best_val_loss = val_loss_avg
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                    self.training_history['best_epoch'] = epoch + 1
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    self.training_history['early_stopped'] = True
                    # ベストモデルの重みを復元
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break
        
        training_time = time.time() - training_start_time
        
        # 訓練履歴の保存
        if save_training_history:
            self._save_training_history(training_history_path)
        
        return training_time
    
    def _save_training_history(self, output_dir):
        """
        訓練履歴を保存し、収束曲線をプロット
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # JSONファイルに保存
        history_file = os.path.join(output_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # 収束曲線のプロット
        plt.figure(figsize=(12, 8))
        
        # 損失曲線
        plt.subplot(2, 2, 1)
        plt.plot(self.training_history['train_loss'], label='Train Loss', marker='o', markersize=3)
        plt.plot(self.training_history['val_loss'], label='Validation Loss', marker='s', markersize=3)
        if self.training_history['best_epoch'] > 0:
            plt.axvline(x=self.training_history['best_epoch'] - 1, color='r', 
                       linestyle='--', label=f"Best Epoch ({self.training_history['best_epoch']})")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # 学習率の推移
        plt.subplot(2, 2, 2)
        plt.plot(self.training_history['learning_rates'], marker='o', markersize=3, color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True)
        
        # バッチサイズの推移
        plt.subplot(2, 2, 3)
        plt.plot(self.training_history['batch_sizes'], marker='o', markersize=3, color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('Batch Size')
        plt.title('Batch Size Evolution')
        plt.grid(True)
        
        # 損失の対数スケール
        plt.subplot(2, 2, 4)
        plt.plot(self.training_history['train_loss'], label='Train Loss', marker='o', markersize=3)
        plt.plot(self.training_history['val_loss'], label='Validation Loss', marker='s', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training and Validation Loss (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'convergence_curves.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history saved to {history_file}")
        print(f"Convergence curves saved to {plot_file}")
