# Generic
MAX_ANTECEDENT = 2

# DL-Generic
EPOCHS = 100  # 最大エポック数を増やす
BATCH_SIZE = 32  # バッチサイズを増やす
LEARNING_RATE = 5e-3

# 訓練戦略の改善
# Early Stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10  # 改善が見られないエポック数
EARLY_STOPPING_MIN_DELTA = 1e-4  # 改善とみなす最小損失変化

# Learning Rate Scheduling
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = "ReduceLROnPlateau"  # "ReduceLROnPlateau", "StepLR", "CosineAnnealingLR"
LR_SCHEDULER_FACTOR = 0.5  # 学習率を減らす倍率
LR_SCHEDULER_PATIENCE = 5  # 学習率を減らすまでの待機エポック数
LR_SCHEDULER_STEP_SIZE = 30  # StepLRの場合のステップサイズ
LR_SCHEDULER_MIN_LR = 1e-6  # 最小学習率

# Adaptive Batch Size
USE_ADAPTIVE_BATCH_SIZE = False
ADAPTIVE_BATCH_SIZE_START = 2
ADAPTIVE_BATCH_SIZE_MAX = 128
ADAPTIVE_BATCH_SIZE_INCREASE_INTERVAL = 10  # エポック間隔

# Training History / Convergence Analysis
SAVE_TRAINING_HISTORY = True
TRAINING_HISTORY_PATH = "training_history"  # 保存先ディレクトリ

# Exhaustive
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.8

# aerial_plus
ANTECEDENT_SIMILARITY = 0.1
CONSEQUENT_SIMILARITY = 0.8

# ARM-AE
SIMILARITY_THRESHOLD = 0.5

# Optimization-based
POPULATION_SIZE = 200
MAX_EVALS = 5000

# discretization
NUM_BINS = 10
