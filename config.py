# Generic
MAX_ANTECEDENT = 2

# DL-Generic
EPOCHS = 1
BATCH_SIZE = 2
LEARNING_RATE = 5e-3

# Exhaustive
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.8

# aerial_plus
ANTECEDENT_SIMILARITY = 0.1
CONSEQUENT_SIMILARITY = 0.8
NOISE_FACTOR = 0.5

# aerial_plus: ハイパーパラメータ自動調整
# 自動調整を有効にすると、データセットごとに最適なパラメータを自動で見つけます
# 注意: 自動調整には時間がかかります（TUNING_TRIALS * モデル学習時間）
ENABLE_AUTO_TUNING = True  # 自動調整を有効にする（False=固定値を使用）
TUNING_TRIALS = 30  # Optunaの試行回数（推奨: 30-100、多いほど精度向上だが時間もかかる）
TUNING_METRIC = 'f1_score'  # 最適化指標
# 利用可能な指標:
#   - 'f1_score': support, confidence, coverageの調和平均（バランス重視）
#   - 'support': ルールの出現頻度を最大化
#   - 'confidence': ルールの信頼度を最大化
#   - 'coverage': データのカバレッジを最大化
#   - 'balanced': 重み付き線形和（confidence重視）

# ARM-AE
SIMILARITY_THRESHOLD = 0.5

# Optimization-based
POPULATION_SIZE = 200
MAX_EVALS = 5000

# discretization
NUM_BINS = 10
