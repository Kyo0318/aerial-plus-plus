# 訓練戦略の改善 - 使用ガイド

## 概要

Aerial++に実装された訓練戦略の改善機能について説明します。

## 実装された機能

### 1. 早期終了（Early Stopping）

検証損失の改善が停止したら訓練を自動的に中断します。

**メリット:**
- 過学習を防止
- 訓練時間の最適化
- 最良モデルの自動保存

**パラメータ:**
```python
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10  # 改善が見られない連続エポック数
EARLY_STOPPING_MIN_DELTA = 1e-4  # 改善とみなす最小損失変化
```

### 2. 学習率スケジューリング（Learning Rate Scheduling）

訓練の進行に応じて学習率を自動調整します。

**サポートされるスケジューラ:**
- **ReduceLROnPlateau**: 検証損失の改善が停滞した際に学習率を削減
- **StepLR**: 固定エポック間隔で学習率を削減
- **CosineAnnealingLR**: コサイン関数に従って学習率を調整

**パラメータ:**
```python
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = "ReduceLROnPlateau"  # または "StepLR", "CosineAnnealingLR"
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_MIN_LR = 1e-6
```

### 3. 適応的バッチサイズ（Adaptive Batch Size）

訓練の進行に応じてバッチサイズを段階的に増加させます。

**メリット:**
- 初期段階: 小さいバッチサイズで細かい勾配更新
- 後期段階: 大きいバッチサイズで安定した収束

**パラメータ:**
```python
USE_ADAPTIVE_BATCH_SIZE = False  # デフォルトはオフ
ADAPTIVE_BATCH_SIZE_START = 2
ADAPTIVE_BATCH_SIZE_MAX = 128
ADAPTIVE_BATCH_SIZE_INCREASE_INTERVAL = 10  # エポック間隔
```

### 4. 訓練履歴と収束曲線の分析

訓練中のメトリクスを自動的に記録・可視化します。

**出力ファイル:**
- `training_history.json`: 詳細な訓練メトリクス
- `convergence_curves.png`: 4つのプロット
  - 訓練損失と検証損失の推移
  - 学習率の推移（対数スケール）
  - バッチサイズの推移
  - 損失の対数スケール表示

**パラメータ:**
```python
SAVE_TRAINING_HISTORY = True
TRAINING_HISTORY_PATH = "training_history"
```

## 使用方法

### 基本的な使用

`config.py`でパラメータを設定してから、通常通り実験を実行します。

```bash
# ルールマイニング実験
python3 rule_mining_experiments.py

# 分類実験
python3 classification_experiments.py
```

### カスタム設定例

#### 例1: 早期終了のみを使用

```python
# config.py
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 1e-5

USE_LR_SCHEDULER = False
USE_ADAPTIVE_BATCH_SIZE = False
```

#### 例2: すべての機能を有効化

```python
# config.py
EPOCHS = 200
BATCH_SIZE = 32

# Early Stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 1e-4

# Learning Rate Scheduling
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = "ReduceLROnPlateau"
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 10

# Adaptive Batch Size
USE_ADAPTIVE_BATCH_SIZE = True
ADAPTIVE_BATCH_SIZE_START = 8
ADAPTIVE_BATCH_SIZE_MAX = 128
ADAPTIVE_BATCH_SIZE_INCREASE_INTERVAL = 20

# Training History
SAVE_TRAINING_HISTORY = True
TRAINING_HISTORY_PATH = "training_history"
```

#### 例3: 高速プロトタイピング用（訓練履歴なし）

```python
# config.py
EPOCHS = 50
BATCH_SIZE = 64

USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5

USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = "StepLR"
LR_SCHEDULER_STEP_SIZE = 15

SAVE_TRAINING_HISTORY = False  # 高速化のため無効化
```

## 訓練履歴の分析

訓練後、`training_history/<dataset_name>/`に以下のファイルが生成されます。

### training_history.json

```json
{
  "train_loss": [0.523, 0.412, 0.389, ...],
  "val_loss": [0.534, 0.425, 0.401, ...],
  "learning_rates": [0.005, 0.005, 0.0025, ...],
  "epochs_completed": 45,
  "early_stopped": true,
  "best_epoch": 38,
  "batch_sizes": [32, 32, 32, ...]
}
```

### convergence_curves.png

4つのサブプロットを含む可視化:
1. **訓練損失と検証損失**: モデルの収束を確認
2. **学習率の推移**: スケジューラの動作を確認
3. **バッチサイズの推移**: 適応的バッチサイズの動作を確認
4. **損失の対数スケール**: 細かい変化を確認

## パフォーマンスへの影響

### 訓練時間

- **早期終了**: 通常20-40%の訓練時間削減
- **適応的バッチサイズ**: 初期は遅いが、後期で高速化
- **訓練履歴の保存**: 約1-2%のオーバーヘッド（無視できるレベル）

### メモリ使用量

- バッチサイズに比例して増加
- 適応的バッチサイズ使用時は最大バッチサイズまで増加

### 推奨設定

**小規模データセット（< 1000サンプル）:**
```python
BATCH_SIZE = 16
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10
USE_LR_SCHEDULER = True
USE_ADAPTIVE_BATCH_SIZE = False
```

**中規模データセット（1000-10000サンプル）:**
```python
BATCH_SIZE = 32
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 15
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = "ReduceLROnPlateau"
USE_ADAPTIVE_BATCH_SIZE = True
```

**大規模データセット（> 10000サンプル）:**
```python
BATCH_SIZE = 64
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 20
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = "CosineAnnealingLR"
USE_ADAPTIVE_BATCH_SIZE = True
ADAPTIVE_BATCH_SIZE_START = 16
ADAPTIVE_BATCH_SIZE_MAX = 256
```

## トラブルシューティング

### 問題1: 訓練がすぐに終了する

**原因**: 早期終了のパラメータが厳しすぎる

**解決策**:
```python
EARLY_STOPPING_PATIENCE = 20  # より大きな値に
EARLY_STOPPING_MIN_DELTA = 1e-5  # より小さな値に
```

### 問題2: 学習率が下がりすぎる

**原因**: 学習率スケジューラのパラメータが積極的すぎる

**解決策**:
```python
LR_SCHEDULER_PATIENCE = 10  # より大きな値に
LR_SCHEDULER_FACTOR = 0.7  # より大きな値に（0.5よりも緩やか）
```

### 問題3: メモリ不足エラー

**原因**: バッチサイズが大きすぎる

**解決策**:
```python
BATCH_SIZE = 16  # より小さな値に
ADAPTIVE_BATCH_SIZE_MAX = 64  # 最大バッチサイズを削減
USE_ADAPTIVE_BATCH_SIZE = False  # 適応的バッチサイズを無効化
```

## まとめ

訓練戦略の改善により、以下のメリットが得られます:

1. ✅ **訓練時間の最適化**: 早期終了により不要な訓練を削減
2. ✅ **性能向上**: 適切な学習率スケジューリングにより収束性能を改善
3. ✅ **過学習の防止**: 検証セットでの早期終了により汎化性能を向上
4. ✅ **トレーサビリティ**: 詳細な訓練履歴により実験の再現性を確保
5. ✅ **ハイパーパラメータ調整の容易化**: 収束曲線の可視化により調整が簡単に

これらの機能を活用して、より効率的な研究開発を行ってください！

