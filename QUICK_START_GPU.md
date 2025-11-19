# 🚀 Aerial+ GPU対応版 クイックスタートガイド

このガイドでは、GPU対応のAerial+を使ってルールマイニング実験を実行する方法を説明します。

## 📋 前提条件

### 必須環境

1. **Python 3.8以上**
2. **CUDA対応GPU**（推奨、なくてもCPUで動作します）
3. **CUDA Toolkit 11.0以上**（GPU使用時）

### パッケージのインストール

```bash
# 依存パッケージをインストール
pip install -r requirements.txt

# GPU対応PyTorchをインストール（GPU使用時）
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### GPUの確認

```bash
# CUDAの利用可能性を確認
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

---

## 🎯 基本的な使い方

### 1. 設定ファイルの確認

`config.py`でパラメータを調整できます：

```python
# Generic
MAX_ANTECEDENT = 2  # 最大前件数

# DL-Generic
EPOCHS = 1  # エポック数（GPU使用時は増やすと精度向上）
BATCH_SIZE = 2  # CPU用（GPU使用時は自動最適化されます）
LEARNING_RATE = 5e-3  # 学習率

# Aerial+
ANTECEDENT_SIMILARITY = 0.1  # 前件類似度閾値
CONSEQUENT_SIMILARITY = 0.8  # 後件類似度閾値
```

### 2. 実験の実行

```bash
python3 rule_mining_experiments.py
```

### 3. 出力の確認

実行後、以下のファイルが生成されます：

- `Aerial_Plus_<データセット名>_<タイムスタンプ>.csv` - 統計情報
- `Aerial_Plus_<データセット名>_rules_<タイムスタンプ>.txt` - サンプルルール（最初の20個）

---

## 📊 実行例

### サンプル出力

```
============================================================
Aerial+ GPU-Accelerated Rule Mining
============================================================

============================================================
Aerial+ GPU-Accelerated Experiment Configuration
============================================================
Antecedent similarity: 0.1
Consequent similarity: 0.8
Maximum antecedents: 2
Epochs: 1
Learning rate: 0.005
Batch size: Auto (GPU-optimized)
Number of bins for discretization: 10
============================================================


============================================================
GPU Configuration
============================================================
✅ CUDA Available: Yes
GPU Device: NVIDIA GeForce RTX 3090
CUDA Version: 11.8
Total GPU Memory: 24.00 GB
Number of GPUs: 1
============================================================

LOADING: Loading the datasets ...
LOADED: Following dataset(s) are loaded: Congressional Voting Records 

🚀 Initializing Aerial+ with GPU support...
Device: cuda:0
GPU: NVIDIA GeForce RTX 3090
GPU Memory: 24.00 GB

============================================================
Processing Dataset: Congressional Voting Records
============================================================

[1/3] Creating input vectors...
✅ Input vectors created

[2/3] Training autoencoder...

[GPU Configuration]
Using GPU: NVIDIA GeForce RTX 3090
Total GPU Memory: 24.00 GB
Auto-selected batch size: 128
✅ Training completed in 2.15s

[3/3] Generating association rules...
✅ Generated 89 rules in 3.42s

📊 Calculating statistics...

============================================================
Rule Mining with Aerial+ (GPU) - COMPLETED
============================================================
Rule Count: 89
Total Time: 5.57s
Average Support: 0.4523
Average Confidence: 0.8912
Data Coverage: 0.9654
============================================================

💾 Saving results to: Aerial_Plus_Congressional Voting Records_11-19-2025_12:30:45.csv
✅ Results saved successfully!
💾 Saving sample rules to: Aerial_Plus_Congressional Voting Records_rules_11-19-2025_12:30:45.txt
✅ Sample rules saved successfully!

============================================================
Experiment Completed Successfully! 🎉
============================================================
```

---

## ⚙️ カスタマイズ

### データセットの変更

`rule_mining_experiments.py`の`get_datasets()`関数を編集：

```python
def get_datasets():
    print("LOADING: Loading the datasets ...")
    congress_voting_records = fetch_ucirepo(id=105)
    mushroom = fetch_ucirepo(id=73)  # 追加
    breast_cancer = discretize_numerical_features(fetch_ucirepo(id=14))  # 追加
    
    datasets = [congress_voting_records, mushroom, breast_cancer]  # 複数指定
    print("LOADED: Following dataset(s) are loaded:", ", ".join([dataset.metadata.name for dataset in datasets]), "\n")
    return datasets
```

### GPU/CPU切り替え

`rule_mining_experiments.py`の`AerialPlus`初期化部分を編集：

```python
# GPU使用
aerial_plus = AerialPlus(
    max_antecedents=config.MAX_ANTECEDENT,
    ant_similarity=config.ANTECEDENT_SIMILARITY,
    cons_similarity=config.CONSEQUENT_SIMILARITY,
    use_gpu=True  # GPU使用
)

# CPU使用
aerial_plus = AerialPlus(
    max_antecedents=config.MAX_ANTECEDENT,
    ant_similarity=config.ANTECEDENT_SIMILARITY,
    cons_similarity=config.CONSEQUENT_SIMILARITY,
    use_gpu=False  # CPU使用
)
```

### バッチサイズの手動設定

```python
# 自動最適化（推奨）
use_batch_size = None

# 手動設定
use_batch_size = 64  # 好みのバッチサイズ
```

---

## 🔧 トラブルシューティング

### 1. GPUが検出されない

**症状**: `CUDA Available: No`

**解決策**:
```bash
# CUDA対応PyTorchを再インストール
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDAのバージョンを確認
nvidia-smi
```

### 2. GPUメモリ不足（OOM）

**症状**: `RuntimeError: CUDA out of memory`

**解決策**:
```python
# config.pyでエポック数を減らす
EPOCHS = 1

# または、バッチサイズを手動で小さく設定
use_batch_size = 16  # デフォルトより小さく
```

### 3. 実行が遅い

**考えられる原因**:
- CPUで実行されている
- バッチサイズが小さすぎる
- データセットが大きすぎる

**解決策**:
```python
# GPU使用を確認
device_info = aerial_plus.get_device_info()
print(device_info)

# エポック数を増やしてより良いモデルを訓練
EPOCHS = 10  # より多くのエポック

# バッチサイズを手動で大きく
use_batch_size = 128  # より大きなバッチサイズ
```

---

## 📈 パフォーマンス最適化のヒント

### 1. エポック数の調整

```python
# config.py
EPOCHS = 10  # より多くのエポックでモデルの品質向上
```

**効果**: より高品質なルールが生成される可能性がありますが、訓練時間が長くなります。

### 2. バッチサイズの最適化

```python
# GPU使用時は自動最適化が推奨
use_batch_size = None

# 手動で設定する場合
use_batch_size = 128  # GPU: 64-256が一般的
use_batch_size = 4    # CPU: 2-8が一般的
```

### 3. 複数GPUの使用

現在の実装は単一GPU対応です。マルチGPU対応は今後の実装予定です。

---

## 📚 さらに詳しく

- **完全なドキュメント**: [README.md](README.md)
- **GPU実装の詳細**: [GPU_IMPLEMENTATION_NOTES.md](GPU_IMPLEMENTATION_NOTES.md)
- **元の論文**: [Neurosymbolic Association Rule Mining from Tabular Data](https://proceedings.mlr.press/v284/karabulut25a.html)

---

## 💡 ヒント

1. **初回実行**: まずデフォルト設定で実行して動作を確認
2. **パラメータチューニング**: `config.py`で類似度閾値を調整してルール数を制御
3. **大規模データセット**: GPU使用時は自動バッチサイズ最適化を活用
4. **結果の分析**: 生成されたCSVとテキストファイルで結果を確認

---

## 🤝 サポート

問題が発生した場合は、以下を確認してください：

1. Python/CUDA/PyTorchのバージョン
2. GPU情報（`nvidia-smi`）
3. エラーメッセージの全文

---

## 📝 ライセンス

このプロジェクトは元のAerial+プロジェクトと同じライセンスに従います。

