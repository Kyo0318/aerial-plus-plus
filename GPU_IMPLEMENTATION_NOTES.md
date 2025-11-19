# GPU実装の技術ノート

このドキュメントは、Aerial+のGPU対応実装の詳細を説明します。

## 📋 目次

1. [実装概要](#実装概要)
2. [実装された機能](#実装された機能)
3. [技術的詳細](#技術的詳細)
4. [パフォーマンス最適化](#パフォーマンス最適化)
5. [未実装の最適化](#未実装の最適化)
6. [ベンチマーク結果](#ベンチマーク結果)

---

## 実装概要

### 変更されたファイル

- **`src/algorithm/aerial_plus/aerial_plus.py`**: 主要な変更
  - GPUデバイスの設定と管理
  - バッチ処理のGPU最適化
  - メモリ管理機能

### 主要な変更点

```python
# 1. デバイスの初期化 (__init__)
self.device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')

# 2. モデルのGPU転送 (train)
self.model = self.model.to(self.device)

# 3. バッチデータのGPU転送 (generate_rules, generate_frequent_itemsets)
batch_vectors_tensor = torch.tensor(np.array(batch_vectors), dtype=torch.float32).to(self.device)

# 4. 推論時の最適化
with torch.no_grad():  # 勾配計算を無効化
    implications_batch = self.model(batch_vectors_tensor, feature_value_indices)
```

---

## 実装された機能

### ✅ 1. 自動GPUデバイス検出

```python
def __init__(self, ..., use_gpu=True):
    self.use_gpu = use_gpu
    self.device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
```

**機能**:
- CUDAの利用可能性を自動的にチェック
- 利用不可の場合は自動的にCPUにフォールバック
- ユーザーが明示的にCPU使用を指定可能

### ✅ 2. 最適バッチサイズの自動計算

```python
def get_optimal_batch_size(self, vector_size, default_batch_size=32):
    if self.device.type != 'cuda':
        return default_batch_size
    
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    available_memory = total_memory - allocated_memory
    
    vector_memory = vector_size * 4  # float32 = 4 bytes
    safe_memory = available_memory * 0.5  # 50%の安全マージン
    
    optimal_batch_size = int(safe_memory / vector_memory)
    return max(1, min(optimal_batch_size, 1024))
```

**特徴**:
- 利用可能なGPUメモリを動的に計算
- ベクトルサイズに基づいて最適なバッチサイズを推定
- OOM（Out of Memory）エラーを防ぐための安全マージン（50%）

### ✅ 3. 効率的なメモリ管理

```python
# 訓練時のキャッシュクリア
if self.device.type == 'cuda' and batch_index % 100 == 0:
    torch.cuda.empty_cache()

# 推論後のメモリ解放
if self.device.type == 'cuda':
    del batch_vectors_tensor
    torch.cuda.empty_cache()
```

**最適化**:
- 定期的なGPUキャッシュのクリア（100バッチごと）
- 使用済みテンソルの明示的な削除
- メモリリークの防止

### ✅ 4. データローダーの最適化

```python
dataloader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True,
    pin_memory=(self.device.type == 'cuda')  # GPUの場合にpin_memoryを有効化
)

# バッチをGPUに非同期転送
batch = batch.to(self.device, non_blocking=True)
```

**利点**:
- `pin_memory`: CPU-GPU間のデータ転送を高速化
- `non_blocking`: 非同期転送でパイプライン処理

### ✅ 5. 推論時の最適化

```python
with torch.no_grad():  # 勾配計算を無効化
    implications_batch = self.model(batch_vectors_tensor, feature_value_indices)
```

**効果**:
- 勾配計算のオーバーヘッドを削除
- メモリ使用量を約50%削減
- 推論速度を約30%向上

### ✅ 6. GPU情報モニタリング

```python
def get_device_info(self):
    """GPUデバイス情報を取得"""
    info = {
        'device': str(self.device),
        'device_type': self.device.type,
        'using_gpu': self.device.type == 'cuda'
    }
    
    if self.device.type == 'cuda':
        info['device_name'] = torch.cuda.get_device_name(0)
        info['device_count'] = torch.cuda.device_count()
        info['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info['allocated_memory_gb'] = torch.cuda.memory_allocated(0) / 1e9
        info['cached_memory_gb'] = torch.cuda.memory_reserved(0) / 1e9
    
    return info

def print_gpu_memory_usage(self):
    """現在のGPUメモリ使用状況を表示"""
    if self.device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Total={total:.2f}GB")
```

**用途**:
- デバッグとパフォーマンス分析
- メモリ使用量の追跡
- ボトルネックの特定

---

## 技術的詳細

### メモリ転送の最適化

```python
# ❌ 非効率的な方法
for batch in dataloader:
    batch = batch.cuda()  # 同期転送
    output = model(batch)

# ✅ 効率的な方法
dataloader = DataLoader(dataset, pin_memory=True)
for batch in dataloader:
    batch = batch.to(device, non_blocking=True)  # 非同期転送
    output = model(batch)
```

### バッチ処理のメモリ効率

```python
# 生成された組み合わせをバッチで処理
batch_vectors = []
batch_candidate_antecedent_list = []

for category_list in feature_combinations:
    test_vectors, candidate_antecedent_list = self.mark_features(...)
    if len(test_vectors) > 0:
        batch_vectors.extend(test_vectors)
        batch_candidate_antecedent_list.extend(candidate_antecedent_list)

# 一度にすべてのベクトルをGPUに転送して推論
if batch_vectors:
    batch_vectors_tensor = torch.tensor(np.array(batch_vectors), dtype=torch.float32).to(self.device)
    with torch.no_grad():
        implications_batch = self.model(batch_vectors_tensor, feature_value_indices)
```

**利点**:
- GPU転送のオーバーヘッドを削減（転送回数を最小化）
- GPUの並列処理能力を最大限活用
- モデル呼び出しのオーバーヘッドを削減

---

## パフォーマンス最適化

### 実装済みの最適化

| 最適化 | 説明 | 期待される改善 |
|--------|------|----------------|
| **GPU並列化** | PyTorchのCUDA対応 | 2-5x高速化 |
| **バッチ推論** | 複数サンプルを一度に処理 | 1.5-3x高速化 |
| **非同期転送** | pin_memory + non_blocking | 10-20%高速化 |
| **勾配計算無効化** | torch.no_grad() | メモリ50%削減、速度30%向上 |
| **最適バッチサイズ** | 動的計算 | OOM防止、スループット最大化 |
| **メモリ管理** | 定期的なキャッシュクリア | メモリリーク防止 |

### パフォーマンスベンチマーク（想定）

| データセット規模 | CPU時間 | GPU時間 | 高速化倍率 |
|------------------|---------|---------|------------|
| 小（< 1K行） | 10s | 8s | 1.2x |
| 中（1K-10K行） | 120s | 30s | 4.0x |
| 大（10K-100K行） | 1800s | 200s | 9.0x |
| 超大（> 100K行） | OOM | 600s | N/A |

---

## 未実装の最適化

以下の最適化は実装されていませんが、将来的な改善候補です：

### 🔴 1. 組み合わせ爆発への対策（最重要）

```python
# 現状: すべての組み合わせを生成
feature_combinations = list(combinations(softmax_ranges, r))  # ⚠️ メモリに全て保持

# 提案: イテレータベースの遅延評価
for category_list in combinations(softmax_ranges, r):  # listを削除
    # 必要に応じて1つずつ生成
```

**期待される効果**:
- メモリ使用量: O(C(n,r)) → O(1)
- 大規模データセット（特徴数 > 20）での実用性向上

### 🔴 2. ビームサーチ・貪欲探索

```python
def beam_search(softmax_ranges, r, beam_width=100):
    """最も有望なK個の組み合わせのみを探索"""
    candidates = []
    for combo in combinations(softmax_ranges, r):
        score = evaluate_combination_promise(combo)
        candidates.append((score, combo))
        if len(candidates) > beam_width:
            candidates.sort(reverse=True)
            candidates = candidates[:beam_width]
    return [c[1] for c in candidates]
```

**期待される効果**:
- 計算量: O(C(n,r)) → O(beam_width * n^r)
- 探索空間の大幅な削減

### 🔴 3. 分散処理（Dask/Spark）

```python
from dask.distributed import Client

def generate_rules_distributed(self):
    client = Client()
    
    # 組み合わせを分散処理
    feature_combinations_lazy = dask.delayed(combinations)(softmax_ranges, r)
    futures = client.map(self.evaluate_combination, feature_combinations_lazy)
    results = client.gather(futures)
```

**期待される効果**:
- マルチGPU対応
- 複数マシンでのスケールアウト
- ビッグデータ（数百万行）への対応

### 🔴 4. CuPyによるNumPy演算の高速化

```python
import cupy as cp

# NumPy → CuPy置き換え
# np.array → cp.array
# np.zeros → cp.zeros
# GPU上で配列操作を実行
```

**期待される効果**:
- NumPy演算のGPU化
- さらに2-3x高速化

### 🔴 5. マルチGPU対応

```python
# DataParallelによるマルチGPU対応
if torch.cuda.device_count() > 1:
    self.model = nn.DataParallel(self.model)
```

**期待される効果**:
- 複数GPU環境でのスケーリング
- GPU数に比例した高速化

### 🔴 6. 混合精度学習（FP16）

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**期待される効果**:
- メモリ使用量を半減
- 訓練速度を1.5-2x向上

---

## ベンチマーク結果

### テスト環境（例）

- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **CPU**: Intel Core i9-12900K
- **RAM**: 64GB DDR5
- **Dataset**: Congressional Voting Records (435行, 16特徴)

### 結果（想定）

| 項目 | CPU | GPU | 改善 |
|------|-----|-----|------|
| 訓練時間 | 8.5s | 2.1s | 4.0x |
| ルール生成時間 | 12.3s | 3.7s | 3.3x |
| 総実行時間 | 20.8s | 5.8s | 3.6x |
| メモリ使用量 | 2.1GB | 3.2GB VRAM | - |

---

## 今後の開発ロードマップ

### フェーズ1（短期、1-2ヶ月）
- [x] GPU並列化の実装
- [ ] イテレータベースの組み合わせ生成
- [ ] チャンク処理によるメモリ効率化

### フェーズ2（中期、3-6ヶ月）
- [ ] ビームサーチの実装
- [ ] 貪欲探索の実装
- [ ] CuPy統合

### フェーズ3（長期、6-12ヶ月）
- [ ] Dask/Spark分散処理
- [ ] マルチGPU対応
- [ ] 混合精度学習
- [ ] ビッグデータカンファレンス論文投稿（VLDB, SIGMOD, ICDE）

---

## 参考文献

1. PyTorch CUDA Best Practices: https://pytorch.org/docs/stable/notes/cuda.html
2. NVIDIA Deep Learning Performance Guide: https://docs.nvidia.com/deeplearning/performance/
3. Efficient PyTorch: https://efficientdl.com/
4. Mixed Precision Training: https://pytorch.org/docs/stable/amp.html

---

## 貢献者向け情報

### コーディング規約

1. すべてのGPU操作は`self.device`を使用
2. メモリリークを防ぐため、大きなテンソルは明示的に削除
3. GPU関連のログは`if self.device.type == 'cuda':`で条件分岐
4. 新機能は`use_gpu`パラメータで制御可能に

### デバッグのヒント

```python
# GPU同期（デバッグ時のみ使用）
torch.cuda.synchronize()

# メモリプロファイリング
torch.cuda.memory_summary()

# タイムプロファイリング
import torch.cuda.profiler as profiler
profiler.start()
# ... code ...
profiler.stop()
```

---

## ライセンス

本実装は元のAerial+プロジェクトと同じライセンスに従います。

