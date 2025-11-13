# Aerial+ 最適化レポート：辞書キャッシュによるルール統計計算の高速化

## 概要

`vector_tracker_list.index(...)`の全面廃止と辞書キャッシュ導入により、ルール統計計算のボトルネックを**O(n) → O(1)**へ削減し、全体の実行時間を大幅短縮しました。

- **実装難易度**: ★☆☆（既存ロジックに辞書を挟むだけ）
- **改善効果**: 大（特にルール数・トランザクション数が多い場合に顕著）
- **実装日**: 2025年11月13日

---

## 問題の詳細

### 改善前の状態

Aerial+のルール統計計算において、文字列キー（例：`"feature__value"`）からベクトル内のインデックスを取得する際、`list.index()`による**O(n)の線形探索**が実行されていました。

#### ボトルネック箇所

1. **`calculate_basic_stats`メソッド（行309-310, 314）**
   ```python
   # 改善前：O(n)の線形探索が毎回実行される
   encoded_transaction[self.input_vectors['vector_tracker_list'].index(antecedent)]
   # → vector_tracker_listの長さがnの場合、平均n/2回の比較が必要
   ```

2. **`calculate_stats`メソッド（行359-360）**
   ```python
   # 改善前：O(n)の線形探索
   antecedents_indices = [vector_tracker_list.index(ant) for ant in rule['antecedents']]
   consequent_index = vector_tracker_list.index(rule['consequent'])
   ```

### 計算量の問題

多重ループ内で`.index()`が呼ばれるため、計算量が爆発的に増加：

```
calculate_basic_stats:
  = トランザクション数 × ルール数 × 前提条件数 × n（線形探索）
  
具体例（トランザクション: 10,000、ルール: 1,000、前提条件: 2、n: 100）:
  = 10,000 × 1,000 × 2 × 50（平均比較回数）
  = 1,000,000,000回の比較操作
```

---

## 解決策：辞書キャッシュの導入

### 基本アイデア

リストの線形探索（O(n)）を辞書のハッシュテーブル探索（O(1)）に置き換えることで、ルックアップを定数時間化します。

### 実装方針

1. データ前処理時（`create_input_vectors`）に一度だけ辞書を作成
2. 辞書を`self.input_vectors`に保存し、全メソッドで再利用
3. すべての`.index()`呼び出しを辞書アクセスに置き換え

---

## 具体的な変更箇所

### 変更1: 辞書キャッシュの作成と保存

**ファイル**: `src/algorithm/aerial_plus/aerial_plus.py`

**行61-62**: 辞書キャッシュの作成（既存コードを活用）
```python
# Map tracker entries to indices for fast lookup (辞書キャッシュ)
tracker_index_map = {key: idx for idx, key in enumerate(vector_tracker)}
```

**行88-93**: 辞書キャッシュの保存
```python
self.input_vectors = {
    "vector_list": vector_list.tolist(),
    "vector_tracker_list": vector_tracker,
    "vector_tracker_index_map": tracker_index_map,  # O(1)ルックアップ用の辞書キャッシュ
    "feature_value_indices": feature_value_indices,
}
```

**変更内容**:
- 既にローカル変数として作成していた`tracker_index_map`を`self.input_vectors`に追加
- 追加メモリ: 約5-50KB（キー数×約50バイト程度）

---

### 変更2: `calculate_basic_stats`メソッドの最適化

**ファイル**: `src/algorithm/aerial_plus/aerial_plus.py`

**行301**: 辞書キャッシュの取得
```python
tracker_index_map = self.input_vectors['vector_tracker_index_map']  # O(1)辞書キャッシュ
```

**行311**: 前提条件のインデックス取得を最適化
```python
# 改善前（削除済み）
encoded_transaction[self.input_vectors['vector_tracker_list'].index(antecedent)]

# 改善後
encoded_transaction[tracker_index_map[antecedent]]
```

**行316**: 帰結のインデックス取得を最適化
```python
# 改善前（削除済み）
encoded_transaction[self.input_vectors['vector_tracker_list'].index(rule['consequent'])]

# 改善後
encoded_transaction[tracker_index_map[rule['consequent']]]
```

**変更内容**:
- 多重ループ内（トランザクション×ルール×前提条件）での`.index()`を削除
- 辞書ルックアップに置き換え
- 計算量: **O(T×R×A×n) → O(T×R×A)**（n倍高速化）

---

### 変更3: `calculate_stats`メソッドの最適化

**ファイル**: `src/algorithm/aerial_plus/aerial_plus.py`

**行356**: 辞書キャッシュの取得
```python
tracker_index_map = self.input_vectors['vector_tracker_index_map']  # O(1)辞書キャッシュ
```

**行361-362**: インデックス取得の最適化
```python
# 改善前（削除済み）
antecedents_indices = [vector_tracker_list.index(ant) for ant in rule['antecedents']]
consequent_index = vector_tracker_list.index(rule['consequent'])

# 改善後
antecedents_indices = [tracker_index_map[ant] for ant in rule['antecedents']]
consequent_index = tracker_index_map[rule['consequent']]
```

**変更内容**:
- 並列処理内での`.index()`を削除
- 辞書ルックアップに置き換え
- 計算量: **O(R×A×n) → O(R×A)**（n倍高速化）

---

## 計算量の改善

### 理論的な改善

| 操作 | 改善前 | 改善後 | 改善倍率 |
|------|--------|--------|----------|
| インデックス取得 | O(n) | O(1) | n倍 |
| `calculate_basic_stats` | O(T×R×A×n) | O(T×R×A) | n倍 |
| `calculate_stats` | O(R×A×n) | O(R×A) | n倍 |

- T: トランザクション数
- R: ルール数
- A: 前提条件数
- n: `vector_tracker_list`の長さ

### 具体例での改善効果

#### シナリオ
- トランザクション数: 10,000
- ルール数: 1,000
- 平均前提条件数: 2
- `vector_tracker_list`の長さ: 100

#### 改善前
```
calculate_basic_stats:
  10,000 × 1,000 × 2 × 50（平均比較回数） = 1,000,000,000回の比較操作

calculate_stats:
  1,000 × 2 × 50（平均比較回数） = 100,000回の比較操作
```

#### 改善後
```
calculate_basic_stats:
  10,000 × 1,000 × 2 × 1（辞書ルックアップ） = 20,000,000回のO(1)操作

calculate_stats:
  1,000 × 2 × 1（辞書ルックアップ） = 2,000回のO(1)操作
```

#### 理論上の改善
- `calculate_basic_stats`: **約50倍高速化**
- `calculate_stats`: **約50倍高速化**

---

## 実際の使用箇所と効果

### 1. ルール品質実験（`rule_mining_experiments.py`）

**使用箇所**: 155行目
```python
aerial_plus_stats, aerial_plus_rules = aerial_plus.calculate_stats(
    aerial_plus_association_rules,
    classical_arm_input,
    aerial_plus_training_time + ae_exec_time
)
```

**効果**: ルール品質評価の実行時間が短縮

---

### 2. CBA分類アルゴリズム（`rule_generation.py`）

#### 使用箇所①: `generateCARs`関数（99行目）
```python
rules = aerial_plus.calculate_basic_stats(
    filtered_rules, 
    prepare_classic_arm_input(aerial_plus_input)
)
```

**用途**:
1. ルールに統計値（support/confidence）を付与
2. CBA形式への変換
3. ルールの優先順位付け（confidence → support → length順）
4. 分類器構築時のルール選択

**効果**: CBA訓練時のルール統計計算が高速化

#### 使用箇所②: `top_rules`関数（208行目）
```python
rules_current = aerial_plus.calculate_basic_stats(
    rules_current,
    prepare_classic_arm_input(aerial_plus_input)
)
```

**用途**: 目標ルール数に達するまで反復処理内で呼ばれる

**効果**: 反復内での計算が高速化され、全体の実行時間が短縮

---

### 3. 分類実験（`classification_experiments.py`）

**実行フロー**:
```
10-fold交差検証
  └─> 各foldで訓練（fit）
      └─> generateCARs()
          └─> calculate_basic_stats() ← ここで最適化が効く
```

**効果**: 訓練時間（Rule mining time）が短縮

**注意**: 予測時（predict）には統計計算は行わないため、予測時間には影響しない

---

## 使用フローでの位置づけ

### Aerial+の全体フロー

```
Phase 1: データ前処理
  └─> create_input_vectors()
      ├─> one-hotエンコード
      └─> ✅ 辞書キャッシュ作成（1回のみ）

Phase 2: ニューラルネットワーク訓練
  └─> train()
      └─> Autoencoderの訓練

Phase 3: ルール生成
  └─> generate_rules()
      └─> ニューラルベースでルール生成
          （この時点では統計値は未計算）

Phase 4: ルール評価・統計計算 ⚡ 改善対象
  └─> calculate_stats() / calculate_basic_stats()
      ├─> support計算 ← ✅ 最適化が効く
      ├─> confidence計算 ← ✅ 最適化が効く
      └─> coverage計算 ← ✅ 最適化が効く
```

### 改善箇所の特徴

- Aerial+のコアアルゴリズム（`generate_rules`）: 変更なし
- ルール統計計算（後処理）: **大幅改善**

---

## CBA分類での用途詳細

### 訓練時の処理フロー

```
1. Aerial+でルール生成
   ↓
2. calculate_basic_stats()でsupport/confidenceを計算 ← ⚡ 最適化対象
   ↓
3. ルールをconfidence → support → lengthでソート
   ↓
4. ソート順にルールを評価し、エラーが最小のルールセットを選択
   ↓
5. 分類器を構築
```

### 予測時の処理フロー（最適化の影響なし）

```
1. ソート済みルールを順にチェック
   ↓
2. 最初にマッチしたルールのconsequentを返す
   （統計計算は行わない）
```

---

## メモリ使用量への影響

### 追加メモリ

- 辞書1つ分: `{key: idx for idx, key in enumerate(vector_tracker)}`
- サイズ: キー数 × 約50バイト
- 例: 100要素の辞書で約5KB
- 1000要素の辞書で約50KB

### トレードオフ

わずかなメモリ増加（5-50KB）で大幅な速度向上を実現

---

## 実装の特徴

### 利点

✅ 既存ロジックへの影響が最小限  
✅ 辞書参照に変更しただけでシンプル  
✅ リンターエラーなし  
✅ コメント付きで保守性向上  
✅ 既存の並列処理との互換性維持  
✅ 追加メモリが無視可能レベル  

### 実装難易度

★☆☆（非常に簡単）

- 既存の辞書作成コードを再利用
- `.index()`を辞書アクセスに置き換えるだけ
- 4箇所の変更で完了

---

## まとめ

| 項目 | 改善前 | 改善後 | 改善効果 |
|------|--------|--------|----------|
| **ルックアップ計算量** | O(n) | O(1) | **定数時間化** |
| **`calculate_basic_stats`** | O(T×R×A×n) | O(T×R×A) | **約n倍高速化** |
| **`calculate_stats`** | O(R×A×n) | O(R×A) | **約n倍高速化** |
| **メモリ増加** | - | 約5-50KB | **無視可能** |
| **実装難易度** | - | ★☆☆ | **非常に簡単** |

### 効果が顕著なケース

- ルール数が多い（1000+）
- トランザクション数が多い（10000+）
- `vector_tracker_list`の長さが大きい（100+）

### 結論

シンプルな実装で大幅な性能改善を実現し、特に大規模データセットや多数のルールを扱う場合に実行時間が大幅に短縮される効果的な最適化です。

