# Kato Lab Graph Visualize

## 概要

実験・測定で得られた数値データ（テキストファイル）から、ラマン分光スペクトルのグラフ作成・ピーク解析を自動化する Python スクリプト群です。

---

## モジュール構成

| ファイル | 役割 |
| --- | --- |
| `spectra_common.py` | I/O ユーティリティ（`SpectrumDataStore`・`PlotStyle`） |
| `spectra_peak_fit.py` | フィッティングモデルとアルゴリズム |
| `spectra_analysis.py` | ラマン分光データの解析オーケストレーション・エントリポイント |
| `spectra_plot.py` | 汎用スペクトル可視化スクリプト |

---

## セットアップ

```bash
uv sync
```

`data/` ディレクトリを作成し、解析対象の `.txt` ファイルを配置してください（2列: X, Y）。

```bash
(プロジェクトルート)/
  ├─ data/
  │   ├─ sample1.txt
  │   └─ sample2.txt
  └─ ...
```

---

## 開発

```bash
uv sync --extra dev

uv run ruff format .
```

---

## 実行

```bash
uv run analyze   # ラマン分光解析
uv run plot      # 汎用スペクトルプロット
```

---

## `analyze` — ラマン分光解析

### 処理フロー

各 `data/**/*.txt` に対して以下を実行します。

1. データ読込（2列: `X` cm⁻¹, `Y` 強度）
2. **Si シフト補正**（`ENABLE_SI_SHIFT = True` のとき）  
   Si ピーク（`480–560 cm⁻¹`）をフィットし、理論値 `520.8 cm⁻¹` との差分で X 軸を補正  
   → 補正後 Si ピーク振幅を `1` に正規化
3. **Si 補正なし**（`ENABLE_SI_SHIFT = False` のとき）  
   `X > 500 cm⁻¹` の範囲を最大値で正規化
4. `D`・`G`・`2D` ピークをフィット
5. 強度比 `I_D/I_G`・`I_2D/I_G`・`I_D/I_2D` を計算

### フィッティングモデル

`spectra_analysis.py` 冒頭の `FIT_METHOD` で手法を切り替えられます。

```python
FIT_METHOD = FitMethod.GAUSSIAN      # ガウスフィッティング（デフォルト）
FIT_METHOD = FitMethod.LORENTZIAN    # ローレンツフィッティング
```

| 手法 | 関数 | ピーク面積 |
| --- | --- | --- |
| Gaussian | `A·exp(-(x-μ)²/(2σ²)) + m·x + b` | `A·σ·√(2π)` |
| Lorentzian | `A·γ²/((x-μ)²+γ²) + m·x + b` | `A·γ·π` |

### ピーク窓（cm⁻¹）

| ピーク | 範囲 |
| --- | --- |
| Si | 480–560 |
| D | 1250–1450 |
| G | 1500–1650 |
| 2D | 2600–2800 |

### 出力

`output/spectra_analysis/` 以下に生成されます。

```bash
output/spectra_analysis/
  ├─ summary.csv                        # 全ファイルの解析結果一覧
  ├─ tables/.../*_corrected_normalized.csv  # 入力フォルダ構造を維持して保存
  └─ plots/.../*_analysis.png               # 入力フォルダ構造を維持して保存
```

#### `summary.csv` の主な列

| 列名 | 内容 |
| --- | --- |
| `relative_path` | `data/` 配下の相対パス |
| `label_xxx` | ファイル名中の `XXX-YYY` 形式を抽出した動的列。値は数値部分のみ保持。例: `anneal-1h` → `label_anneal=1`, `position-645mm` → `label_position=645` |
| `fit_method` | 使用したフィッティング手法 |
| `si_shift_enabled` | Si シフト補正の有無 |
| `si_center_raw_cm-1` | 補正前 Si ピーク中心 |
| `si_shift_correction_cm-1` | 適用した補正量 |
| `D_amplitude_norm` / `G_amplitude_norm` / `2D_amplitude_norm` | 正規化後ピーク振幅 |
| `I_D/I_G` / `I_2D/I_G` / `I_D/I_2D` | 強度比 |
| `R2_D` / `R2_G` / `R2_2D` | 各フィットの決定係数 R² |

---

## `plot` — 汎用スペクトルプロット

各スペクトルを 1 枚の PNG として保存するシンプルなプロッタです。  
フィッティングは行いません。

`spectra_plot.py` 冒頭の定数で表示範囲を変更できます。

```python
X_MIN = 1000   # X 軸最小値 (cm⁻¹)
X_MAX = 3000   # X 軸最大値 (cm⁻¹)
```

出力先: `output/spectra_plot/`
