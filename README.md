# Kato Lab Graph Visualize

## 概要

実験・測定で得られた数値データ（テキストファイル）から、自動でグラフを作成し画像として保存するPythonスクリプトです。

特に、分光データのように  **「特定の範囲だけを拡大して見たいが、範囲外のノイズや外れ値のせいでY軸のスケールが潰れてしまう」**  という問題を解決するために開発されました。

指定したX軸の範囲に基づいてY軸を自動で最適化し、見やすいグラフを生成します。

---

## 使い方 (uv ネイティブワークフロー)

本プロジェクトは、高速なPythonパッケージ管理ツール [uv](https://github.com/astral-sh/uv) のネイティブな利用を前提としています。

### 1. 準備

**a. 仮想環境の作成と有効化**  

```bash
# 仮想環境を作成
uv venv

source .venv/bin/activate
```

**b. 依存パッケージのインストール**
`pyproject.toml` と `uv.lock` ファイルから依存関係をインストールします。

```bash
uv sync
```

**c. 解析したいデータの配置**
`data` ディレクトリを作成し、その中に解析対象のテキストファイル（`.txt`）を配置してください。

```bash
(プロジェクトルート)/
  ├─ main.py
  ├─ data/  <-- ここにテキストファイルを置く
  │   └─ sample1.txt
  │   └─ sample2.txt
  └─ ...
```

### 2. 実行

```bash
uv run main.py
```

このコマンドは、`main.py` を実行します。

### 3. 結果の確認

処理が完了すると、`output/data` ディレクトリにグラフの画像ファイル（`.png`）が生成されています。

```bash
(プロジェクトルート)/
  └─ output/
      └─ data/
          └─ sample1.png
          └─ sample2.png
```

---

## 設定の変更

グラフのX軸範囲などを変更したい場合は、`main.py` の冒頭にある `Constants` セクションを直接編集してください。

```python
# main.py

# Constants
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
X_MIN = 500  # グラフのX軸 最小値
X_MAX = 3000 # グラフのX軸 最大値
MARGIN_RATIO = 0.05
# ...
```

---

## ラマン分光解析 (`spectram_analysis.py`)

`main.py` が「可視化用」なのに対して、`spectram_analysis.py` はラマン分光データのピーク解析を行うスクリプトです。

### 目的

ラマンシフト（`cm^-1`）を用いて、以下を自動処理します。

1. シリコン（Si）ピーク位置の理論値 `520.8 cm^-1` からのズレを算出し、X軸を補正
2. 補正後Siピークの強度（振幅）を `1` に正規化
3. ガウシアン + 一次ベースラインでピークフィッティング
4. グラフェンの `D`, `G`, `2D` ピークの強度比を算出

### 実行方法

```bash
uv run python spectram_analysis.py
```

### 処理フロー

各 `data/*.txt` に対して次の順で解析します。

1. データ読込（2列: `X`, `Y`）
2. Si領域（`480-560 cm^-1`）をフィットして、理論値 `520.8` との差分 `shift` を取得
3. `X_corrected = X + shift` で軸補正
4. 補正後Siを再フィットし、その振幅 `A_Si` で `Y_normalized = Y / A_Si`
5. `D(1250-1450)`, `G(1500-1650)`, `2D(2600-2800)` をフィット
6. 強度比 `I_D/I_G`, `I_2D/I_G`, `I_D/I_2D` を計算

### フィットモデル

各ピークは以下でフィットします。

```text
y(x) = A * exp(-(x - mu)^2 / (2*sigma^2)) + (m*x + b)
```

- `A`: ピーク振幅（強度）
- `mu`: ピーク中心
- `sigma`: 幅
- `m*x + b`: 一次ベースライン

### 出力

実行後に `output/spectram_analysis/` 以下を生成します。

- `summary.csv`
  - ファイルごとの補正量、ピーク位置、正規化後ピーク強度、強度比、各フィットの `R^2`
- `tables/*_corrected_normalized.csv`
  - 軸補正 + Si正規化済みスペクトル
- `plots/*_analysis.png`
  - 上段: 生データ + Siフィット（補正量表示）
  - 下段: 補正・正規化データ + D/G/2Dフィット

### 主なサマリ列 (`summary.csv`)

- `si_center_raw_cm-1`: 補正前Si中心
- `si_shift_correction_cm-1`: 適用した補正量（理論値との差）
- `si_center_corrected_cm-1`: 補正後Si中心
- `si_normalization_factor`: 正規化に使ったSi振幅
- `D_amplitude_norm`, `G_amplitude_norm`, `2D_amplitude_norm`: 正規化後ピーク強度
- `I_D/I_G`, `I_2D/I_G`, `I_D/I_2D`: 強度比
- `R2_D`, `R2_G`, `R2_2D`: 各ピークフィットの決定係数
