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
