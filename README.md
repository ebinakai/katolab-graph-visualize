# Kato Lab Graph Visualize

## 概要

実験・測定で得られた数値データ（テキストファイル）から、分光スペクトルのグラフ作成・ピーク解析を行うツール群です。
従来のスクリプトベースの構成から、柔軟な解析と可視化が可能な **Jupyter Notebook** ベースの構成に移行しました。

---

## 解析ノートブック

| ファイル | 役割 |
| --- | --- |
| `raman.ipynb` | **ラマン分光マッピング解析**<br>多点観測データ (.txt) の読込、ベースライン補正 (ALS)、ピークフィッティング、マッピング可視化。 |
| `pl.ipynb` | **PL (Photoluminescence) 解析**<br>単点スペクトルおよびマッピングデータの解析。Voigt 関数を用いた精密なピークフィッティングに対応。 |
| `graphen.ipynb` | **グラフェン品質評価・比較**<br>解析済みデータ (summary.csv) を集計し、2D/G比やG/D比の統計的比較（ヒストグラム、相関図）を実行。 |

---

## 補助スクリプト

| ファイル | 役割 |
| --- | --- |
| `spectra_analysis_single.py` | 単一スペクトル向けの段階的パイプライン解析エントリポイント。 |

---

## セットアップ

本プロジェクトは [uv](https://github.com/astral-sh/uv) を使用してパッケージ管理を行っています。

```bash
uv sync
```

---

## 実行方法

### 1. ノートブックによる解析

Jupyter Lab を起動して、各 `.ipynb` ファイルを実行します。

```bash
uv run jupyter lab
```

### 2. 単一スペクトル解析（CLI）

```bash
uv run spectra_analysis_single.py
```

---

## データ構造

`data/` ディレクトリに、解析対象のテキストファイルを配置します。

```text
data/
  ├─ spectra_mapping/   # マッピングデータ (Raman 等)
  ├─ pl/                # PL データ
  └─ spectra_single/    # 単一スペクトルデータ
```

---

## 解析フロー (Raman/PL Mapping)

1. **データ読込**: ASCII 形式等のテキストデータから波長・座標・強度を抽出。
2. **前処理**: 
   - **ALS (Asymmetric Least Squares)** によるベースライン補正。
   - Savitzky-Golay フィルタによる平滑化。
3. **ピークフィッティング**:
   - `lmfit` を使用した Lorentzian/Gaussian/Voigt 関数による最適化。
   - D, G, 2D 等の主要ピークの強度・中心位置・半値幅を算出。
4. **可視化**:
   - 2D 強度マップの生成。
   - フィッティング結果のデバッグ用プロット。
5. **統計解析 (`graphen.ipynb`)**:
   - 複数サンプル間での品質（2D/G 比等）の比較。

---

## 出力

`output/` ディレクトリ以下に、解析結果の画像および CSV が生成されます。

```text
output/
  ├─ spectra_mapping/   # マッピング結果 (2D Map, quantitative_results.csv)
  ├─ pl/                # PL 解析結果
  └─ spectra_analysis/  # 単一スペクトル解析結果
```
