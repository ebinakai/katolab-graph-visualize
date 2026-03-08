import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Tuple, Optional

from spectra_common import PlotStyle, SpectrumDataStore, prepare_input_files

# Constants
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output/spectra_plot")
X_MIN = 1000
X_MAX = 3000
MARGIN_RATIO = 0.05
PLOT_STYLE = PlotStyle(figure_size=(8, 5), dpi=300, x_label="Raman shift (cm^-1)", y_label="Intensity (a.u.)")
DATA_STORE = SpectrumDataStore(DATA_DIR)

def calculate_y_range(df: pd.DataFrame, x_min: float, x_max: float) -> Optional[Tuple[float, float]]:
    """
    指定されたX軸の範囲に基づいてY軸の範囲を計算する。
    計算された最小値/最大値にマージンを追加する。
    """
    if df.empty:
        return None

    # Xの範囲内のデータをフィルタリング
    mask = (df['X'] >= x_min) & (df['X'] <= x_max)
    filtered_df = df[mask]

    if filtered_df.empty:
        return None

    y_min = filtered_df['Y'].min()
    y_max = filtered_df['Y'].max()
    
    margin = (y_max - y_min) * MARGIN_RATIO
    return y_min - margin, y_max + margin

def plot_graph(
    df: pd.DataFrame,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    title: str,
    save_path: Path,
):
    """
    データをプロットし、グラフを画像として保存する。
    """
    fig, ax = plt.subplots(figsize=PLOT_STYLE.figure_size)
    
    # 全データをプロット
    ax.plot(df['X'], df['Y'], color='blue', label='Data line')

    ax.set_title(title)
    ax.set_xlabel(PLOT_STYLE.x_label)
    ax.set_ylabel(PLOT_STYLE.y_label)
    
    # 軸の制限を設定
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    ax.grid(True)
    ax.legend()

    try:
        DATA_STORE.save_figure(fig, save_path, dpi=PLOT_STYLE.dpi)
        logging.info(f"保存しました: {save_path}")
    except Exception as e:
        logging.error(f"保存エラー {save_path}: {e}")
    finally:
        # メモリ解放のために図を明示的に閉じる
        plt.close(fig)

def main():
    input_files = prepare_input_files(DATA_STORE, DATA_DIR)
    if not input_files:
        return

    for file_path in input_files:
        df = DATA_STORE.load_txt(file_path)
        if df.empty:
            continue
            
        y_range = calculate_y_range(df, X_MIN, X_MAX)
        if y_range is None:
            logging.warning(f"スキップ {file_path.name}: 範囲 {X_MIN}-{X_MAX} にデータがありません")
            continue

        y_min, y_max = y_range
        
        # 保存パスを構築: output/spectra_plot/<data配下の相対パス>.png
        relative_path = file_path.relative_to(DATA_DIR).with_suffix(".png")
        save_path = OUTPUT_DIR / relative_path
        
        plot_graph(df, X_MIN, X_MAX, y_min, y_max, file_path.stem, save_path)

if __name__ == "__main__":
    main()
