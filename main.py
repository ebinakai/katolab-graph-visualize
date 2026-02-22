import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Tuple, Optional

# Constants
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
X_MIN = 500
X_MAX = 3000
MARGIN_RATIO = 0.05
DPI = 300
FIGURE_SIZE = (8, 5)
X_LABEL = "Raman shift (cm⁻¹)"
Y_LABEL = "Intensity (a.u.)"

def load_data(file_path: Path) -> pd.DataFrame:
    """
    テキストファイルからデータを読み込む。
    空白区切りの値で、ヘッダーがないことを前提とする。
    """
    try:
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=['X', 'Y'])
        return df
    except Exception as e:
        logging.error(f"読み込みエラー {file_path}: {e}")
        return pd.DataFrame()

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

def plot_graph(df: pd.DataFrame, x_min: float, x_max: float, y_min: float, y_max: float, title: str, save_path: Path):
    """
    データをプロットし、グラフを画像として保存する。
    """
    # 出力ディレクトリが存在することを確認
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # 全データをプロット
    ax.plot(df['X'], df['Y'], color='blue', label='Data line')

    ax.set_title(title)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    
    # 軸の制限を設定
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    ax.grid(True)
    ax.legend()

    try:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
        logging.info(f"保存しました: {save_path}")
    except Exception as e:
        logging.error(f"保存エラー {save_path}: {e}")
    finally:
        # メモリ解放のために図を明示的に閉じる
        plt.close(fig)

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # pathlib を使ってファイルを見つける
    input_files = list(DATA_DIR.glob("*.txt"))
    
    if not input_files:
        logging.warning(f"{DATA_DIR} に .txt ファイルが見つかりませんでした")
        return

    logging.info(f"{len(input_files)} 個のファイルが見つかりました。処理を開始します...")

    for file_path in input_files:
        df = load_data(file_path)
        if df.empty:
            continue
            
        y_range = calculate_y_range(df, X_MIN, X_MAX)
        if y_range is None:
            logging.warning(f"スキップ {file_path.name}: 範囲 {X_MIN}-{X_MAX} にデータがありません")
            continue

        y_min, y_max = y_range
        
        # 保存パスを構築: output/data/filename.png
        save_path = OUTPUT_DIR / file_path.parent.name / (file_path.stem + ".png")
        
        plot_graph(df, X_MIN, X_MAX, y_min, y_max, file_path.stem, save_path)

if __name__ == "__main__":
    main()
