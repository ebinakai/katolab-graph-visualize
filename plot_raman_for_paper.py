import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def configure_publication_style():
    """Matplotlibのプロットスタイルを論文用（セリフ系・高解像度）に設定する"""
    plt.rcParams['font.family'] = 'Times New Roman'  # セリフ系
    plt.rcParams['font.size'] = 10                  # 学会用標準サイズ
    plt.rcParams['axes.linewidth'] = 1.0            # 軸の太さ
    plt.rcParams['xtick.direction'] = 'in'          # 目盛り内向き
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['xtick.top'] = True                # 上部にも目盛りを表示
    plt.rcParams['ytick.right'] = True              # 右側にも目盛りを表示

def plot_raman_spectrum(csv_path: Path, output_dir: Path, xlim: list | None = None):
    """CSVから必要なデータ列（補正後強度とフィット）のみを抽出し、指定されたX軸範囲でプロットする"""
    if not csv_path.exists():
        print(f"Error: Target data file {csv_path} does not exist.")
        return
        
    # データの読み込み
    df = pd.read_csv(csv_path)
    
    # 1. X軸列の自動検出
    x_col = None
    for candidate in ['Wavenumber', 'wavenumber', 'X', 'x', 'Wavelength', 'wavelength']:
        if candidate in df.columns:
            x_col = candidate
            break
    if x_col is None:
        x_col = df.columns[0]
        
    configure_publication_style()
    
    # アブストラクト用に最適化したサイズ
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    
    # 2. 表示対象の列（補正後スペクトル強度と全体のフィットカーブ）のみに限定
    # フォーマット: {列名: (色, 線幅, 線種, 凡例ラベル)}
    allowed_cols = {
        'Processed_Intensity': ('#000000', 1.0, '-', 'Experiment'),
        # 'Fit_Curve': ('#0f172a', 1.5, '-', 'Fit')
    }
    
    # 3. 指定した表示対象の列のみをプロット（生データなどは非表示にする）
    for col in df.columns:
        if col in allowed_cols:
            color, lw, ls, label = allowed_cols[col]
            ax.plot(
                df[x_col], 
                df[col], 
                color=color, 
                linewidth=lw, 
                linestyle=ls, 
                label=label
            )
        
    # 学術用の枠線処理（上と右を消してスッキリさせる）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 4. X軸の範囲（指定があれば適用）
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    
    # 軸ラベルの設定
    if 'wave' in x_col.lower():
        ax.set_xlabel('Raman shift (cm$^{-1}$)')
    else:
        ax.set_xlabel(x_col)
    ax.set_ylabel('Intensity (a.u.)')
    
    # 凡例の設定（枠なし）
    if len(allowed_cols) > 1:
        ax.legend(frameon=False, loc='upper left')
    
    # 保存処理
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_output = output_dir / f"{csv_path.stem}_plot.pdf"
    png_output = output_dir / f"{csv_path.stem}_plot.png"
    
    plt.savefig(pdf_output, format='pdf', bbox_inches='tight')
    plt.savefig(png_output, format='png', dpi=300, bbox_inches='tight')
    
    print(f"Plot generated (xlim={xlim}):\n - {pdf_output}\n - {png_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Raman spectrum with specific columns and xlim.")
    parser.add_argument(
        "--input", 
        type=str, 
        default="output/spectra_mapping/S7/representative_spectrum_49.csv",
        help="Path to the processed spectrum CSV file."
    )
    parser.add_argument(
        "--outdir", 
        type=str, 
        default="output/plots",
        help="Directory to save the generated plots."
    )
    parser.add_argument(
        "--xlim", 
        type=float, 
        nargs=2, 
        default=None,
        help="X-axis display limits (min max), e.g., --xlim 1200 3000"
    )
    args = parser.parse_args()
    
    plot_raman_spectrum(Path(args.input), Path(args.outdir), args.xlim)
