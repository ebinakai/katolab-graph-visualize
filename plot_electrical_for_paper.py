import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def configure_publication_style():
    """Matplotlibのプロットスタイルを論文用（セリフ系・高解像度）に設定する"""
    plt.rcParams['font.family'] = 'Times New Roman'  # セリフ系
    plt.rcParams['font.size'] = 14                  # 学会用標準サイズ（拡大）
    plt.rcParams['axes.linewidth'] = 1.0            # 軸の太さ
    plt.rcParams['xtick.direction'] = 'in'          # 目盛り内向き
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['xtick.top'] = True                # 上部にも目盛りを表示
    plt.rcParams['ytick.right'] = True              # 右側にも目盛りを表示

def plot_electrical_properties(csv_path: Path, output_dir: Path, xlim: list = None, ylim: list = None):
    """電気特性 (Id-Vg) データを読み込んでプロットを生成する"""
    if not csv_path.exists():
        print(f"Error: Target data file {csv_path} does not exist.")
        return
        
    # データの読み込み
    df = pd.read_csv(csv_path)
    
    configure_publication_style()
    
    # アブストラクト用に最適化したサイズ (ラマンスペクトルと統一)
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    
    # Y軸データの決定 (Id*10^6 があれば優先、なければ Id * 1e6)
    if 'Id*10^6' in df.columns:
        id_ua = df['Id*10^6']
    else:
        id_ua = df['Id'] * 1e6
        
    # 1. 実験データのプロット (Id-Vg 曲線) - Okabe-Ito Black
    ax.plot(
        df['Vg'], 
        id_ua, 
        color='#000000', 
        linewidth=1.5, 
        label=r'$I_\mathrm{d}$'
    )
    
    # 2. 接線のプロット (liner_fit) - Okabe-Ito Vermillion
    if 'liner_fit' in df.columns:
        # liner_fit も同様に uA 単位で入っているためそのままプロット
        ax.plot(
            df['Vg'], 
            df['liner_fit'], 
            color='#d55e00', 
            linewidth=1.2, 
            linestyle='--',   # 破線
            label='Linear Fit'
        )
        
    # 学術用の枠線処理（上と右を消してスッキリさせる。ラマンと統一）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # X軸の範囲指定（適用）
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
        
    # Y軸の範囲指定（適用）
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
        
    # 軸ラベルの設定（LaTeX形式で単位を記述）
    ax.set_xlabel(r'Gate voltage (V)')
    ax.set_ylabel(r'Drain current ($\mu$A)')
    
    # 凡例の設定（枠なし）
    ax.legend(frameon=False, loc='upper right')

    # 測定条件 (Vd = 100 mV) を凡例の下に記入
    ax.text(
        0.95, 0.68, 
        r'$V_\mathrm{d} = 100$ mV', 
        transform=ax.transAxes, 
        horizontalalignment='right',
        verticalalignment='top'
    )
    
    # 保存処理
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_output = output_dir / f"{csv_path.stem}_plot.pdf"
    png_output = output_dir / f"{csv_path.stem}_plot.png"
    
    plt.savefig(pdf_output, format='pdf', bbox_inches='tight')
    plt.savefig(pdf_output, format='pdf', bbox_inches='tight', transparent=True)
    # plt.savefig(png_output, format='png', dpi=300, bbox_inches='tight', transparent=True)
    
    print(f"Plot generated (xlim={xlim}, ylim={ylim}):\n - {pdf_output}\n - {png_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot electrical properties (Id-Vg) from CSV.")
    parser.add_argument(
        "--input", 
        type=str, 
        default="/Users/kaiebina/agents/0040_projects/20260702_fntg71_abstract/ref/d9_Id-Vg_Vg--40-60_Vd100mV_RP-27Pa.csv",
        help="Path to the Id-Vg CSV file."
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
        help="X-axis display limits (min max), e.g., --xlim -40 40"
    )
    parser.add_argument(
        "--ylim", 
        type=float, 
        nargs=2, 
        default=[200.0, 500.0],
        help="Y-axis display limits (min max), e.g., --ylim 200 500"
    )
    args = parser.parse_args()
    
    plot_electrical_properties(Path(args.input), Path(args.outdir), args.xlim, args.ylim)
