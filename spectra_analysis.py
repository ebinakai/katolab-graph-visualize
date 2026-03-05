import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from spectra_common import PlotStyle, SpectrumDataStore

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output") / "spectra_analysis"
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"

# Constants
PLOT_STYLE = PlotStyle(figure_size=(12, 8), dpi=300, x_label="Raman shift (cm^-1)", y_label="Intensity (a.u.)")
DATA_STORE = SpectrumDataStore(DATA_DIR)

# Peak settings (cm^-1)
SI_THEORETICAL_CENTER = 520.8
PEAK_WINDOWS: Dict[str, Tuple[float, float]] = {
    "Si": (480.0, 560.0),
    "D": (1250.0, 1450.0),
    "G": (1500.0, 1650.0),
    "2D": (2600.0, 2800.0),
}


@dataclass
class PeakFitResult:
    name: str
    amplitude: float
    center: float
    sigma: float
    slope: float
    intercept: float
    r_squared: float

    @property
    def area(self) -> float:
        return self.amplitude * self.sigma * np.sqrt(2.0 * np.pi)


def gaussian_with_linear_baseline(
    x: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
    slope: float,
    intercept: float,
) -> np.ndarray:
    gauss = amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma**2))
    baseline = slope * x + intercept
    return gauss + baseline


def fit_peak(
    df: pd.DataFrame,
    peak_name: str,
    window: Tuple[float, float],
    center_bounds: Optional[Tuple[float, float]] = None,
) -> Optional[PeakFitResult]:
    x_min, x_max = window
    mask = (df["X"] >= x_min) & (df["X"] <= x_max)
    sub = df.loc[mask]

    if len(sub) < 8:
        logging.warning("%s フィット失敗: データ点不足", peak_name)
        return None

    x = sub["X"].to_numpy(dtype=float)
    y = sub["Y"].to_numpy(dtype=float)

    slope_guess = float((y[-1] - y[0]) / (x[-1] - x[0])) if x[-1] != x[0] else 0.0
    intercept_guess = float(np.median(y))
    amp_guess = max(float(y.max() - y.min()), 1.0)
    center_guess = float(x[np.argmax(y)])
    sigma_guess = max((x_max - x_min) / 10.0, 3.0)

    # Baselineを簡易推定して再初期化パラメータを作る
    baseline = slope_guess * x + intercept_guess
    y_corr = y - baseline
    top_idx = int(np.argmax(y_corr))
    center_guess_alt = float(x[top_idx])
    amp_guess_alt = max(float(np.max(y_corr)), 1.0)
    sigma_guess_alt = max((x_max - x_min) / 20.0, 2.0)

    p0_candidates = [
        [amp_guess, center_guess, sigma_guess, slope_guess, intercept_guess],
        [amp_guess_alt, center_guess_alt, sigma_guess_alt, slope_guess, intercept_guess],
    ]
    center_lower, center_upper = (x_min, x_max)
    if center_bounds is not None:
        center_lower = max(x_min, center_bounds[0])
        center_upper = min(x_max, center_bounds[1])
        if center_lower >= center_upper:
            logging.warning("%s フィット失敗: center_bounds が不正です", peak_name)
            return None

    lower = [0.0, center_lower, 0.5, -np.inf, -np.inf]
    upper = [np.inf, center_upper, (x_max - x_min), np.inf, np.inf]

    # center初期値が拘束範囲外のときは範囲内に寄せる
    for p0 in p0_candidates:
        p0[1] = float(np.clip(p0[1], center_lower + 1e-6, center_upper - 1e-6))

    popt = None
    last_error: Optional[Exception] = None
    for p0 in p0_candidates:
        try:
            popt, _ = curve_fit(
                gaussian_with_linear_baseline,
                x,
                y,
                p0=p0,
                bounds=(lower, upper),
                maxfev=100000,
            )
            break
        except Exception as e:
            last_error = e

    if popt is None:
        logging.warning("%s フィット失敗: %s", peak_name, last_error)
        return None

    y_fit = gaussian_with_linear_baseline(x, *popt)
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return PeakFitResult(
        name=peak_name,
        amplitude=float(popt[0]),
        center=float(popt[1]),
        sigma=float(abs(popt[2])),
        slope=float(popt[3]),
        intercept=float(popt[4]),
        r_squared=r_squared,
    )


def build_fitted_curve(x: np.ndarray, result: PeakFitResult) -> np.ndarray:
    return gaussian_with_linear_baseline(
        x,
        result.amplitude,
        result.center,
        result.sigma,
        result.slope,
        result.intercept,
    )


def calculate_ratios(fits: Dict[str, PeakFitResult]) -> Dict[str, float]:
    d = fits["D"].amplitude
    g = fits["G"].amplitude
    two_d = fits["2D"].amplitude

    return {
        "I_D/I_G": d / g if g != 0 else np.nan,
        "I_2D/I_G": two_d / g if g != 0 else np.nan,
        "I_D/I_2D": d / two_d if two_d != 0 else np.nan,
    }


def analyze_file(file_path: Path) -> Optional[Dict[str, float]]:
    df = DATA_STORE.load_txt(file_path)
    if df.empty:
        return None

    # 1) Siピークで軸シフト補正
    si_fit_raw = fit_peak(df, "Si(raw)", PEAK_WINDOWS["Si"])
    if si_fit_raw is None:
        logging.warning("%s: Siピークが取得できないためスキップ", file_path.name)
        return None

    shift_cm1 = SI_THEORETICAL_CENTER - si_fit_raw.center
    df_corrected = df.copy()
    df_corrected["X"] = df_corrected["X"] + shift_cm1

    # 2) 補正後Siピークの振幅で正規化
    si_fit_corrected = fit_peak(
        df_corrected,
        "Si(corrected)",
        PEAK_WINDOWS["Si"],
        center_bounds=(SI_THEORETICAL_CENTER - 8.0, SI_THEORETICAL_CENTER + 8.0),
    )
    if si_fit_corrected is None or si_fit_corrected.amplitude <= 0:
        logging.warning("%s: 正規化用Siピークが取得できないためスキップ", file_path.name)
        return None

    normalization_factor = si_fit_corrected.amplitude
    df_normalized = df_corrected.copy()
    df_normalized["Y"] = df_normalized["Y"] / normalization_factor

    # 3) D/G/2Dをガウシアン+一次ベースラインでフィット
    peak_fits: Dict[str, PeakFitResult] = {}
    for peak_name in ("D", "G", "2D"):
        fit = fit_peak(df_normalized, peak_name, PEAK_WINDOWS[peak_name])
        if fit is None:
            logging.warning("%s: %sピークのフィットに失敗", file_path.name, peak_name)
            return None
        peak_fits[peak_name] = fit

    # 4) 強度比計算
    ratios = calculate_ratios(peak_fits)

    # 保存: 補正+正規化データ
    out_table = TABLE_DIR / f"{file_path.stem}_corrected_normalized.csv"
    DATA_STORE.save_dataframe(df_normalized, out_table, index=False)

    # 保存: 図
    plot_path = PLOT_DIR / f"{file_path.stem}_analysis.png"
    save_plot(file_path.stem, df, si_fit_raw, shift_cm1, df_normalized, peak_fits, plot_path)

    result = {
        "file": file_path.name,
        "si_center_raw_cm-1": si_fit_raw.center,
        "si_shift_correction_cm-1": shift_cm1,
        "si_center_corrected_cm-1": si_fit_corrected.center,
        "si_normalization_factor": normalization_factor,
        "D_center_cm-1": peak_fits["D"].center,
        "G_center_cm-1": peak_fits["G"].center,
        "2D_center_cm-1": peak_fits["2D"].center,
        "D_amplitude_norm": peak_fits["D"].amplitude,
        "G_amplitude_norm": peak_fits["G"].amplitude,
        "2D_amplitude_norm": peak_fits["2D"].amplitude,
        "I_D/I_G": ratios["I_D/I_G"],
        "I_2D/I_G": ratios["I_2D/I_G"],
        "I_D/I_2D": ratios["I_D/I_2D"],
        "R2_D": peak_fits["D"].r_squared,
        "R2_G": peak_fits["G"].r_squared,
        "R2_2D": peak_fits["2D"].r_squared,
        "normalized_csv": str(out_table),
        "analysis_plot": str(plot_path),
    }
    return result


def save_plot(
    title: str,
    df_raw: pd.DataFrame,
    si_fit_raw: PeakFitResult,
    shift_cm1: float,
    df_normalized: pd.DataFrame,
    peak_fits: Dict[str, PeakFitResult],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=PLOT_STYLE.figure_size, sharex=False)

    # 上段: 生データ + Siフィット
    ax1 = axes[0]
    ax1.plot(df_raw["X"], df_raw["Y"], color="tab:blue", linewidth=1.0, label="Raw")

    si_window = PEAK_WINDOWS["Si"]
    x_si = np.linspace(si_window[0], si_window[1], 400)
    y_si = build_fitted_curve(x_si, si_fit_raw)
    ax1.plot(x_si, y_si, color="tab:red", linewidth=1.5, label="Si fit")

    ax1.axvline(si_fit_raw.center, color="tab:red", linestyle="--", linewidth=1.0)
    ax1.axvline(SI_THEORETICAL_CENTER, color="tab:green", linestyle=":", linewidth=1.0)
    ax1.set_title(f"{title} | Raw and Si calibration")
    ax1.set_xlabel(PLOT_STYLE.x_label)
    ax1.set_ylabel(PLOT_STYLE.y_label)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    ax1.text(
        0.02,
        0.95,
        f"Si raw center={si_fit_raw.center:.3f} cm^-1\nShift correction={shift_cm1:+.3f} cm^-1",
        transform=ax1.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    # 下段: 補正+正規化 + D/G/2Dフィット
    ax2 = axes[1]
    ax2.plot(
        df_normalized["X"],
        df_normalized["Y"],
        color="tab:blue",
        linewidth=1.0,
        label="Corrected + Si-normalized",
    )

    color_map = {"D": "tab:orange", "G": "tab:green", "2D": "tab:purple"}
    for peak_name, fit in peak_fits.items():
        x_min, x_max = PEAK_WINDOWS[peak_name]
        x_fit = np.linspace(x_min, x_max, 400)
        y_fit = build_fitted_curve(x_fit, fit)
        ax2.plot(x_fit, y_fit, color=color_map[peak_name], linewidth=1.5, label=f"{peak_name} fit")
        ax2.axvline(fit.center, color=color_map[peak_name], linestyle="--", linewidth=1.0)

    ax2.set_xlim(1000, 3000)
    ax2.set_title(f"{title} | Graphene peak fitting")
    ax2.set_xlabel(PLOT_STYLE.x_label)
    ax2.set_ylabel("Normalized intensity (Si=1)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    DATA_STORE.save_figure(fig, save_path, dpi=PLOT_STYLE.dpi)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_files = DATA_STORE.list_input_files()
    if not input_files:
        logging.warning("%s に .txt ファイルが見つかりません", DATA_DIR)
        return

    logging.info("%d 個のファイルを解析します", len(input_files))

    records = []
    for file_path in input_files:
        logging.info("処理中: %s", file_path.name)
        result = analyze_file(file_path)
        if result is not None:
            records.append(result)

    if not records:
        logging.warning("有効な解析結果がありませんでした")
        return

    summary_path = OUTPUT_DIR / "summary.csv"
    DATA_STORE.save_dataframe(pd.DataFrame(records), summary_path, index=False)
    logging.info("完了: %s", summary_path)


if __name__ == "__main__":
    main()
