import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from spectra_common import PlotStyle, SpectrumDataStore, prepare_input_files
from spectra_peak_fit import FitMethod, PeakFitResult, build_fitted_curve, fit_peak

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output") / "spectra_analysis"
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"

# Constants
PLOT_STYLE = PlotStyle(figure_size=(12, 8), dpi=300, x_label="Raman shift (cm^-1)", y_label="Intensity (a.u.)")

FIT_METHOD = FitMethod.GAUSSIAN
DATA_STORE = SpectrumDataStore(DATA_DIR)

# Peak settings (cm^-1)
SI_THEORETICAL_CENTER = 520.8
ENABLE_SI_SHIFT = False
MIN_X_WITHOUT_SI_SHIFT = 500.0
PEAK_WINDOWS: Dict[str, Tuple[float, float]] = {
    "Si": (480.0, 560.0),
    "D": (1270.0, 1450.0),
    "G": (1500.0, 1650.0),
    "2D": (2600.0, 2800.0),
}
PEAK_CENTER_BOUNDS: Dict[str, Tuple[float, float]] = {
    "D": (1290.0, 1435.0),
    "G": (1565.0, 1625.0),
    "2D": (2660.0, 2750.0),
}
PEAK_WIDTH_BOUNDS: Dict[str, Tuple[float, float]] = {
    "D": (2.0, 120.0),
    "G": (2.0, 80.0),
    "2D": (2.0, 120.0),
}
FIT_CENTER_EDGE_MARGIN_CM1 = 0.0
FIT_BOUNDARY_TOLERANCE_CM1: Dict[str, float] = {
    "D": 2.0,
    "G": 0.5,
    "2D": 0.5,
}
MIN_R2_FOR_RATIO: Dict[str, float] = {
    "D": 0.30,
    "G": 0.80,
    "2D": 0.80,
}
COMPACT_SUMMARY_COLUMNS = [
    "file",
    "D_center_cm-1",
    "G_center_cm-1",
    "2D_center_cm-1",
    "I_D/I_G",
    "I_2D/I_G",
    "I_D/I_2D",
    "R2_D",
    "R2_G",
    "R2_2D",
]
LONG_SUMMARY_RATIO_COLUMNS = ["I_D/I_G", "I_2D/I_G"]
LONG_SUMMARY_ID_COLUMNS = ["growth_time_hour", "file", "sample", "measurement_id"]
ENABLE_BASELINE_CORRECTION = True
BASELINE_ASLS_LAMBDA = 1.0e6
BASELINE_ASLS_P = 0.01
BASELINE_ASLS_ITER = 10


def calculate_ratios(fits: Dict[str, PeakFitResult]) -> Dict[str, float]:
    d = fits["D"].amplitude
    g = fits["G"].amplitude
    two_d = fits["2D"].amplitude

    return {
        "I_D/I_G": d / g if g != 0 else np.nan,
        "I_2D/I_G": two_d / g if g != 0 else np.nan,
        "I_D/I_2D": d / two_d if two_d != 0 else np.nan,
    }


def is_center_near_bound(center: float, bounds: Tuple[float, float], tolerance: float) -> bool:
    return center <= bounds[0] + tolerance or center >= bounds[1] - tolerance


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return np.nan
    return numerator / denominator


def baseline_correct_asls(
    y: np.ndarray,
    lam: float = BASELINE_ASLS_LAMBDA,
    p: float = BASELINE_ASLS_P,
    n_iter: int = BASELINE_ASLS_ITER,
) -> np.ndarray:
    if y.size < 3:
        return y.copy()

    n = y.size
    d = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    w = np.ones(n, dtype=float)

    baseline = np.zeros(n, dtype=float)
    for _ in range(n_iter):
        w_matrix = sparse.diags(w, 0, shape=(n, n), format="csc")
        z_matrix = w_matrix + lam * (d.T @ d)
        baseline = spsolve(z_matrix, w * y)
        w = np.where(y > baseline, p, 1.0 - p)

    corrected = y - baseline
    return np.maximum(corrected, 0.0)


def parse_growth_time_hour(file_path: Path) -> float:
    path_text = str(file_path)
    growth_match = re.search(r"growth_(\d+(?:\.\d+)?)h", path_text, flags=re.IGNORECASE)
    if growth_match:
        return float(growth_match.group(1))

    generic_match = re.search(r"(\d+(?:\.\d+)?)h", path_text, flags=re.IGNORECASE)
    if generic_match:
        return float(generic_match.group(1))

    return np.nan


def parse_measurement_id(file_name: str) -> float:
    stem = Path(file_name).stem
    match = re.search(r"_(\d+)$", stem)
    if not match:
        return np.nan
    return float(match.group(1))


def parse_sample_name(file_name: str) -> str:
    stem = Path(file_name).stem
    return re.sub(r"_\d+$", "", stem)


def build_long_summary(summary_full_df: pd.DataFrame) -> pd.DataFrame:
    id_columns = [col for col in LONG_SUMMARY_ID_COLUMNS if col in summary_full_df.columns]
    ratio_columns = [col for col in LONG_SUMMARY_RATIO_COLUMNS if col in summary_full_df.columns]
    if not ratio_columns:
        return pd.DataFrame(columns=id_columns + ["ratio_type", "value"])

    return summary_full_df.melt(
        id_vars=id_columns,
        value_vars=ratio_columns,
        var_name="ratio_type",
        value_name="value",
    )


def analyze_file(
    file_path: Path,
    enable_si_shift: bool = ENABLE_SI_SHIFT,
    fit_method: FitMethod = FIT_METHOD,
) -> Optional[Dict[str, float]]:
    df = DATA_STORE.load_txt(file_path)
    if df.empty:
        return None

    si_fit_raw: Optional[PeakFitResult] = None
    si_fit_corrected: Optional[PeakFitResult] = None
    shift_cm1 = 0.0
    normalization_factor = 1.0
    growth_time_hour = parse_growth_time_hour(file_path)
    measurement_id = parse_measurement_id(file_path.name)
    sample_name = parse_sample_name(file_path.name)

    if enable_si_shift:
        # 1) Siピークで軸シフト補正
        si_fit_raw = fit_peak(df, "Si(raw)", PEAK_WINDOWS["Si"], method=fit_method)
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
            method=fit_method,
        )
        if si_fit_corrected is None or si_fit_corrected.amplitude <= 0:
            logging.warning("%s: 正規化用Siピークが取得できないためスキップ", file_path.name)
            return None

        normalization_factor = si_fit_corrected.amplitude
        df_normalized = df_corrected.copy()
        df_normalized["Y"] = df_normalized["Y"] / normalization_factor
    else:
        # Si補正なし: 低波数域を除外してから最大値で正規化
        df_filtered = df.loc[df["X"] > MIN_X_WITHOUT_SI_SHIFT].copy()
        if df_filtered.empty:
            logging.warning(
                "%s: X > %.1f cm^-1 のデータがないためスキップ",
                file_path.name,
                MIN_X_WITHOUT_SI_SHIFT,
            )
            return None

        normalization_scale = float(df_filtered["Y"].abs().max())
        if normalization_scale <= 0:
            logging.warning("%s: 正規化スケールが0のためスキップ", file_path.name)
            return None

        df_normalized = df_filtered.copy()
        df_normalized["Y"] = df_normalized["Y"] / normalization_scale

    if ENABLE_BASELINE_CORRECTION:
        y_values = df_normalized["Y"].to_numpy(dtype=float)
        y_corrected = baseline_correct_asls(y_values)
        corrected_scale = float(np.max(np.abs(y_corrected)))
        if corrected_scale <= 0.0:
            logging.warning("%s: ベースライン補正後の信号がゼロのためスキップ", file_path.name)
            return None
        df_normalized = df_normalized.copy()
        df_normalized["Y"] = y_corrected / corrected_scale

    # 3) D/G/2Dをフィット
    peak_fits: Dict[str, PeakFitResult] = {}
    center_at_bound: Dict[str, bool] = {}
    peak_reliable: Dict[str, bool] = {}
    for peak_name in ("D", "G", "2D"):
        fit = fit_peak(
            df_normalized,
            peak_name,
            PEAK_WINDOWS[peak_name],
            center_bounds=PEAK_CENTER_BOUNDS[peak_name],
            width_bounds=PEAK_WIDTH_BOUNDS[peak_name],
            center_edge_margin=FIT_CENTER_EDGE_MARGIN_CM1,
            method=fit_method,
        )
        if fit is None:
            logging.warning("%s: %sピークのフィットに失敗", file_path.name, peak_name)
            return None
        peak_fits[peak_name] = fit
        center_at_bound[peak_name] = is_center_near_bound(
            fit.center,
            PEAK_CENTER_BOUNDS[peak_name],
            FIT_BOUNDARY_TOLERANCE_CM1[peak_name],
        )
        peak_reliable[peak_name] = (not center_at_bound[peak_name]) and (fit.r_squared >= MIN_R2_FOR_RATIO[peak_name])
        if center_at_bound[peak_name]:
            logging.warning(
                "%s: %sピークcenterが拘束境界に近いため、信頼性フラグを付与します (center=%.3f)",
                file_path.name,
                peak_name,
                fit.center,
            )
        if fit.r_squared < MIN_R2_FOR_RATIO[peak_name]:
            logging.warning(
                "%s: %sピークR^2が低いため、信頼性フラグを付与します (R^2=%.3f, min=%.2f)",
                file_path.name,
                peak_name,
                fit.r_squared,
                MIN_R2_FOR_RATIO[peak_name],
            )

    # 4) 強度比計算
    d = peak_fits["D"].amplitude
    g = peak_fits["G"].amplitude
    two_d = peak_fits["2D"].amplitude
    ratios = {
        "I_D/I_G": safe_ratio(d, g),
        "I_2D/I_G": safe_ratio(two_d, g),
        "I_D/I_2D": safe_ratio(d, two_d),
    }

    # 保存: 補正+正規化データ
    out_table = TABLE_DIR / f"{file_path.stem}_corrected_normalized.csv"
    DATA_STORE.save_dataframe(df_normalized, out_table, index=False)

    # 保存: 図
    plot_path = PLOT_DIR / f"{file_path.stem}_analysis.png"
    save_plot(
        file_path.stem,
        df,
        si_fit_raw,
        shift_cm1,
        df_normalized,
        peak_fits,
        plot_path,
        si_shift_enabled=enable_si_shift,
    )

    result = {
        "file": file_path.name,
        "sample": sample_name,
        "measurement_id": measurement_id,
        "growth_time_hour": growth_time_hour,
        "fit_method": fit_method.value,
        "si_shift_enabled": enable_si_shift,
        "baseline_correction_enabled": ENABLE_BASELINE_CORRECTION,
        "si_center_raw_cm-1": si_fit_raw.center if si_fit_raw is not None else np.nan,
        "si_shift_correction_cm-1": shift_cm1,
        "si_center_corrected_cm-1": si_fit_corrected.center if si_fit_corrected is not None else np.nan,
        "si_normalization_factor": normalization_factor if si_fit_corrected is not None else np.nan,
        "x_filter_min_cm-1": np.nan if enable_si_shift else MIN_X_WITHOUT_SI_SHIFT,
        "D_center_cm-1": peak_fits["D"].center,
        "G_center_cm-1": peak_fits["G"].center,
        "2D_center_cm-1": peak_fits["2D"].center,
        "D_center_at_bound": center_at_bound["D"],
        "G_center_at_bound": center_at_bound["G"],
        "2D_center_at_bound": center_at_bound["2D"],
        "D_peak_reliable": peak_reliable["D"],
        "G_peak_reliable": peak_reliable["G"],
        "2D_peak_reliable": peak_reliable["2D"],
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
    si_fit_raw: Optional[PeakFitResult],
    shift_cm1: float,
    df_normalized: pd.DataFrame,
    peak_fits: Dict[str, PeakFitResult],
    save_path: Path,
    si_shift_enabled: bool,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=PLOT_STYLE.figure_size, sharex=False)

    # 上段: 生データ + Siフィット
    ax1 = axes[0]
    ax1.plot(df_raw["X"], df_raw["Y"], color="tab:blue", linewidth=1.0, label="Raw")

    if si_shift_enabled and si_fit_raw is not None:
        si_window = PEAK_WINDOWS["Si"]
        x_si = np.linspace(si_window[0], si_window[1], 400)
        y_si = build_fitted_curve(x_si, si_fit_raw)
        ax1.plot(x_si, y_si, color="tab:red", linewidth=1.5, label="Si fit")
        ax1.axvline(si_fit_raw.center, color="tab:red", linestyle="--", linewidth=1.0)
        ax1.axvline(SI_THEORETICAL_CENTER, color="tab:green", linestyle=":", linewidth=1.0)
        si_text = f"Si raw center={si_fit_raw.center:.3f} cm^-1\nShift correction={shift_cm1:+.3f} cm^-1"
        ax1.set_title(f"{title} | Raw and Si calibration")
    else:
        si_text = "Si calibration skipped"
        ax1.set_title(f"{title} | Raw (Si calibration OFF)")

    ax1.set_xlabel(PLOT_STYLE.x_label)
    ax1.set_ylabel(PLOT_STYLE.y_label)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    ax1.text(
        0.02,
        0.95,
        si_text,
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
        label=(
            "Corrected + baseline-corrected + Si-normalized"
            if si_shift_enabled
            else "X>500 filtered + baseline-corrected + max-normalized"
        ),
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
    ax2.set_ylabel("Normalized intensity (Si=1)" if si_shift_enabled else "Normalized intensity (max=1)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    DATA_STORE.save_figure(fig, save_path, dpi=PLOT_STYLE.dpi)
    plt.close(fig)


def main() -> None:
    input_files = prepare_input_files(DATA_STORE, DATA_DIR)
    if not input_files:
        return

    logging.info("Siシフト補正: %s", "ON" if ENABLE_SI_SHIFT else "OFF")
    logging.info("ベースライン補正: %s", "ON" if ENABLE_BASELINE_CORRECTION else "OFF")
    logging.info("フィッティング手法: %s", FIT_METHOD.value)

    records = []
    for file_path in input_files:
        logging.info("処理中: %s", file_path.name)
        result = analyze_file(file_path, enable_si_shift=ENABLE_SI_SHIFT, fit_method=FIT_METHOD)
        if result is not None:
            records.append(result)

    summary_path = OUTPUT_DIR / "summary.csv"
    summary_wide_path = OUTPUT_DIR / "summary_wide.csv"
    summary_full_path = OUTPUT_DIR / "summary_full.csv"
    summary_full_df = pd.DataFrame(records)
    compact_columns = [col for col in COMPACT_SUMMARY_COLUMNS if col in summary_full_df.columns]
    summary_compact_df = summary_full_df.loc[:, compact_columns]
    summary_long_df = build_long_summary(summary_full_df)

    DATA_STORE.save_dataframe(summary_long_df, summary_path, index=False)
    DATA_STORE.save_dataframe(summary_compact_df, summary_wide_path, index=False)
    DATA_STORE.save_dataframe(summary_full_df, summary_full_path, index=False)
    if not records:
        logging.warning(
            "有効な解析結果がありませんでした。空のサマリーを保存しました: %s, %s, %s",
            summary_path,
            summary_wide_path,
            summary_full_path,
        )
        return

    logging.info("完了: %s (long), %s (wide), %s (full)", summary_path, summary_wide_path, summary_full_path)


if __name__ == "__main__":
    main()
