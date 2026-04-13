import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

formatter = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=formatter)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

DATA_DIR = Path("data") / "spectra_single"
OUTPUT_DIR = Path("output") / "spectra_analysis_single"
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"


@dataclass(frozen=True)
class PlotStyle:
    figure_size: tuple[float, float] = (8, 5)
    dpi: int = 300
    x_label: str = "Raman shift (cm^-1)"
    y_label: str = "Intensity (a.u.)"


PLOT_STYLE = PlotStyle(
    figure_size=(12, 8),
    dpi=300,
    x_label="Raman shift (cm^-1)",
    y_label="Intensity (a.u.)",
)

SI_PEAK_RANGE = (500.0, 540.0)
SI_TARGET = 520.7
NORMALIZATION_RANGE = (1000.0, 3000.0)
D_PEAK_RANGE = (1300.0, 1400.0)
G_PEAK_RANGE = (1550.0, 1600.0)
TWO_D_PEAK_RANGE = (2650.0, 2750.0)

ENABLE_SI_SHIFT = True
ENABLE_BASELINE = True
ENABLE_SMOOTHING = False
SMOOTH_WINDOW = 11
SMOOTH_POLY = 3
PEAK_INTENSITY_METHOD = "topn"
PEAK_TOPN = 3


def list_input_files(
    target_dir: Path, pattern: str = "*.txt", recursive: bool = True
) -> list[Path]:
    paths = target_dir.rglob(pattern) if recursive else target_dir.glob(pattern)
    return sorted(path for path in paths if path.is_file())


def load_txt(file_path: Path, sort_x: bool = True) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, sep=r"\s+", header=None, names=["X", "Y"])
        df = df.dropna()
        if sort_x:
            df = df.sort_values("X")
        return df.reset_index(drop=True)
    except Exception as exc:
        logger.error("Failed to load %s: %s", file_path, exc)
        return pd.DataFrame(columns=["X", "Y"])


def save_dataframe(df: pd.DataFrame, save_path: Path, index: bool = False) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=index)


def save_figure(fig, save_path: Path, dpi: int, bbox_inches: str = "tight") -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)


def prepare_input_files(data_dir: Path, pattern: str = "*.txt") -> list[Path]:
    input_files = list_input_files(data_dir, pattern=pattern, recursive=True)

    if not input_files:
        logger.warning("%s has no .txt files", data_dir)
        return []

    logger.info("%d files found. Starting analysis.", len(input_files))
    return input_files


def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10):
    if niter <= 0 or y.size < 3:
        return y.copy()

    length = len(y)
    diff = sparse.diags(
        [1.0, -2.0, 1.0],
        [0, -1, -2],  # type: ignore[arg-type]
        shape=(length, length - 2),
        dtype=float,
    )
    weights = sparse.eye(length, dtype=float)
    baseline = y.copy()

    for _ in range(niter):
        weights.setdiag(p * (y > baseline) + (1 - p) * (y < baseline))
        system = (weights + lam * diff @ diff.T).tocsc()
        baseline = spsolve(system, weights @ y)

    return baseline


def smooth_signal(y: np.ndarray, window: int = SMOOTH_WINDOW, poly: int = SMOOTH_POLY):
    if y.size < window:
        return y.copy()
    if window % 2 == 0:
        window += 1
    if window <= poly:
        return y.copy()
    return savgol_filter(y, window_length=window, polyorder=poly)


def filter_positive_x(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.loc[df["X"] > 0].copy()
    return filtered.reset_index(drop=True)


def apply_si_shift(df: pd.DataFrame, si_range=SI_PEAK_RANGE, target=SI_TARGET):
    mask = (df["X"] >= si_range[0]) & (df["X"] <= si_range[1])
    sub = df.loc[mask]

    if sub.empty:
        logger.warning("Si range %.1f-%.1f cm^-1 has no data", si_range[0], si_range[1])
        return df.copy(), np.nan, np.nan

    peak_idx = sub["Y"].to_numpy(dtype=float).argmax()
    peak_pos = float(sub.iloc[peak_idx]["X"])
    shift = target - peak_pos

    corrected = df.copy()
    corrected["X"] = corrected["X"] + shift
    return corrected, peak_pos, shift


def subtract_baseline(df: pd.DataFrame):
    corrected = df.copy()
    baseline = np.asarray(
        baseline_als(corrected["Y"].to_numpy(dtype=float)), dtype=float
    )
    corrected_y = corrected["Y"].to_numpy(dtype=float) - baseline
    corrected["Y"] = pd.Series(corrected_y, index=corrected.index, dtype=float)
    corrected["baseline"] = pd.Series(baseline, index=corrected.index, dtype=float)
    return corrected


def apply_smoothing(df: pd.DataFrame):
    smoothed = df.copy()
    smoothed_y = np.asarray(
        smooth_signal(smoothed["Y"].to_numpy(dtype=float)),
        dtype=float,
    )
    smoothed["Y"] = pd.Series(smoothed_y, index=smoothed.index, dtype=float)
    return smoothed


def extract_range(df: pd.DataFrame, x_range: tuple[float, float]) -> pd.DataFrame:
    mask = (df["X"] >= x_range[0]) & (df["X"] <= x_range[1])
    return df.loc[mask].copy()


def _extract_intensity(values: np.ndarray, method: str = "topn", n: int = 3) -> float:
    if values.size == 0:
        return np.nan
    if method == "max":
        return float(np.max(values))
    if method == "mean":
        return float(np.mean(values))
    if method == "topn":
        count = min(n, values.size)
        return float(np.mean(np.sort(values)[-count:]))
    if method == "percentile":
        return float(np.percentile(values, 95))
    raise ValueError(f"Unknown method: {method}")


def normalize_by_si_peak(
    df: pd.DataFrame,
    si_range=SI_PEAK_RANGE,
    method: str = PEAK_INTENSITY_METHOD,
    n: int = PEAK_TOPN,
):
    sub = extract_range(df, si_range)
    if sub.empty:
        normalized = df.copy()
        normalized["Y"] = np.nan
        return normalized, np.nan, np.nan

    peak_idx = sub["Y"].to_numpy(dtype=float).argmax()
    peak_x = float(sub.iloc[peak_idx]["X"])
    scale = _extract_intensity(sub["Y"].to_numpy(dtype=float), method=method, n=n)

    normalized = df.copy()
    if not np.isfinite(scale) or scale == 0:
        normalized["Y"] = np.nan
        return normalized, peak_x, np.nan

    normalized["Y"] = normalized["Y"] / scale
    return normalized, peak_x, scale


def normalize_by_max_in_range(df: pd.DataFrame, x_range=NORMALIZATION_RANGE):
    sub = extract_range(df, x_range)
    if sub.empty:
        normalized = df.copy()
        normalized["Y"] = np.nan
        return normalized, np.nan, np.nan

    idx = sub["Y"].to_numpy(dtype=float).argmax()
    peak_x = float(sub.iloc[idx]["X"])
    scale = float(sub.iloc[idx]["Y"])

    normalized = df.copy()
    if scale == 0:
        normalized["Y"] = np.nan
        return normalized, peak_x, np.nan

    normalized["Y"] = normalized["Y"] / scale
    return normalized, peak_x, scale


def summarize_peak(
    df: pd.DataFrame,
    peak_range: tuple[float, float],
    label: str,
    method: str = PEAK_INTENSITY_METHOD,
    n: int = PEAK_TOPN,
):
    sub = extract_range(df, peak_range)
    if sub.empty:
        return {
            f"{label}_peak_x_cm-1": np.nan,
            f"{label}_peak_y": np.nan,
            f"{label}_range_min_cm-1": peak_range[0],
            f"{label}_range_max_cm-1": peak_range[1],
        }

    y = sub["Y"].to_numpy(dtype=float)
    idx = y.argmax()
    return {
        f"{label}_peak_x_cm-1": float(sub.iloc[idx]["X"]),
        f"{label}_peak_y": _extract_intensity(y, method=method, n=n),
        f"{label}_range_min_cm-1": peak_range[0],
        f"{label}_range_max_cm-1": peak_range[1],
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def build_summary(file_path: Path, norm_label: str, df: pd.DataFrame, scale_info: dict):
    d = summarize_peak(df, D_PEAK_RANGE, "D")
    g = summarize_peak(df, G_PEAK_RANGE, "G")
    two_d = summarize_peak(df, TWO_D_PEAK_RANGE, "2D")

    return {
        "file": file_path.name,
        "relative_path": str(file_path.relative_to(DATA_DIR)),
        "normalization": norm_label,
        **scale_info,
        **d,
        **g,
        **two_d,
        "I_D/I_G": safe_ratio(d["D_peak_y"], g["G_peak_y"]),
        "I_2D/I_G": safe_ratio(two_d["2D_peak_y"], g["G_peak_y"]),
        "I_D/I_2D": safe_ratio(d["D_peak_y"], two_d["2D_peak_y"]),
    }


def save_plot(file_path: Path, stages: dict[str, pd.DataFrame]) -> Path:
    relative_parent = file_path.relative_to(DATA_DIR).parent
    save_path = PLOT_DIR / relative_parent / f"{file_path.stem}_pipeline.png"

    fig, axes = plt.subplots(3, 1, figsize=PLOT_STYLE.figure_size, sharex=False)

    axes[0].plot(stages["raw"]["X"], stages["raw"]["Y"], label="raw", linewidth=1.0)
    axes[0].plot(
        stages["positive"]["X"], stages["positive"]["Y"], label="x > 0", linewidth=1.0
    )
    axes[0].axvspan(*SI_PEAK_RANGE, alpha=0.15, label="Si range")
    axes[0].set_title("Raw and positive-X filtered")
    axes[0].set_xlabel(PLOT_STYLE.x_label)
    axes[0].set_ylabel(PLOT_STYLE.y_label)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    corrected_label = "Si-shift corrected" if ENABLE_SI_SHIFT else "No Si shift"
    axes[1].plot(
        stages["shifted"]["X"],
        stages["shifted"]["Y"],
        label=corrected_label,
        linewidth=1.0,
    )
    axes[1].plot(
        stages["processed"]["X"],
        stages["processed"]["Y"],
        label="baseline corrected",
        linewidth=1.0,
    )
    axes[1].axvspan(*NORMALIZATION_RANGE, alpha=0.15, label="1000-3000 range")
    axes[1].set_title("Shift correction and preprocessing")
    axes[1].set_xlabel(PLOT_STYLE.x_label)
    axes[1].set_ylabel(PLOT_STYLE.y_label)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(
        stages["norm_si"]["X"],
        stages["norm_si"]["Y"],
        label="normalized by Si",
        linewidth=1.0,
    )
    axes[2].plot(
        stages["norm_max"]["X"],
        stages["norm_max"]["Y"],
        label="normalized by max(1000-3000)",
        linewidth=1.0,
    )
    for peak_range, label in (
        (D_PEAK_RANGE, "D"),
        (G_PEAK_RANGE, "G"),
        (TWO_D_PEAK_RANGE, "2D"),
    ):
        axes[2].axvspan(*peak_range, alpha=0.12, label=f"{label} range")
    axes[2].set_xlim(0, 3200)
    axes[2].set_title("Normalized spectra")
    axes[2].set_xlabel(PLOT_STYLE.x_label)
    axes[2].set_ylabel("Normalized intensity")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", ncol=2)

    fig.tight_layout()
    save_figure(fig, save_path, dpi=PLOT_STYLE.dpi)
    plt.close(fig)
    return save_path


def save_processed_tables(
    file_path: Path, tables: dict[str, pd.DataFrame]
) -> dict[str, Path]:
    relative_parent = file_path.relative_to(DATA_DIR).parent
    saved_paths: dict[str, Path] = {}

    for label, df in tables.items():
        save_path = TABLE_DIR / relative_parent / f"{file_path.stem}_{label}.csv"
        save_dataframe(df, save_path, index=False)
        saved_paths[label] = save_path

    return saved_paths


def analyze_spectrum(file_path: Path) -> list[dict]:
    df_raw = load_txt(file_path)
    if df_raw.empty:
        return []

    stages: dict[str, pd.DataFrame] = {"raw": df_raw.copy()}

    df_positive = filter_positive_x(df_raw)
    if df_positive.empty:
        logger.warning("%s: positive X data not found", file_path.name)
        return []
    stages["positive"] = df_positive

    if ENABLE_SI_SHIFT:
        df_shifted, si_peak_raw, si_shift = apply_si_shift(df_positive)
    else:
        df_shifted = df_positive.copy()
        si_peak_raw = np.nan
        si_shift = 0.0
    stages["shifted"] = df_shifted

    df_processed = df_shifted.copy()
    if ENABLE_BASELINE:
        df_processed = subtract_baseline(df_processed)
    if ENABLE_SMOOTHING:
        df_processed = apply_smoothing(df_processed)
    stages["processed"] = df_processed

    df_norm_si, si_peak_x, si_scale = normalize_by_si_peak(df_processed)
    df_norm_max, max_peak_x, max_scale = normalize_by_max_in_range(df_processed)
    stages["norm_si"] = df_norm_si
    stages["norm_max"] = df_norm_max

    plot_path = save_plot(file_path, stages)
    table_paths = save_processed_tables(
        file_path,
        {
            "processed": df_processed,
            "normalized_si": df_norm_si,
            "normalized_max": df_norm_max,
        },
    )

    common_info = {
        "si_shift_enabled": ENABLE_SI_SHIFT,
        "si_peak_raw_cm-1": si_peak_raw,
        "si_shift_cm-1": si_shift,
        "baseline_enabled": ENABLE_BASELINE,
        "smoothing_enabled": ENABLE_SMOOTHING,
        "processed_csv": str(table_paths["processed"]),
        "analysis_plot": str(plot_path),
    }

    summary_si = build_summary(
        file_path,
        "si",
        df_norm_si,
        {
            **common_info,
            "normalization_scale": si_scale,
            "normalization_peak_x_cm-1": si_peak_x,
            "normalized_csv": str(table_paths["normalized_si"]),
        },
    )
    summary_max = build_summary(
        file_path,
        "max_1000_3000",
        df_norm_max,
        {
            **common_info,
            "normalization_scale": max_scale,
            "normalization_peak_x_cm-1": max_peak_x,
            "normalized_csv": str(table_paths["normalized_max"]),
        },
    )
    return [summary_si, summary_max]


def main():
    input_files = prepare_input_files(DATA_DIR)
    if not input_files:
        return

    records: list[dict] = []
    for file_path in input_files:
        logger.info("Processing %s", file_path)
        records.extend(analyze_spectrum(file_path))

    summary_df = pd.DataFrame(records)
    summary_path = OUTPUT_DIR / "summary.csv"
    save_dataframe(summary_df, summary_path, index=False)

    if summary_df.empty:
        logger.warning("No valid results. Empty summary saved to %s", summary_path)
        return

    logger.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    main()
