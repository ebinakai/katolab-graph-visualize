import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from pathlib import Path
from matplotlib import pyplot as plt
import logging

formatter = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=formatter)
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
SOURCE_DIR = DATA_DIR / "spectra_mapping"
G_PEAK_RANGE = (1550, 1600)
D_PEAK_RANGE = (1300, 1400)
TWO_D_PEAK_RANGE = (2650, 2750)
G_D_RATIO_MIN = 0
G_D_RATIO_MAX = None
TWO_D_G_RATIO_MIN = 0
TWO_D_G_RATIO_MAX = None


def list_input_files(
    target_dir: Path, pattern: str = "*.txt", recursive: bool = True
) -> list[Path]:
    paths = target_dir.rglob(pattern) if recursive else target_dir.glob(pattern)
    return sorted(path for path in paths if path.is_file())


def load_mapping_ascii(
    filepath: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(filepath, "r") as f:
        lines = f.readlines()

    # --- 1行目：波数 ---
    header = lines[0].strip().split()
    wavenumber = np.array([float(x) for x in header])

    # --- データ部分 ---
    data = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 0:
            continue

        try:
            row = [float(x) for x in parts]
            data.append(row)
        except ValueError:
            logger.warning(f"Skipping malformed line: {line.strip()}")
            continue

    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]
    spectra = data[:, 2:]

    return x, y, wavenumber, spectra


def _extract_intensity(values, method="topn", n=3):
    if method == "max":
        return values.max(axis=1)

    elif method == "mean":
        return values.mean(axis=1)

    elif method == "topn":
        sorted_vals = np.sort(values, axis=1)
        return sorted_vals[:, -n:].mean(axis=1)

    elif method == "percentile":
        return np.percentile(values, 95, axis=1)

    else:
        raise ValueError(f"Unknown method: {method}")


def correct_wavenumber_shift(
    wavenumber,
    spectra,
    si_range=(500, 540),
    target=520.7,
):
    """
    Siピークを基準に波数シフト補正

    Parameters
    ----------
    wavenumber : (N_wn,)
    spectra    : (N_pixel, N_wn)
    si_range   : tuple
        Siピーク探索範囲
    target     : float
        目標ピーク位置（520.7）

    Returns
    -------
    spectra_corrected : (N_pixel, N_wn)
    shifts            : (N_pixel,)
    """

    spectra = np.atleast_2d(spectra)
    N_pixel, N_wn = spectra.shape

    # --- Siピーク範囲 ---
    mask = (wavenumber >= si_range[0]) & (wavenumber <= si_range[1])
    wn_si = wavenumber[mask]

    corrected = np.zeros_like(spectra)
    shifts = np.zeros(N_pixel)

    for i in range(N_pixel):
        spec = spectra[i, mask]

        # --- ピーク位置 ---
        peak_idx = np.argmax(spec)
        peak_pos = wn_si[peak_idx]

        # --- シフト量 ---
        shift = peak_pos - target
        shifts[i] = shift

        # --- 補間関数 ---
        f = interp1d(
            wavenumber - shift,
            spectra[i],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",  # type: ignore
        )

        corrected[i] = f(wavenumber)

    return corrected, shifts


def calc_peak_ratio(
    wavenumber,
    spectra,
    num_range,
    den_range,
    method="topn",
    n=3,
    eps=1e-12,
):
    """
    任意ピーク比を計算する汎用関数
    """

    # --- マスク ---
    num_mask = (wavenumber >= num_range[0]) & (wavenumber <= num_range[1])
    den_mask = (wavenumber >= den_range[0]) & (wavenumber <= den_range[1])

    # --- 強度抽出 ---
    num_vals = spectra[:, num_mask]
    den_vals = spectra[:, den_mask]

    I_num = _extract_intensity(num_vals, method=method, n=n)
    I_den = _extract_intensity(den_vals, method=method, n=n)

    # --- 比 ---
    ratio = I_num / (I_den + eps)

    return ratio, I_num, I_den


def calc_2d_g_ratio(wavenumber, spectra, **kwargs):
    return calc_peak_ratio(
        wavenumber, spectra, TWO_D_PEAK_RANGE, G_PEAK_RANGE, **kwargs
    )


def calc_g_d_ratio(wavenumber, spectra, **kwargs):
    return calc_peak_ratio(wavenumber, spectra, G_PEAK_RANGE, D_PEAK_RANGE, **kwargs)


def baseline_als(y, lam=1e5, p=0.01, niter=10):
    if niter <= 0:
        return y.copy()

    L = len(y)
    D = sparse.diags(
        [1.0, -2.0, 1.0],
        [0, -1, -2],  # type: ignore
        shape=(L, L - 2),
        dtype=float,
    )
    W = sparse.eye(L, dtype=float)
    z = y.copy()

    for _ in range(niter):
        W.setdiag(p * (y > 0) + (1 - p) * (y < 0))
        Z = (W + lam * D @ D.T).tocsc()  # ←重要
        z = spsolve(Z, W @ y)

    return z


def preprocess_baseline(spectra):
    corrected = []

    for spec in spectra:
        baseline = baseline_als(spec)
        corrected.append(spec - baseline)

    return np.array(corrected)


def smooth_spectra(spectra, window=11, poly=3):
    return savgol_filter(spectra, window_length=window, polyorder=poly, axis=1)


def create_ig_mask(IG, threshold=None, method="absolute", ratio=0.1):
    """
    IGベースのマスクを生成

    Parameters
    ----------
    IG : (N_pixel,)
        Gピーク強度
    method : str
        "absolute" or "relative"
    threshold : float or None
        絶対閾値（method="absolute" の場合使用）
    ratio : float
        最大値に対する割合（method="relative" の場合使用）

    Returns
    -------
    mask : (N_pixel,) bool
    """

    if method == "absolute":
        if threshold is None:
            raise ValueError("threshold must be specified for absolute mode")
        mask = IG > threshold

    elif method == "relative":
        thr = IG.max() * ratio
        mask = IG > thr

    else:
        raise ValueError(f"Unknown method: {method}")

    return mask


def apply_mask(data, mask):
    """
    無効ピクセルをNaNにする
    """
    data = data.copy()
    data[~mask] = np.nan
    return data


def save_ratio_map_figure(
    filepath: Path,
    x,
    y,
    ratio,
    title="2D/G Ratio Map",
    cmap="viridis",
    vmin=0.0,
    vmax=2.0,
):
    filepath.parent.mkdir(parents=True, exist_ok=True)

    plt.scatter(x, y, c=ratio, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label=title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.savefig(filepath)
    plt.close()


def spectra_draw(
    wavenumber,
    spectra,
    spectra_ref=None,
    n=5,
    mask_range=(-100, 3000),
    g_range=None,
    d_range=None,
    two_d_range=None,
    show_peak=False,
    output_dir=Path("output/spectra_mapping/debug"),
    output_prefix="spectra_debug",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    spectra = np.atleast_2d(spectra)

    if spectra_ref is not None:
        spectra_ref = np.atleast_2d(spectra_ref)

    idx = np.random.choice(len(spectra), n, replace=False)

    mask = (wavenumber >= mask_range[0]) & (wavenumber <= mask_range[1])
    wn = wavenumber[mask]

    for i in idx:
        spec = spectra[i, mask]

        plt.figure()

        # --- メインスペクトル ---
        plt.plot(wn, spec, alpha=0.9, label="spectra")

        # --- 比較用（任意） ---
        if spectra_ref is not None:
            spec_ref = spectra_ref[i, mask]
            plt.plot(wn, spec_ref, alpha=0.5, label="reference")

        # --- 範囲表示 ---
        if g_range is not None:
            plt.axvspan(*g_range, alpha=0.2, label="G range")

        if d_range is not None:
            plt.axvspan(*d_range, alpha=0.2, label="D range")

        if two_d_range is not None:
            plt.axvspan(*two_d_range, alpha=0.2, label="2D range")

        # --- ピーク位置 ---
        if show_peak:

            def draw_peak(rng, label):
                if rng is None:
                    return
                m = (wn >= rng[0]) & (wn <= rng[1])
                if m.any():
                    peak = wn[m][np.argmax(spec[m])]
                    plt.axvline(peak, linestyle="--", label=label)

            draw_peak(g_range, "G peak")
            draw_peak(d_range, "D peak")
            draw_peak(two_d_range, "2D peak")

        plt.title(f"Pixel {i}")
        plt.xlabel("Wavenumber")
        plt.ylabel("Intensity")
        plt.legend()

        plt.savefig(output_dir / f"{output_prefix}_pixel_{i}.png")
        plt.close()


def analyze_spectra(filepath: Path):
    save_dir = OUTPUT_DIR / filepath.relative_to(DATA_DIR).parent
    debug_path = save_dir / "debug"
    if debug_path.exists():
        for file in debug_path.glob("*.png"):
            file.unlink()

    x, y, wavenumber, spectra = load_mapping_ascii(filepath)

    def run_stage(name, process_func, *args, **kwargs):
        nonlocal spectra
        spectra_before = spectra.copy()
        spectra = process_func(spectra, *args, **kwargs)

        spectra_draw(
            wavenumber,
            spectra,
            spectra_before,
            output_dir=debug_path,
            output_prefix=name,
        )

    # --- シフト ---
    spectra, shifts = correct_wavenumber_shift(wavenumber, spectra)

    run_stage("baseline", preprocess_baseline)
    # run_stage("smoothed", smooth_spectra, window=11, poly=3)

    # -- 比計算 ---
    ratio_2dg, I2D, IG = calc_2d_g_ratio(wavenumber, spectra)
    ratio_gd, IG_gd, ID = calc_g_d_ratio(wavenumber, spectra)

    # --- IGベースのマスク適用 ---
    mask = create_ig_mask(IG, method="absolute", threshold=5)
    ratio_2dg = apply_mask(ratio_2dg, mask)
    ratio_gd = apply_mask(ratio_gd, mask)

    # --- 2D/G比マップの保存 ---
    save_ratio_map_figure(
        save_dir / "2d_g_ratio_map.png",
        x,
        y,
        ratio_2dg,
        title="2D/G Ratio Map",
        vmin=np.nanpercentile(ratio_2dg, 25)
        if TWO_D_G_RATIO_MIN is None
        else TWO_D_G_RATIO_MIN,
        vmax=np.nanpercentile(ratio_2dg, 75)
        if TWO_D_G_RATIO_MAX is None
        else TWO_D_G_RATIO_MAX,
    )

    # --- G/D比マップの保存 ---
    save_ratio_map_figure(
        save_dir / "g_d_ratio_map.png",
        x,
        y,
        ratio_gd,
        title="G/D Ratio Map",
        vmin=np.nanpercentile(ratio_gd, 25) if G_D_RATIO_MIN is None else G_D_RATIO_MIN,
        vmax=np.nanpercentile(ratio_gd, 75) if G_D_RATIO_MAX is None else G_D_RATIO_MAX,
    )

    # --- デバッグ用スペクトル描画 ---
    spectra_draw(
        wavenumber,
        spectra,
        mask_range=(1000, 3000),
        g_range=G_PEAK_RANGE,
        d_range=D_PEAK_RANGE,
        two_d_range=TWO_D_PEAK_RANGE,
        show_peak=True,
        output_dir=debug_path,
        output_prefix="final",
    )


def main():
    for filepath in list_input_files(SOURCE_DIR):
        logger.info(f"Processing {filepath}...")
        analyze_spectra(filepath)


if __name__ == "__main__":
    main()
