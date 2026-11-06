import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_spectra_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """波形CSV (Wavenumber, Pixel_0, ...) を読み込み、(wavenumber, spectra[n_pixels, n_wn]) を返す"""
    df = pd.read_csv(csv_path)
    wavenumber = df["Wavenumber"].to_numpy(dtype=float)
    pixel_cols = [c for c in df.columns if c != "Wavenumber"]
    spectra = df[pixel_cols].to_numpy(dtype=float).T
    return wavenumber, spectra


def plot_overlay(
    save_path: Path,
    wavenumber: np.ndarray,
    raw: np.ndarray,
    processed: np.ndarray,
    fit: np.ndarray | None = None,
    title: str = "Spectra Overlay",
    edge_trim: float = 20.0,
) -> None:
    """上段にRaw波形、下段にProcessed波形＋Fit曲線を重ねて保存する"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })

    raw = np.atleast_2d(raw)
    processed = np.atleast_2d(processed)
    valid_mask = wavenumber >= (wavenumber.min() + edge_trim)
    wn_valid = wavenumber[valid_mask]

    fig, (ax_raw, ax_proc) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # --- 上段: Raw Spectra ---
    for i in range(len(raw)):
        ax_raw.plot(wavenumber, raw[i], alpha=0.35, linewidth=0.8)

    raw_valid = raw[:, valid_mask]
    if raw_valid.size > 0:
        y_min = float(np.percentile(raw_valid, 0.5))
        y_max = float(np.percentile(raw_valid, 99.0))
        span = max(y_max - y_min, 1.0)
        ax_raw.set_ylim(y_min - 0.08 * span, y_max + 0.15 * span)

    ax_raw.set_title(f"{title} - Raw Spectra")
    ax_raw.set_ylabel("Intensity (a.u.)")
    ax_raw.grid(True, linestyle="--", alpha=0.4)

    # --- 下段: Processed + Fit Overlay ---
    for i in range(len(processed)):
        lbl_proc = "Processed (Despiked & Bkg-subtracted)" if i == 0 else None
        ax_proc.plot(
             wn_valid,
            processed[i, valid_mask],
            color="#0072b2",
            alpha=0.3,
            linewidth=0.8,
            label=lbl_proc,
        )

    if fit is not None:
        fit = np.atleast_2d(fit)
        for i in range(len(fit)):
            lbl_fit = "Lorentzian Fit" if i == 0 else None
            ax_proc.plot(
                wn_valid,
                fit[i, valid_mask],
                color="#d55e00",
                alpha=0.55,
                linewidth=1.1,
                label=lbl_fit,
            )

    proc_valid = processed[:, valid_mask]
    if proc_valid.size > 0:
        p_min = float(np.percentile(proc_valid, 1.0))
        p_max = float(np.percentile(np.max(proc_valid, axis=1), 98.0))
        if fit is not None:
            p_max = max(p_max, float(np.max(fit[:, valid_mask])))
        span = max(p_max - p_min, 1.0)
        ax_proc.set_ylim(p_min - 0.1 * span, p_max + 0.15 * span)

    ax_proc.axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_proc.set_title(f"{title} - Processed & Fit Overlay")
    ax_proc.set_xlabel(r"Wavenumber ($\mathrm{cm}^{-1}$)")
    ax_proc.set_ylabel("Intensity (a.u.)")
    ax_proc.set_xlim( float(wn_valid.min()), float(wn_valid.max()) )
    ax_proc.legend(loc="upper right", frameon=False)
    ax_proc.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_from_dir(target_dir: Path) -> list[Path]:
    """指定ディレクトリ配下の *_raw.csv を自動探索し、対応する processed / fit CSV と統合して画像出力する"""
    saved_files: list[Path] = []
    for raw_csv in sorted(target_dir.rglob("*_raw.csv")):
        prefix = raw_csv.stem[:-4]  # "{sample}_{band}" ("_raw" を除去)
        sample_root = raw_csv.parent.parent if raw_csv.parent.name == "raw" else raw_csv.parent

        proc_matches = list(sample_root.rglob(f"{prefix}_processed.csv"))
        if not proc_matches:
            continue
        fit_matches = list(sample_root.rglob(f"{prefix}_fit.csv"))

        wn, raw_spec = _load_spectra_csv(raw_csv)
        _, proc_spec = _load_spectra_csv(proc_matches[0])
        fit_spec = _load_spectra_csv(fit_matches[0])[1] if fit_matches else None

        save_path = sample_root / f"{prefix}_all_spectra.png"
        plot_overlay(
            save_path=save_path,
            wavenumber=wn,
            raw=raw_spec,
            processed=proc_spec,
            fit=fit_spec,
            title=prefix.replace("_", " "),
        )
        saved_files.append(save_path)
        print(f"Saved: {save_path}")

    return saved_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay raw, processed, and fitted Raman spectra.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("output/spectra_mapping_high_resolution"),
        help="Target output directory (sample folder or parent folder).",
    )
    args = parser.parse_args()
    plot_from_dir(args.dir)
