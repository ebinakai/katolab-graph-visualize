import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class FitMethod(Enum):
    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"


@dataclass
class PeakFitResult:
    name: str
    method: FitMethod
    amplitude: float
    center: float
    sigma: float  # Gaussian: 標準偏差σ / Lorentzian: 半値半幅γ
    slope: float
    intercept: float
    r_squared: float

    @property
    def area(self) -> float:
        if self.method == FitMethod.GAUSSIAN:
            return self.amplitude * self.sigma * np.sqrt(2.0 * np.pi)
        else:  # Lorentzian
            return self.amplitude * self.sigma * np.pi


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


def lorentzian_with_linear_baseline(
    x: np.ndarray,
    amplitude: float,
    center: float,
    gamma: float,
    slope: float,
    intercept: float,
) -> np.ndarray:
    lorentz = amplitude * gamma**2 / ((x - center) ** 2 + gamma**2)
    baseline = slope * x + intercept
    return lorentz + baseline


def _get_fit_func(method: FitMethod) -> Callable:
    if method == FitMethod.GAUSSIAN:
        return gaussian_with_linear_baseline
    else:
        return lorentzian_with_linear_baseline


def fit_peak(
    df: pd.DataFrame,
    peak_name: str,
    window: Tuple[float, float],
    center_bounds: Optional[Tuple[float, float]] = None,
    width_bounds: Optional[Tuple[float, float]] = None,
    center_edge_margin: float = 0.0,
    min_points: int = 6,
    method: FitMethod = FitMethod.GAUSSIAN,
) -> Optional[PeakFitResult]:
    x_min, x_max = window
    mask = (df["X"] >= x_min) & (df["X"] <= x_max)
    sub = df.loc[mask]

    if len(sub) < min_points:
        logging.warning("%s フィット失敗: データ点不足 (%d < %d)", peak_name, len(sub), min_points)
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

    width_lower, width_upper = (0.5, x_max - x_min)
    if width_bounds is not None:
        width_lower = max(width_lower, width_bounds[0])
        width_upper = min(width_upper, width_bounds[1])
        if width_lower >= width_upper:
            logging.warning("%s フィット失敗: width_bounds が不正です", peak_name)
            return None

    lower = [0.0, center_lower, width_lower, -np.inf, -np.inf]
    upper = [np.inf, center_upper, width_upper, np.inf, np.inf]

    # center初期値が拘束範囲外のときは範囲内に寄せる
    for p0 in p0_candidates:
        p0[1] = float(np.clip(p0[1], center_lower + 1e-6, center_upper - 1e-6))

    popt = None
    last_error: Optional[Exception] = None
    fit_func = _get_fit_func(method)
    for p0 in p0_candidates:
        try:
            popt, _ = curve_fit(
                fit_func,
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

    fitted_center = float(popt[1])
    fitted_width = float(abs(popt[2]))

    if center_edge_margin > 0.0:
        edge_low = center_lower + center_edge_margin
        edge_high = center_upper - center_edge_margin
        if edge_low >= edge_high:
            logging.warning("%s フィット失敗: center_edge_margin が大きすぎます", peak_name)
            return None
        if fitted_center <= edge_low or fitted_center >= edge_high:
            logging.warning(
                "%s フィット失敗: center が拘束範囲の端に寄りすぎています (center=%.3f)",
                peak_name,
                fitted_center,
            )
            return None

    y_fit = fit_func(x, *popt)
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return PeakFitResult(
        name=peak_name,
        method=method,
        amplitude=float(popt[0]),
        center=fitted_center,
        sigma=fitted_width,
        slope=float(popt[3]),
        intercept=float(popt[4]),
        r_squared=r_squared,
    )


def build_fitted_curve(x: np.ndarray, result: PeakFitResult) -> np.ndarray:
    fit_func = _get_fit_func(result.method)
    return fit_func(
        x,
        result.amplitude,
        result.center,
        result.sigma,
        result.slope,
        result.intercept,
    )
