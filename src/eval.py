"""Evaluation metrics and protocol (single source of truth for IEEE SP Cup 2027).

Protocol:
- HR from BVP: 3rd-order Butterworth bandpass 0.7-3.0 Hz, Welch PSD, peak frequency x 60.
- Windows: 30 s, 1 s stride.
- Ground truth: HR derived from the GT PPG with the same estimator; compare to provided HR line.
  Flag videos where the two disagree by > 5 BPM.
- Metrics: MAE (BPM), RMSE (BPM), Pearson r, Bland-Altman limits of agreement.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from src.hr import (
    estimate_hr_from_bvp,
    estimate_hr_video,
    estimate_hr_windowed,
)
from src.io import resample_to_uniform


def calc_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (MAE) in BPM."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    if len(y_t) == 0 or len(y_p) == 0:
        return float("nan")
    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_t)} != len(y_pred)={len(y_p)}")
    return float(np.mean(np.abs(y_p - y_t)))


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error (RMSE) in BPM."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    if len(y_t) == 0 or len(y_p) == 0:
        return float("nan")
    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_t)} != len(y_pred)={len(y_p)}")
    return float(np.sqrt(np.mean((y_p - y_t) ** 2)))


def calc_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient (r)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    if len(y_t) < 2 or len(y_p) < 2:
        return 0.0
    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_t)} != len(y_pred)={len(y_p)}")

    std_t = np.std(y_t)
    std_p = np.std(y_p)
    if std_t < 1e-9 or std_p < 1e-9:
        # Constant signal has zero correlation
        return 0.0

    r = float(np.corrcoef(y_t, y_p)[0, 1])
    return r if not np.isnan(r) else 0.0


def calc_bland_altman(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate Bland-Altman statistics.

    Difference is defined as diff = y_pred - y_true.
    LoA = bias +/- 1.96 * std_diff.

    Returns:
        Dict with keys:
            - bias: mean difference
            - std: standard deviation of difference
            - upper_loa: bias + 1.96 * std
            - lower_loa: bias - 1.96 * std
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    if len(y_t) == 0 or len(y_p) == 0:
        return {"bias": float("nan"), "std": float("nan"), "upper_loa": float("nan"), "lower_loa": float("nan")}
    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_t)} != len(y_pred)={len(y_p)}")

    diff = y_p - y_t
    bias = float(np.mean(diff))
    std = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    upper_loa = float(bias + 1.96 * std)
    lower_loa = float(bias - 1.96 * std)

    return {
        "bias": bias,
        "std": std,
        "upper_loa": upper_loa,
        "lower_loa": lower_loa,
    }


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute complete metric suite: MAE, RMSE, Pearson r, Bland-Altman."""
    mae = calc_mae(y_true, y_pred)
    rmse = calc_rmse(y_true, y_pred)
    r = calc_pearson_r(y_true, y_pred)
    ba = calc_bland_altman(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson_r": r,
        "bias": ba["bias"],
        "std": ba["std"],
        "upper_loa": ba["upper_loa"],
        "lower_loa": ba["lower_loa"],
    }


def evaluate_ground_truth_consistency(
    gt_ppg: np.ndarray,
    gt_hr_line: np.ndarray,
    time: Optional[np.ndarray] = None,
    fs: float = 30.0,
    window_sec: float = 30.0,
    stride_sec: float = 1.0,
    threshold_bpm: float = 5.0,
) -> Dict[str, Any]:
    """Verify consistency between GT PPG-derived HR and provided GT HR line.

    Per AGENTS.md:
        - Ground truth: HR derived from the GT PPG with the same estimator;
          also compare to the provided HR line. Flag videos where the two disagree
          by more than 5 BPM.

    Args:
        gt_ppg: Raw or uniform GT PPG signal.
        gt_hr_line: Hardware provided HR values per sample.
        time: Timestamps in seconds.
        fs: Sampling rate.
        window_sec: Window length (default: 30.0 s).
        stride_sec: Stride length (default: 1.0 s).
        threshold_bpm: Flagging threshold in BPM (default: 5.0 BPM).

    Returns:
        Dict with video-level and window-level discrepancy analysis and flag.
    """
    # 1. Video-level comparison
    hr_ppg_vid = estimate_hr_video(gt_ppg, fs=fs, time=time)
    mean_hr_line = float(np.mean(gt_hr_line))
    video_discrepancy = float(abs(hr_ppg_vid - mean_hr_line))
    flagged_video = video_discrepancy > threshold_bpm

    # 2. Windowed comparison
    if time is not None:
        t_u, ppg_u = resample_to_uniform(time, gt_ppg, target_fs=fs)
        _, hr_u = resample_to_uniform(time, gt_hr_line, target_fs=fs, method="linear")
    else:
        ppg_u = gt_ppg
        hr_u = gt_hr_line
        t_u = np.arange(len(gt_ppg)) / fs

    win_times, hrs_ppg_win = estimate_hr_windowed(
        ppg_u, fs=fs, window_sec=window_sec, stride_sec=stride_sec
    )

    win_samples = int(round(window_sec * fs))
    stride_samples = int(round(stride_sec * fs))

    hrs_line_win = []
    total_samples = len(ppg_u)
    if total_samples >= win_samples:
        for start in range(0, total_samples - win_samples + 1, stride_samples):
            end = start + win_samples
            hrs_line_win.append(float(np.mean(hr_u[start:end])))
    else:
        hrs_line_win.append(float(np.mean(hr_u)))

    hrs_line_win = np.array(hrs_line_win, dtype=np.float64)

    window_metrics = calc_metrics(hrs_line_win, hrs_ppg_win)
    window_diffs = np.abs(hrs_ppg_win - hrs_line_win)
    max_window_diff = float(np.max(window_diffs)) if len(window_diffs) > 0 else 0.0
    flagged_window_count = int(np.sum(window_diffs > threshold_bpm))

    return {
        "video_level": {
            "hr_ppg": hr_ppg_vid,
            "hr_line": mean_hr_line,
            "discrepancy_bpm": video_discrepancy,
            "flagged": flagged_video,
        },
        "windowed": {
            "num_windows": len(hrs_ppg_win),
            "window_times": win_times,
            "hrs_ppg": hrs_ppg_win,
            "hrs_line": hrs_line_win,
            "metrics": window_metrics,
            "max_discrepancy_bpm": max_window_diff,
            "flagged_windows_count": flagged_window_count,
            "flagged": (flagged_window_count > 0),
        },
        "threshold_bpm": threshold_bpm,
    }


def plot_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Bland-Altman Plot",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Generate and optionally save a Bland-Altman plot."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    mean_vals = (y_t + y_p) / 2.0
    diff = y_p - y_t

    ba = calc_bland_altman(y_t, y_p)
    bias = ba["bias"]
    upper = ba["upper_loa"]
    lower = ba["lower_loa"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mean_vals, diff, alpha=0.6, edgecolors="none", color="#1f77b4")
    ax.axhline(bias, color="red", linestyle="--", label=f"Bias: {bias:.2f} BPM")
    ax.axhline(upper, color="gray", linestyle=":", label=f"+1.96 SD: {upper:.2f} BPM")
    ax.axhline(lower, color="gray", linestyle=":", label=f"-1.96 SD: {lower:.2f} BPM")

    ax.set_title(title)
    ax.set_xlabel("Mean of Ground Truth and Estimate (BPM)")
    ax.set_ylabel("Estimate - Ground Truth (BPM)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")

    return fig


def log_exclusion(
    subject_id: str,
    reason: str,
    exclusions_file: Union[str, Path] = "results/exclusions.csv",
) -> None:
    """Log an excluded video/subject to exclusions.csv per Hard Rule 4."""
    path = Path(exclusions_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = "subject_id,reason,timestamp\n"
    now_iso = datetime.now(timezone.utc).isoformat()
    line = f"{subject_id},{reason},{now_iso}\n"

    if not path.is_file():
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + line)
    else:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
