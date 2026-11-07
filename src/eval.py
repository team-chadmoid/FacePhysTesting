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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate, signal as sp_signal

from src.hr import (
    butterworth_bandpass_filter,
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


def calc_medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median Absolute Error (MedAE) in BPM."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    if len(y_t) == 0 or len(y_p) == 0:
        return float("nan")
    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_t)} != len(y_pred)={len(y_p)}")
    return float(np.median(np.abs(y_p - y_t)))


def calc_fraction_within_threshold(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 5.0
) -> float:
    """Percentage of estimates within threshold (default: 5.0 BPM)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    if len(y_t) == 0 or len(y_p) == 0:
        return float("nan")
    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_t)} != len(y_pred)={len(y_p)}")
    return float(np.mean(np.abs(y_p - y_t) <= threshold) * 100.0)


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
    """Compute complete metric suite: MAE, MedAE, RMSE, Pearson r, Fraction <= 5 BPM, Bland-Altman."""
    mae = calc_mae(y_true, y_pred)
    medae = calc_medae(y_true, y_pred)
    rmse = calc_rmse(y_true, y_pred)
    r = calc_pearson_r(y_true, y_pred)
    frac_5bpm = calc_fraction_within_threshold(y_true, y_pred, threshold=5.0)
    ba = calc_bland_altman(y_true, y_pred)

    return {
        "mae": mae,
        "medae": medae,
        "rmse": rmse,
        "pearson_r": r,
        "fraction_within_5bpm": frac_5bpm,
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


def plot_bland_altman_across_subjects(
    df: Any,
    true_col: str,
    pred_col: str,
    subject_col: str = "subject_id",
    res_col: Optional[str] = "resolution",
    title: str = "Bland-Altman Across Subjects (Reference vs Reference, Not Model Error)",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot Bland-Altman across subjects (one independent point per subject).

    Avoids spurious degrees-of-freedom inflation from overlapping sliding windows.
    Labels clearly state that comparison is reference vs reference, not model error.

    Args:
        df: pandas DataFrame containing subject-level metrics.
        true_col: Column name for ground truth reference (e.g. oximeter line HR).
        pred_col: Column name for comparison reference (e.g. PPG-derived HR).
        subject_col: Column for subject identifier.
        res_col: Optional column for video resolution.
        title: Plot title.
        save_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    y_t = np.asarray(df[true_col], dtype=np.float64)
    y_p = np.asarray(df[pred_col], dtype=np.float64)

    mean_vals = (y_t + y_p) / 2.0
    diff = y_p - y_t

    ba = calc_bland_altman(y_t, y_p)
    bias = ba["bias"]
    upper = ba["upper_loa"]
    lower = ba["lower_loa"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot subject points
    for i, row in df.iterrows():
        x = (row[true_col] + row[pred_col]) / 2.0
        y = row[pred_col] - row[true_col]
        sid = row[subject_col]
        lbl = f"{sid}"
        if res_col and res_col in row and pd.notna(row[res_col]):
            lbl += f" ({row[res_col]})"
        ax.scatter([x], [y], s=80, zorder=5, label=lbl)
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(8, 5), fontsize=9)

    ax.axhline(bias, color="red", linestyle="--", linewidth=1.5, label=f"Bias: {bias:.2f} BPM")
    ax.axhline(upper, color="gray", linestyle=":", linewidth=1.5, label=f"+1.96 SD: {upper:.2f} BPM")
    ax.axhline(lower, color="gray", linestyle=":", linewidth=1.5, label=f"-1.96 SD: {lower:.2f} BPM")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Mean of [{true_col}] and [{pred_col}] (BPM)\n(Reference vs Reference, Not Model Error)", fontsize=10)
    ax.set_ylabel(f"[{pred_col}] - [{true_col}] (BPM)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
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


def calculate_snr(
    psd: np.ndarray,
    freqs: np.ndarray,
    ref_hr_bpm: float,
    delta_hz: float = 0.1,
    harmonic: bool = True,
    band: Tuple[float, float] = (0.7, 3.0),
) -> float:
    """Calculate Signal-to-Noise Ratio (SNR) in dB around reference HR frequency.

    Signal power is integrated in [f0 - delta_hz, f0 + delta_hz] and optionally
    [2*f0 - 2*delta_hz, 2*f0 + 2*delta_hz] (if <= band[1]).
    Noise power is the integrated power across the remainder of the band [0.7, 3.0] Hz.

    Args:
        psd: Power spectral density array.
        freqs: Frequency array in Hz.
        ref_hr_bpm: Reference heart rate in BPM.
        delta_hz: Half-width around fundamental in Hz (default: 0.1 Hz ~ 6 BPM).
        harmonic: Whether to include the first harmonic 2*f0 (default: True).
        band: In-band search range in Hz (default: (0.7, 3.0)).

    Returns:
        SNR in dB: 10 * log10(P_signal / P_noise).
    """
    low_hz, high_hz = band
    in_band = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(in_band):
        return float("nan")

    f0 = ref_hr_bpm / 60.0
    sig_mask = np.abs(freqs - f0) <= delta_hz
    if harmonic:
        f1 = 2.0 * f0
        if f1 <= high_hz:
            sig_mask = sig_mask | (np.abs(freqs - f1) <= 2.0 * delta_hz)

    signal_bins = in_band & sig_mask
    noise_bins = in_band & (~sig_mask)

    p_sig = float(np.sum(psd[signal_bins]))
    p_noise = float(np.sum(psd[noise_bins]))

    if p_sig <= 0.0:
        return -50.0
    if p_noise <= 0.0:
        return 50.0

    return float(10.0 * np.log10(p_sig / p_noise))


def get_top_spectral_peaks(
    psd: np.ndarray,
    freqs: np.ndarray,
    n_peaks: int = 3,
    band: Tuple[float, float] = (0.7, 3.0),
) -> List[Dict[str, float]]:
    """Extract top-N spectral peaks within frequency band with relative power.

    Args:
        psd: Power spectral density array.
        freqs: Frequency array in Hz.
        n_peaks: Number of peaks to retrieve (default: 3).
        band: Frequency range in Hz (default: (0.7, 3.0)).

    Returns:
        List of dicts sorted by power descending, each containing:
            - rank: 1-based rank
            - freq_hz: peak frequency in Hz
            - freq_bpm: peak frequency in BPM
            - power: absolute PSD power at peak
            - rel_power_inband: power / total_inband_psd
            - rel_power_max: power / max_peak_power
    """
    low_hz, high_hz = band
    in_band = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(in_band):
        return []

    f_sub = freqs[in_band]
    p_sub = psd[in_band]
    total_inband_power = float(np.sum(p_sub))

    peaks, _ = sp_signal.find_peaks(p_sub)
    if len(peaks) == 0:
        peaks = np.array([np.argmax(p_sub)])

    sorted_peak_indices = peaks[np.argsort(p_sub[peaks])[::-1]]
    top_indices = sorted_peak_indices[:n_peaks]

    max_p = float(p_sub[top_indices[0]]) if len(top_indices) > 0 else 1.0

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        p_val = float(p_sub[idx])
        f_val = float(f_sub[idx])
        rel_inband = float(p_val / total_inband_power) if total_inband_power > 0 else 0.0
        rel_max = float(p_val / max_p) if max_p > 0 else 0.0
        results.append({
            "rank": rank,
            "freq_hz": f_val,
            "freq_bpm": f_val * 60.0,
            "power": p_val,
            "rel_power_inband": rel_inband,
            "rel_power_max": rel_max,
        })

    return results


def max_lag_waveform_correlation(
    pred_bvp: np.ndarray,
    ref_bvp: np.ndarray,
    fs: float = 30.0,
    max_lag_sec: float = 2.0,
    band: Optional[Tuple[float, float]] = (0.7, 3.0),
) -> Dict[str, float]:
    """Calculate cross-correlation between predicted BVP and GT PPG across lags.

    Args:
        pred_bvp: Predicted BVP signal (1D).
        ref_bvp: Reference PPG/BVP signal (1D).
        fs: Sampling rate in Hz (default: 30.0).
        max_lag_sec: Maximum time lag to search in seconds (default: 2.0 s).
        band: Optional bandpass filtering range in Hz (default: (0.7, 3.0)).

    Returns:
        Dict with keys:
            - max_r: maximum positive Pearson r across lags
            - optimal_lag_sec: optimal lag tau in seconds (pred shifted by tau)
            - optimal_lag_frames: optimal lag in integer frames
            - max_abs_r: maximum absolute Pearson |r| across lags
            - abs_optimal_lag_sec: lag at maximum absolute correlation
    """
    p = np.asarray(pred_bvp, dtype=np.float64)
    r = np.asarray(ref_bvp, dtype=np.float64)

    min_len = min(len(p), len(r))
    if min_len < int(round(fs * 2.0)):
        return {
            "max_r": 0.0,
            "optimal_lag_sec": 0.0,
            "optimal_lag_frames": 0,
            "max_abs_r": 0.0,
            "abs_optimal_lag_sec": 0.0,
        }

    p = p[:min_len]
    r = r[:min_len]

    if band is not None:
        p = butterworth_bandpass_filter(p, fs=fs, lowcut=band[0], highcut=band[1], order=3)
        r = butterworth_bandpass_filter(r, fs=fs, lowcut=band[0], highcut=band[1], order=3)

    max_k = int(round(max_lag_sec * fs))
    lags = np.arange(-max_k, max_k + 1)
    corrs = []

    for k in lags:
        if k > 0:
            p_slice = p[k:]
            r_slice = r[:-k]
        elif k < 0:
            p_slice = p[:k]
            r_slice = r[-k:]
        else:
            p_slice = p
            r_slice = r
        corr = calc_pearson_r(r_slice, p_slice)
        corrs.append(corr)

    corrs = np.array(corrs)
    best_pos_idx = np.argmax(corrs)
    best_abs_idx = np.argmax(np.abs(corrs))

    return {
        "max_r": float(corrs[best_pos_idx]),
        "optimal_lag_sec": float(lags[best_pos_idx] / fs),
        "optimal_lag_frames": int(lags[best_pos_idx]),
        "max_abs_r": float(corrs[best_abs_idx]),
        "abs_optimal_lag_sec": float(lags[best_abs_idx] / fs),
    }

