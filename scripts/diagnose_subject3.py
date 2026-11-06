"""Diagnostic script for Subject 3 discrepancy and reference comparison.

Evaluates:
(1) GT PPG waveform & spectrogram for subject3 from 0-50s with detected beats overlaid.
(2) hr_from_peaks vs FFT-derived HR (Welch) vs pulse-oximeter line for all 3 subjects.
(3) HR on raw frame-index signal vs timestamp-resampled signal, especially around subject3's 6 gaps.
(4) Agreement with oximeter line on flagged windows (reference vs reference, not model error).
(5) Across-subject Bland-Altman plots.
"""

from pathlib import Path
import subprocess
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.io import (
    find_ubfc_subjects,
    load_ubfc_subject,
    check_ubfc_alignment,
    resample_to_uniform,
)
from src.hr import (
    butterworth_bandpass_filter,
    detect_peaks_ppg,
    hr_from_peaks,
    estimate_hr_from_bvp,
    estimate_hr_video,
    estimate_hr_windowed,
    estimate_hr_peaks_windowed,
    estimate_hr_raw_frames,
    estimate_hr_windowed_raw_frames,
)
from src.eval import (
    calc_metrics,
    calc_bland_altman,
    plot_bland_altman_across_subjects,
)


def get_git_hash() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def main():
    config_path = repo_root / "configs" / "subject3_diagnostic.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["experiment"]["seed"]
    np.random.seed(seed)
    git_hash = get_git_hash()

    print(f"=== Running Diagnosis (git hash: {git_hash}, seed: {seed}) ===")

    data_root = repo_root / cfg["dataset"]["root"]
    subjects_map = find_ubfc_subjects(data_root)
    target_subjects = cfg["dataset"]["subjects"]

    fs = float(cfg["parameters"]["fs"])
    win_sec = float(cfg["parameters"]["window_sec"])
    stride_sec = float(cfg["parameters"]["stride_sec"])
    threshold_bpm = float(cfg["parameters"]["discrepancy_threshold_bpm"])

    subject_rows = []
    window_rows = []
    subject3_data = None

    for subj_id in target_subjects:
        if subj_id not in subjects_map:
            print(f"Warning: {subj_id} not found in {data_root}")
            continue

        sdir = subjects_map[subj_id]
        sdata = load_ubfc_subject(sdir, subject_id=subj_id)
        if subj_id == "subject3":
            subject3_data = sdata

        align = check_ubfc_alignment(sdata)
        res_str = (
            f"{sdata.video_metadata['width']}x{sdata.video_metadata['height']}"
            if sdata.video_metadata
            else "unknown"
        )

        # Video-level reference values
        hr_ox_vid = float(np.mean(sdata.hr))
        hr_welch_resampled = float(estimate_hr_video(sdata.ppg, fs=fs, time=sdata.time))
        hr_peaks_resampled = float(hr_from_peaks(sdata.ppg, fs=fs, time=sdata.time))
        hr_welch_raw_30fps = float(estimate_hr_raw_frames(sdata.ppg, fs_nominal=30.0))

        exact_fps = float(len(sdata.ppg) / (sdata.time[-1] - sdata.time[0]))
        hr_welch_raw_exact_fps = float(estimate_hr_raw_frames(sdata.ppg, fs_nominal=exact_fps))

        # Windowed references
        t_u, ppg_u = resample_to_uniform(sdata.time, sdata.ppg, target_fs=fs)
        _, hr_u = resample_to_uniform(sdata.time, sdata.hr, target_fs=fs, method="linear")

        win_times, win_welch_res = estimate_hr_windowed(
            ppg_u, fs=fs, window_sec=win_sec, stride_sec=stride_sec
        )
        _, win_peaks_res = estimate_hr_peaks_windowed(
            ppg_u, fs=fs, window_sec=win_sec, stride_sec=stride_sec
        )

        # Windowed raw frame-index
        _, win_welch_raw = estimate_hr_windowed_raw_frames(
            sdata.ppg, fs_nominal=30.0, window_sec=win_sec, stride_sec=stride_sec
        )

        win_samples = int(round(win_sec * fs))
        stride_samples = int(round(stride_sec * fs))
        win_ox = []
        for start in range(0, len(ppg_u) - win_samples + 1, stride_samples):
            win_ox.append(float(np.mean(hr_u[start : start + win_samples])))
        win_ox = np.array(win_ox, dtype=np.float64)

        # Metrics: Welch vs Oximeter
        m_welch_ox = calc_metrics(win_ox, win_welch_res)
        # Metrics: Peaks vs Oximeter
        m_peaks_ox = calc_metrics(win_ox, win_peaks_res)
        # Metrics: Welch vs Peaks
        m_welch_peaks = calc_metrics(win_peaks_res, win_welch_res)

        # Count flagged windows where |welch - oximeter| > threshold
        diff_welch_ox = np.abs(win_welch_res - win_ox)
        flagged_count = int(np.sum(diff_welch_ox > threshold_bpm))

        subject_rows.append({
            "git_hash": git_hash,
            "comparison_type": "reference vs reference, not model error",
            "subject_id": subj_id,
            "resolution": res_str,
            "video_frames": align["video_frames"],
            "gt_samples": align["n_samples"],
            "duration_sec": align["duration_sec"],
            "nominal_fps": align["nominal_fps"],
            "video_fps": align["video_fps"],
            "non_positive_dt_count": align["non_positive_count"],
            "large_gaps_count": align["large_gap_count"],
            # Video-level comparisons
            "hr_oximeter_vid_bpm": hr_ox_vid,
            "hr_welch_resampled_vid_bpm": hr_welch_resampled,
            "hr_peaks_resampled_vid_bpm": hr_peaks_resampled,
            "hr_welch_raw_30fps_vid_bpm": hr_welch_raw_30fps,
            "hr_welch_raw_exact_fps_vid_bpm": hr_welch_raw_exact_fps,
            "diff_welch_ox_vid_bpm": hr_welch_resampled - hr_ox_vid,
            "abs_diff_welch_ox_vid_bpm": abs(hr_welch_resampled - hr_ox_vid),
            "diff_peaks_ox_vid_bpm": hr_peaks_resampled - hr_ox_vid,
            "abs_diff_peaks_ox_vid_bpm": abs(hr_peaks_resampled - hr_ox_vid),
            # Windowed comparisons vs oximeter
            "num_windows": len(win_ox),
            "flagged_windows_count": flagged_count,
            "welch_vs_ox_mae_bpm": m_welch_ox["mae"],
            "welch_vs_ox_rmse_bpm": m_welch_ox["rmse"],
            "welch_vs_ox_r": m_welch_ox["pearson_r"],
            "welch_vs_ox_bias": m_welch_ox["bias"],
            "peaks_vs_ox_mae_bpm": m_peaks_ox["mae"],
            "peaks_vs_ox_rmse_bpm": m_peaks_ox["rmse"],
            "peaks_vs_ox_r": m_peaks_ox["pearson_r"],
            "peaks_vs_ox_bias": m_peaks_ox["bias"],
        })

        for i in range(len(win_ox)):
            h_ox = win_ox[i]
            h_w_res = win_welch_res[i]
            h_p_res = win_peaks_res[i]
            h_w_raw = win_welch_raw[i] if i < len(win_welch_raw) else np.nan

            d_w_ox = abs(h_w_res - h_ox)
            d_p_ox = abs(h_p_res - h_ox)
            flagged = d_w_ox > threshold_bpm

            # Determine which reference is closer to oximeter
            closer_ref = "welch" if d_w_ox < d_p_ox else ("peaks" if d_p_ox < d_w_ox else "tie")

            window_rows.append({
                "git_hash": git_hash,
                "comparison_type": "reference vs reference, not model error",
                "subject_id": subj_id,
                "resolution": res_str,
                "window_idx": i,
                "center_time_sec": win_times[i],
                "hr_oximeter_bpm": h_ox,
                "hr_welch_resampled_bpm": h_w_res,
                "hr_peaks_resampled_bpm": h_p_res,
                "hr_welch_raw_30fps_bpm": h_w_raw,
                "diff_welch_ox_bpm": h_w_res - h_ox,
                "abs_diff_welch_ox_bpm": d_w_ox,
                "diff_peaks_ox_bpm": h_p_res - h_ox,
                "abs_diff_peaks_ox_bpm": d_p_ox,
                "diff_resampled_vs_raw_bpm": h_w_res - h_w_raw,
                "flagged_welch_discrepancy": flagged,
                "closer_reference_to_oximeter": closer_ref,
            })

    # Save summary dataframe
    df_subjects = pd.DataFrame(subject_rows)
    sub_path = repo_root / cfg["outputs"]["subject_summary_csv"]
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    df_subjects.to_csv(sub_path, index=False)
    print(f"\nSaved subject-level summary to {sub_path}")

    # Save windowed dataframe
    df_windows = pd.DataFrame(window_rows)
    win_path = repo_root / cfg["outputs"]["window_comparison_csv"]
    df_windows.to_csv(win_path, index=False)
    print(f"Saved window-level details to {win_path}")

    # -------------------------------------------------------------
    # (1) Spectrogram and GT PPG plot for Subject 3 (0-50s)
    # -------------------------------------------------------------
    if subject3_data is not None:
        print("\nGenerating Subject 3 diagnostic plot (0-50s)...")
        t_orig = subject3_data.time
        ppg_orig = subject3_data.ppg
        hr_orig = subject3_data.hr

        # 30Hz resampled
        t_u, ppg_u = resample_to_uniform(t_orig, ppg_orig, target_fs=fs)
        _, hr_u = resample_to_uniform(t_orig, hr_orig, target_fs=fs, method="linear")
        ppg_filt = butterworth_bandpass_filter(ppg_u, fs=fs, lowcut=0.7, highcut=3.0)

        # 0-50s mask
        mask_50 = t_u <= 50.0
        t_50 = t_u[mask_50]
        ppg_50 = ppg_filt[mask_50]
        hr_50 = hr_u[mask_50]

        # Detect peaks on 0-50s
        peaks, t_peaks_rel = detect_peaks_ppg(ppg_50, fs=fs, refine_subsample=True)
        t_peaks = t_50[0] + t_peaks_rel
        ibis = np.diff(t_peaks)
        hr_instant = 60.0 / ibis
        t_instant_mid = (t_peaks[:-1] + t_peaks[1:]) / 2.0

        # Find gaps in 0-50s
        diffs_orig = np.diff(t_orig)
        mean_dt = np.mean(diffs_orig)
        gap_indices = np.where((diffs_orig > 2.0 * mean_dt) & (t_orig[:-1] <= 50.0))[0]

        # Spectrogram: window 8s, 95% overlap
        nperseg = int(fs * cfg["parameters"]["spectrogram_window_sec"])
        noverlap = int(fs * cfg["parameters"]["spectrogram_overlap_sec"])
        f_spec, t_spec, Sxx = sp_signal.spectrogram(
            ppg_50, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=4096, window="hann"
        )
        mask_f = (f_spec >= 0.7) & (f_spec <= 3.0)
        f_bpm = f_spec[mask_f] * 60.0
        Sxx_db = 10 * np.log10(Sxx[mask_f, :] + 1e-12)

        # Filter windowed HR for subject 3 in 0-50s
        df_w3 = df_windows[(df_windows["subject_id"] == "subject3") & (df_windows["center_time_sec"] <= 50.0)]

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            4, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [1.2, 2.0, 1.2, 0.8]}
        )

        # Panel 1: GT PPG waveform with detected systolic peaks
        ax1.plot(t_50, ppg_50, color="#1f77b4", linewidth=1.2, label="Filtered GT PPG (0.7-3.0 Hz)")
        peak_idx_int = np.clip(np.round(t_peaks_rel * fs).astype(int), 0, len(ppg_50) - 1)
        ax1.scatter(t_peaks, ppg_50[peak_idx_int], color="red", s=25, zorder=5, label=f"Detected Beats (N={len(t_peaks)})")

        for gi in gap_indices:
            t_gap_start = t_orig[gi]
            t_gap_end = t_orig[gi + 1]
            ax1.axvspan(t_gap_start, t_gap_end, color="orange", alpha=0.35, label="Timestamp Gap" if gi == gap_indices[0] else "")

        ax1.set_ylabel("PPG (AU)", fontsize=10)
        ax1.set_title("Subject 3 Diagnostic: GT PPG Waveform, Spectrogram, Reference Comparison, and Gaps (0–50 s)\n[Reference vs Reference, Not Model Error | Resolution: 640x480]", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Panel 2: Spectrogram with reference curves overlaid
        im = ax2.pcolormesh(t_spec, f_bpm, Sxx_db, shading="gouraud", cmap="inferno")
        cb = fig.colorbar(im, ax=ax2, pad=0.015, aspect=20)
        cb.set_label("PSD (dB)", fontsize=9)

        # Overlay oximeter line
        ax2.plot(t_50, hr_50, color="cyan", linewidth=2.5, label="Oximeter HR Line")
        # Overlay Welch windowed HR
        ax2.plot(df_w3["center_time_sec"], df_w3["hr_welch_resampled_bpm"], color="white", linestyle="--", linewidth=2.0, marker="o", markersize=4, label="Welch PSD HR (30s Window)")
        # Overlay Peak windowed HR
        ax2.plot(df_w3["center_time_sec"], df_w3["hr_peaks_resampled_bpm"], color="lime", linestyle=":", linewidth=2.2, marker="s", markersize=4, label="Peak-Derived HR (Median IBI)")

        for gi in gap_indices:
            ax2.axvline(t_orig[gi], color="orange", linestyle="--", alpha=0.7)

        ax2.set_ylabel("Frequency (BPM)", fontsize=10)
        ax2.set_ylim(50, 150)
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, linestyle=":", alpha=0.4, color="white")

        # Panel 3: Discrepancy comparison across time
        ax3.plot(t_50, hr_50, color="black", linewidth=2.0, label="Oximeter Reference")
        ax3.plot(df_w3["center_time_sec"], df_w3["hr_welch_resampled_bpm"], color="red", linestyle="--", marker="o", markersize=4, label="Welch PSD (30s win)")
        ax3.plot(df_w3["center_time_sec"], df_w3["hr_peaks_resampled_bpm"], color="green", linestyle="-.", marker="^", markersize=4, label="Peaks Median IBI (30s win)")
        ax3.plot(t_instant_mid, hr_instant, color="blue", alpha=0.35, linewidth=1.0, label="Instantaneous Beat HR (60/IBI)")

        # Highlight flagged window centers
        flagged_w3 = df_w3[df_w3["flagged_welch_discrepancy"]]
        if len(flagged_w3) > 0:
            ax3.scatter(flagged_w3["center_time_sec"], flagged_w3["hr_welch_resampled_bpm"], color="darkred", s=60, edgecolors="black", zorder=6, label=f"Flagged Windows (|Diff|>5 BPM, N={len(flagged_w3)})")

        ax3.set_ylabel("Heart Rate (BPM)", fontsize=10)
        ax3.set_ylim(60, 140)
        ax3.legend(loc="upper left", fontsize=8, ncol=2)
        ax3.grid(True, linestyle="--", alpha=0.5)

        # Panel 4: Timestamp difference dt
        t_mid_orig = (t_orig[:-1] + t_orig[1:]) / 2.0
        mask_dt_50 = t_mid_orig <= 50.0
        ax4.plot(t_mid_orig[mask_dt_50], diffs_orig[mask_dt_50] * 1000.0, color="purple", linewidth=1.0, label=r"Interval $\Delta t$ (ms)")
        ax4.axhline(mean_dt * 1000.0, color="gray", linestyle="--", label=f"Mean dt ({mean_dt*1000:.1f} ms)")
        ax4.axhline(2.0 * mean_dt * 1000.0, color="red", linestyle=":", label="Gap Threshold (2x mean dt)")
        ax4.set_ylabel(r"$\Delta t$ (ms)", fontsize=10)
        ax4.set_xlabel("Time (seconds)", fontsize=10)
        ax4.set_xlim(0, 50)
        ax4.set_ylim(0, 260)
        ax4.legend(loc="upper right", fontsize=8)
        ax4.grid(True, linestyle="--", alpha=0.5)

        spec_plot_path = repo_root / cfg["outputs"]["spectrogram_plot"]
        fig.tight_layout()
        fig.savefig(spec_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Subject 3 diagnostic plot to {spec_plot_path}")

    # -------------------------------------------------------------
    # (3) Subject 3 Timestamp Gaps Analysis
    # -------------------------------------------------------------
    if subject3_data is not None:
        print("\nAnalyzing Subject 3's 6 timestamp gaps...")
        t3 = subject3_data.time
        ppg3 = subject3_data.ppg
        diffs3 = np.diff(t3)
        mean_dt3 = np.mean(diffs3)
        gap_idxs = np.where(diffs3 > 2.0 * mean_dt3)[0]

        gap_rows = []
        for i, idx in enumerate(gap_idxs):
            t_start = t3[idx]
            t_end = t3[idx + 1]
            dt_ms = diffs3[idx] * 1000.0
            missed_frames = int(round(diffs3[idx] / mean_dt3)) - 1

            # Find windows containing this gap
            overlapping_wins = df_windows[
                (df_windows["subject_id"] == "subject3")
                & (df_windows["center_time_sec"] - win_sec / 2.0 <= t_start)
                & (df_windows["center_time_sec"] + win_sec / 2.0 >= t_end)
            ]

            gap_rows.append({
                "git_hash": git_hash,
                "gap_idx": i + 1,
                "frame_start": idx,
                "frame_end": idx + 1,
                "time_start_sec": t_start,
                "time_end_sec": t_end,
                "gap_duration_ms": dt_ms,
                "estimated_missed_frames": missed_frames,
                "overlapping_windows_count": len(overlapping_wins),
                "mean_welch_resampled_in_gap_windows": overlapping_wins["hr_welch_resampled_bpm"].mean() if len(overlapping_wins) > 0 else np.nan,
                "mean_welch_raw_in_gap_windows": overlapping_wins["hr_welch_raw_30fps_bpm"].mean() if len(overlapping_wins) > 0 else np.nan,
                "mean_raw_minus_resampled_in_gap_windows": overlapping_wins["diff_resampled_vs_raw_bpm"].mean() if len(overlapping_wins) > 0 else np.nan,
            })

        df_gaps = pd.DataFrame(gap_rows)
        gaps_path = repo_root / cfg["outputs"]["subject3_gaps_csv"]
        df_gaps.to_csv(gaps_path, index=False)
        print(f"Saved Subject 3 gaps analysis to {gaps_path}")

    # -------------------------------------------------------------
    # (5) Bland-Altman Across Subjects (One point per subject)
    # -------------------------------------------------------------
    print("\nGenerating across-subject Bland-Altman plots...")
    # 1. Welch vs Oximeter
    ba_welch_path = repo_root / cfg["outputs"]["ba_plot_welch"]
    plot_bland_altman_across_subjects(
        df_subjects,
        true_col="hr_oximeter_vid_bpm",
        pred_col="hr_welch_resampled_vid_bpm",
        subject_col="subject_id",
        res_col="resolution",
        title="Across-Subject Bland-Altman: Welch PSD HR vs Oximeter Line\n[Reference vs Reference, Not Model Error]",
        save_path=ba_welch_path,
    )
    print(f"Saved across-subject Bland-Altman (Welch vs Oximeter) to {ba_welch_path}")

    # 2. Peaks vs Oximeter
    ba_peaks_path = repo_root / cfg["outputs"]["ba_plot_peaks"]
    plot_bland_altman_across_subjects(
        df_subjects,
        true_col="hr_oximeter_vid_bpm",
        pred_col="hr_peaks_resampled_vid_bpm",
        subject_col="subject_id",
        res_col="resolution",
        title="Across-Subject Bland-Altman: Peaks Median IBI HR vs Oximeter Line\n[Reference vs Reference, Not Model Error]",
        save_path=ba_peaks_path,
    )
    print(f"Saved across-subject Bland-Altman (Peaks vs Oximeter) to {ba_peaks_path}")

    # -------------------------------------------------------------
    # Print formatted tables to console
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print("TABLE 1: SUBJECT-LEVEL REFERENCE COMPARISON (REFERENCE VS REFERENCE, NOT MODEL ERROR)")
    print("=" * 95)
    cols_sub = [
        "subject_id",
        "resolution",
        "nominal_fps",
        "hr_oximeter_vid_bpm",
        "hr_welch_resampled_vid_bpm",
        "hr_peaks_resampled_vid_bpm",
        "abs_diff_welch_ox_vid_bpm",
        "abs_diff_peaks_ox_vid_bpm",
        "welch_vs_ox_mae_bpm",
        "peaks_vs_ox_mae_bpm",
        "flagged_windows_count",
    ]
    print(df_subjects[cols_sub].to_string(index=False))

    print("\n" + "=" * 95)
    print("TABLE 2: SUBJECT 3 FLAGGED WINDOWS COMPARISON (REFERENCE VS REFERENCE, NOT MODEL ERROR)")
    print("=" * 95)
    df_s3_flagged = df_windows[(df_windows["subject_id"] == "subject3") & (df_windows["flagged_welch_discrepancy"])]
    cols_win = [
        "window_idx",
        "center_time_sec",
        "hr_oximeter_bpm",
        "hr_welch_resampled_bpm",
        "abs_diff_welch_ox_bpm",
        "hr_peaks_resampled_bpm",
        "abs_diff_peaks_ox_bpm",
        "closer_reference_to_oximeter",
    ]
    print(df_s3_flagged[cols_win].to_string(index=False))

    print("\nDiagnosis completed successfully.")


if __name__ == "__main__":
    main()
