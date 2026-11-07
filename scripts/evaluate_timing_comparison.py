"""Phase A: Controlled Timing Comparison across 3 Paths on Identical Frames.

Compares:
1) Path 1: Library default (open-rppg model.process_video using PyAV container timestamps)
2) Path 2: Uniform-synthetic (ts = i / 30.0 passed to update_frame)
3) Path 3: GT-timestamp (ts = t_gt[i] passed to update_frame - PRE-REGISTERED BASELINE-OF-RECORD)

All three outputs and Ground Truth PPG are resampled onto the EXACT same uniform timeline
at fs=30.0 Hz with equal length, enabling direct sample-to-sample comparison, cross-path
correlations, and lag-aligned waveform correlations against GT.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rppg
from src.eval import (
    calc_mae,
    calc_medae,
    calc_rmse,
    calc_pearson_r,
    calc_fraction_within_threshold,
    calculate_snr,
    max_lag_waveform_correlation,
)
from src.hr import (
    butterworth_bandpass_filter,
    estimate_hr_video,
    estimate_hr_welch,
    estimate_hr_windowed,
)
from src.io import find_ubfc_subjects, load_ubfc_subject, resample_to_uniform
from src.models import FacePhysWrapper


def get_git_hash() -> str:
    try:
        res = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
        )
        return res.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN"


def main():
    git_hash = get_git_hash()
    print("=" * 80)
    print("=== Phase A: Timing & 3-Path Comparison ===")
    print(f"Git commit hash: {git_hash}")
    print("Pre-registered Baseline of Record: GT-timestamp path (timing_mode='gt_timestamp')")
    print("=" * 80)

    config_path = PROJECT_ROOT / "configs" / "timing_comparison.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    target_fs = float(cfg.get("preprocessing", {}).get("target_fs", 30.0))
    lowcut = float(cfg.get("preprocessing", {}).get("bandpass", {}).get("lowcut", 0.7))
    highcut = float(cfg.get("preprocessing", {}).get("bandpass", {}).get("highcut", 3.0))
    order = int(cfg.get("preprocessing", {}).get("bandpass", {}).get("order", 3))
    win_sec = float(cfg.get("preprocessing", {}).get("windowing", {}).get("window_sec", 30.0))
    stride_sec = float(cfg.get("preprocessing", {}).get("windowing", {}).get("stride_sec", 1.0))
    threshold_bpm = float(cfg.get("metrics", {}).get("threshold_bpm", 5.0))
    max_lag_sec = float(cfg.get("metrics", {}).get("lag_search_max_sec", 2.0))
    snr_delta_hz = float(cfg.get("metrics", {}).get("snr_delta_hz", 0.1))

    dataset_root = PROJECT_ROOT / cfg.get("dataset", {}).get("root_dir", "data/ubfc/Data Set 2")
    subjects_map = find_ubfc_subjects(dataset_root)
    target_subject_ids = cfg.get("dataset", {}).get("subjects", ["subject1", "subject3", "subject4"])

    results_rows = []
    crosscheck_rows = []

    plot_dir = PROJECT_ROOT / cfg.get("outputs", {}).get("plot_dir", "results/plots/timing_comparison")
    plot_dir.mkdir(parents=True, exist_ok=True)

    for sid in target_subject_ids:
        if sid not in subjects_map:
            print(f"Warning: Subject {sid} not found in {dataset_root}. Skipping.")
            continue

        sdir = subjects_map[sid]
        sdata = load_ubfc_subject(sdir, subject_id=sid)
        vid_path = sdata.video_path
        res_str = (
            f"{sdata.video_metadata['width']}x{sdata.video_metadata['height']}"
            if sdata.video_metadata
            else "unknown"
        )
        print(f"\nProcessing {sid} (Resolution: {res_str}, Video: {vid_path})...")

        # -------------------------------------------------------------
        # Path 1: Library default (open-rppg process_video via PyAV container PTS)
        # -------------------------------------------------------------
        print("  Running Path 1: Library Default (process_video with container PTS)...")
        m_lib = rppg.Model("FacePhys.rlap")
        lib_hr_dict = m_lib.process_video(str(vid_path))
        lib_builtin_hr = float(lib_hr_dict.get("hr", float("nan"))) if lib_hr_dict else float("nan")
        bvp_p1_raw, ts_p1_raw = m_lib.bvp(raw=True)
        bvp_p1_raw = np.asarray(bvp_p1_raw, dtype=np.float64)
        ts_p1_raw = np.asarray(ts_p1_raw, dtype=np.float64)
        stats_p1 = dict(m_lib.statistic)

        # -------------------------------------------------------------
        # Path 2: Uniform-synthetic (ts = i / 30.0 passed to update_frame)
        # -------------------------------------------------------------
        print("  Running Path 2: Uniform-Synthetic (ts = i / 30.0)...")
        wrapper_p2 = FacePhysWrapper()
        bvp_p2_u, t_p2_u, meta_p2 = wrapper_p2.predict_video(
            vid_path, timing_mode="uniform", fs_target=target_fs
        )
        bvp_p2_raw = meta_p2["raw_bvp"]
        ts_p2_raw = meta_p2["raw_timestamps"]
        stats_p2 = meta_p2["face_stats"]

        # -------------------------------------------------------------
        # Path 3: GT-timestamp (ts = t_gt[i] passed to update_frame - BASELINE OF RECORD)
        # -------------------------------------------------------------
        print("  Running Path 3: GT-Timestamp [PRE-REGISTERED BASELINE] (ts = t_gt[i])...")
        wrapper_p3 = FacePhysWrapper()
        bvp_p3_u, t_p3_u, meta_p3 = wrapper_p3.predict_video(
            vid_path, timestamps=sdata.time, timing_mode="gt_timestamp", fs_target=target_fs
        )
        bvp_p3_raw = meta_p3["raw_bvp"]
        ts_p3_raw = meta_p3["raw_timestamps"]
        stats_p3 = meta_p3["face_stats"]

        # -------------------------------------------------------------
        # Align all signals to the same uniform timeline with equal length
        # -------------------------------------------------------------
        # Standardize time base to start at 0.0
        t_gt_rel = sdata.time - sdata.time[0]
        ts_p1_rel = (ts_p1_raw - ts_p1_raw[0]) if len(ts_p1_raw) > 0 else np.arange(len(bvp_p1_raw)) / target_fs
        ts_p2_rel = (ts_p2_raw - ts_p2_raw[0]) if len(ts_p2_raw) > 0 else np.arange(len(bvp_p2_raw)) / target_fs
        ts_p3_rel = (ts_p3_raw - ts_p3_raw[0]) if len(ts_p3_raw) > 0 else np.arange(len(bvp_p3_raw)) / target_fs

        # Common duration: minimum of all signal spans
        max_duration = min(t_gt_rel[-1], ts_p1_rel[-1], ts_p2_rel[-1], ts_p3_rel[-1])
        n_common = int(np.floor(max_duration * target_fs))
        t_common = np.arange(n_common) / target_fs

        # Resample onto common grid
        _, ppg_common = resample_to_uniform(t_gt_rel, sdata.ppg, target_fs=target_fs)
        ppg_common = ppg_common[:n_common]

        _, hr_gt_common = resample_to_uniform(t_gt_rel, sdata.hr, target_fs=target_fs, method="linear")
        hr_gt_common = hr_gt_common[:n_common]

        _, bvp_p1_common = resample_to_uniform(ts_p1_rel, bvp_p1_raw, target_fs=target_fs)
        bvp_p1_common = bvp_p1_common[:n_common]

        _, bvp_p2_common = resample_to_uniform(ts_p2_rel, bvp_p2_raw, target_fs=target_fs)
        bvp_p2_common = bvp_p2_common[:n_common]

        _, bvp_p3_common = resample_to_uniform(ts_p3_rel, bvp_p3_raw, target_fs=target_fs)
        bvp_p3_common = bvp_p3_common[:n_common]

        # Ground truth references
        primary_ref_hr = estimate_hr_video(ppg_common, fs=target_fs, lowcut=lowcut, highcut=highcut, order=order)
        secondary_ref_hr = float(np.mean(hr_gt_common))

        # Windowed ground truth
        _, gt_win_hrs = estimate_hr_windowed(
            ppg_common, fs=target_fs, window_sec=win_sec, stride_sec=stride_sec,
            lowcut=lowcut, highcut=highcut, order=order
        )

        paths_dict = {
            "path1_library_default": {
                "bvp": bvp_p1_common,
                "stats": stats_p1,
                "raw_len": len(bvp_p1_raw),
                "builtin_hr": lib_builtin_hr,
            },
            "path2_uniform_synthetic": {
                "bvp": bvp_p2_common,
                "stats": stats_p2,
                "raw_len": len(bvp_p2_raw),
                "builtin_hr": float("nan"),
            },
            "path3_gt_timestamp_baseline": {
                "bvp": bvp_p3_common,
                "stats": stats_p3,
                "raw_len": len(bvp_p3_raw),
                "builtin_hr": float("nan"),
            },
        }

        # Calculate metrics for each path
        path_metrics = {}
        for pname, pdata in paths_dict.items():
            bvp_sig = pdata["bvp"]
            bvp_filt = butterworth_bandpass_filter(
                bvp_sig, fs=target_fs, lowcut=lowcut, highcut=highcut, order=order
            )
            video_hr, f_bvp, psd_bvp = estimate_hr_welch(
                bvp_filt, fs=target_fs, low_bpm=lowcut * 60.0, high_bpm=highcut * 60.0, nfft=16384
            )
            abs_err_primary = abs(video_hr - primary_ref_hr)
            abs_err_secondary = abs(video_hr - secondary_ref_hr)

            # Waveform correlation with GT PPG:
            direct_r = calc_pearson_r(bvp_sig, ppg_common)
            lag_dict = max_lag_waveform_correlation(
                bvp_sig, ppg_common, fs=target_fs, max_lag_sec=max_lag_sec
            )
            max_lag_r = lag_dict["max_r"]
            best_lag_s = lag_dict["optimal_lag_sec"]

            # Windowed HR metrics:
            _, win_hrs = estimate_hr_windowed(
                bvp_sig, fs=target_fs, window_sec=win_sec, stride_sec=stride_sec,
                lowcut=lowcut, highcut=highcut, order=order
            )

            valid_mask = ~np.isnan(win_hrs) & ~np.isnan(gt_win_hrs)
            if np.sum(valid_mask) > 0:
                p_win = win_hrs[valid_mask]
                g_win = gt_win_hrs[valid_mask]
                w_mae = calc_mae(p_win, g_win)
                w_medae = calc_medae(p_win, g_win)
                w_rmse = calc_rmse(p_win, g_win)
                w_pearson = calc_pearson_r(p_win, g_win)
                w_acc5 = calc_fraction_within_threshold(p_win, g_win, threshold=threshold_bpm)
            else:
                w_mae = w_medae = w_rmse = w_pearson = w_acc5 = float("nan")

            snr_val = calculate_snr(
                psd_bvp, f_bvp, ref_hr_bpm=primary_ref_hr, delta_hz=snr_delta_hz, band=(lowcut, highcut)
            )

            path_metrics[pname] = {
                "video_hr": video_hr,
                "abs_err_primary": abs_err_primary,
                "abs_err_secondary": abs_err_secondary,
                "direct_waveform_r": direct_r,
                "max_lag_waveform_r": max_lag_r,
                "best_lag_sec": best_lag_s,
                "windowed_mae": w_mae,
                "windowed_medae": w_medae,
                "windowed_rmse": w_rmse,
                "windowed_pearson_r": w_pearson,
                "windowed_acc5": w_acc5,
                "snr_db": snr_val,
            }

            results_rows.append({
                "git_hash": git_hash,
                "subject_id": sid,
                "video_path": str(vid_path),
                "video_size_bytes": os.path.getsize(str(vid_path)),
                "video_size_mb": round(os.path.getsize(str(vid_path)) / (1024.0 * 1024.0), 2),
                "resolution": res_str,
                "timing_path": pname,
                "is_preregistered_baseline": (pname == "path3_gt_timestamp_baseline"),
                "raw_bvp_len": pdata["raw_len"],
                "common_len": n_common,
                "stat_filled": pdata["stats"].get("filled", 0),
                "stat_skipped": pdata["stats"].get("skipped", 0),
                "stat_null": pdata["stats"].get("null", 0),
                "primary_ref_hr_bpm": primary_ref_hr,
                "secondary_ref_hr_bpm": secondary_ref_hr,
                "video_hr_bpm": video_hr,
                "error_vs_primary_bpm": abs_err_primary,
                "error_vs_secondary_bpm": abs_err_secondary,
                "waveform_r_direct": direct_r,
                "waveform_r_max_lag": max_lag_r,
                "waveform_best_lag_sec": best_lag_s,
                "windowed_mae_bpm": w_mae,
                "windowed_medae_bpm": w_medae,
                "windowed_rmse_bpm": w_rmse,
                "windowed_pearson_r": w_pearson,
                "windowed_acc5_pct": w_acc5,
                "snr_db": snr_val,
            })

            print(f"    [{pname}] Video HR: {video_hr:.2f} BPM (Err: {abs_err_primary:.2f} BPM), "
                  f"r_direct: {direct_r:.4f}, r_maxlag: {max_lag_r:.4f} (lag {best_lag_s:+.2f}s), "
                  f"MedAE: {w_medae:.2f} BPM, Acc5: {w_acc5:.1f}%, SNR: {snr_val:.2f} dB, "
                  f"Filled: {pdata['stats'].get('filled', 0)}")

        # -------------------------------------------------------------
        # Pairwise Cross-Check Correlations between the 3 paths
        # -------------------------------------------------------------
        r_p1_p2 = calc_pearson_r(bvp_p1_common, bvp_p2_common)
        r_p1_p3 = calc_pearson_r(bvp_p1_common, bvp_p3_common)
        r_p2_p3 = calc_pearson_r(bvp_p2_common, bvp_p3_common)

        res_p1_p2 = max_lag_waveform_correlation(bvp_p1_common, bvp_p2_common, fs=target_fs, max_lag_sec=max_lag_sec)
        max_r_p1_p2, lag_p1_p2 = res_p1_p2["max_r"], res_p1_p2["optimal_lag_sec"]

        res_p1_p3 = max_lag_waveform_correlation(bvp_p1_common, bvp_p3_common, fs=target_fs, max_lag_sec=max_lag_sec)
        max_r_p1_p3, lag_p1_p3 = res_p1_p3["max_r"], res_p1_p3["optimal_lag_sec"]

        res_p2_p3 = max_lag_waveform_correlation(bvp_p2_common, bvp_p3_common, fs=target_fs, max_lag_sec=max_lag_sec)
        max_r_p2_p3, lag_p2_p3 = res_p2_p3["max_r"], res_p2_p3["optimal_lag_sec"]

        crosscheck_rows.append({
            "git_hash": git_hash,
            "subject_id": sid,
            "resolution": res_str,
            "common_samples": n_common,
            "r_direct_lib_vs_uniform": r_p1_p2,
            "r_maxlag_lib_vs_uniform": max_r_p1_p2,
            "lag_lib_vs_uniform_sec": lag_p1_p2,
            "r_direct_lib_vs_gttime": r_p1_p3,
            "r_maxlag_lib_vs_gttime": max_r_p1_p3,
            "lag_lib_vs_gttime_sec": lag_p1_p3,
            "r_direct_uniform_vs_gttime": r_p2_p3,
            "r_maxlag_uniform_vs_gttime": max_r_p2_p3,
            "lag_uniform_vs_gttime_sec": lag_p2_p3,
        })

        print(f"    --> Cross-Path Waveform Correlations for {sid}:")
        print(f"        Path1 (Lib) vs Path2 (Uniform): direct r = {r_p1_p2:.4f}, max-lag r = {max_r_p1_p2:.4f} (lag {lag_p1_p2:+.2f}s)")
        print(f"        Path1 (Lib) vs Path3 (GT-Time): direct r = {r_p1_p3:.4f}, max-lag r = {max_r_p1_p3:.4f} (lag {lag_p1_p3:+.2f}s)")
        print(f"        Path2 (Uniform) vs Path3 (GT-Time): direct r = {r_p2_p3:.4f}, max-lag r = {max_r_p2_p3:.4f} (lag {lag_p2_p3:+.2f}s)")

        # -------------------------------------------------------------
        # Plot waveforms overlay for diagnostic audit
        # -------------------------------------------------------------
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        t_plot = t_common[:600]  # First 20 seconds
        axes[0].plot(t_plot, ppg_common[:600], label="Ground Truth PPG", color="green")
        axes[0].set_title(f"{sid} - Ground Truth PPG (Ref HR: {primary_ref_hr:.2f} BPM)")
        axes[0].set_ylabel("Normalized PPG")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t_plot, bvp_p3_common[:600], label="Path 3: GT-Timestamp [Baseline of Record]", color="blue")
        axes[1].plot(t_plot, bvp_p2_common[:600], label="Path 2: Uniform-Synthetic", color="orange", alpha=0.7, linestyle="--")
        axes[1].set_title(f"{sid} - Model Predictions (GT-Timestamp vs Uniform)")
        axes[1].set_ylabel("Raw BVP")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t_plot, bvp_p3_common[:600], label="Path 3: GT-Timestamp [Baseline of Record]", color="blue")
        axes[2].plot(t_plot, bvp_p1_common[:600], label="Path 1: Library Default (process_video)", color="red", alpha=0.7, linestyle=":")
        axes[2].set_title(f"{sid} - Model Predictions (GT-Timestamp vs Library Default)")
        axes[2].set_xlabel("Time (seconds)")
        axes[2].set_ylabel("Raw BVP")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = plot_dir / f"{sid}_timing_3path_overlay.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"    Saved diagnostic overlay plot: {fig_path}")

    df_results = pd.DataFrame(results_rows)
    results_csv = PROJECT_ROOT / cfg.get("outputs", {}).get("results_csv", "results/timing_comparison.csv")
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(results_csv, index=False)
    print(f"\n--> Successfully saved timing comparison results to: {results_csv}")

    df_crosscheck = pd.DataFrame(crosscheck_rows)
    crosscheck_csv = PROJECT_ROOT / cfg.get("outputs", {}).get("crosscheck_csv", "results/crosscheck_3paths.csv")
    df_crosscheck.to_csv(crosscheck_csv, index=False)
    print(f"--> Successfully saved crosscheck correlation results to: {crosscheck_csv}")

    print("\n" + "=" * 80)
    print("=== SUMMARY OF 3-PATH TIMING EVALUATION ===")
    print("=" * 80)
    summary_cols = [
        "subject_id", "timing_path", "video_hr_bpm", "primary_ref_hr_bpm",
        "error_vs_primary_bpm", "waveform_r_direct", "waveform_r_max_lag",
        "windowed_mae_bpm", "windowed_medae_bpm", "windowed_acc5_pct", "stat_filled"
    ]
    print(df_results[summary_cols].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()

