"""Phase C: Pilot Evaluation on Next 4 Subjects in Sequential ID Order.

Runs frozen baseline pipeline (GT-timestamp timing + synchronous feeding + verified spatial settings)
on the next 4 subjects in sequential numerical ID order (subject5, subject8, subject9, subject10)
without cherry-picking.

Evaluation Gate:
- Median video-level error <= 2.0 BPM
- At least 70% of subjects with waveform r >= 0.7 (lag search +/- 0.5s)
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def extract_subject_number(subject_id: str) -> int:
    """Extract numeric integer from subject ID (e.g. 'subject12' -> 12)."""
    digits = re.findall(r"\d+", subject_id)
    return int(digits[0]) if digits else 999999


def main():
    parser = argparse.ArgumentParser(description="Phase C Pilot Evaluation (4 Subjects)")
    parser.add_argument("--data-dir", type=str, default=None, help="Root dataset directory")
    parser.add_argument("--subjects", nargs="+", default=None, help="Explicit list of subjects")
    parser.add_argument("--count", type=int, default=4, help="Number of pilot subjects to evaluate (default: 4)")
    args = parser.parse_args()

    git_hash = get_git_hash()
    print("=" * 80)
    print("=== Phase C: Pilot Evaluation on Next Sequential Subjects (4 Subjects) ===")
    print(f"Git commit hash: {git_hash}")
    print("Pre-registered Baseline of Record: GT-timestamp path (timing_mode='gt_timestamp')")
    print("Synchronous worker execution: ENABLED (100% deterministic)")
    print("Lag search window: +/- 0.5 s")
    print("=" * 80)

    config_path = PROJECT_ROOT / "configs" / "pilot_4_subjects.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Determine dataset directory
    if args.data_dir:
        dataset_root = Path(args.data_dir)
    else:
        local_dir = PROJECT_ROOT / cfg.get("dataset", {}).get("root_dir", "data/ubfc/Data Set 2")
        fallback_dir = Path(cfg.get("dataset", {}).get("fallback_dir", "/content/drive/MyDrive/DATASET_2"))
        if local_dir.is_dir():
            dataset_root = local_dir
        elif fallback_dir.is_dir():
            dataset_root = fallback_dir
        else:
            dataset_root = local_dir

    print(f"Looking for dataset in: {dataset_root}")
    subjects_map = find_ubfc_subjects(dataset_root)

    # Select pilot subjects
    if args.subjects:
        pilot_subject_ids = args.subjects
    else:
        excluded = set(cfg.get("dataset", {}).get("exclude_exploration_subjects", ["subject1", "subject3", "subject4"]))
        all_sids = sorted(
            [sid for sid in subjects_map.keys() if sid not in excluded],
            key=extract_subject_number,
        )
        target_count = args.count if args.count else int(cfg.get("dataset", {}).get("pilot_subject_count", 4))
        pilot_subject_ids = all_sids[:target_count]

    print(f"Discovered subjects in numerical ID order (excluding exploration set {list(excluded) if not args.subjects else []}):")
    print(f"Pilot targets ({len(pilot_subject_ids)} subjects): {pilot_subject_ids}")

    if len(pilot_subject_ids) == 0:
        print("Error: No eligible pilot subjects found!")
        sys.exit(1)

    target_fs = float(cfg.get("preprocessing", {}).get("target_fs", 30.0))
    lowcut = float(cfg.get("preprocessing", {}).get("bandpass", {}).get("lowcut", 0.7))
    highcut = float(cfg.get("preprocessing", {}).get("bandpass", {}).get("highcut", 3.0))
    order = int(cfg.get("preprocessing", {}).get("bandpass", {}).get("order", 3))
    win_sec = float(cfg.get("preprocessing", {}).get("windowing", {}).get("window_sec", 30.0))
    stride_sec = float(cfg.get("preprocessing", {}).get("windowing", {}).get("stride_sec", 1.0))
    threshold_bpm = float(cfg.get("metrics", {}).get("threshold_bpm", 5.0))
    max_lag_sec = float(cfg.get("metrics", {}).get("lag_search_max_sec", 0.5))
    pad_px = int(cfg.get("padding_rule_hypothesis", {}).get("fixed_border_px", 60))

    gate_max_medae = float(cfg.get("gate", {}).get("max_median_video_error_bpm", 2.0))
    gate_min_pct_r = float(cfg.get("gate", {}).get("min_pct_waveform_r_above_07", 70.0))

    wrapper = FacePhysWrapper()
    results_rows = []

    for sid in pilot_subject_ids:
        sdir = subjects_map[sid]
        sdata = load_ubfc_subject(sdir, subject_id=sid)
        vid_path = sdata.video_path
        vid_size_bytes = os.path.getsize(str(vid_path))
        vid_size_mb = vid_size_bytes / (1024.0 * 1024.0)

        orig_w = sdata.video_metadata["width"]
        orig_h = sdata.video_metadata["height"]
        print(f"\nProcessing {sid}:")
        print(f"  Exact Video Path: {vid_path}")
        print(f"  Resolution: {orig_w}x{orig_h}, File Size: {vid_size_bytes:,} bytes ({vid_size_mb:.2f} MB)")

        # Read frames
        cap = cv2.VideoCapture(str(vid_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        n_frames = len(frames)
        if n_frames == 0:
            print(f"Error: 0 frames read from {vid_path}. Recording failure.")
            results_rows.append({
                "git_hash": git_hash,
                "subject_id": sid,
                "video_path": str(vid_path),
                "video_size_bytes": vid_size_bytes,
                "video_size_mb": vid_size_mb,
                "status": "FAIL_NO_FRAMES",
            })
            continue

        # If tight crop (< 350px), apply padding hypothesis rule
        is_tight_crop = (orig_w < 350 or orig_h < 350)
        if is_tight_crop:
            print(f"  Tight crop detected ({orig_w}x{orig_h}). Applying +{pad_px}px border padding.")
            frames_to_feed = [
                cv2.copyMakeBorder(f, pad_px, pad_px, pad_px, pad_px, cv2.BORDER_REFLECT)
                for f in frames
            ]
        else:
            frames_to_feed = frames

        # Predict with baseline of record (GT-timestamp timing + synchronous worker execution)
        bvp_u, t_u, meta = wrapper.predict_from_frames(
            frames=frames_to_feed,
            timestamps=sdata.time,
            timing_mode="gt_timestamp",
            synchronous=True,
            fs_target=target_fs,
        )

        bvp_raw = meta["raw_bvp"]
        ts_raw = meta["raw_timestamps"]
        ts_rel = (ts_raw - ts_raw[0]) if len(ts_raw) > 0 else np.arange(len(bvp_raw)) / target_fs

        t_gt_rel = sdata.time - sdata.time[0]
        n_common = int(np.floor(min(t_gt_rel[-1], ts_rel[-1]) * target_fs))

        _, ppg_common = resample_to_uniform(t_gt_rel, sdata.ppg, target_fs=target_fs)
        ppg_common = ppg_common[:n_common]

        _, hr_gt_common = resample_to_uniform(t_gt_rel, sdata.hr, target_fs=target_fs, method="linear")
        hr_gt_common = hr_gt_common[:n_common]

        _, bvp_common = resample_to_uniform(ts_rel, bvp_raw, target_fs=target_fs)
        bvp_common = bvp_common[:n_common]

        primary_ref_hr = estimate_hr_video(ppg_common, fs=target_fs, lowcut=lowcut, highcut=highcut, order=order)
        secondary_ref_hr = float(np.mean(hr_gt_common))

        bvp_filt = butterworth_bandpass_filter(
            bvp_common, fs=target_fs, lowcut=lowcut, highcut=highcut, order=order
        )
        video_hr, f_bvp, psd_bvp = estimate_hr_welch(
            bvp_filt, fs=target_fs, low_bpm=lowcut * 60.0, high_bpm=highcut * 60.0, nfft=16384
        )
        abs_err_primary = abs(video_hr - primary_ref_hr)
        abs_err_secondary = abs(video_hr - secondary_ref_hr)

        direct_r = calc_pearson_r(bvp_common, ppg_common)
        # Limited lag search (+/- 0.5s) per protocol
        lag_dict = max_lag_waveform_correlation(bvp_common, ppg_common, fs=target_fs, max_lag_sec=max_lag_sec)
        max_lag_r = lag_dict["max_r"]
        best_lag_s = lag_dict["optimal_lag_sec"]

        _, win_hrs = estimate_hr_windowed(
            bvp_common, fs=target_fs, window_sec=win_sec, stride_sec=stride_sec,
            lowcut=lowcut, highcut=highcut, order=order
        )
        _, gt_win_hrs = estimate_hr_windowed(
            ppg_common, fs=target_fs, window_sec=win_sec, stride_sec=stride_sec,
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
            psd_bvp, f_bvp, ref_hr_bpm=primary_ref_hr, delta_hz=0.1, band=(lowcut, highcut)
        )

        results_rows.append({
            "git_hash": git_hash,
            "subject_id": sid,
            "status": "SUCCESS",
            "video_path": str(vid_path),
            "video_size_bytes": vid_size_bytes,
            "video_size_mb": round(vid_size_mb, 2),
            "video_resolution": f"{orig_w}x{orig_h}",
            "video_frames": n_frames,
            "is_tight_crop": is_tight_crop,
            "padding_applied": is_tight_crop,
            "primary_ref_hr_bpm": primary_ref_hr,
            "secondary_ref_hr_bpm": secondary_ref_hr,
            "video_hr_bpm": video_hr,
            "error_vs_primary_bpm": abs_err_primary,
            "error_vs_secondary_bpm": abs_err_secondary,
            "waveform_r_direct": direct_r,
            "waveform_r_max_lag_05s": max_lag_r,
            "waveform_best_lag_sec": best_lag_s,
            "windowed_mae_bpm": w_mae,
            "windowed_medae_bpm": w_medae,
            "windowed_rmse_bpm": w_rmse,
            "windowed_pearson_r": w_pearson,
            "windowed_acc5_pct": w_acc5,
            "snr_db": snr_val,
            "stat_null": meta["face_null_frames"],
            "stat_filled": meta["filled_frames"],
            "stat_skipped": meta["skipped_frames"],
        })

        print(f"  --> {sid}: Video HR: {video_hr:.2f} BPM (GT: {primary_ref_hr:.2f} BPM, Err: {abs_err_primary:.2f} BPM), "
              f"r_maxlag: {max_lag_r:.4f} (lag {best_lag_s:+.2f}s), MedAE: {w_medae:.2f} BPM, Acc5: {w_acc5:.1f}%, SNR: {snr_val:.2f} dB")

    df_pilot = pd.DataFrame(results_rows)
    results_csv = PROJECT_ROOT / cfg.get("outputs", {}).get("results_csv", "results/pilot_4_subjects.csv")
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    df_pilot.to_csv(results_csv, index=False)
    print(f"\n--> Successfully saved pilot results to: {results_csv}")

    # =========================================================================
    # GATE EVALUATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("=== PILOT GATE EVALUATION (4 SUBJECTS) ===")
    print("=" * 80)
    print(f"Total Pilot Subjects Evaluated: {len(df_pilot)}")
    success_df = df_pilot[df_pilot["status"] == "SUCCESS"]

    if len(success_df) == 0:
        print("GATE FAILED: No successful subject runs.")
        return

    median_error = float(success_df["error_vs_primary_bpm"].median())
    mean_error = float(success_df["error_vs_primary_bpm"].mean())
    frac_r_07 = float(np.mean(success_df["waveform_r_max_lag_05s"] >= 0.7) * 100.0)

    gate1_pass = median_error <= gate_max_medae
    gate2_pass = frac_r_07 >= gate_min_pct_r
    overall_pass = gate1_pass and gate2_pass

    print(f"Gate 1 (Median Video Error <= {gate_max_medae:.1f} BPM): "
          f"{median_error:.2f} BPM --> {'PASS [OK]' if gate1_pass else 'FAIL [X]'}")
    print(f"Gate 2 (Waveform r >= 0.7 on >= {gate_min_pct_r:.0f}% of subjects): "
          f"{frac_r_07:.1f}% ({int(np.sum(success_df['waveform_r_max_lag_05s'] >= 0.7))}/{len(success_df)}) --> {'PASS [OK]' if gate2_pass else 'FAIL [X]'}")
    print(f"Mean Video Error: {mean_error:.2f} BPM")
    print(f"Overall Gate Verdict: {'ALL GATES PASSED [OK]' if overall_pass else 'GATE NOT MET [FAIL]'}")
    print("=" * 80)

    summary_cols = [
        "subject_id", "video_resolution", "video_size_mb", "video_hr_bpm",
        "primary_ref_hr_bpm", "error_vs_primary_bpm", "waveform_r_max_lag_05s",
        "windowed_medae_bpm", "windowed_acc5_pct", "snr_db"
    ]
    print(df_pilot[[c for c in summary_cols if c in df_pilot.columns]].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()

