"""Phase B: Controlled Face-Crop & Padding Evaluation on Identical Frames.

Evaluates the effect of tight face cropping (face touching frame boundaries) vs
border padding across subjects on identical frames:
- 640x480 subjects (subject1, subject3):
  1) Tight face crop: face bounding box cropped tightly (0px background margin, mirroring subject4).
  2) Padded face crop: tight face crop + symmetric pre-set padding (+60px reflect border).
  3) Full uncropped frame (640x480).
- subject4 (native 242x242 tight face crop):
  1) Unpadded tight face crop (native 242x242).
  2) Padded face crop (+60px reflect border -> 362x362).

All conditions are evaluated using the pre-registered baseline (Path 3: GT-timestamp timing)
on identical frames, resampled to uniform 30 Hz with equal length.
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


def detect_face_bbox(frames: List[np.ndarray]) -> Tuple[int, int, int, int]:
    """Detect frontal face bounding box (x, y, w, h) using open-rppg detector."""
    import rppg
    m = rppg.Model()
    # Try first 30 frames to find a stable detection
    for frame in frames[:30]:
        res = m.detector.detect(frame)
        if len(res) > 0:
            box = res[0][0]
            x1, y1, x2, y2 = int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3]))
            w = x2 - x1
            h = y2 - y1
            if w > 40 and h > 40:
                return max(0, x1), max(0, y1), w, h

    # Fallback to central 60% crop if no face detected
    h, w, _ = frames[0].shape
    crop_w = int(w * 0.6)
    crop_h = int(h * 0.6)
    x = (w - crop_w) // 2
    y = (h - crop_h) // 2
    return x, y, crop_w, crop_h


def main():
    git_hash = get_git_hash()
    print("=" * 80)
    print("=== Phase B: Controlled Face-Crop & Border Padding Test ===")
    print(f"Git commit hash: {git_hash}")
    print("Pre-registered Baseline of Record: GT-timestamp path (timing_mode='gt_timestamp')")
    print("=" * 80)

    config_path = PROJECT_ROOT / "configs" / "padding_controlled.yaml"
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
    pad_px = int(cfg.get("padding_rule", {}).get("fixed_border_px", 60))

    dataset_root = PROJECT_ROOT / cfg.get("dataset", {}).get("root_dir", "data/ubfc/Data Set 2")
    subjects_map = find_ubfc_subjects(dataset_root)
    target_subject_ids = cfg.get("dataset", {}).get("subjects", ["subject1", "subject3", "subject4"])

    results_rows = []
    plot_dir = PROJECT_ROOT / cfg.get("outputs", {}).get("plot_dir", "results/plots/padding_controlled")
    plot_dir.mkdir(parents=True, exist_ok=True)

    wrapper = FacePhysWrapper()

    for sid in target_subject_ids:
        if sid not in subjects_map:
            print(f"Warning: Subject {sid} not found. Skipping.")
            continue

        sdir = subjects_map[sid]
        sdata = load_ubfc_subject(sdir, subject_id=sid)
        vid_path = sdata.video_path
        orig_w = sdata.video_metadata["width"]
        orig_h = sdata.video_metadata["height"]
        print(f"\nProcessing {sid} (Original Resolution: {orig_w}x{orig_h})...")

        # Read all video frames into memory
        cap = cv2.VideoCapture(str(vid_path))
        frames_orig = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_orig.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        n_frames = len(frames_orig)
        print(f"  Loaded {n_frames} frames.")

        # Ground truth references on uniform grid
        t_gt_rel = sdata.time - sdata.time[0]
        n_common = int(np.floor(t_gt_rel[-1] * target_fs))
        t_common = np.arange(n_common) / target_fs

        _, ppg_common = resample_to_uniform(t_gt_rel, sdata.ppg, target_fs=target_fs)
        ppg_common = ppg_common[:n_common]

        _, hr_gt_common = resample_to_uniform(t_gt_rel, sdata.hr, target_fs=target_fs, method="linear")
        hr_gt_common = hr_gt_common[:n_common]

        primary_ref_hr = estimate_hr_video(ppg_common, fs=target_fs, lowcut=lowcut, highcut=highcut, order=order)
        secondary_ref_hr = float(np.mean(hr_gt_common))

        _, gt_win_hrs = estimate_hr_windowed(
            ppg_common, fs=target_fs, window_sec=win_sec, stride_sec=stride_sec,
            lowcut=lowcut, highcut=highcut, order=order
        )

        conditions = {}

        if sid == "subject4":
            # Subject 4 is natively a tight face crop (242x242)
            frames_padded = [
                cv2.copyMakeBorder(f, pad_px, pad_px, pad_px, pad_px, cv2.BORDER_REFLECT)
                for f in frames_orig
            ]
            conditions["unpadded_tight_crop"] = {
                "frames": frames_orig,
                "res": f"{orig_w}x{orig_h}",
                "desc": "Native tight crop (face touches boundaries)",
            }
            conditions["padded_tight_crop"] = {
                "frames": frames_padded,
                "res": f"{orig_w + 2*pad_px}x{orig_h + 2*pad_px}",
                "desc": f"Native tight crop + {pad_px}px reflect border",
            }
        else:
            # 640x480 subjects (subject1, subject3): create tight crop and padded version
            x, y, w, h = detect_face_bbox(frames_orig)
            print(f"  Detected face bounding box: x={x}, y={y}, w={w}, h={h}")

            frames_tight = [f[y : y + h, x : x + w] for f in frames_orig]
            frames_padded = [
                cv2.copyMakeBorder(f, pad_px, pad_px, pad_px, pad_px, cv2.BORDER_REFLECT)
                for f in frames_tight
            ]

            conditions["full_frame_original"] = {
                "frames": frames_orig,
                "res": f"{orig_w}x{orig_h}",
                "desc": "Original full frame (natural background)",
            }
            conditions["unpadded_tight_crop"] = {
                "frames": frames_tight,
                "res": f"{w}x{h}",
                "desc": "Tight face crop (face touches boundaries, 0px margin)",
            }
            conditions["padded_tight_crop"] = {
                "frames": frames_padded,
                "res": f"{w + 2*pad_px}x{h + 2*pad_px}",
                "desc": f"Tight face crop + {pad_px}px reflect border",
            }

        for cond_name, cdata in conditions.items():
            print(f"  Evaluating Condition: {cond_name} ({cdata['res']})...")
            bvp_u, t_u, meta = wrapper.predict_from_frames(
                frames=cdata["frames"],
                timestamps=sdata.time,
                timing_mode="gt_timestamp",
                fs_target=target_fs,
            )

            bvp_raw = meta["raw_bvp"]
            ts_raw = meta["raw_timestamps"]
            ts_rel = (ts_raw - ts_raw[0]) if len(ts_raw) > 0 else np.arange(len(bvp_raw)) / target_fs

            _, bvp_common = resample_to_uniform(ts_rel, bvp_raw, target_fs=target_fs)
            bvp_common = bvp_common[:n_common]

            bvp_filt = butterworth_bandpass_filter(
                bvp_common, fs=target_fs, lowcut=lowcut, highcut=highcut, order=order
            )
            video_hr, f_bvp, psd_bvp = estimate_hr_welch(
                bvp_filt, fs=target_fs, low_bpm=lowcut * 60.0, high_bpm=highcut * 60.0, nfft=16384
            )
            abs_err_primary = abs(video_hr - primary_ref_hr)
            abs_err_secondary = abs(video_hr - secondary_ref_hr)

            direct_r = calc_pearson_r(bvp_common, ppg_common)
            lag_dict = max_lag_waveform_correlation(bvp_common, ppg_common, fs=target_fs, max_lag_sec=max_lag_sec)
            max_lag_r = lag_dict["max_r"]
            best_lag_s = lag_dict["optimal_lag_sec"]

            _, win_hrs = estimate_hr_windowed(
                bvp_common, fs=target_fs, window_sec=win_sec, stride_sec=stride_sec,
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

            results_rows.append({
                "git_hash": git_hash,
                "subject_id": sid,
                "condition": cond_name,
                "resolution": cdata["res"],
                "description": cdata["desc"],
                "primary_ref_hr_bpm": primary_ref_hr,
                "video_hr_bpm": video_hr,
                "error_vs_primary_bpm": abs_err_primary,
                "waveform_r_direct": direct_r,
                "waveform_r_max_lag": max_lag_r,
                "waveform_best_lag_sec": best_lag_s,
                "windowed_mae_bpm": w_mae,
                "windowed_medae_bpm": w_medae,
                "windowed_rmse_bpm": w_rmse,
                "windowed_pearson_r": w_pearson,
                "windowed_acc5_pct": w_acc5,
                "snr_db": snr_val,
                "stat_null": meta["face_null_frames"],
                "stat_filled": meta["filled_frames"],
            })

            print(f"    --> [{cond_name}] Video HR: {video_hr:.2f} BPM (Err: {abs_err_primary:.2f} BPM), "
                  f"r_maxlag: {max_lag_r:.4f}, MedAE: {w_medae:.2f} BPM, Acc5: {w_acc5:.1f}%, SNR: {snr_val:.2f} dB")

    df_results = pd.DataFrame(results_rows)
    results_csv = PROJECT_ROOT / cfg.get("outputs", {}).get("results_csv", "results/padding_controlled_test.csv")
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(results_csv, index=False)
    print(f"\n--> Saved controlled padding test results to: {results_csv}")

    print("\n" + "=" * 80)
    print("=== SUMMARY OF CONTROLLED PADDING TEST ===")
    print("=" * 80)
    summary_cols = [
        "subject_id", "condition", "resolution", "video_hr_bpm",
        "primary_ref_hr_bpm", "error_vs_primary_bpm", "waveform_r_max_lag",
        "windowed_medae_bpm", "windowed_acc5_pct", "snr_db"
    ]
    print(df_results[summary_cols].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()

