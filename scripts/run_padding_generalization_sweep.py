"""Task 3: Padding Generalization Sweep across 6 Subjects.

Evaluates whether border padding generalizes across subjects 1, 3, 5, 8, 9, 10
by creating artificial tight face crops (face touching image boundaries, 0px margin)
and testing padding fractions of crop width: [0%, 15%, 30%, 45%, 60%] with cv2.BORDER_REFLECT.

Evaluation Rule:
  Only propose a general rule in terms of face-to-frame ratio if the trend replicates
  on at least 4 of 6 subjects; otherwise keep it strictly as an observation for subject4 only.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import (
    calc_fraction_within_threshold,
    calc_mae,
    calc_medae,
    calc_pearson_r,
    calc_rmse,
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
    """Detect frontal face bounding box (x, y, w, h) tightly using open-rppg detector."""
    import rppg
    m = rppg.Model()
    for frame in frames[:30]:
        res = m.detector.detect(frame)
        if len(res) > 0:
            box = res[0][0]
            x1, y1, x2, y2 = int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3]))
            w = x2 - x1
            h = y2 - y1
            if w > 50 and h > 50:
                # Clamp to frame bounds
                ih, iw, _ = frame.shape
                x1 = max(0, min(iw - 1, x1))
                y1 = max(0, min(ih - 1, y1))
                w = min(iw - x1, w)
                h = min(ih - y1, h)
                return x1, y1, w, h

    # Fallback to central 60% crop if detector returns empty
    ih, iw, _ = frames[0].shape
    crop_w = int(iw * 0.6)
    crop_h = int(ih * 0.6)
    x = (iw - crop_w) // 2
    y = (ih - crop_h) // 2
    return x, y, crop_w, crop_h


def main():
    git_hash = get_git_hash()
    print("=" * 90)
    print("=== TASK 3: PADDING GENERALIZATION SWEEP (0%, 15%, 30%, 45%, 60%) ===")
    print(f"Git commit hash: {git_hash}")
    print("Pre-registered Decision Criterion:")
    print("  Only propose a rule in terms of face-to-frame ratio if trend replicates on >= 4 of 6 subjects.")
    print("  Otherwise, maintain strictly as an observation for subject4 only.")
    print("=" * 90)

    config_path = PROJECT_ROOT / "configs" / "padding_generalization_sweep.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    target_fs = float(cfg.get("preprocessing", {}).get("target_fs", 30.0))
    lowcut = float(cfg.get("preprocessing", {}).get("bandpass", {}).get("lowcut", 0.7))
    highcut = float(cfg.get("preprocessing", {}).get("bandpass", {}).get("highcut", 3.0))
    order = int(cfg.get("preprocessing", {}).get("bandpass", {}).get("order", 3))
    win_sec = float(cfg.get("preprocessing", {}).get("windowing", {}).get("window_sec", 30.0))
    stride_sec = float(cfg.get("preprocessing", {}).get("windowing", {}).get("stride_sec", 1.0))
    threshold_bpm = float(cfg.get("metrics", {}).get("threshold_bpm", 5.0))
    max_lag_sec = float(cfg.get("metrics", {}).get("lag_search_max_sec", 0.5))
    snr_delta_hz = float(cfg.get("metrics", {}).get("snr_delta_hz", 0.1))

    fractions = cfg.get("padding_sweep", {}).get("fractions_of_crop_width", [0.0, 0.15, 0.30, 0.45, 0.60])
    target_subjects = cfg.get("dataset", {}).get("subjects", ["subject1", "subject3", "subject5", "subject8", "subject9", "subject10"])

    dataset_root = PROJECT_ROOT / cfg.get("dataset", {}).get("root_dir", "data/ubfc/Data Set 2")
    subjects_map = find_ubfc_subjects(dataset_root)

    wrapper = FacePhysWrapper()
    results_rows = []

    for sid in target_subjects:
        if sid not in subjects_map:
            print(f"Warning: Subject {sid} not found. Skipping.")
            continue

        sdir = subjects_map[sid]
        sdata = load_ubfc_subject(sdir, subject_id=sid)
        vid_path = sdata.video_path
        orig_w = sdata.video_metadata["width"]
        orig_h = sdata.video_metadata["height"]
        vid_size_bytes = os.path.getsize(str(vid_path))
        vid_size_mb = round(vid_size_bytes / (1024.0 * 1024.0), 2)

        print(f"\nProcessing {sid} (Native Resolution: {orig_w}x{orig_h})...")

        # Load video frames
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

        # Detect tight face bounding box (face touching boundaries, 0px margin)
        fx, fy, fw, fh = detect_face_bbox(frames_orig)
        print(f"  Tight face crop detected: x={fx}, y={fy}, w={fw}, h={fh}")

        frames_tight = [f[fy : fy + fh, fx : fx + fw] for f in frames_orig]

        for frac in fractions:
            pad_px = int(round(frac * fw))
            res_w = fw + 2 * pad_px
            res_h = fh + 2 * pad_px
            face_ratio = fw / res_w
            cond_label = f"crop_pad_{int(round(frac*100)):02d}pct"
            desc = (
                f"Tight face crop (0px margin, face touches boundary, ratio={face_ratio:.2f})"
                if pad_px == 0
                else f"Tight crop + {pad_px}px ({int(round(frac*100))}%) reflect border (ratio={face_ratio:.2f})"
            )

            print(f"  [Evaluating {cond_label}] pad={pad_px}px -> {res_w}x{res_h} (face ratio: {face_ratio:.2f})...")

            if pad_px == 0:
                frames_eval = frames_tight
            else:
                frames_eval = [
                    cv2.copyMakeBorder(f, pad_px, pad_px, pad_px, pad_px, cv2.BORDER_REFLECT)
                    for f in frames_tight
                ]

            bvp_u, t_u, meta = wrapper.predict_from_frames(
                frames=frames_eval,
                timestamps=sdata.time,
                timing_mode="gt_timestamp",
                synchronous=True,
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

            direct_r = calc_pearson_r(bvp_common, ppg_common)
            lag_dict = max_lag_waveform_correlation(
                bvp_common, ppg_common, fs=target_fs, max_lag_sec=max_lag_sec
            )
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

            row = {
                "git_hash": git_hash,
                "subject_id": sid,
                "crop_w": fw,
                "crop_h": fh,
                "pad_fraction": frac,
                "pad_px": pad_px,
                "effective_resolution": f"{res_w}x{res_h}",
                "face_to_frame_ratio": face_ratio,
                "condition": cond_label,
                "description": desc,
                "primary_ref_hr_bpm": primary_ref_hr,
                "video_hr_bpm": video_hr,
                "error_vs_primary_bpm": abs_err_primary,
                "waveform_r_direct": direct_r,
                "waveform_r_max_lag_05s": max_lag_r,
                "waveform_best_lag_sec": best_lag_s,
                "windowed_mae_bpm": w_mae,
                "windowed_medae_bpm": w_medae,
                "windowed_rmse_bpm": w_rmse,
                "windowed_acc5_pct": w_acc5,
                "snr_db": snr_val,
                "stat_null": meta["face_null_frames"],
                "stat_filled": meta["filled_frames"],
                "stat_skipped": meta.get("skipped_frames", 0),
            }
            results_rows.append(row)

            print(
                f"    --> [{cond_label}] HR={video_hr:.2f} (Err={abs_err_primary:.2f} BPM), "
                f"r_maxlag={max_lag_r:.4f}, SNR={snr_val:.2f} dB, Acc5={w_acc5:.1f}%"
            )

    df = pd.DataFrame(results_rows)
    out_csv = PROJECT_ROOT / cfg.get("outputs", {}).get("results_csv", "results/padding_generalization_sweep.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n--> Successfully saved padding generalization results to: {out_csv}")

    # Generalization Analysis: Check if padding improves metric across subjects
    print("\n" + "=" * 90)
    print("=== HYPOTHESIS TESTING: DOES BORDER PADDING GENERALIZE? ===")
    print("=" * 90)

    improved_count = 0
    decision_summary = []

    for sid in target_subjects:
        sub_df = df[df["subject_id"] == sid].sort_values("pad_fraction")
        row_0 = sub_df[sub_df["pad_fraction"] == 0.0].iloc[0]
        # Best padded configuration (from fractions > 0)
        padded_df = sub_df[sub_df["pad_fraction"] > 0.0]
        row_best_padded = padded_df.sort_values("error_vs_primary_bpm").iloc[0]

        err_0 = row_0["error_vs_primary_bpm"]
        err_best = row_best_padded["error_vs_primary_bpm"]
        r_0 = row_0["waveform_r_max_lag_05s"]
        r_best = row_best_padded["waveform_r_max_lag_05s"]
        snr_0 = row_0["snr_db"]
        snr_best = row_best_padded["snr_db"]

        # Improvement criterion: error decreased by >= 2.0 BPM OR waveform r improved by >= 0.15
        improved = (err_best < err_0 - 2.0) or (r_best > r_0 + 0.15) or (snr_best > snr_0 + 2.0)
        if improved:
            improved_count += 1

        decision_summary.append({
            "subject_id": sid,
            "crop_box": f"{row_0['crop_w']}x{row_0['crop_h']}",
            "err_0pct": err_0,
            "err_best_padded": err_best,
            "r_0pct": r_0,
            "r_best_padded": r_best,
            "snr_0pct": snr_0,
            "snr_best_padded": snr_best,
            "best_fraction": row_best_padded["pad_fraction"],
            "padding_helps": improved,
        })

    df_dec = pd.DataFrame(decision_summary)
    print(df_dec.to_string(index=False))
    print("-" * 90)
    print(f"Subjects where padding meaningfully helped: {improved_count} / {len(target_subjects)}")
    passed_rule = improved_count >= 4
    if passed_rule:
        conclusion = "REPLICATED (>=4/6 subjects): Propose general rule in terms of face-to-frame ratio."
    else:
        conclusion = "NOT REPLICATED (<4/6 subjects): Maintain strictly as an observation for subject4 only."
    print("Decision:", conclusion)
    print("=" * 90)


if __name__ == "__main__":
    main()

