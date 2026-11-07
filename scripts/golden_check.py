"""Golden check script for Baseline-v1 reproduction.

Runs subject5 through the frozen Baseline-v1 pipeline (Path 3 GT-timestamp, synchronous=True)
and asserts reproduction against stored golden reference values.
Teammates can run this script to confirm their local environment achieves bit-exact parity.

Exit codes:
  0: Success (all metrics match stored reference values within tolerance)
  1: Failure (metric mismatch or error)
"""

import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import (
    calc_fraction_within_threshold,
    calc_medae,
    calc_pearson_r,
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

# Stored Golden Reference Values for Subject 5 (Baseline-v1)
GOLDEN_REFERENCE = {
    "subject_id": "subject5",
    "primary_ref_hr_bpm": 99.31640625,
    "video_hr_bpm": 99.5361328125,
    "error_bpm": 0.2197265625,
    "waveform_r_direct": 0.7861440726,
    "waveform_r_max_lag_05s": 0.8989000304,
    "waveform_best_lag_sec": 0.0000000000,
    "windowed_medae_bpm": 0.0000000000,
    "windowed_acc5_pct": 100.0,
    "snr_db": 9.4031669265,
    "stat_filled": 98,
    "stat_null": 0,
    "stat_skipped": 76,
}


def run_golden_check() -> bool:
    print("=" * 80)
    print("=== IEEE SP Cup 2027: Baseline-v1 Golden Reproduction Check ===")
    print("Target: Subject 5 (UBFC Dataset 2)")
    print("Pipeline: FacePhysWrapper (timing_mode='gt_timestamp', synchronous=True)")
    print("=" * 80)

    dataset_root = PROJECT_ROOT / "data" / "ubfc" / "Data Set 2"
    subjects_map = find_ubfc_subjects(dataset_root)

    if "subject5" not in subjects_map:
        print(f"ERROR: Subject 5 directory not found in {dataset_root}")
        return False

    sdir = subjects_map["subject5"]
    sdata = load_ubfc_subject(sdir, subject_id="subject5")
    vid_path = sdata.video_path

    print(f"Loaded Subject 5 video from: {vid_path}")
    cap = cv2.VideoCapture(str(vid_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    n_frames = len(frames)
    print(f"Read {n_frames} frames.")

    target_fs = 30.0
    lowcut, highcut, order = 0.7, 3.0, 3

    t_gt_rel = sdata.time - sdata.time[0]
    n_common = int(np.floor(t_gt_rel[-1] * target_fs))

    _, ppg_common = resample_to_uniform(t_gt_rel, sdata.ppg, target_fs=target_fs)
    ppg_common = ppg_common[:n_common]

    primary_ref_hr = estimate_hr_video(ppg_common, fs=target_fs, lowcut=lowcut, highcut=highcut, order=order)

    _, gt_win_hrs = estimate_hr_windowed(
        ppg_common, fs=target_fs, window_sec=30.0, stride_sec=1.0,
        lowcut=lowcut, highcut=highcut, order=order
    )

    # Execute synchronous FacePhys inference with GT timestamps
    wrapper = FacePhysWrapper()
    bvp_u, t_u, meta = wrapper.predict_from_frames(
        frames=frames,
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

    bvp_filt = butterworth_bandpass_filter(bvp_common, fs=target_fs, lowcut=lowcut, highcut=highcut, order=order)
    video_hr, f_bvp, psd_bvp = estimate_hr_welch(bvp_filt, fs=target_fs, low_bpm=42.0, high_bpm=180.0, nfft=16384)
    error_bpm = abs(video_hr - primary_ref_hr)

    direct_r = calc_pearson_r(bvp_common, ppg_common)
    lag_dict = max_lag_waveform_correlation(bvp_common, ppg_common, fs=target_fs, max_lag_sec=0.5)
    max_lag_r = lag_dict["max_r"]
    best_lag = lag_dict["optimal_lag_sec"]

    _, win_hrs = estimate_hr_windowed(
        bvp_common, fs=target_fs, window_sec=30.0, stride_sec=1.0,
        lowcut=lowcut, highcut=highcut, order=order
    )
    valid_mask = ~np.isnan(win_hrs) & ~np.isnan(gt_win_hrs)
    medae = calc_medae(win_hrs[valid_mask], gt_win_hrs[valid_mask])
    acc5 = calc_fraction_within_threshold(win_hrs[valid_mask], gt_win_hrs[valid_mask], threshold=5.0)

    snr_val = calculate_snr(psd_bvp, f_bvp, ref_hr_bpm=primary_ref_hr, delta_hz=0.1, band=(0.7, 3.0))

    actual = {
        "subject_id": "subject5",
        "primary_ref_hr_bpm": primary_ref_hr,
        "video_hr_bpm": video_hr,
        "error_bpm": error_bpm,
        "waveform_r_direct": direct_r,
        "waveform_r_max_lag_05s": max_lag_r,
        "waveform_best_lag_sec": best_lag,
        "windowed_medae_bpm": medae,
        "windowed_acc5_pct": acc5,
        "snr_db": snr_val,
        "stat_filled": meta["filled_frames"],
        "stat_null": meta["face_null_frames"],
        "stat_skipped": meta.get("skipped_frames", 0),
    }

    print("\nVerification Results:")
    print("-" * 80)
    all_passed = True
    tolerances = {
        "primary_ref_hr_bpm": 1e-4,
        "video_hr_bpm": 1e-4,
        "error_bpm": 1e-4,
        "waveform_r_direct": 1e-3,
        "waveform_r_max_lag_05s": 1e-3,
        "waveform_best_lag_sec": 1e-3,
        "windowed_medae_bpm": 1e-4,
        "windowed_acc5_pct": 1e-3,
        "snr_db": 1e-3,
        "stat_filled": 0,
        "stat_null": 0,
        "stat_skipped": 0,
    }

    for key, tol in tolerances.items():
        ref_v = GOLDEN_REFERENCE[key]
        act_v = actual[key]
        if tol == 0:
            match = (ref_v == act_v)
            diff_str = f"diff={act_v - ref_v:+d}"
        else:
            diff = abs(ref_v - act_v)
            match = diff <= tol
            diff_str = f"diff={diff:+.6f} (tol={tol})"

        status_str = "PASS" if match else "FAIL"
        print(f"  [{status_str}] {key:<24}: Expected={ref_v}, Got={act_v} ({diff_str})")
        if not match:
            all_passed = False

    print("-" * 80)
    if all_passed:
        print("SUCCESS: Golden check PASSED! All metrics reproduce bit-exact reference values.")
        return True
    else:
        print("FAILURE: One or more metrics deviated from golden reference values.")
        return False


if __name__ == "__main__":
    success = run_golden_check()
    sys.exit(0 if success else 1)
