"""Determinism Test: Run Baseline Pipeline 3 Times Each in Fresh Subprocesses.

Evaluates:
- Subjects: subject1, subject3, subject4 (padded +60px reflect border), subject5
- 3 independent runs per subject, each in a clean fresh Python subprocess
- Baseline: GT-timestamp timing (Path 3) + synchronous frame feeding
- Metrics per run: Video HR (BPM), Waveform direct r, Waveform max-lag r (limited to +/-0.5s),
  SNR (dB), Filled frames, Skipped frames
- Spread (max - min) across the 3 runs for each metric
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import calculate_snr, calc_pearson_r, max_lag_waveform_correlation
from src.hr import butterworth_bandpass_filter, estimate_hr_welch, estimate_hr_video
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


def run_single_subject(subject_id: str, pad_px: int = 0) -> Dict[str, float]:
    """Execute a single run of the baseline pipeline on subject_id."""
    dataset_root = PROJECT_ROOT / "data" / "ubfc" / "Data Set 2"
    subjects_map = find_ubfc_subjects(dataset_root)
    if subject_id not in subjects_map:
        raise FileNotFoundError(f"Subject {subject_id} not found in {dataset_root}")

    sdir = subjects_map[subject_id]
    sdata = load_ubfc_subject(sdir, subject_id=subject_id)
    vid_path = sdata.video_path

    cap = cv2.VideoCapture(str(vid_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if pad_px > 0:
        frames = [
            cv2.copyMakeBorder(f, pad_px, pad_px, pad_px, pad_px, cv2.BORDER_REFLECT)
            for f in frames
        ]

    target_fs = 30.0
    lowcut = 0.7
    highcut = 3.0
    order = 3

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

    t_gt_rel = sdata.time - sdata.time[0]
    n_common = int(np.floor(min(t_gt_rel[-1], ts_rel[-1]) * target_fs))

    _, ppg_common = resample_to_uniform(t_gt_rel, sdata.ppg, target_fs=target_fs)
    ppg_common = ppg_common[:n_common]

    _, bvp_common = resample_to_uniform(ts_rel, bvp_raw, target_fs=target_fs)
    bvp_common = bvp_common[:n_common]

    primary_ref_hr = estimate_hr_video(ppg_common, fs=target_fs, lowcut=lowcut, highcut=highcut, order=order)

    bvp_filt = butterworth_bandpass_filter(
        bvp_common, fs=target_fs, lowcut=lowcut, highcut=highcut, order=order
    )
    video_hr, f_bvp, psd_bvp = estimate_hr_welch(
        bvp_filt, fs=target_fs, low_bpm=lowcut * 60.0, high_bpm=highcut * 60.0, nfft=16384
    )

    direct_r = calc_pearson_r(bvp_common, ppg_common)
    # Limited lag search to +/- 0.5s per protocol
    lag_dict = max_lag_waveform_correlation(bvp_common, ppg_common, fs=target_fs, max_lag_sec=0.5)
    max_lag_r = lag_dict["max_r"]
    best_lag_s = lag_dict["optimal_lag_sec"]

    snr_val = calculate_snr(
        psd_bvp, f_bvp, ref_hr_bpm=primary_ref_hr, delta_hz=0.1, band=(lowcut, highcut)
    )

    return {
        "subject_id": subject_id,
        "padded": (pad_px > 0),
        "video_hr_bpm": float(video_hr),
        "ref_hr_bpm": float(primary_ref_hr),
        "error_bpm": float(abs(video_hr - primary_ref_hr)),
        "waveform_r_direct": float(direct_r),
        "waveform_r_max_lag": float(max_lag_r),
        "waveform_best_lag_sec": float(best_lag_s),
        "snr_db": float(snr_val),
        "stat_filled": int(meta["filled_frames"]),
        "stat_skipped": int(meta["skipped_frames"]),
        "stat_null": int(meta["face_null_frames"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Determinism Test Runner")
    parser.add_argument("--worker-subject", type=str, default=None, help="Worker mode: run single subject")
    parser.add_argument("--worker-pad", type=int, default=0, help="Worker mode: padding px")
    args = parser.parse_args()

    # Worker subprocess mode
    if args.worker_subject:
        res = run_single_subject(args.worker_subject, pad_px=args.worker_pad)
        print(f"WORKER_JSON_OUTPUT:{json.dumps(res)}")
        return

    # Master orchestrator mode: launches fresh subprocesses
    git_hash = get_git_hash()
    print("=" * 80)
    print("=== DETERMINISM TEST: 3 Independent Runs per Subject in Fresh Processes ===")
    print(f"Git commit hash: {git_hash}")
    print("Baseline: Path 3 (GT-timestamp timing) with Synchronous Worker Execution")
    print("Lag search range: +/- 0.5 s")
    print("=" * 80)

    test_targets = [
        {"sid": "subject1", "pad": 0, "desc": "subject1 (640x480, full frame)"},
        {"sid": "subject3", "pad": 0, "desc": "subject3 (640x480, full frame)"},
        {"sid": "subject4", "pad": 60, "desc": "subject4 (padded +60px border -> 362x362)"},
        {"sid": "subject5", "pad": 0, "desc": "subject5 (640x480, full frame)"},
    ]

    all_rows = []
    spread_rows = []

    for target in test_targets:
        sid = target["sid"]
        pad = target["pad"]
        desc = target["desc"]
        print(f"\n--> Running Determinism Test on {desc} (3 independent runs)...")

        runs_data = []
        for run_idx in range(1, 4):
            print(f"  [Run {run_idx}/3] Launching fresh Python process for {sid}...")
            cmd = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--worker-subject",
                sid,
                "--worker-pad",
                str(pad),
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                print(f"Error in worker process for {sid} run {run_idx}:")
                print(proc.stderr)
                sys.exit(1)

            # Extract JSON output
            json_line = None
            for line in proc.stdout.split("\n"):
                if line.startswith("WORKER_JSON_OUTPUT:"):
                    json_line = line.replace("WORKER_JSON_OUTPUT:", "").strip()
                    break

            if not json_line:
                print(f"Error: No JSON output found from worker for {sid} run {run_idx}")
                print(proc.stdout)
                sys.exit(1)

            res = json.loads(json_line)
            res["run_index"] = run_idx
            res["git_hash"] = git_hash
            runs_data.append(res)
            all_rows.append(res)

            print(f"    Run {run_idx}: Video HR = {res['video_hr_bpm']:.4f} BPM, "
                  f"r_maxlag = {res['waveform_r_max_lag']:.4f}, SNR = {res['snr_db']:.4f} dB, "
                  f"Filled = {res['stat_filled']}, Skipped = {res['stat_skipped']}")

        # Compute spread across the 3 runs
        hr_vals = [r["video_hr_bpm"] for r in runs_data]
        r_vals = [r["waveform_r_max_lag"] for r in runs_data]
        snr_vals = [r["snr_db"] for r in runs_data]
        filled_vals = [r["stat_filled"] for r in runs_data]
        skipped_vals = [r["stat_skipped"] for r in runs_data]

        hr_spread = float(np.max(hr_vals) - np.min(hr_vals))
        r_spread = float(np.max(r_vals) - np.min(r_vals))
        snr_spread = float(np.max(snr_vals) - np.min(snr_vals))
        filled_spread = int(np.max(filled_vals) - np.min(filled_vals))
        skipped_spread = int(np.max(skipped_vals) - np.min(skipped_vals))

        is_deterministic = (
            hr_spread < 1e-5
            and r_spread < 1e-5
            and snr_spread < 1e-5
            and filled_spread == 0
            and skipped_spread == 0
        )

        spread_rows.append({
            "git_hash": git_hash,
            "subject_id": sid,
            "description": desc,
            "run1_hr": hr_vals[0],
            "run2_hr": hr_vals[1],
            "run3_hr": hr_vals[2],
            "hr_spread_bpm": hr_spread,
            "r_maxlag_spread": r_spread,
            "snr_spread_db": snr_spread,
            "filled_spread": filled_spread,
            "skipped_spread": skipped_spread,
            "is_identical": is_deterministic,
        })

        verdict_str = "IDENTICAL (Spread = 0.000) [PASS]" if is_deterministic else f"DIFFERENT (HR Spread = {hr_spread:.4f} BPM) [FAIL]"
        print(f"  ==> Result for {sid}: {verdict_str}")

    df_all = pd.DataFrame(all_rows)
    out_all_csv = PROJECT_ROOT / "results" / "determinism_runs.csv"
    out_all_csv.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_all_csv, index=False)

    df_spread = pd.DataFrame(spread_rows)
    out_spread_csv = PROJECT_ROOT / "results" / "determinism_spread.csv"
    df_spread.to_csv(out_spread_csv, index=False)
    print(f"\n--> Successfully saved determinism results to: {out_spread_csv}")

    print("\n" + "=" * 80)
    print("=== DETERMINISM SPREAD SUMMARY ===")
    print("=" * 80)
    summary_cols = [
        "subject_id", "run1_hr", "run2_hr", "run3_hr",
        "hr_spread_bpm", "r_maxlag_spread", "snr_spread_db", "filled_spread", "is_identical"
    ]
    print(df_spread[summary_cols].to_string(index=False))
    print("=" * 80)

    overall_pass = all(df_spread["is_identical"])
    print(f"Overall Determinism Verdict: {'ALL RUNS 100% IDENTICAL [PASS]' if overall_pass else 'NON-DETERMINISM DETECTED [FAIL]'}")
    print("=" * 80)


if __name__ == "__main__":
    main()

