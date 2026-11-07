"""Extended lag search (+/- 2.0 s) for the 5 clean subjects.

Evaluates waveform Pearson correlation r across lags tau in [-2.0 s, +2.0 s] in 1/30s steps.
Identifies best lag within +/-0.5s and +/-2.0s.
Investigates subject9's optimal lag behavior (cardiac cycle periodicity vs time offset).
Saves results to results/clean_subjects_lag_search_2s.csv and saves missing prediction CSVs.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io import find_ubfc_subjects, load_ubfc_subject, resample_to_uniform
from src.hr import butterworth_bandpass_filter
from src.models import FacePhysWrapper


def compute_cross_correlation(sig1: np.ndarray, sig2: np.ndarray, fs: float = 30.0, max_lag_sec: float = 2.0):
    """Compute Pearson correlation r as a function of lag tau in [-max_lag_sec, +max_lag_sec]."""
    s1 = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-8)
    s2 = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-8)

    max_shift = int(round(max_lag_sec * fs))
    shifts = np.arange(-max_shift, max_shift + 1)
    lags_sec = shifts / fs
    r_values = np.zeros(len(shifts))

    for idx, shift in enumerate(shifts):
        if shift < 0:
            c1 = s1[-shift:]
            c2 = s2[:shift]
        elif shift > 0:
            c1 = s1[:-shift]
            c2 = s2[shift:]
        else:
            c1 = s1
            c2 = s2
        
        if len(c1) > 10:
            r = np.corrcoef(c1, c2)[0, 1]
            r_values[idx] = r if not np.isnan(r) else 0.0

    return lags_sec, r_values


def read_all_frames(video_path: Path):
    """Read all video frames into memory as RGB numpy arrays."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def get_or_compute_prediction(sid: str, sdata, wrapper: FacePhysWrapper, target_fs: float = 30.0):
    pred_dir = PROJECT_ROOT / "results" / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = pred_dir / f"{sid}_facephys_pred.csv"

    if pred_csv.exists():
        df_pred = pd.read_csv(pred_csv)
        bvp_col = [c for c in df_pred.columns if "bvp" in c.lower() or "pred" in c.lower()][0]
        ts_col = [c for c in df_pred.columns if "time" in c.lower() or "ts" in c.lower()][0]
        return df_pred[bvp_col].values, df_pred[ts_col].values

    print(f"Prediction CSV not found for {sid}, running FacePhys inference...")
    frames = read_all_frames(sdata.video_path)
    bvp_u, t_u, meta = wrapper.predict_from_frames(
        frames=frames,
        timestamps=sdata.time,
        fs_target=target_fs,
        timing_mode="gt_timestamp",
        synchronous=True,
    )
    raw_bvp = meta["raw_bvp"]
    raw_ts = meta.get("raw_timestamps", sdata.time[:len(raw_bvp)])
    
    # Save prediction CSV
    df_pred = pd.DataFrame({
        "frame_idx": np.arange(len(raw_bvp)),
        "timestamp_sec": raw_ts[:len(raw_bvp)],
        "facephys_bvp_raw": raw_bvp,
    })
    df_pred.to_csv(pred_csv, index=False)
    print(f"Saved predictions to {pred_csv}")
    return raw_bvp, raw_ts[:len(raw_bvp)]


def main():
    print("=" * 90)
    print("=== EXTENDED LAG SEARCH (+/- 2.0s) ON 5 CLEAN SUBJECTS ===")
    print("=" * 90)

    dataset_root = PROJECT_ROOT / "data" / "ubfc" / "Data Set 2"
    subjects_map = find_ubfc_subjects(dataset_root)

    clean_subjects = ["subject1", "subject5", "subject8", "subject9", "subject10"]
    wrapper = FacePhysWrapper()
    rows = []
    subject9_details = {}

    for sid in clean_subjects:
        sdir = subjects_map[sid]
        sdata = load_ubfc_subject(sdir, subject_id=sid)

        pred_bvp, pred_ts = get_or_compute_prediction(sid, sdata, wrapper)

        gt_ppg = sdata.ppg
        gt_ts = sdata.time

        fs = 30.0
        t_u_pred, bvp_u = resample_to_uniform(pred_ts, pred_bvp, target_fs=fs)
        t_u_gt, ppg_u = resample_to_uniform(gt_ts, gt_ppg, target_fs=fs)

        min_len = min(len(bvp_u), len(ppg_u))
        bvp_filt = butterworth_bandpass_filter(bvp_u[:min_len], fs=fs, lowcut=0.7, highcut=3.0, order=3)
        ppg_filt = butterworth_bandpass_filter(ppg_u[:min_len], fs=fs, lowcut=0.7, highcut=3.0, order=3)

        lags, r_vals = compute_cross_correlation(bvp_filt, ppg_filt, fs=fs, max_lag_sec=2.0)

        # Direct (tau = 0)
        idx_0 = np.argmin(np.abs(lags))
        r_at_0 = r_vals[idx_0]

        # Best in +/-0.5s
        mask_05 = np.abs(lags) <= 0.501
        best_idx_05 = np.argmax(r_vals[mask_05])
        best_lag_05 = lags[mask_05][best_idx_05]
        best_r_05 = r_vals[mask_05][best_idx_05]

        # Best in +/-2.0s
        best_idx_20 = np.argmax(r_vals)
        best_lag_20 = lags[best_idx_20]
        best_r_20 = r_vals[best_idx_20]

        hr_bpm = np.median(sdata.hr)
        cardiac_period_sec = 60.0 / hr_bpm if hr_bpm > 0 else 0.0

        rows.append({
            "subject_id": sid,
            "hr_median_bpm": round(hr_bpm, 2),
            "cardiac_period_sec": round(cardiac_period_sec, 3),
            "r_direct_tau0": round(r_at_0, 4),
            "best_lag_05s": round(best_lag_05, 3),
            "best_r_05s": round(best_r_05, 4),
            "best_lag_20s": round(best_lag_20, 3),
            "best_r_20s": round(best_r_20, 4),
        })

        if sid == "subject9":
            subject9_details = {
                "lags": lags,
                "r_vals": r_vals,
                "hr_bpm": hr_bpm,
                "T": cardiac_period_sec,
            }

    df_out = pd.DataFrame(rows)
    out_csv = PROJECT_ROOT / "results" / "clean_subjects_lag_search_2s.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved extended lag search results to: {out_csv}")
    print("\n" + "=" * 90)
    print("=== SUMMARY: LAG SEARCH ACROSS 5 CLEAN SUBJECTS ===")
    print("=" * 90)
    print(df_out.to_string(index=False))
    print("=" * 90)

    # Detailed Subject 9 Investigation
    if subject9_details:
        lags = subject9_details["lags"]
        r_vals = subject9_details["r_vals"]
        T = subject9_details["T"]
        hr = subject9_details["hr_bpm"]

        print("\n" + "=" * 90)
        print("=== INVESTIGATION OF SUBJECT 9 OPTIMAL LAG (+0.5s) ===")
        print("=" * 90)
        print(f"Median Heart Rate: {hr:.2f} BPM")
        print(f"Cardiac Fundamental Period T = 60 / {hr:.2f} = {T:.4f} s ({T * 30:.1f} frames at 30 fps)")
        print("\nCross-correlation r(tau) at specific key delays:")
        key_delays = [-1.5, -1.0, -T, -0.5, -0.25, 0.0, 0.25, 0.5, T, 1.0, 1.5, 2.0]
        for d in key_delays:
            idx = np.argmin(np.abs(lags - d))
            cycles = lags[idx] / T
            print(f"  lag tau = {lags[idx]:+6.3f} s ({cycles:+5.2f} cardiac cycles) => r = {r_vals[idx]:+.4f}")

        # Find all local maxima in r(tau) across [-2.0s, +2.0s]
        from scipy.signal import find_peaks
        peaks, props = find_peaks(r_vals, height=0.3, distance=int(round(0.7 * T * 30)))
        print("\nLocal maxima (periodic cardiac peaks) in cross-correlation curve:")
        for p in peaks:
            tau = lags[p]
            print(f"  Peak at lag tau = {tau:+6.3f} s (distance in cycles: {tau / T:+.2f}) => r = {r_vals[p]:+.4f}")


if __name__ == "__main__":
    main()
