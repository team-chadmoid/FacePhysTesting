"""Build Cohort v2 Report & Table.

Updates the benchmark reporting to:
1. Include verified true video frame counts (1547, 1801, 1368, 1550, 2022, 2016, 2024),
   resolving the receptive-field artifact that produced 1,515/1,977/1,971/1,979 in raw logs.
2. Report subject4 BOTH unpadded (native 242x242) and padded (362x362).
3. Compute official cohort statistics under both Variant A (with subject4 padded)
   and Variant B (with subject4 unpadded).
4. Save results/cohort_v2_7_subjects.csv, results/cohort_v2_summary.csv, and docs/cohort_report_v2.md.
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".cache" / "matplotlib")
(PROJECT_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import (
    calc_bland_altman,
    calc_mae,
    calc_medae,
    calc_pearson_r,
    calc_rmse,
)


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
    print("=== BUILDING COHORT V2 TABLE (DUAL SUBJECT 4 REPORTING) ===")
    print(f"Git commit hash: {git_hash}")
    print("=" * 80)

    # Source CSVs
    pad_csv = PROJECT_ROOT / "results" / "padding_controlled_test.csv"
    pilot_csv = PROJECT_ROOT / "results" / "pilot_4_subjects.csv"
    reconcile_csv = PROJECT_ROOT / "results" / "frame_count_reconciliation.csv"

    df_pad = pd.read_csv(pad_csv)
    df_pilot = pd.read_csv(pilot_csv)
    df_rec = pd.read_csv(reconcile_csv).set_index("subject_id")

    rows = []

    # Subject 1 (full frame original)
    r1 = df_pad[(df_pad["subject_id"] == "subject1") & (df_pad["condition"] == "full_frame_original")].iloc[0]
    s1_frames = int(df_rec.loc["subject1", "opencv_frames"])
    rows.append({
        "git_hash": r1["git_hash"],
        "subject_id": "subject1",
        "variant": "standard",
        "video_path": r1["video_path"],
        "video_size_mb": float(r1["video_size_mb"]),
        "verified_video_frames": s1_frames,
        "resolution": r1["resolution"],
        "condition_notes": "640x480 uncropped natural framing",
        "primary_ref_hr_bpm": float(r1["primary_ref_hr_bpm"]),
        "video_hr_bpm": float(r1["video_hr_bpm"]),
        "error_bpm": float(r1["error_vs_primary_bpm"]),
        "waveform_r_direct": float(r1["waveform_r_direct"]),
        "waveform_r_max_lag": float(r1["waveform_r_max_lag"]),
        "waveform_best_lag_sec": float(r1["waveform_best_lag_sec"]),
        "windowed_mae_bpm": float(r1["windowed_mae_bpm"]),
        "windowed_medae_bpm": float(r1["windowed_medae_bpm"]),
        "windowed_rmse_bpm": float(r1["windowed_rmse_bpm"]),
        "windowed_pearson_r": float(r1["windowed_pearson_r"]),
        "windowed_acc5_pct": float(r1["windowed_acc5_pct"]),
        "snr_db": float(r1["snr_db"]),
        "stat_filled": int(r1["stat_filled"]),
    })

    # Subject 3 (full frame original)
    r3 = df_pad[(df_pad["subject_id"] == "subject3") & (df_pad["condition"] == "full_frame_original")].iloc[0]
    s3_frames = int(df_rec.loc["subject3", "opencv_frames"])
    rows.append({
        "git_hash": r3["git_hash"],
        "subject_id": "subject3",
        "variant": "standard",
        "video_path": r3["video_path"],
        "video_size_mb": float(r3["video_size_mb"]),
        "verified_video_frames": s3_frames,
        "resolution": r3["resolution"],
        "condition_notes": "640x480 uncropped natural framing",
        "primary_ref_hr_bpm": float(r3["primary_ref_hr_bpm"]),
        "video_hr_bpm": float(r3["video_hr_bpm"]),
        "error_bpm": float(r3["error_vs_primary_bpm"]),
        "waveform_r_direct": float(r3["waveform_r_direct"]),
        "waveform_r_max_lag": float(r3["waveform_r_max_lag"]),
        "waveform_best_lag_sec": float(r3["waveform_best_lag_sec"]),
        "windowed_mae_bpm": float(r3["windowed_mae_bpm"]),
        "windowed_medae_bpm": float(r3["windowed_medae_bpm"]),
        "windowed_rmse_bpm": float(r3["windowed_rmse_bpm"]),
        "windowed_pearson_r": float(r3["windowed_pearson_r"]),
        "windowed_acc5_pct": float(r3["windowed_acc5_pct"]),
        "snr_db": float(r3["snr_db"]),
        "stat_filled": int(r3["stat_filled"]),
    })

    # Subject 4 (UNPADDED native 242x242)
    r4_unpadded = df_pad[(df_pad["subject_id"] == "subject4") & (df_pad["condition"] == "unpadded_tight_crop")].iloc[0]
    s4_frames = int(df_rec.loc["subject4", "opencv_frames"])
    rows.append({
        "git_hash": r4_unpadded["git_hash"],
        "subject_id": "subject4_unpadded",
        "variant": "unpadded",
        "video_path": r4_unpadded["video_path"],
        "video_size_mb": float(r4_unpadded["video_size_mb"]),
        "verified_video_frames": s4_frames,
        "resolution": "242x242 (0px padding)",
        "condition_notes": "Native tight face crop (face touches boundaries)",
        "primary_ref_hr_bpm": float(r4_unpadded["primary_ref_hr_bpm"]),
        "video_hr_bpm": float(r4_unpadded["video_hr_bpm"]),
        "error_bpm": float(r4_unpadded["error_vs_primary_bpm"]),
        "waveform_r_direct": float(r4_unpadded["waveform_r_direct"]),
        "waveform_r_max_lag": float(r4_unpadded["waveform_r_max_lag"]),
        "waveform_best_lag_sec": float(r4_unpadded["waveform_best_lag_sec"]),
        "windowed_mae_bpm": float(r4_unpadded["windowed_mae_bpm"]),
        "windowed_medae_bpm": float(r4_unpadded["windowed_medae_bpm"]),
        "windowed_rmse_bpm": float(r4_unpadded["windowed_rmse_bpm"]),
        "windowed_pearson_r": float(r4_unpadded["windowed_pearson_r"]),
        "windowed_acc5_pct": float(r4_unpadded["windowed_acc5_pct"]),
        "snr_db": float(r4_unpadded["snr_db"]),
        "stat_filled": int(r4_unpadded["stat_filled"]),
    })

    # Subject 4 (PADDED +60px reflect border -> 362x362)
    r4_padded = df_pad[(df_pad["subject_id"] == "subject4") & (df_pad["condition"] == "padded_tight_crop")].iloc[0]
    rows.append({
        "git_hash": r4_padded["git_hash"],
        "subject_id": "subject4_padded",
        "variant": "padded",
        "video_path": r4_padded["video_path"],
        "video_size_mb": float(r4_padded["video_size_mb"]),
        "verified_video_frames": s4_frames,
        "resolution": "242x242 (+60px reflect -> 362x362)",
        "condition_notes": "Native tight crop with +60px reflect border padding",
        "primary_ref_hr_bpm": float(r4_padded["primary_ref_hr_bpm"]),
        "video_hr_bpm": float(r4_padded["video_hr_bpm"]),
        "error_bpm": float(r4_padded["error_vs_primary_bpm"]),
        "waveform_r_direct": float(r4_padded["waveform_r_direct"]),
        "waveform_r_max_lag": float(r4_padded["waveform_r_max_lag"]),
        "waveform_best_lag_sec": float(r4_padded["waveform_best_lag_sec"]),
        "windowed_mae_bpm": float(r4_padded["windowed_mae_bpm"]),
        "windowed_medae_bpm": float(r4_padded["windowed_medae_bpm"]),
        "windowed_rmse_bpm": float(r4_padded["windowed_rmse_bpm"]),
        "windowed_pearson_r": float(r4_padded["windowed_pearson_r"]),
        "windowed_acc5_pct": float(r4_padded["windowed_acc5_pct"]),
        "snr_db": float(r4_padded["snr_db"]),
        "stat_filled": int(r4_padded["stat_filled"]),
    })

    # Pilot subjects: 5, 8, 9, 10
    for sid in ["subject5", "subject8", "subject9", "subject10"]:
        rp = df_pilot[df_pilot["subject_id"] == sid].iloc[0]
        sp_frames = int(df_rec.loc[sid, "opencv_frames"])
        rows.append({
            "git_hash": rp["git_hash"],
            "subject_id": sid,
            "variant": "standard",
            "video_path": rp["video_path"],
            "video_size_mb": float(rp["video_size_mb"]),
            "verified_video_frames": sp_frames,
            "resolution": rp["video_resolution"],
            "condition_notes": "640x480 uncropped natural framing",
            "primary_ref_hr_bpm": float(rp["primary_ref_hr_bpm"]),
            "video_hr_bpm": float(rp["video_hr_bpm"]),
            "error_bpm": float(rp["error_vs_primary_bpm"]),
            "waveform_r_direct": float(rp["waveform_r_direct"]),
            "waveform_r_max_lag": float(rp["waveform_r_max_lag_05s"]),
            "waveform_best_lag_sec": float(rp["waveform_best_lag_sec"]),
            "windowed_mae_bpm": float(rp["windowed_mae_bpm"]),
            "windowed_medae_bpm": float(rp["windowed_medae_bpm"]),
            "windowed_rmse_bpm": float(rp["windowed_rmse_bpm"]),
            "windowed_pearson_r": float(rp["windowed_pearson_r"]),
            "windowed_acc5_pct": float(rp["windowed_acc5_pct"]),
            "snr_db": float(rp["snr_db"]),
            "stat_filled": int(rp["stat_filled"]),
        })

    df_all = pd.DataFrame(rows)
    out_table_csv = PROJECT_ROOT / "results" / "cohort_v2_7_subjects.csv"
    df_all.to_csv(out_table_csv, index=False)
    print(f"--> Saved Cohort v2 table to: {out_table_csv}")

    # Compute Statistics for Variant A (subject4 padded)
    df_a = df_all[df_all["variant"].isin(["standard", "padded"])]
    preds_a = df_a["video_hr_bpm"].values
    refs_a = df_a["primary_ref_hr_bpm"].values
    mae_a = calc_mae(preds_a, refs_a)
    medae_a = calc_medae(preds_a, refs_a)
    rmse_a = calc_rmse(preds_a, refs_a)
    r_a = calc_pearson_r(preds_a, refs_a)
    ba_a = calc_bland_altman(refs_a, preds_a)
    pct_r7_a = float(np.mean(df_a["waveform_r_max_lag"].values >= 0.7) * 100.0)

    # Compute Statistics for Variant B (subject4 unpadded)
    df_b = df_all[df_all["variant"].isin(["standard", "unpadded"])]
    preds_b = df_b["video_hr_bpm"].values
    refs_b = df_b["primary_ref_hr_bpm"].values
    mae_b = calc_mae(preds_b, refs_b)
    medae_b = calc_medae(preds_b, refs_b)
    rmse_b = calc_rmse(preds_b, refs_b)
    r_b = calc_pearson_r(preds_b, refs_b)
    ba_b = calc_bland_altman(refs_b, preds_b)
    pct_r7_b = float(np.mean(df_b["waveform_r_max_lag"].values >= 0.7) * 100.0)

    summary_rows = [
        {
            "cohort_variant": "Variant A: subject4 padded (+60px reflect)",
            "n_subjects": len(df_a),
            "video_medae_bpm": medae_a,
            "video_mae_bpm": mae_a,
            "video_rmse_bpm": rmse_a,
            "video_pearson_r": r_a,
            "bland_altman_bias_bpm": ba_a["bias"],
            "bland_altman_lower_loa": ba_a["lower_loa"],
            "bland_altman_upper_loa": ba_a["upper_loa"],
            "pct_subjects_waveform_r_ge_07": pct_r7_a,
            "gate_medae_le_2bpm": medae_a <= 2.0,
            "gate_pct_r_ge_07": pct_r7_a >= 70.0,
        },
        {
            "cohort_variant": "Variant B: subject4 unpadded (native 242x242)",
            "n_subjects": len(df_b),
            "video_medae_bpm": medae_b,
            "video_mae_bpm": mae_b,
            "video_rmse_bpm": rmse_b,
            "video_pearson_r": r_b,
            "bland_altman_bias_bpm": ba_b["bias"],
            "bland_altman_lower_loa": ba_b["lower_loa"],
            "bland_altman_upper_loa": ba_b["upper_loa"],
            "pct_subjects_waveform_r_ge_07": pct_r7_b,
            "gate_medae_le_2bpm": medae_b <= 2.0,
            "gate_pct_r_ge_07": pct_r7_b >= 70.0,
        },
    ]

    df_summary = pd.DataFrame(summary_rows)
    out_summary_csv = PROJECT_ROOT / "results" / "cohort_v2_summary.csv"
    df_summary.to_csv(out_summary_csv, index=False)
    print(f"--> Saved Cohort v2 Summary to: {out_summary_csv}")

    print("\n" + "=" * 95)
    print("=== DUAL-VARIANT COHORT COMPARISON ===")
    print("=" * 95)
    print(df_summary.to_string(index=False))
    print("=" * 95)


if __name__ == "__main__":
    main()

