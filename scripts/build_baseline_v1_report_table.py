"""Consolidation script to build the official Baseline-v1 7-subject table.

Consolidates metrics for:
- subject1 (640x480 full frame, Path 3 GT-timestamp)
- subject3 (640x480 full frame, Path 3 GT-timestamp)
- subject4 (242x242 natively tight crop + 60px reflect border, Path 3 GT-timestamp)
- subject5 (640x480 full frame, Path 3 GT-timestamp)
- subject8 (640x480 full frame, Path 3 GT-timestamp)
- subject9 (640x480 full frame, Path 3 GT-timestamp)
- subject10 (640x480 full frame, Path 3 GT-timestamp)

Strictly sourced from:
- results/padding_controlled_test.csv
- results/pilot_4_subjects.csv
- results/subject4_padding_sweep.csv

Calculates:
- Video-level MAE, MedAE, RMSE, Pearson r
- Bland-Altman mean bias and 95% limits of agreement
- Percentage of subjects with waveform r >= 0.7
- Saves results/baseline_v1_7_subjects.csv and results/baseline_v1_cohort_summary.csv
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".cache" / "matplotlib")
(PROJECT_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import (
    calc_bland_altman,
    calc_mae,
    calc_medae,
    calc_rmse,
    calc_pearson_r,
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
    print("=== Consolidating Official Baseline-v1 7-Subject Benchmark ===")
    print(f"Current Git commit hash: {git_hash}")
    print("=" * 80)

    # 1. Load source CSVs
    pad_csv = PROJECT_ROOT / "results" / "padding_controlled_test.csv"
    pilot_csv = PROJECT_ROOT / "results" / "pilot_4_subjects.csv"

    if not pad_csv.exists() or not pilot_csv.exists():
        raise FileNotFoundError("Source CSVs do not exist in results/")

    df_pad = pd.read_csv(pad_csv)
    df_pilot = pd.read_csv(pilot_csv)

    rows = []

    # Subject 1 (full frame original)
    r1 = df_pad[(df_pad["subject_id"] == "subject1") & (df_pad["condition"] == "full_frame_original")].iloc[0]
    rows.append({
        "git_hash": r1["git_hash"],
        "subject_id": "subject1",
        "video_path": r1["video_path"],
        "video_size_bytes": int(r1["video_size_bytes"]),
        "video_size_mb": float(r1["video_size_mb"]),
        "resolution": r1["resolution"],
        "condition_notes": "640x480 uncropped natural framing",
        "padding_applied": False,
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
        "stat_null": int(r1["stat_null"]),
    })

    # Subject 3 (full frame original)
    r3 = df_pad[(df_pad["subject_id"] == "subject3") & (df_pad["condition"] == "full_frame_original")].iloc[0]
    rows.append({
        "git_hash": r3["git_hash"],
        "subject_id": "subject3",
        "video_path": r3["video_path"],
        "video_size_bytes": int(r3["video_size_bytes"]),
        "video_size_mb": float(r3["video_size_mb"]),
        "resolution": r3["resolution"],
        "condition_notes": "640x480 uncropped natural framing",
        "padding_applied": False,
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
        "stat_null": int(r3["stat_null"]),
    })

    # Subject 4 (padded tight crop: +60px reflect border)
    r4 = df_pad[(df_pad["subject_id"] == "subject4") & (df_pad["condition"] == "padded_tight_crop")].iloc[0]
    rows.append({
        "git_hash": r4["git_hash"],
        "subject_id": "subject4",
        "video_path": r4["video_path"],
        "video_size_bytes": int(r4["video_size_bytes"]),
        "video_size_mb": float(r4["video_size_mb"]),
        "resolution": "242x242 (+60px reflect -> 362x362)",
        "condition_notes": "Native tight crop with +60px reflect border padding",
        "padding_applied": True,
        "primary_ref_hr_bpm": float(r4["primary_ref_hr_bpm"]),
        "video_hr_bpm": float(r4["video_hr_bpm"]),
        "error_bpm": float(r4["error_vs_primary_bpm"]),
        "waveform_r_direct": float(r4["waveform_r_direct"]),
        "waveform_r_max_lag": float(r4["waveform_r_max_lag"]),
        "waveform_best_lag_sec": float(r4["waveform_best_lag_sec"]),
        "windowed_mae_bpm": float(r4["windowed_mae_bpm"]),
        "windowed_medae_bpm": float(r4["windowed_medae_bpm"]),
        "windowed_rmse_bpm": float(r4["windowed_rmse_bpm"]),
        "windowed_pearson_r": float(r4["windowed_pearson_r"]),
        "windowed_acc5_pct": float(r4["windowed_acc5_pct"]),
        "snr_db": float(r4["snr_db"]),
        "stat_filled": int(r4["stat_filled"]),
        "stat_null": int(r4["stat_null"]),
    })

    # Pilot subjects: 5, 8, 9, 10
    for sid in ["subject5", "subject8", "subject9", "subject10"]:
        rp = df_pilot[df_pilot["subject_id"] == sid].iloc[0]
        rows.append({
            "git_hash": rp["git_hash"],
            "subject_id": sid,
            "video_path": rp["video_path"],
            "video_size_bytes": int(rp["video_size_bytes"]),
            "video_size_mb": float(rp["video_size_mb"]),
            "resolution": rp["video_resolution"],
            "condition_notes": "640x480 uncropped natural framing",
            "padding_applied": bool(rp["padding_applied"]),
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
            "stat_null": int(rp["stat_null"]),
        })

    df_7 = pd.DataFrame(rows)
    out_7_csv = PROJECT_ROOT / "results" / "baseline_v1_7_subjects.csv"
    df_7.to_csv(out_7_csv, index=False)
    print(f"--> Saved 7-subject table to: {out_7_csv}")

    # Compute Cohort Statistics
    preds = df_7["video_hr_bpm"].values
    refs = df_7["primary_ref_hr_bpm"].values
    errors = df_7["error_bpm"].values
    r_waveforms = df_7["waveform_r_max_lag"].values

    mae = float(calc_mae(preds, refs))
    medae = float(calc_medae(preds, refs))
    rmse = float(calc_rmse(preds, refs))
    cohort_pearson = float(calc_pearson_r(preds, refs))
    ba_dict = calc_bland_altman(refs, preds)
    ba_mean = ba_dict["bias"]
    ba_lower = ba_dict["lower_loa"]
    ba_upper = ba_dict["upper_loa"]
    frac_r_ge_07 = float(np.mean(r_waveforms >= 0.7) * 100.0)
    median_snr = float(np.median(df_7["snr_db"].values))

    summary_data = [{
        "cohort_size": len(df_7),
        "video_mae_bpm": mae,
        "video_medae_bpm": medae,
        "video_rmse_bpm": rmse,
        "video_pearson_r": cohort_pearson,
        "bland_altman_mean_bias_bpm": ba_mean,
        "bland_altman_lower_loa_bpm": ba_lower,
        "bland_altman_upper_loa_bpm": ba_upper,
        "pct_subjects_waveform_r_ge_07": frac_r_ge_07,
        "median_snr_db": median_snr,
        "gate_passed_medae_le_2bpm": medae <= 2.0,
        "gate_passed_pct_r_ge_07": frac_r_ge_07 >= 70.0,
    }]
    df_summary = pd.DataFrame(summary_data)
    out_summary_csv = PROJECT_ROOT / "results" / "baseline_v1_cohort_summary.csv"
    df_summary.to_csv(out_summary_csv, index=False)
    print(f"--> Saved cohort summary to: {out_summary_csv}")

    # Generate Bland-Altman and Scatter Plots
    plot_dir = PROJECT_ROOT / "results" / "plots" / "baseline_v1"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    ax = axes[0]
    ax.scatter(refs, preds, color="#1f77b4", s=80, edgecolors="black", zorder=3)
    min_v = min(min(refs), min(preds)) - 5
    max_v = max(max(refs), max(preds)) + 5
    ax.plot([min_v, max_v], [min_v, max_v], "r--", label="Identity (x=y)")
    ax.set_title(f"Baseline-v1: Video HR vs Reference GT HR (r={cohort_pearson:.3f})")
    ax.set_xlabel("Reference GT PPG HR (BPM)")
    ax.set_ylabel("FacePhys Video HR (BPM)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # Bland-Altman plot
    ax2 = axes[1]
    means = (preds + refs) / 2.0
    diffs = preds - refs
    ax2.scatter(means, diffs, color="#2ca02c", s=80, edgecolors="black", zorder=3)
    ax2.axhline(ba_mean, color="black", linestyle="-", label=f"Mean Bias: {ba_mean:+.2f} BPM")
    ax2.axhline(ba_lower, color="red", linestyle="--", label=f"Lower LoA: {ba_lower:+.2f} BPM")
    ax2.axhline(ba_upper, color="red", linestyle="--", label=f"Upper LoA: {ba_upper:+.2f} BPM")
    ax2.set_title(f"Bland-Altman Agreement (95% LoA: [{ba_lower:.2f}, {ba_upper:.2f}] BPM)")
    ax2.set_xlabel("Mean of Estimate and Reference (BPM)")
    ax2.set_ylabel("Difference (Estimate - Reference) (BPM)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    ba_path = plot_dir / "baseline_v1_bland_altman_and_scatter.png"
    plt.savefig(ba_path, dpi=200)
    plt.close()
    print(f"--> Saved plot to: {ba_path}")

    print("\n" + "=" * 90)
    print("=== BASELINE-V1 CONSOLIDATED 7-SUBJECT TABLE ===")
    print("=" * 90)
    display_cols = [
        "subject_id", "resolution", "video_hr_bpm", "primary_ref_hr_bpm",
        "error_bpm", "waveform_r_max_lag", "windowed_medae_bpm", "windowed_acc5_pct", "snr_db"
    ]
    print(df_7[display_cols].to_string(index=False))
    print("=" * 90)
    print(f"Cohort MAE:        {mae:.4f} BPM")
    print(f"Cohort MedAE:      {medae:.4f} BPM (Gate <= 2.0 BPM: {medae <= 2.0})")
    print(f"Cohort RMSE:       {rmse:.4f} BPM")
    print(f"Cohort Pearson r:  {cohort_pearson:.4f}")
    print(f"Bland-Altman Bias: {ba_mean:+.4f} BPM (95% LoA: [{ba_lower:+.2f}, {ba_upper:+.2f}] BPM)")
    print(f"Waveform r >= 0.7: {frac_r_ge_07:.1f}% ({int(np.sum(r_waveforms >= 0.7))}/7 subjects, Gate >= 70%: {frac_r_ge_07 >= 70.0})")
    print("=" * 90)


if __name__ == "__main__":
    main()
