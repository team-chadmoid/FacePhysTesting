"""Recompute padding generalization summary using mean across padded levels and delta metrics.

Saves results to results/padding_generalization_mean_summary.csv.
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    sweep_path = PROJECT_ROOT / "results" / "padding_generalization_sweep.csv"
    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep file not found: {sweep_path}")

    df = pd.read_csv(sweep_path)
    
    subjects = ["subject1", "subject3", "subject5", "subject8", "subject9", "subject10"]
    rows = []

    for sid in subjects:
        sub_df = df[df["subject_id"] == sid]
        base_row = sub_df[sub_df["pad_fraction"] == 0.0].iloc[0]
        pad_rows = sub_df[sub_df["pad_fraction"] > 0.0]

        r0 = base_row["waveform_r_max_lag_05s"]
        r0_direct = base_row["waveform_r_direct"]
        snr0 = base_row["snr_db"]
        err0 = base_row["error_vs_primary_bpm"]
        mae0 = base_row["windowed_mae_bpm"]
        acc5_0 = base_row["windowed_acc5_pct"]

        mean_r_pad = pad_rows["waveform_r_max_lag_05s"].mean()
        mean_r_direct_pad = pad_rows["waveform_r_direct"].mean()
        mean_snr_pad = pad_rows["snr_db"].mean()
        mean_err_pad = pad_rows["error_vs_primary_bpm"].mean()
        mean_mae_pad = pad_rows["windowed_mae_bpm"].mean()
        mean_acc5_pad = pad_rows["windowed_acc5_pct"].mean()

        delta_r = mean_r_pad - r0
        delta_r_direct = mean_r_direct_pad - r0_direct
        delta_snr = mean_snr_pad - snr0
        delta_err = mean_err_pad - err0
        delta_mae = mean_mae_pad - mae0
        delta_acc5 = mean_acc5_pad - acc5_0

        # Does padding improve or degrade?
        # Higher r is better (+), higher SNR is better (+), lower err is better (-), higher acc5 is better (+)
        improved_r = delta_r > 0.01
        improved_snr = delta_snr > 0.5
        improved_err = delta_err < -0.1

        rows.append({
            "subject_id": sid,
            "crop_size": f"{int(base_row['crop_w'])}x{int(base_row['crop_h'])}",
            "r_0_lag05": round(r0, 4),
            "mean_r_pad_lag05": round(mean_r_pad, 4),
            "delta_r_lag05": round(delta_r, 4),
            "r_0_direct": round(r0_direct, 4),
            "mean_r_pad_direct": round(mean_r_direct_pad, 4),
            "delta_r_direct": round(delta_r_direct, 4),
            "snr_0_db": round(snr0, 2),
            "mean_snr_pad_db": round(mean_snr_pad, 2),
            "delta_snr_db": round(delta_snr, 2),
            "err_0_bpm": round(err0, 4),
            "mean_err_pad_bpm": round(mean_err_pad, 4),
            "delta_err_bpm": round(delta_err, 4),
            "mae_0_bpm": round(mae0, 4),
            "mean_mae_pad_bpm": round(mean_mae_pad, 4),
            "delta_mae_bpm": round(delta_mae, 4),
            "acc5_0_pct": round(acc5_0, 2),
            "mean_acc5_pad_pct": round(mean_acc5_pad, 2),
            "delta_acc5_pct": round(delta_acc5, 2),
            "r_improved": improved_r,
            "snr_improved": improved_snr,
        })

    summary_df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "results" / "padding_generalization_mean_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"Saved recomputed mean summary to {out_path}")
    print("\n=== PADDING GENERALIZATION: MEAN ACROSS PADDING LEVELS VS BASELINE TIGHT CROP ===")
    disp_cols = [
        "subject_id", "r_0_lag05", "mean_r_pad_lag05", "delta_r_lag05",
        "snr_0_db", "mean_snr_pad_db", "delta_snr_db",
        "err_0_bpm", "mean_err_pad_bpm", "delta_err_bpm",
        "acc5_0_pct", "mean_acc5_pad_pct", "delta_acc5_pct"
    ]
    print(summary_df[disp_cols].to_string(index=False))

    # Cohort averages
    print("\n--- Cohort Summary ---")
    print(f"Subjects where r improved with padding: {summary_df['r_improved'].sum()} / 6")
    print(f"Subjects where SNR improved with padding: {summary_df['snr_improved'].sum()} / 6")
    print(f"Mean delta r across cohort: {summary_df['delta_r_lag05'].mean():+.4f}")
    print(f"Mean delta SNR across cohort: {summary_df['delta_snr_db'].mean():+.2f} dB")
    print(f"Mean delta Video Err across cohort: {summary_df['delta_err_bpm'].mean():+.4f} BPM")
    print(f"Mean delta Acc5 across cohort: {summary_df['delta_acc5_pct'].mean():+.2f}%")

if __name__ == "__main__":
    main()

