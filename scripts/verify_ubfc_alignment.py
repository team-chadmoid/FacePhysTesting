"""Verification script for UBFC Dataset 2 ground-truth timestamp alignment and consistency.

Adheres strictly to AGENTS.md rules:
- Outputs exact CSVs to results/ with git hash recorded.
- Fixed seed from config.
- Single source of truth evaluation protocol from src/eval.py and src/hr.py.
"""

from pathlib import Path
import subprocess
import sys
import yaml
import numpy as np
import pandas as pd

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.io import find_ubfc_subjects, load_ubfc_subject, check_ubfc_alignment
from src.eval import (
    evaluate_ground_truth_consistency,
    plot_bland_altman,
    log_exclusion,
)


def get_git_hash() -> str:
    """Retrieve current git hash or return 'uncommitted'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "initial_commit_pending"


def main():
    config_path = repo_root / "configs" / "ubfc_alignment.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["evaluation"]["seed"]
    np.random.seed(seed)

    git_hash = get_git_hash()
    print(f"Running verification (git hash: {git_hash}, seed: {seed})")

    data_root = repo_root / cfg["dataset"]["root"]
    subjects_map = find_ubfc_subjects(data_root)
    target_subjects = cfg["dataset"]["subjects"]

    summary_rows = []
    window_rows = []

    plot_dir = repo_root / cfg["output"]["plot_dir"]
    plot_dir.mkdir(parents=True, exist_ok=True)

    fs = float(cfg["evaluation"]["fs"])
    win_sec = float(cfg["evaluation"]["window_sec"])
    stride_sec = float(cfg["evaluation"]["stride_sec"])
    threshold_bpm = float(cfg["evaluation"]["discrepancy_threshold_bpm"])

    for subj_id in target_subjects:
        if subj_id not in subjects_map:
            print(f"Warning: Subject {subj_id} not found in {data_root}")
            continue

        sdir = subjects_map[subj_id]
        print(f"\nProcessing {subj_id} from {sdir}...")
        sdata = load_ubfc_subject(sdir, subject_id=subj_id)

        # 1. Alignment diagnostics
        align_diag = check_ubfc_alignment(sdata)

        # 2. GT consistency (PPG-derived HR vs provided HR line)
        gt_eval = evaluate_ground_truth_consistency(
            sdata.ppg,
            sdata.hr,
            time=sdata.time,
            fs=fs,
            window_sec=win_sec,
            stride_sec=stride_sec,
            threshold_bpm=threshold_bpm,
        )

        vid_res = gt_eval["video_level"]
        win_res = gt_eval["windowed"]
        win_metrics = win_res["metrics"]

        # Save Bland-Altman plot
        ba_path = plot_dir / f"{subj_id}_bland_altman_gt.png"
        plot_bland_altman(
            y_true=win_res["hrs_line"],
            y_pred=win_res["hrs_ppg"],
            title=f"{subj_id} GT PPG vs HR Line (30s Windows)",
            save_path=ba_path,
        )

        summary_rows.append({
            "git_hash": git_hash,
            "subject_id": subj_id,
            "video_width": sdata.video_metadata.get("width") if sdata.video_metadata else None,
            "video_height": sdata.video_metadata.get("height") if sdata.video_metadata else None,
            "video_frames": align_diag["video_frames"],
            "gt_samples": align_diag["n_samples"],
            "frames_match": align_diag["frames_match"],
            "duration_sec": align_diag["duration_sec"],
            "mean_dt_ms": align_diag["mean_dt"] * 1000.0,
            "std_dt_ms": align_diag["std_dt"] * 1000.0,
            "min_dt_ms": align_diag["min_dt"] * 1000.0,
            "max_dt_ms": align_diag["max_dt"] * 1000.0,
            "nominal_fps": align_diag["nominal_fps"],
            "video_fps": align_diag["video_fps"],
            "non_positive_dt_count": align_diag["non_positive_count"],
            "large_gaps_count": align_diag["large_gap_count"],
            "video_hr_ppg_bpm": vid_res["hr_ppg"],
            "video_hr_line_bpm": vid_res["hr_line"],
            "video_discrepancy_bpm": vid_res["discrepancy_bpm"],
            "video_flagged": vid_res["flagged"],
            "window_count": win_res["num_windows"],
            "window_mae_bpm": win_metrics["mae"],
            "window_rmse_bpm": win_metrics["rmse"],
            "window_pearson_r": win_metrics["pearson_r"],
            "window_ba_bias": win_metrics["bias"],
            "window_ba_upper": win_metrics["upper_loa"],
            "window_ba_lower": win_metrics["lower_loa"],
            "window_max_discrepancy_bpm": win_res["max_discrepancy_bpm"],
            "window_flagged_count": win_res["flagged_windows_count"],
            "flags": "; ".join(align_diag["flags"]),
        })

        # Record individual window details
        for i, (t_mid, h_ppg, h_line) in enumerate(
            zip(win_res["window_times"], win_res["hrs_ppg"], win_res["hrs_line"])
        ):
            window_rows.append({
                "git_hash": git_hash,
                "subject_id": subj_id,
                "window_idx": i,
                "center_time_sec": t_mid,
                "hr_ppg_bpm": h_ppg,
                "hr_line_bpm": h_line,
                "diff_bpm": h_ppg - h_line,
                "abs_diff_bpm": abs(h_ppg - h_line),
                "flagged": abs(h_ppg - h_line) > threshold_bpm,
            })

    # Save summary dataframe
    df_summary = pd.DataFrame(summary_rows)
    summary_path = repo_root / cfg["output"]["results_csv"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary CSV to {summary_path}")

    # Save windowed dataframe
    df_window = pd.DataFrame(window_rows)
    window_path = repo_root / cfg["output"]["window_results_csv"]
    df_window.to_csv(window_path, index=False)
    print(f"Saved windowed details CSV to {window_path}")

    # Print summary table
    print("\n=== SUMMARY OF VERIFICATION RESULTS ===")
    cols_to_print = [
        "subject_id",
        "video_frames",
        "gt_samples",
        "frames_match",
        "nominal_fps",
        "non_positive_dt_count",
        "video_hr_ppg_bpm",
        "video_hr_line_bpm",
        "video_discrepancy_bpm",
        "video_flagged",
        "window_mae_bpm",
        "window_flagged_count",
    ]
    print(df_summary[cols_to_print].to_string(index=False))


if __name__ == "__main__":
    main()

