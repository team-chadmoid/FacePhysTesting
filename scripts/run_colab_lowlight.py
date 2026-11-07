"""Evaluate FacePhys baseline across UBFC low-light variations (Condition C1).

Supports Google Colab (mounted Drive) and local execution.
Discovers all conditions (e.g. original, 50lux, 30lux, etc.) and subjects dynamically.
Calculates:
- Video HR error vs Primary (Welch GT PPG) and Secondary (oximeter line)
- Windowed MAE, MedAE, RMSE, Pearson r, and Fraction <= 5 BPM (all and reliable windows)
- Spectral SNR and top-3 spectral peaks
- Waveform lag correlation
Outputs:
- results/preds/{condition}_{subject}_pred.csv
- results/ubfc_lowlight_conditions.csv
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
    get_top_spectral_peaks,
    max_lag_waveform_correlation,
)
from src.hr import (
    butterworth_bandpass_filter,
    estimate_hr_from_bvp,
    estimate_hr_video,
    estimate_hr_welch,
    estimate_hr_windowed,
)
from src.io import load_ubfc_ground_truth, resample_to_uniform
from src.models import FacePhysWrapper, predict_bvp


def get_git_hash() -> str:
    try:
        res = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
        )
        return res.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN"


def find_lowlight_directories(root_dir: Path, original_dir: Optional[Path] = None) -> List[Dict[str, Path]]:
    """Recursively discover all video/GT subject folders under root_dir and optional original_dir.

    Uses os.walk(followlinks=True) so Google Drive shortcuts / symlinks are traversed.
    Standardizes conditions (e.g. 'Data Set 2' / 'original' -> 'original_baseline').

    Returns a list of dicts:
        - condition: name of variation (e.g. 'original_baseline', '50lux', '30lux')
        - subject_id: subject name (e.g. 'subject1')
        - subject_dir: Path to directory
        - gt_path: Path to ground_truth.txt
        - video_path: Path to video file (.avi / .mp4)
    """
    entries = []

    # 1. Walk root_dir with followlinks=True
    if root_dir.is_dir():
        for root, dirs, files in os.walk(str(root_dir), followlinks=True):
            if "ground_truth.txt" in files:
                sdir = Path(root)
                vids = [sdir / f for f in files if f.endswith(".avi") or f.endswith(".mp4")]
                if not vids:
                    continue
                vid_path = vids[0]
                gt_path = sdir / "ground_truth.txt"

                # Determine condition from relative path
                try:
                    rel_parts = sdir.relative_to(root_dir).parts
                except ValueError:
                    rel_parts = (sdir.name,)

                if len(rel_parts) >= 2:
                    raw_cond = rel_parts[0]
                    subj_name = rel_parts[-1]
                elif len(rel_parts) == 1:
                    raw_cond = "default"
                    subj_name = rel_parts[0]
                else:
                    raw_cond = "root"
                    subj_name = sdir.name

                # Standardize condition name
                cond_clean = raw_cond.lower().replace("_", " ").replace("-", " ")
                if any(x in cond_clean for x in ["data set 2", "dataset 2", "dataset2", "original", "baseline"]):
                    condition = "original_baseline"
                else:
                    condition = raw_cond

                match = re.search(r"subject\d+", subj_name, re.IGNORECASE)
                clean_sub = match.group(0).lower() if match else subj_name.lower()

                entries.append({
                    "condition": condition,
                    "subject_id": clean_sub,
                    "subject_dir": sdir,
                    "gt_path": gt_path,
                    "video_path": vid_path,
                })

    # 2. Walk original_dir if provided separately
    if original_dir and original_dir.is_dir():
        existing_orig_subs = {e["subject_id"] for e in entries if e["condition"] == "original_baseline"}
        for root, dirs, files in os.walk(str(original_dir), followlinks=True):
            if "ground_truth.txt" in files:
                sdir = Path(root)
                vids = [sdir / f for f in files if f.endswith(".avi") or f.endswith(".mp4")]
                if not vids:
                    continue
                match = re.search(r"subject\d+", sdir.name, re.IGNORECASE)
                clean_sub = match.group(0).lower() if match else sdir.name.lower()
                if clean_sub not in existing_orig_subs:
                    entries.append({
                        "condition": "original_baseline",
                        "subject_id": clean_sub,
                        "subject_dir": sdir,
                        "gt_path": sdir / "ground_truth.txt",
                        "video_path": vids[0],
                    })
                    existing_orig_subs.add(clean_sub)

    # Sort by condition and subject
    entries.sort(key=lambda x: (x["condition"], x["subject_id"]))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Evaluate FacePhys on UBFC low-light variations.")
    parser.add_argument("--config", type=str, default="configs/colab_lowlight.yaml", help="Path to config file")
    parser.add_argument("--data-dir", type=str, default=None, help="Root directory of UBFC_lowlight")
    parser.add_argument("--original-dir", type=str, default=None, help="Directory of original Data Set 2 if shortcut or stored separately")
    args = parser.parse_args()

    git_hash = get_git_hash()
    print("=" * 80)
    print("=== FacePhys Evaluation on UBFC Low-Light (C1 Condition) ===")
    print(f"Git commit hash: {git_hash}")
    print("=" * 80)

    cfg_file = PROJECT_ROOT / args.config
    if cfg_file.is_file():
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # Dataset path resolution
    if args.data_dir:
        data_root = Path(args.data_dir)
    else:
        colab_path = Path(cfg.get("dataset", {}).get("root_dir", "/content/drive/MyDrive/UBFC_lowlight"))
        local_path = PROJECT_ROOT / cfg.get("dataset", {}).get("fallback_local_dir", "data/ubfc/UBFC_lowlight")
        if colab_path.is_dir():
            data_root = colab_path
        elif local_path.is_dir():
            data_root = local_path
        else:
            data_root = colab_path

    # Check original dataset directory
    orig_root: Optional[Path] = None
    if args.original_dir:
        orig_root = Path(args.original_dir)
    else:
        cfg_orig = cfg.get("dataset", {}).get("original_dir")
        if cfg_orig and Path(cfg_orig).is_dir():
            orig_root = Path(cfg_orig)
        else:
            # Check common Colab MyDrive paths for Data Set 2
            candidates = [
                Path("/content/drive/MyDrive/Data Set 2"),
                Path("/content/drive/MyDrive/UBFC/Data Set 2"),
                Path("/content/drive/MyDrive/DATASET_2"),
                PROJECT_ROOT / "data/ubfc/Data Set 2",
            ]
            for cand in candidates:
                if cand.is_dir():
                    orig_root = cand
                    break

    print(f"Looking for low-light dataset in: {data_root}")
    if orig_root:
        print(f"Original dataset found in: {orig_root}")
    else:
        print("Note: Original dataset folder not specified separately; scanning low-light root for all conditions/shortcuts.")

    if not data_root.is_dir():
        print(f"ERROR: Dataset directory '{data_root}' not found!")
        print("In Google Colab, please make sure you ran:")
        print("  from google.colab import drive; drive.mount('/content/drive')")
        print("and added 'UBFC_lowlight' to My Drive (Right click folder -> Organize -> Add shortcut -> My Drive).")
        sys.exit(1)

    entries = find_lowlight_directories(data_root, original_dir=orig_root)
    if not entries:
        print(f"No valid subjects with ground_truth.txt and video files found in {data_root}.")
        sys.exit(1)

    print(f"Found {len(entries)} subject runs across conditions:")
    conditions_found = sorted(list(set(e["condition"] for e in entries)))
    print(f"Conditions detected: {conditions_found}")

    params = cfg.get("parameters", {})
    fs_u = float(params.get("fs_uniform", 30.0))
    win_sec = float(params.get("window_sec", 30.0))
    stride_sec = float(params.get("stride_sec", 1.0))
    lowcut = float(params.get("bandpass_low", 0.7))
    highcut = float(params.get("bandpass_high", 3.0))
    order = int(params.get("filter_order", 3))
    thresh_bpm = float(params.get("reliability_threshold_bpm", 5.0))
    snr_delta = float(params.get("snr_delta_hz", 0.1))
    max_lag = float(params.get("lag_search_max_sec", 2.0))

    preds_dir = PROJECT_ROOT / cfg.get("outputs", {}).get("preds_dir", "results/preds")
    preds_dir.mkdir(parents=True, exist_ok=True)
    summary_path = PROJECT_ROOT / cfg.get("outputs", {}).get("summary_csv", "results/ubfc_lowlight_conditions.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, item in enumerate(entries, start=1):
        cond = item["condition"]
        sid = item["subject_id"]
        vid_p = item["video_path"]
        gt_p = item["gt_path"]

        print(f"\n[{idx}/{len(entries)}] Processing Condition='{cond}', Subject='{sid}'...")
        print(f"  Video: {vid_p}")

        # Load GT
        gt_data = load_ubfc_ground_truth(gt_p)
        t_gt = gt_data["time"]
        ppg_gt = gt_data["ppg"]
        hr_gt = gt_data["hr"]

        # Run FacePhys
        bvp_u, t_u, meta = predict_bvp(
            video_path=vid_p,
            timestamps=t_gt,
            fs_target=fs_u,
            model_name="FacePhys.rlap",
        )

        # Save prediction CSV
        clean_cond = re.sub(r"[^\w\-]", "_", cond)
        pred_csv = preds_dir / f"{clean_cond}_{sid}_facephys_pred.csv"
        df_pred = pd.DataFrame({
            "frame_idx": np.arange(len(meta.get("raw_bvp", bvp_u))),
            "gt_timestamp_sec": meta.get("orig_timestamps", t_u)[:len(meta.get("raw_bvp", bvp_u))],
            "facephys_bvp_raw": meta.get("raw_bvp", bvp_u),
        })
        df_pred.to_csv(pred_csv, index=False)

        # Uniform GT
        _, ppg_u = resample_to_uniform(t_gt, ppg_gt, target_fs=fs_u)
        _, hr_line_u = resample_to_uniform(t_gt, hr_gt, target_fs=fs_u, method="linear")

        # Video-level HR
        hr_pred_vid, f_pred, psd_pred = estimate_hr_welch(
            butterworth_bandpass_filter(bvp_u, fs=fs_u, lowcut=lowcut, highcut=highcut, order=order),
            fs=fs_u,
            low_bpm=lowcut * 60.0,
            high_bpm=highcut * 60.0,
            nfft=16384,
        )
        primary_ref_hr = estimate_hr_video(ppg_u, fs=fs_u)
        secondary_ref_hr = float(np.mean(hr_line_u))

        # Spectral metrics
        snr_prim = calculate_snr(psd_pred, f_pred, ref_hr_bpm=primary_ref_hr, delta_hz=snr_delta)
        snr_sec = calculate_snr(psd_pred, f_pred, ref_hr_bpm=secondary_ref_hr, delta_hz=snr_delta)
        peaks = get_top_spectral_peaks(psd_pred, f_pred, n_peaks=3)
        lag_res = max_lag_waveform_correlation(bvp_u, ppg_u, fs=fs_u, max_lag_sec=max_lag)

        # Windowed evaluation
        _, hr_p_win = estimate_hr_windowed(bvp_u, fs=fs_u, window_sec=win_sec, stride_sec=stride_sec)
        _, hr_prim_win = estimate_hr_windowed(ppg_u, fs=fs_u, window_sec=win_sec, stride_sec=stride_sec)

        win_samp = int(round(win_sec * fs_u))
        stride_samp = int(round(stride_sec * fs_u))
        hr_sec_win = []
        for st in range(0, len(hr_line_u) - win_samp + 1, stride_samp):
            hr_sec_win.append(float(np.mean(hr_line_u[st:st + win_samp])))
        hr_sec_win = np.array(hr_sec_win)

        w_len = min(len(hr_p_win), len(hr_prim_win), len(hr_sec_win))
        hr_p_sub = hr_p_win[:w_len]
        hr_prim_sub = hr_prim_win[:w_len]
        hr_sec_sub = hr_sec_win[:w_len]

        # All windows
        mae_all_prim = calc_mae(hr_prim_sub, hr_p_sub)
        medae_all_prim = calc_medae(hr_prim_sub, hr_p_sub)
        rmse_all_prim = calc_rmse(hr_prim_sub, hr_p_sub)
        r_all_prim = calc_pearson_r(hr_prim_sub, hr_p_sub)
        acc5_all_prim = calc_fraction_within_threshold(hr_prim_sub, hr_p_sub, threshold=thresh_bpm)

        # Reliable windows (|Primary - Secondary| <= 5 BPM)
        rel_mask = np.abs(hr_prim_sub - hr_sec_sub) <= thresh_bpm
        n_rel = int(np.sum(rel_mask))

        if n_rel > 0:
            mae_rel_prim = calc_mae(hr_prim_sub[rel_mask], hr_p_sub[rel_mask])
            medae_rel_prim = calc_medae(hr_prim_sub[rel_mask], hr_p_sub[rel_mask])
            rmse_rel_prim = calc_rmse(hr_prim_sub[rel_mask], hr_p_sub[rel_mask])
            r_rel_prim = calc_pearson_r(hr_prim_sub[rel_mask], hr_p_sub[rel_mask])
            acc5_rel_prim = calc_fraction_within_threshold(hr_prim_sub[rel_mask], hr_p_sub[rel_mask], threshold=thresh_bpm)
        else:
            mae_rel_prim = medae_rel_prim = rmse_rel_prim = r_rel_prim = acc5_rel_prim = float("nan")

        print(f"  Pred HR: {hr_pred_vid:.2f} BPM | GT Primary: {primary_ref_hr:.2f} BPM | Error: {abs(hr_pred_vid - primary_ref_hr):.2f} BPM")
        print(f"  Window MAE: {mae_all_prim:.2f} BPM | MedAE: {medae_all_prim:.2f} BPM | Acc<=5BPM: {acc5_all_prim:.1f}% | SNR: {snr_prim:.2f} dB")

        results.append({
            "git_hash": git_hash,
            "condition": cond,
            "subject_id": sid,
            "resolution": f"{meta.get('video_width', 'unknown')}x{meta.get('video_height', 'unknown')}",
            "face_detected_ratio": meta.get("face_detected_ratio", 1.0),
            "video_hr_pred_bpm": hr_pred_vid,
            "primary_ref_hr_bpm": primary_ref_hr,
            "secondary_ref_hr_bpm": secondary_ref_hr,
            "video_error_vs_primary_bpm": abs(hr_pred_vid - primary_ref_hr),
            "snr_primary_db": snr_prim,
            "waveform_lag_max_r": lag_res["max_r"],
            "waveform_optimal_lag_sec": lag_res["optimal_lag_sec"],
            "peak1_bpm": peaks[0]["freq_bpm"] if len(peaks) > 0 else float("nan"),
            "peak1_rel_power_inband": peaks[0]["rel_power_inband"] if len(peaks) > 0 else float("nan"),
            "all_windows_count": w_len,
            "all_win_mae_vs_primary": mae_all_prim,
            "all_win_medae_vs_primary": medae_all_prim,
            "all_win_rmse_vs_primary": rmse_all_prim,
            "all_win_r_vs_primary": r_all_prim,
            "all_win_acc5_vs_primary": acc5_all_prim,
            "reliable_windows_count": n_rel,
            "reliable_win_mae_vs_primary": mae_rel_prim,
            "reliable_win_medae_vs_primary": medae_rel_prim,
            "reliable_win_rmse_vs_primary": rmse_rel_prim,
            "reliable_win_r_vs_primary": r_rel_prim,
            "reliable_win_acc5_vs_primary": acc5_rel_prim,
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(summary_path, index=False)
    print("\n" + "=" * 80)
    print(f"Saved complete low-light summary to: {summary_path}")
    print("=" * 80)

    # Print Condition-Level Aggregates
    print("\nCondition-Level Summary (Mean across subjects in condition):")
    cond_summary = df_out.groupby("condition").agg({
        "video_error_vs_primary_bpm": ["mean", "median"],
        "all_win_mae_vs_primary": "mean",
        "all_win_medae_vs_primary": "mean",
        "all_win_acc5_vs_primary": "mean",
        "snr_primary_db": "mean",
        "subject_id": "count",
    })
    print(cond_summary.to_string())


if __name__ == "__main__":
    main()

