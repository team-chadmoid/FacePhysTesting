"""Evaluate FacePhys baseline across UBFC low-light variations (Condition C1).

Supports Google Colab (mounted Drive) and local execution.
Discovers all conditions (e.g. original_baseline, lux50, lux30, lux10, lux5, lux1) and subjects dynamically.
Calculates:
- Video HR error vs Primary (Welch GT PPG) and Secondary (oximeter line)
- Windowed MAE, MedAE, RMSE, Pearson r, and Fraction <= 5 BPM (all and reliable windows)
- Spectral SNR and top-3 spectral peaks
- Waveform lag correlation
Outputs:
- results/preds/{condition}_{subject}_pred.csv
- results/ubfc_lowlight_conditions.csv
- results/exclusions.csv (logs any excluded/failed subject per AGENTS.md rule 4)

Cohort Size & Deduplication:
- UBFC Dataset 2 contains exactly 42 unique subjects (subject1 through subject49, with a
  few non-contiguous indices missing in the benchmark).
- Across the 6 benchmark conditions (5 low-light variations: lux1, lux5, lux10, lux30, lux50
  plus original_baseline), the clean evaluation cohort is exactly 42 x 6 = 252 video runs.
- Duplicate folders created by Google Drive / cloud download artifacts (e.g. 'subject22 (1)',
  'subject31 (1)', 'subject34 (1)') and redundant directory scans (e.g. DATASET_2 traversed
  both inside UBFC_lowlight and at MyDrive root) are automatically deduplicated by default.
  All excluded duplicate folders are logged to results/exclusions.csv per AGENTS.md rule 4.
- Passing ``--include-duplicates`` allows processing all 309 raw discovered folders if desired.

Memory safety & Subprocess Isolation:
- FacePhys (JAX / Mamba SSM) creates background threads and C-extension buffers.
  Running hundreds of subjects in a single Python process corrupts glibc heap metadata,
  resulting in ``malloc(): invalid size (unsorted)`` or SIGABRT after ~150-180 subjects.
- Fix: Each subject is executed in a fresh isolated worker subprocess (via ``--worker``).
  When the worker finishes inference and writes its JSON result, the OS kernel reclaims
  100% of heap allocations and thread resources. The coordinator process never accumulates
  memory across subjects.
"""

import os
# Prevent OpenMP and OpenCV multi-threaded allocator contention and bound JAX memory on Linux
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.20"

import argparse
import gc
import json
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
cv2.setNumThreads(0)
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


def get_git_hash() -> str:
    try:
        res = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
        )
        return res.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN"


def log_exclusion(subject_id: str, reason: str, exclusions_csv: Path) -> None:
    """Log an excluded or failed video to results/exclusions.csv per AGENTS.md rule 4."""
    exclusions_csv.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "subject_id": subject_id,
        "reason": reason.replace(",", ";"),
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    df_new = pd.DataFrame([row])
    if not exclusions_csv.is_file():
        df_new.to_csv(exclusions_csv, index=False)
    else:
        df_new.to_csv(exclusions_csv, mode="a", header=False, index=False)


def normalize_condition_name(raw_name: str) -> str:
    """Normalize directory string to standardized condition name.
    
    Examples:
        '50lux', 'lux50', 'lux_50' -> 'lux50'
        'DATASET_2', 'Data Set 2', 'original', 'baseline' -> 'original_baseline'
    """
    clean = raw_name.lower().replace("_", " ").replace("-", " ").strip()
    if any(x in clean for x in ["data set 2", "dataset 2", "dataset2", "original", "baseline"]):
        return "original_baseline"
    
    m_num_first = re.search(r"(\d+)\s*lux", clean)
    if m_num_first:
        return f"lux{m_num_first.group(1)}"
    m_lux_first = re.search(r"lux\s*(\d+)", clean)
    if m_lux_first:
        return f"lux{m_lux_first.group(1)}"
    
    return raw_name.strip()


def is_canonical_folder(path: Path) -> int:
    """Return penalty score for duplicate/copy artifacts (lower score is more canonical)."""
    score = 0
    name = path.name.lower()
    path_str = str(path).lower()
    if re.search(r"\(\d+\)", name) or re.search(r"\(\d+\)", path_str):
        score += 100
    if "copy" in name or "copy" in path_str:
        score += 50
    if "dup" in name or "dup" in path_str:
        score += 50
    score += len(path_str)  # Prefer cleaner / shorter paths
    return score


def find_lowlight_directories(
    root_dir: Path,
    original_dir: Optional[Path] = None,
    include_duplicates: bool = False,
    exclusions_csv: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Recursively discover all video/GT subject folders under root_dir and optional original_dir.

    Handles:
    - Follows Google Drive shortcuts / symlinks (followlinks=True).
    - Case-insensitive video and GT file extensions (.avi, .mp4, .mov, etc.).
    - By default (include_duplicates=False), automatically filters out duplicate copies
      (such as Google Drive '(1)' artifacts and redundant DATASET_2 scans) to produce
      the clean 42 subjects x 6 conditions = 252 run benchmark cohort.
    - Excluded duplicates are logged to results/exclusions.csv per AGENTS.md rule 4.
    """
    search_roots: List[Path] = []
    if root_dir.is_dir():
        search_roots.append(root_dir)
        # Explicitly add each condition subfolder as a separate search root.
        # Google Drive's FUSE filesystem sometimes returns incomplete directory
        # listings for os.walk when the parent directory has many entries.
        # Walking each lux subfolder independently avoids this.
        for child in sorted(root_dir.iterdir()):
            if child.is_dir() and child not in search_roots:
                search_roots.append(child)
    
    if original_dir and original_dir.is_dir() and original_dir not in search_roots:
        search_roots.append(original_dir)
    
    # If DATASET_2 is sibling to root_dir in Drive, include it if not already found
    if root_dir.is_dir() and root_dir.parent.is_dir():
        for sibling_name in ["DATASET_2", "Data Set 2", "dataset_2"]:
            cand = root_dir.parent / sibling_name
            if cand.is_dir() and cand not in search_roots:
                search_roots.append(cand)

    vid_exts = (".avi", ".mp4", ".mov", ".mkv", ".m4v")
    raw_discovered: List[Dict[str, Any]] = []

    for sroot in search_roots:
        for root, dirs, files in os.walk(str(sroot), followlinks=True):
            gt_matches = [f for f in files if f.lower() in ("ground_truth.txt", "groundtruth.txt", "gt.txt")]
            if not gt_matches:
                continue
            
            vid_matches = [f for f in files if f.lower().endswith(vid_exts)]
            if not vid_matches:
                continue
            
            sdir = Path(root)
            gt_path = sdir / gt_matches[0]
            # Prefer vid.avi / vid.mp4 if available, otherwise first match
            preferred_vids = [f for f in vid_matches if f.lower().startswith("vid")]
            vid_path = sdir / (preferred_vids[0] if preferred_vids else vid_matches[0])

            # Determine condition from path relative to sroot
            try:
                rel_parts = sdir.relative_to(sroot).parts
            except ValueError:
                rel_parts = (sdir.name,)

            # Check rel_parts for condition keyword
            detected_cond = None
            for part in rel_parts:
                norm_part = normalize_condition_name(part)
                if norm_part in ("original_baseline", "lux1", "lux5", "lux10", "lux30", "lux50"):
                    detected_cond = norm_part
                    break

            if detected_cond is None:
                if len(rel_parts) >= 2:
                    detected_cond = normalize_condition_name(rel_parts[0])
                else:
                    detected_cond = normalize_condition_name(sroot.name)

            # Determine base subject name
            match = re.search(r"subject\s*(\d+)", sdir.name, re.IGNORECASE)
            if not match and len(rel_parts) >= 2:
                # Check parent folder name if sdir is nested
                match = re.search(r"subject\s*(\d+)", rel_parts[-1], re.IGNORECASE)
            
            if match:
                base_sub = f"subject{int(match.group(1))}"
            else:
                base_sub = sdir.name.lower()

            raw_discovered.append({
                "condition": detected_cond,
                "base_subject_id": base_sub,
                "subject_dir": sdir,
                "gt_path": gt_path,
                "video_path": vid_path,
            })

    # Group by (condition, base_subject_id)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for item in raw_discovered:
        key = (item["condition"], item["base_subject_id"])
        grouped.setdefault(key, []).append(item)

    entries: List[Dict[str, Any]] = []
    excluded_logs: List[str] = []

    def sort_key_fn(item_tuple):
        cond, sid = item_tuple[0]
        m = re.search(r"\d+", sid)
        num = int(m.group(0)) if m else 9999
        return (cond, num, sid)

    for (cond, base_sub), items in sorted(grouped.items(), key=sort_key_fn):
        # Sort items: canonical (lowest penalty score) first
        items.sort(key=lambda x: (is_canonical_folder(x["subject_dir"]), str(x["subject_dir"])))

        canonical_item = items[0]
        entries.append({
            "condition": cond,
            "subject_id": base_sub,
            "base_subject_id": base_sub,
            "is_duplicate": False,
            "subject_dir": canonical_item["subject_dir"],
            "gt_path": canonical_item["gt_path"],
            "video_path": canonical_item["video_path"],
        })

        # Process any duplicate folders for this subject in this condition
        for dup_idx, dup_item in enumerate(items[1:], start=1):
            if include_duplicates:
                dup_sid = f"{base_sub}_dup{dup_idx}"
                entries.append({
                    "condition": cond,
                    "subject_id": dup_sid,
                    "base_subject_id": base_sub,
                    "is_duplicate": True,
                    "subject_dir": dup_item["subject_dir"],
                    "gt_path": dup_item["gt_path"],
                    "video_path": dup_item["video_path"],
                })
            else:
                # Log exclusion per AGENTS.md rule 4
                reason = f"Excluded duplicate folder artifact (canonical: '{canonical_item['subject_dir'].name}')"
                if exclusions_csv:
                    log_exclusion(
                        subject_id=f"{cond}_{base_sub}_copy{dup_idx}",
                        reason=f"{reason} at {dup_item['subject_dir']}",
                        exclusions_csv=exclusions_csv,
                    )
                excluded_logs.append(
                    f"  [EXCLUDED DUPLICATE] Condition '{cond}': '{base_sub}' duplicate folder '{dup_item['subject_dir'].name}' excluded in favor of canonical '{canonical_item['subject_dir'].name}'"
                )

    if excluded_logs and not include_duplicates:
        print("\nExcluded duplicate folder artifacts (logged to results/exclusions.csv):")
        # Print summary per condition
        cond_counts: Dict[str, int] = {}
        for line in excluded_logs:
            m = re.search(r"Condition '([^']+)':", line)
            if m:
                c = m.group(1)
                cond_counts[c] = cond_counts.get(c, 0) + 1
        for c, cnt in sorted(cond_counts.items()):
            print(f"  • Condition '{c}': {cnt} duplicate folder(s) excluded")

    return entries


def run_worker_task(
    condition: str,
    subject_id: str,
    video_path: Path,
    gt_path: Path,
    output_json: Path,
    config_path: Path,
    git_hash: str,
) -> None:
    """Isolated worker function executing one subject video and saving metrics JSON.
    
    This function is executed inside a fresh subprocess to prevent memory leaks and
    glibc allocator corruption across hundreds of sequential runs.
    """
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

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

    # Import FacePhys model ONLY inside isolated worker subprocess
    from src.models import predict_bvp

    # Load GT (returns ppg, hr, time)
    ppg_gt, hr_gt, t_gt = load_ubfc_ground_truth(gt_path)

    # Run FacePhys inference
    bvp_u, t_u, meta = predict_bvp(
        video_path=video_path,
        timestamps=t_gt,
        fs_target=fs_u,
        model_name="FacePhys.rlap",
    )

    # Save prediction CSV
    clean_cond = re.sub(r"[^\w\-]", "_", condition)
    pred_csv = preds_dir / f"{clean_cond}_{subject_id}_facephys_pred.csv"
    raw_bvp = meta.get("raw_bvp", bvp_u)
    raw_ts = meta.get("raw_timestamps", None)
    if raw_ts is None or len(raw_ts) != len(raw_bvp):
        if len(t_gt) == len(raw_bvp):
            raw_ts = t_gt
        else:
            raw_ts = np.arange(len(raw_bvp)) / fs_u

    min_len = min(len(raw_bvp), len(raw_ts))
    df_pred = pd.DataFrame({
        "frame_idx": np.arange(min_len),
        "gt_timestamp_sec": np.asarray(raw_ts)[:min_len],
        "facephys_bvp_raw": np.asarray(raw_bvp)[:min_len],
    })
    df_pred.to_csv(pred_csv, index=False)

    # Resample GT to uniform time grid
    _, ppg_u = resample_to_uniform(t_gt, ppg_gt, target_fs=fs_u)
    _, hr_line_u = resample_to_uniform(t_gt, hr_gt, target_fs=fs_u, method="linear")

    common_len = min(len(bvp_u), len(ppg_u), len(hr_line_u))
    bvp_u = bvp_u[:common_len]
    ppg_u = ppg_u[:common_len]
    hr_line_u = hr_line_u[:common_len]

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

    row_dict = {
        "git_hash": git_hash,
        "condition": condition,
        "subject_id": subject_id,
        "video_path": str(video_path),
        "resolution": f"{meta.get('video_width', 'unknown')}x{meta.get('video_height', 'unknown')}",
        "face_detected_ratio": meta.get("face_detected_ratio", 1.0),
        "video_hr_pred_bpm": float(hr_pred_vid),
        "primary_ref_hr_bpm": float(primary_ref_hr),
        "secondary_ref_hr_bpm": float(secondary_ref_hr),
        "video_error_vs_primary_bpm": float(abs(hr_pred_vid - primary_ref_hr)),
        "snr_primary_db": float(snr_prim),
        "waveform_lag_max_r": float(lag_res["max_r"]),
        "waveform_optimal_lag_sec": float(lag_res["optimal_lag_sec"]),
        "peak1_bpm": float(peaks[0]["freq_bpm"]) if len(peaks) > 0 else float("nan"),
        "peak1_rel_power_inband": float(peaks[0]["rel_power_inband"]) if len(peaks) > 0 else float("nan"),
        "all_windows_count": int(w_len),
        "all_win_mae_vs_primary": float(mae_all_prim),
        "all_win_medae_vs_primary": float(medae_all_prim),
        "all_win_rmse_vs_primary": float(rmse_all_prim),
        "all_win_r_vs_primary": float(r_all_prim),
        "all_win_acc5_vs_primary": float(acc5_all_prim),
        "reliable_windows_count": int(n_rel),
        "reliable_win_mae_vs_primary": float(mae_rel_prim),
        "reliable_win_medae_vs_primary": float(medae_rel_prim),
        "reliable_win_rmse_vs_primary": float(rmse_rel_prim),
        "reliable_win_r_vs_primary": float(r_rel_prim),
        "reliable_win_acc5_vs_primary": float(acc5_rel_prim),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(row_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FacePhys on UBFC low-light variations.")
    parser.add_argument("--config", type=str, default="configs/colab_lowlight.yaml", help="Path to config file")
    parser.add_argument("--data-dir", type=str, default=None, help="Root directory of UBFC_lowlight")
    parser.add_argument("--original-dir", type=str, default=None, help="Directory of original Data Set 2 if shortcut or stored separately")
    parser.add_argument("--conditions", nargs="+", default=None, help="Filter specific conditions (e.g. lux50 lux30 original_baseline)")
    parser.add_argument("--subjects", nargs="+", default=None, help="Filter specific subjects (e.g. subject1 subject3 subject4)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of subject runs to process")
    parser.add_argument("--no-subprocess", action="store_true", help="Run directly in-process instead of spawning isolated worker processes")
    parser.add_argument("--include-duplicates", action="store_true", help="Include duplicate folder recordings (e.g. subject22 (1)) instead of excluding them")

    # Internal worker mode flags
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-condition", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-subject-id", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-video-path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-gt-path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-json", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    git_hash = get_git_hash()

    # Worker dispatch branch: run single isolated subject and exit immediately.
    # os._exit(0) bypasses Python's interpreter shutdown (GC finalization, atexit
    # handlers, module cleanup) which triggers heap corruption in open-rppg's
    # C extensions / JAX / ONNX runtime when processing large rawvideo frames.
    # The JSON result file is already written; the OS kernel reclaims everything.
    if args.worker:
        run_worker_task(
            condition=args.worker_condition,
            subject_id=args.worker_subject_id,
            video_path=Path(args.worker_video_path),
            gt_path=Path(args.worker_gt_path),
            output_json=Path(args.worker_output_json),
            config_path=PROJECT_ROOT / args.config,
            git_hash=git_hash,
        )
        os._exit(0)

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

    orig_root: Optional[Path] = None
    if args.original_dir:
        orig_root = Path(args.original_dir)
    else:
        cfg_orig = cfg.get("dataset", {}).get("original_dir")
        if cfg_orig and Path(cfg_orig).is_dir():
            orig_root = Path(cfg_orig)
        else:
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

    if not data_root.is_dir():
        print(f"ERROR: Dataset directory '{data_root}' not found!")
        print("In Google Colab, please make sure you ran:")
        print("  from google.colab import drive; drive.mount('/content/drive')")
        print("and added 'UBFC_lowlight' to My Drive.")
        sys.exit(1)

    exclusions_path = PROJECT_ROOT / "results/exclusions.csv"
    entries = find_lowlight_directories(
        root_dir=data_root,
        original_dir=orig_root,
        include_duplicates=args.include_duplicates,
        exclusions_csv=exclusions_path,
    )
    if not entries:
        print(f"No valid subjects with ground_truth.txt and video files found in {data_root}.")
        sys.exit(1)

    # Optional filtering
    if args.conditions:
        entries = [e for e in entries if e["condition"] in args.conditions]
    if args.subjects:
        entries = [e for e in entries if e["subject_id"] in args.subjects or e["base_subject_id"] in args.subjects]
    if args.limit:
        entries = entries[:args.limit]

    # Discovery summary breakdown
    conditions_found = sorted(list(set(e["condition"] for e in entries)))
    print("\n" + "=" * 80)
    print("=== Dataset Discovery Summary ===")
    lux_total = 0
    orig_total = 0
    for cond in conditions_found:
        cond_entries = [e for e in entries if e["condition"] == cond]
        dup_entries = [e for e in cond_entries if e["is_duplicate"]]
        dup_desc = f" ({len(dup_entries)} duplicate recordings: {', '.join(e['subject_id'] for e in dup_entries)})" if dup_entries else ""
        print(f"  • Condition '{cond}': {len(cond_entries)} subjects{dup_desc}")
        if cond == "original_baseline":
            orig_total += len(cond_entries)
        else:
            lux_total += len(cond_entries)

    # Cross-condition consistency check: identify missing subjects per condition
    all_base_subs = sorted(set(e["base_subject_id"] for e in entries))
    cond_subject_map: Dict[str, set] = {}
    for e in entries:
        cond_subject_map.setdefault(e["condition"], set()).add(e["base_subject_id"])

    missing_report: List[str] = []
    for cond in conditions_found:
        cond_subs = cond_subject_map.get(cond, set())
        missing = sorted(set(all_base_subs) - cond_subs)
        if missing and len(cond_subs) < len(all_base_subs):
            missing_report.append(
                f"  ⚠ Condition '{cond}': MISSING {len(missing)} subjects vs full cohort of {len(all_base_subs)}: "
                + ", ".join(missing)
            )

    if missing_report:
        print("\nCross-condition consistency check:")
        for line in missing_report:
            print(line)

    print("-" * 80)
    print(f"Total runs to process: {len(entries)} (Lux: {lux_total}, Original: {orig_total})")
    if len(entries) == 252:
        print("  ✓ Clean cohort confirmed: Exactly 42 subjects x 6 conditions = 252 runs.")
    elif len(entries) == 267:
        print("  ✓ Full dataset with lux duplicates: 225 lux runs (45 x 5) + 42 original = 267 runs.")
    else:
        expected = len(all_base_subs) * len(conditions_found)
        print(f"  ℹ Discovered {len(entries)} runs across {len(conditions_found)} conditions (expected {expected} = {len(all_base_subs)} subjects x {len(conditions_found)} conditions).")
        if missing_report:
            print(f"  ⚠ {len(entries)} < {expected}: see missing subject report above. Check if folders exist in Drive with ground_truth.txt + vid.avi.")
    print("=" * 80 + "\n")

    summary_path = PROJECT_ROOT / cfg.get("outputs", {}).get("summary_csv", "results/ubfc_lowlight_conditions.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: check previously completed runs
    completed_keys = set()
    if summary_path.is_file():
        try:
            df_prev = pd.read_csv(summary_path)
            for _, r in df_prev.iterrows():
                completed_keys.add((str(r["condition"]), str(r["subject_id"])))
            print(f"Resume: Found {len(completed_keys)} already completed runs in {summary_path}")
        except Exception:
            pass

    for idx, item in enumerate(entries, start=1):
        cond = item["condition"]
        sid = item["subject_id"]
        vid_p = item["video_path"]
        gt_p = item["gt_path"]

        if (cond, sid) in completed_keys:
            print(f"[{idx}/{len(entries)}] Skipping already completed Condition='{cond}', Subject='{sid}'")
            continue

        dup_flag = " [Duplicate Entry]" if item["is_duplicate"] else ""
        print(f"\n[{idx}/{len(entries)}] Processing Condition='{cond}', Subject='{sid}'{dup_flag}...")
        print(f"  Video: {vid_p}")

        if args.no_subprocess:
            # In-process execution mode
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_f:
                tmp_json = Path(tmp_f.name)
            try:
                run_worker_task(
                    condition=cond,
                    subject_id=sid,
                    video_path=vid_p,
                    gt_path=gt_p,
                    output_json=tmp_json,
                    config_path=cfg_file,
                    git_hash=git_hash,
                )
                with open(tmp_json, "r", encoding="utf-8") as f:
                    row_dict = json.load(f)
                tmp_json.unlink(missing_ok=True)

                df_row = pd.DataFrame([row_dict])
                if not summary_path.is_file():
                    df_row.to_csv(summary_path, index=False)
                else:
                    df_row.to_csv(summary_path, mode="a", header=False, index=False)

                print(f"  Pred HR: {row_dict['video_hr_pred_bpm']:.2f} BPM | GT Primary: {row_dict['primary_ref_hr_bpm']:.2f} BPM | Error: {row_dict['video_error_vs_primary_bpm']:.2f} BPM")
                print(f"  Window MAE: {row_dict['all_win_mae_vs_primary']:.2f} BPM | MedAE: {row_dict['all_win_medae_vs_primary']:.2f} BPM | Acc<=5BPM: {row_dict['all_win_acc5_vs_primary']:.1f}% | SNR: {row_dict['snr_primary_db']:.2f} dB")
            except Exception as e:
                err_text = traceback.format_exc()
                print(f"  ERROR: In-process run failed for Condition='{cond}', Subject='{sid}': {e}")
                log_exclusion(f"{cond}_{sid}", f"Execution failed: {str(e)}", exclusions_path)
            continue

        # Default isolated subprocess execution mode (memory-safe, prevents malloc corruption)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_f:
            tmp_json = Path(tmp_f.name)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--worker-condition", str(cond),
            "--worker-subject-id", str(sid),
            "--worker-video-path", str(vid_p),
            "--worker-gt-path", str(gt_p),
            "--worker-output-json", str(tmp_json),
            "--config", str(args.config),
        ]

        worker_env = os.environ.copy()
        worker_env["OMP_NUM_THREADS"] = "1"
        worker_env["OPENBLAS_NUM_THREADS"] = "1"
        worker_env["MKL_NUM_THREADS"] = "1"
        worker_env["VECLIB_MAXIMUM_THREADS"] = "1"
        worker_env["NUMEXPR_NUM_THREADS"] = "1"
        worker_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        worker_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.20"

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=worker_env)
            if proc.returncode == 0 and tmp_json.is_file():
                with open(tmp_json, "r", encoding="utf-8") as f:
                    row_dict = json.load(f)
                tmp_json.unlink(missing_ok=True)

                df_row = pd.DataFrame([row_dict])
                if not summary_path.is_file():
                    df_row.to_csv(summary_path, index=False)
                else:
                    df_row.to_csv(summary_path, mode="a", header=False, index=False)

                print(f"  Pred HR: {row_dict['video_hr_pred_bpm']:.2f} BPM | GT Primary: {row_dict['primary_ref_hr_bpm']:.2f} BPM | Error: {row_dict['video_error_vs_primary_bpm']:.2f} BPM")
                print(f"  Window MAE: {row_dict['all_win_mae_vs_primary']:.2f} BPM | MedAE: {row_dict['all_win_medae_vs_primary']:.2f} BPM | Acc<=5BPM: {row_dict['all_win_acc5_vs_primary']:.1f}% | SNR: {row_dict['snr_primary_db']:.2f} dB")
            else:
                err_msg = (proc.stderr or proc.stdout or "Unknown subprocess crash").strip()
                print(f"  ERROR: Subprocess worker failed for Condition='{cond}', Subject='{sid}' (exit code {proc.returncode}):")
                print(f"  {err_msg[-400:]}")
                log_exclusion(
                    subject_id=f"{cond}_{sid}",
                    reason=f"Worker process failed (exit {proc.returncode}): {err_msg[-150:]}",
                    exclusions_csv=exclusions_path,
                )
                tmp_json.unlink(missing_ok=True)
        except Exception as exc:
            print(f"  ERROR: Subprocess launch failed: {exc}")
            log_exclusion(
                subject_id=f"{cond}_{sid}",
                reason=f"Subprocess launch exception: {exc}",
                exclusions_csv=exclusions_path,
            )
            tmp_json.unlink(missing_ok=True)

    # Print final cohort summaries
    if summary_path.is_file():
        df_out = pd.read_csv(summary_path)
        print("\n" + "=" * 80)
        print(f"Total evaluated runs in summary ({summary_path}): {len(df_out)}")
        print("=" * 80)

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
    else:
        print("No results recorded.")


if __name__ == "__main__":
    main()
