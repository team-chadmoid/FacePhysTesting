"""Cross-check FacePhys against library path and run spatial diagnostics on subject4.

Evaluates:
(1) Library path cross-check: model.process_video(path) vs update_frame path
    - Compare raw BVPs (correlation)
    - Compare video HRs
    - Record exact timestamps passed to update_frame vs process_video
(2) Subject4 spatial tests:
    - Upscale to 640x480
    - Pad with border so face is not edge-to-edge
    - Compare Welch PSD spectra against raw and GT PPG
(3) Spectrum-level metrics:
    - Top-3 spectral peaks with relative power
    - SNR around reference HR
    - Waveform correlation with GT PPG after lag search
    - Fraction of windows within 5 BPM
(4) Error distribution:
    - Report median window error (MedAE) alongside MAE
    - No hyperparameter tuning
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
from src.io import find_ubfc_subjects, load_ubfc_subject, resample_to_uniform
from src.models import FacePhysWrapper, predict_bvp
import rppg


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
    print("=== Running FacePhys Cross-Check & Extended Diagnostics ===")
    print(f"Git commit hash: {git_hash}")

    config_path = PROJECT_ROOT / "configs" / "diagnostics_and_crosscheck.yaml"
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    target_fs = float(cfg.get("preprocessing", {}).get("target_fs", 30.0))
    lowcut = float(cfg.get("preprocessing", {}).get("bandpass", {}).get("lowcut", 0.7))
    highcut = float(cfg.get("preprocessing", {}).get("bandpass", {}).get("highcut", 3.0))
    order = int(cfg.get("preprocessing", {}).get("bandpass", {}).get("order", 3))
    win_sec = float(cfg.get("preprocessing", {}).get("windowing", {}).get("window_sec", 30.0))
    stride_sec = float(cfg.get("preprocessing", {}).get("windowing", {}).get("stride_sec", 1.0))
    threshold_bpm = float(cfg.get("metrics", {}).get("threshold_bpm", 5.0))
    max_lag_sec = float(cfg.get("metrics", {}).get("lag_search_max_sec", 2.0))
    snr_delta_hz = float(cfg.get("metrics", {}).get("snr_delta_hz", 0.1))

    dataset_root = PROJECT_ROOT / cfg.get("dataset", {}).get("root_dir", "data/ubfc/Data Set 2")
    subjects_map = find_ubfc_subjects(dataset_root)

    target_subject_ids = cfg.get("dataset", {}).get("subjects", ["subject1", "subject3", "subject4"])
    print(f"Target subjects: {target_subject_ids}")

    # =========================================================================
    # PART 1: Cross-check against library's own path: model.process_video(path)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: Library Path Cross-Check (model.process_video vs update_frame)")
    print("=" * 80)

    crosscheck_rows = []

    for sid in target_subject_ids:
        sdir = subjects_map[sid]
        sdata = load_ubfc_subject(sdir, subject_id=sid)
        vid_path = sdata.video_path
        res_str = (
            f"{sdata.video_metadata['width']}x{sdata.video_metadata['height']}"
            if sdata.video_metadata
            else "unknown"
        )
        print(f"\n--> Running model.process_video on {sid} ({vid_path})...")

        # 1. Run open-rppg model.process_video(vid_path)
        m_lib = rppg.Model(model="FacePhys.rlap")
        lib_hr_dict = m_lib.process_video(str(vid_path))
        lib_hr = float(lib_hr_dict["hr"]) if lib_hr_dict and "hr" in lib_hr_dict and lib_hr_dict["hr"] is not None else float("nan")
        lib_sqi = float(lib_hr_dict["SQI"]) if lib_hr_dict and "SQI" in lib_hr_dict and lib_hr_dict["SQI"] is not None else float("nan")
        bvp_lib_raw, ts_lib_raw = m_lib.bvp(raw=True)
        bvp_lib_raw = np.asarray(bvp_lib_raw, dtype=np.float64)
        ts_lib_arr = np.asarray(ts_lib_raw, dtype=np.float64)
        stats_lib = m_lib.statistic

        # 2. Load our update_frame predictions
        pred_csv = PROJECT_ROOT / "results" / "preds" / f"{sid}_facephys_pred.csv"
        df_update = pd.read_csv(pred_csv)
        bvp_update_raw = df_update["facephys_bvp_raw"].to_numpy()
        t_gt = df_update["gt_timestamp_sec"].to_numpy()
        t_uniform, bvp_update_uniform = resample_to_uniform(t_gt, bvp_update_raw, target_fs=target_fs)

        # 3. Analyze timestamps passed to update_frame vs process_video
        # Our update_frame path passed: ts = i / fps (uniform 0, 1/30, 2/30, ...)
        fps_assumed = 30.0
        n_frames = len(bvp_update_raw)
        ts_passed_update_frame = np.arange(n_frames, dtype=float) / fps_assumed

        # In process_video, timestamps were decoded from container:
        # Note: stats_lib['filled'] indicates if open-rppg inserted forward-filled frames
        ts_lib_arr = np.asarray(ts_lib_raw) if len(ts_lib_raw) > 0 else np.array([])

        # Calculate time deviation between container decoded timestamps and uniform grid
        if len(ts_lib_arr) == n_frames:
            max_ts_diff = float(np.max(np.abs(ts_lib_arr - ts_passed_update_frame)))
            mean_ts_diff = float(np.mean(np.abs(ts_lib_arr - ts_passed_update_frame)))
        else:
            # Different length due to filled/skipped frames in open-rppg
            max_ts_diff = float("nan")
            mean_ts_diff = float("nan")

        # 4. Correlation between BVPs
        # If lengths match:
        if len(bvp_lib_raw) == len(bvp_update_raw):
            corr_raw = calc_pearson_r(bvp_update_raw, bvp_lib_raw)
        else:
            # Resample lib_bvp onto the uniform time grid of update_frame
            t_lib_valid = np.linspace(0, ts_lib_arr[-1] if len(ts_lib_arr) > 0 else 1.0, len(bvp_lib_raw))
            _, bvp_lib_resamp = resample_to_uniform(t_lib_valid, bvp_lib_raw, target_fs=target_fs)
            min_l = min(len(bvp_update_uniform), len(bvp_lib_resamp))
            corr_raw = calc_pearson_r(bvp_update_uniform[:min_l], bvp_lib_resamp[:min_l])

        # 5. Welch video HR on library BVP
        # Resample library BVP to uniform grid
        if len(ts_lib_arr) > 1:
            t_u_lib, bvp_lib_u = resample_to_uniform(ts_lib_arr, bvp_lib_raw, target_fs=target_fs)
            hr_lib_welch = estimate_hr_video(bvp_lib_u, fs=target_fs)
        else:
            hr_lib_welch = estimate_hr_video(bvp_lib_raw, fs=target_fs)

        # Welch video HR on our update_frame BVP
        hr_update_welch = estimate_hr_video(bvp_update_uniform, fs=target_fs)

        # Ground truth references
        _, ppg_u = resample_to_uniform(sdata.time, sdata.ppg, target_fs=target_fs)
        _, hr_line_u = resample_to_uniform(sdata.time, sdata.hr, target_fs=target_fs, method="linear")
        primary_ref_hr = estimate_hr_video(ppg_u, fs=target_fs)
        secondary_ref_hr = float(np.mean(hr_line_u))

        print(f"  process_video HR: {lib_hr:.2f} BPM, SQI: {lib_sqi:.3f}")
        print(f"  update_frame Welch HR: {hr_update_welch:.2f} BPM")
        print(f"  process_video Welch HR: {hr_lib_welch:.2f} BPM")
        print(f"  Primary Ref HR (GT Welch): {primary_ref_hr:.2f} BPM")
        print(f"  Secondary Ref HR (GT Line): {secondary_ref_hr:.2f} BPM")
        print(f"  BVP Correlation (lib vs update): r = {corr_raw:.4f}")
        print(f"  open-rppg stats: {stats_lib}")
        print(f"  update_frame timestamp passed: uniform i/30.0 s (range: [0.0, {ts_passed_update_frame[-1]:.3f}] s)")
        print(f"  process_video timestamp passed: container frame.time (range: [0.0, {ts_lib_arr[-1] if len(ts_lib_arr)>0 else 0.0:.3f}] s)")

        crosscheck_rows.append({
            "git_hash": git_hash,
            "subject_id": sid,
            "resolution": res_str,
            "frames_update_path": len(bvp_update_raw),
            "frames_lib_process_video": len(bvp_lib_raw),
            "lib_stat_key_frames": stats_lib.get("key", 0),
            "lib_stat_filled_frames": stats_lib.get("filled", 0),
            "lib_stat_skipped_frames": stats_lib.get("skipped", 0),
            "lib_stat_null_frames": stats_lib.get("null", 0),
            "timestamp_passed_to_update_frame": "uniform_i_div_30",
            "timestamp_passed_by_process_video": "pyav_frame_time",
            "bvp_correlation_lib_vs_update": corr_raw,
            "hr_lib_builtin_bpm": lib_hr,
            "hr_lib_welch_bpm": hr_lib_welch,
            "hr_update_frame_welch_bpm": hr_update_welch,
            "primary_ref_hr_bpm": primary_ref_hr,
            "secondary_ref_hr_bpm": secondary_ref_hr,
            "error_update_vs_primary_bpm": abs(hr_update_welch - primary_ref_hr),
            "error_lib_welch_vs_primary_bpm": abs(hr_lib_welch - primary_ref_hr),
            "error_lib_builtin_vs_primary_bpm": abs(lib_hr - primary_ref_hr),
        })

    df_crosscheck = pd.DataFrame(crosscheck_rows)
    crosscheck_out = PROJECT_ROOT / cfg.get("outputs", {}).get("crosscheck_csv", "results/crosscheck_process_video.csv")
    crosscheck_out.parent.mkdir(parents=True, exist_ok=True)
    df_crosscheck.to_csv(crosscheck_out, index=False)
    print(f"\nSaved cross-check results to {crosscheck_out}")

    # =========================================================================
    # PART 2: Subject 4 Spatial Tests (Upscale to 640x480 & Border Padding)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: Subject4 Spatial Experiments (Upscale vs Border Padding)")
    print("=" * 80)

    s4_dir = subjects_map["subject4"]
    s4_data = load_ubfc_subject(s4_dir, subject_id="subject4")
    _, s4_ppg_u = resample_to_uniform(s4_data.time, s4_data.ppg, target_fs=target_fs)
    _, s4_hr_line_u = resample_to_uniform(s4_data.time, s4_data.hr, target_fs=target_fs, method="linear")
    s4_primary_ref_hr = estimate_hr_video(s4_ppg_u, fs=target_fs)
    s4_secondary_ref_hr = float(np.mean(s4_hr_line_u))

    # Read original frames of subject4
    cap = cv2.VideoCapture(str(s4_data.video_path))
    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    n_s4_frames = len(raw_frames)
    orig_h, orig_w = raw_frames[0].shape[:2]
    print(f"Loaded {n_s4_frames} frames of subject4 (original size: {orig_w}x{orig_h})")

    # Load baseline subject4 predictions
    s4_base_csv = PROJECT_ROOT / "results" / "preds" / "subject4_facephys_pred.csv"
    df_s4_base = pd.read_csv(s4_base_csv)
    _, bvp_s4_baseline = resample_to_uniform(
        df_s4_base["gt_timestamp_sec"].to_numpy(),
        df_s4_base["facephys_bvp_raw"].to_numpy(),
        target_fs=target_fs,
    )

    spatial_results = {}

    # 1. Baseline
    hr_base_vid, f_base, psd_base = estimate_hr_welch(
        butterworth_bandpass_filter(bvp_s4_baseline, fs=target_fs, lowcut=lowcut, highcut=highcut, order=order),
        fs=target_fs,
        low_bpm=lowcut * 60.0,
        high_bpm=highcut * 60.0,
        nfft=16384,
    )
    spatial_results["baseline_242x242"] = {
        "bvp": bvp_s4_baseline,
        "hr_video": hr_base_vid,
        "freqs": f_base,
        "psd": psd_base,
        "resolution": f"{orig_w}x{orig_h}",
        "description": "Original raw video (242x242, face edge-to-edge)",
    }

    # 2. Upscale to 640x480
    print("\n--> Running FacePhys on Subject 4 upscaled to 640x480 (INTER_CUBIC)...")
    upscaled_frames = [
        cv2.resize(f, (640, 480), interpolation=cv2.INTER_CUBIC)
        for f in raw_frames
    ]
    upscale_model = FacePhysWrapper()
    bvp_upscaled_u, _, meta_up = upscale_model.predict_from_frames(
        upscaled_frames, timestamps=s4_data.time[:n_s4_frames], fs_target=target_fs
    )
    hr_upscaled_vid, f_up, psd_up = estimate_hr_welch(
        butterworth_bandpass_filter(bvp_upscaled_u, fs=target_fs, lowcut=lowcut, highcut=highcut, order=order),
        fs=target_fs,
        low_bpm=lowcut * 60.0,
        high_bpm=highcut * 60.0,
        nfft=16384,
    )
    spatial_results["upscaled_640x480"] = {
        "bvp": bvp_upscaled_u,
        "hr_video": hr_upscaled_vid,
        "freqs": f_up,
        "psd": psd_up,
        "resolution": "640x480",
        "description": "Bicubic upscaled to 640x480",
        "null_frames": meta_up.get("face_null_frames", 0),
    }
    print(f"  Upscaled video HR: {hr_upscaled_vid:.2f} BPM (error vs primary: {abs(hr_upscaled_vid - s4_primary_ref_hr):.2f} BPM)")

    # 3. Pad with border so face is not edge-to-edge
    pad_px = int(cfg.get("subject4_experiments", {}).get("pad_border_pixels", 60))
    print(f"\n--> Running FacePhys on Subject 4 padded with border (+{pad_px}px on all sides)...")
    padded_frames = [
        cv2.copyMakeBorder(f, pad_px, pad_px, pad_px, pad_px, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        for f in raw_frames
    ]
    new_h, new_w = padded_frames[0].shape[:2]
    print(f"  Padded frame size: {new_w}x{new_h}")

    pad_model = FacePhysWrapper()
    bvp_padded_u, _, meta_pad = pad_model.predict_from_frames(
        padded_frames, timestamps=s4_data.time[:n_s4_frames], fs_target=target_fs
    )
    hr_padded_vid, f_pad, psd_pad = estimate_hr_welch(
        butterworth_bandpass_filter(bvp_padded_u, fs=target_fs, lowcut=lowcut, highcut=highcut, order=order),
        fs=target_fs,
        low_bpm=lowcut * 60.0,
        high_bpm=highcut * 60.0,
        nfft=16384,
    )
    spatial_results["padded_border"] = {
        "bvp": bvp_padded_u,
        "hr_video": hr_padded_vid,
        "freqs": f_pad,
        "psd": psd_pad,
        "resolution": f"{new_w}x{new_h}",
        "description": f"Border padded +{pad_px}px constant margin ({new_w}x{new_h})",
        "null_frames": meta_pad.get("face_null_frames", 0),
    }
    print(f"  Padded video HR: {hr_padded_vid:.2f} BPM (error vs primary: {abs(hr_padded_vid - s4_primary_ref_hr):.2f} BPM)")

    # GT PPG Spectrum for subject4
    _, f_gt, psd_gt = estimate_hr_welch(
        butterworth_bandpass_filter(s4_ppg_u, fs=target_fs, lowcut=lowcut, highcut=highcut, order=order),
        fs=target_fs,
        low_bpm=lowcut * 60.0,
        high_bpm=highcut * 60.0,
        nfft=16384,
    )

    # Plot Spectra Comparison for Subject 4
    fig, ax = plt.subplots(figsize=(10, 6))
    mask_band = (f_gt >= lowcut) & (f_gt <= highcut)
    bpm_axis = f_gt[mask_band] * 60.0

    # Normalize PSDs to peak = 1.0 for visual comparison
    ax.plot(bpm_axis, psd_gt[mask_band] / np.max(psd_gt[mask_band]), "k-", linewidth=2.0, label=f"GT PPG (Peak={s4_primary_ref_hr:.2f} BPM)")
    ax.plot(f_base[mask_band] * 60.0, psd_base[mask_band] / np.max(psd_base[mask_band]), "r--", linewidth=1.5, label=f"Baseline 242x242 (Peak={hr_base_vid:.2f} BPM)")
    ax.plot(f_up[mask_band] * 60.0, psd_up[mask_band] / np.max(psd_up[mask_band]), "b-.", linewidth=1.5, label=f"Upscaled 640x480 (Peak={hr_upscaled_vid:.2f} BPM)")
    ax.plot(f_pad[mask_band] * 60.0, psd_pad[mask_band] / np.max(psd_pad[mask_band]), "g:", linewidth=1.8, label=f"Border Padded {new_w}x{new_h} (Peak={hr_padded_vid:.2f} BPM)")

    ax.axvline(s4_primary_ref_hr, color="black", linestyle="--", alpha=0.5, label=f"GT Primary Ref: {s4_primary_ref_hr:.1f} BPM")
    ax.axvline(s4_secondary_ref_hr, color="gray", linestyle=":", alpha=0.5, label=f"GT Oximeter Line: {s4_secondary_ref_hr:.1f} BPM")

    ax.set_title("Subject 4: Normalized PSD Spectra Across Spatial Transformations\n(Baseline vs Upscaled 640x480 vs Border-Padded vs GT PPG)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Heart Rate (BPM)", fontsize=10)
    ax.set_ylabel("Normalized PSD (Peak = 1.0)", fontsize=10)
    ax.set_xlim(lowcut * 60.0, highcut * 60.0)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    s4_plot_out = PROJECT_ROOT / cfg.get("outputs", {}).get("subject4_spectra_plot", "results/plots/subject4_spectra_comparison.png")
    s4_plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(s4_plot_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Subject 4 spectra comparison plot to {s4_plot_out}")

    # Build Subject 4 Spatial Comparison Table
    s4_rows = []
    for cond_name, c_data in spatial_results.items():
        bvp_c = c_data["bvp"]
        # Windowed evaluation against GT references
        _, hr_c_win = estimate_hr_windowed(bvp_c, fs=target_fs, window_sec=win_sec, stride_sec=stride_sec)
        _, hr_gt_win = estimate_hr_windowed(s4_ppg_u, fs=target_fs, window_sec=win_sec, stride_sec=stride_sec)

        # Truncate to common window length
        w_min = min(len(hr_c_win), len(hr_gt_win))
        hr_c_win = hr_c_win[:w_min]
        hr_gt_win = hr_gt_win[:w_min]

        mae_c = calc_mae(hr_gt_win, hr_c_win)
        medae_c = calc_medae(hr_gt_win, hr_c_win)
        rmse_c = calc_rmse(hr_gt_win, hr_c_win)
        r_c = calc_pearson_r(hr_gt_win, hr_c_win)
        acc5_c = calc_fraction_within_threshold(hr_gt_win, hr_c_win, threshold=5.0)

        # Lag search waveform correlation with GT PPG
        corr_res = max_lag_waveform_correlation(bvp_c, s4_ppg_u, fs=target_fs, max_lag_sec=max_lag_sec)
        # SNR
        snr_c = calculate_snr(c_data["psd"], c_data["freqs"], ref_hr_bpm=s4_primary_ref_hr, delta_hz=snr_delta_hz)
        # Top-3 peaks
        peaks_c = get_top_spectral_peaks(c_data["psd"], c_data["freqs"], n_peaks=3)

        s4_rows.append({
            "git_hash": git_hash,
            "condition": cond_name,
            "resolution": c_data["resolution"],
            "description": c_data["description"],
            "video_hr_pred_bpm": c_data["hr_video"],
            "primary_ref_hr_bpm": s4_primary_ref_hr,
            "video_error_bpm": abs(c_data["hr_video"] - s4_primary_ref_hr),
            "window_count": w_min,
            "window_mae_bpm": mae_c,
            "window_medae_bpm": medae_c,
            "window_rmse_bpm": rmse_c,
            "window_pearson_r": r_c,
            "fraction_within_5bpm": acc5_c,
            "waveform_lag_max_r": corr_res["max_r"],
            "waveform_optimal_lag_sec": corr_res["optimal_lag_sec"],
            "snr_db": snr_c,
            "peak1_bpm": peaks_c[0]["freq_bpm"] if len(peaks_c) > 0 else float("nan"),
            "peak1_rel_power_inband": peaks_c[0]["rel_power_inband"] if len(peaks_c) > 0 else float("nan"),
            "peak2_bpm": peaks_c[1]["freq_bpm"] if len(peaks_c) > 1 else float("nan"),
            "peak2_rel_power_inband": peaks_c[1]["rel_power_inband"] if len(peaks_c) > 1 else float("nan"),
            "peak3_bpm": peaks_c[2]["freq_bpm"] if len(peaks_c) > 2 else float("nan"),
            "peak3_rel_power_inband": peaks_c[2]["rel_power_inband"] if len(peaks_c) > 2 else float("nan"),
        })

    df_s4 = pd.DataFrame(s4_rows)
    s4_csv_out = PROJECT_ROOT / cfg.get("outputs", {}).get("subject4_experiments_csv", "results/subject4_spatial_experiments.csv")
    df_s4.to_csv(s4_csv_out, index=False)
    print(f"Saved Subject 4 spatial experiment summary to {s4_csv_out}")

    # =========================================================================
    # PART 3 & 4: Spectrum-Level Metrics & Median Window Error for All Subjects
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3 & 4: Spectrum-Level Metrics & Median Error for All Subjects")
    print("=" * 80)

    extended_rows = []

    for sid in target_subject_ids:
        sdir = subjects_map[sid]
        s_data = load_ubfc_subject(sdir, subject_id=sid)
        res_str = (
            f"{s_data.video_metadata['width']}x{s_data.video_metadata['height']}"
            if s_data.video_metadata
            else "unknown"
        )
        _, ppg_u = resample_to_uniform(s_data.time, s_data.ppg, target_fs=target_fs)
        _, hr_line_u = resample_to_uniform(s_data.time, s_data.hr, target_fs=target_fs, method="linear")

        # Load baseline predictions
        pred_csv = PROJECT_ROOT / "results" / "preds" / f"{sid}_facephys_pred.csv"
        df_pred = pd.read_csv(pred_csv)
        _, bvp_u = resample_to_uniform(
            df_pred["gt_timestamp_sec"].to_numpy(),
            df_pred["facephys_bvp_raw"].to_numpy(),
            target_fs=target_fs,
        )

        # Compute video-level HR and PSD
        hr_pred_vid, f_pred, psd_pred = estimate_hr_welch(
            butterworth_bandpass_filter(bvp_u, fs=target_fs, lowcut=lowcut, highcut=highcut, order=order),
            fs=target_fs,
            low_bpm=lowcut * 60.0,
            high_bpm=highcut * 60.0,
            nfft=16384,
        )
        primary_ref_hr = estimate_hr_video(ppg_u, fs=target_fs)
        secondary_ref_hr = float(np.mean(hr_line_u))

        # 1. Top-3 spectral peaks
        peaks = get_top_spectral_peaks(psd_pred, f_pred, n_peaks=3, band=(lowcut, highcut))

        # 2. SNR around reference HR
        snr_primary = calculate_snr(psd_pred, f_pred, ref_hr_bpm=primary_ref_hr, delta_hz=snr_delta_hz, harmonic=True)
        snr_secondary = calculate_snr(psd_pred, f_pred, ref_hr_bpm=secondary_ref_hr, delta_hz=snr_delta_hz, harmonic=True)

        # 3. Waveform correlation with GT PPG after lag search
        lag_corr = max_lag_waveform_correlation(bvp_u, ppg_u, fs=target_fs, max_lag_sec=max_lag_sec)

        # 4. Windowed calculations
        win_times, hr_pred_win = estimate_hr_windowed(bvp_u, fs=target_fs, window_sec=win_sec, stride_sec=stride_sec)
        _, hr_primary_win = estimate_hr_windowed(ppg_u, fs=target_fs, window_sec=win_sec, stride_sec=stride_sec)

        # Secondary windowed (mean oximeter line)
        win_samples = int(round(win_sec * target_fs))
        stride_samples = int(round(stride_sec * target_fs))
        hr_secondary_win = []
        for start in range(0, len(hr_line_u) - win_samples + 1, stride_samples):
            hr_secondary_win.append(float(np.mean(hr_line_u[start:start + win_samples])))
        hr_secondary_win = np.array(hr_secondary_win)

        w_len = min(len(hr_pred_win), len(hr_primary_win), len(hr_secondary_win))
        hr_p = hr_pred_win[:w_len]
        hr_prim = hr_primary_win[:w_len]
        hr_sec = hr_secondary_win[:w_len]

        # All windows
        mae_all_prim = calc_mae(hr_prim, hr_p)
        medae_all_prim = calc_medae(hr_prim, hr_p)
        rmse_all_prim = calc_rmse(hr_prim, hr_p)
        r_all_prim = calc_pearson_r(hr_prim, hr_p)
        acc5_all_prim = calc_fraction_within_threshold(hr_prim, hr_p, threshold=threshold_bpm)

        mae_all_sec = calc_mae(hr_sec, hr_p)
        medae_all_sec = calc_medae(hr_sec, hr_p)
        rmse_all_sec = calc_rmse(hr_sec, hr_p)
        r_all_sec = calc_pearson_r(hr_sec, hr_p)
        acc5_all_sec = calc_fraction_within_threshold(hr_sec, hr_p, threshold=threshold_bpm)

        # Reference-reliable windows (|Primary - Secondary| <= 5.0 BPM)
        rel_mask = np.abs(hr_prim - hr_sec) <= threshold_bpm
        n_rel = int(np.sum(rel_mask))

        if n_rel > 0:
            hr_p_rel = hr_p[rel_mask]
            hr_prim_rel = hr_prim[rel_mask]
            hr_sec_rel = hr_sec[rel_mask]

            mae_rel_prim = calc_mae(hr_prim_rel, hr_p_rel)
            medae_rel_prim = calc_medae(hr_prim_rel, hr_p_rel)
            rmse_rel_prim = calc_rmse(hr_prim_rel, hr_p_rel)
            r_rel_prim = calc_pearson_r(hr_prim_rel, hr_p_rel)
            acc5_rel_prim = calc_fraction_within_threshold(hr_prim_rel, hr_p_rel, threshold=threshold_bpm)

            mae_rel_sec = calc_mae(hr_sec_rel, hr_p_rel)
            medae_rel_sec = calc_medae(hr_sec_rel, hr_p_rel)
            rmse_rel_sec = calc_rmse(hr_sec_rel, hr_p_rel)
            r_rel_sec = calc_pearson_r(hr_sec_rel, hr_p_rel)
            acc5_rel_sec = calc_fraction_within_threshold(hr_sec_rel, hr_p_rel, threshold=threshold_bpm)
        else:
            mae_rel_prim = medae_rel_prim = rmse_rel_prim = r_rel_prim = acc5_rel_prim = float("nan")
            mae_rel_sec = medae_rel_sec = rmse_rel_sec = r_rel_sec = acc5_rel_sec = float("nan")

        extended_rows.append({
            "git_hash": git_hash,
            "subject_id": sid,
            "resolution": res_str,
            "video_hr_pred_bpm": hr_pred_vid,
            "primary_ref_hr_bpm": primary_ref_hr,
            "secondary_ref_hr_bpm": secondary_ref_hr,
            "video_error_vs_primary_bpm": abs(hr_pred_vid - primary_ref_hr),
            "video_error_vs_secondary_bpm": abs(hr_pred_vid - secondary_ref_hr),
            # Spectral Peaks
            "peak1_freq_bpm": peaks[0]["freq_bpm"] if len(peaks) > 0 else float("nan"),
            "peak1_rel_power_inband": peaks[0]["rel_power_inband"] if len(peaks) > 0 else float("nan"),
            "peak1_rel_power_max": peaks[0]["rel_power_max"] if len(peaks) > 0 else float("nan"),
            "peak2_freq_bpm": peaks[1]["freq_bpm"] if len(peaks) > 1 else float("nan"),
            "peak2_rel_power_inband": peaks[1]["rel_power_inband"] if len(peaks) > 1 else float("nan"),
            "peak2_rel_power_max": peaks[1]["rel_power_max"] if len(peaks) > 1 else float("nan"),
            "peak3_freq_bpm": peaks[2]["freq_bpm"] if len(peaks) > 2 else float("nan"),
            "peak3_rel_power_inband": peaks[2]["rel_power_inband"] if len(peaks) > 2 else float("nan"),
            "peak3_rel_power_max": peaks[2]["rel_power_max"] if len(peaks) > 2 else float("nan"),
            # SNR
            "snr_around_primary_ref_db": snr_primary,
            "snr_around_secondary_ref_db": snr_secondary,
            # Waveform lag correlation
            "waveform_lag_max_r": lag_corr["max_r"],
            "waveform_optimal_lag_sec": lag_corr["optimal_lag_sec"],
            "waveform_max_abs_r": lag_corr["max_abs_r"],
            # Windows counts
            "all_windows_count": w_len,
            "reliable_windows_count": n_rel,
            # All Windows Metrics vs Primary Ref
            "all_win_mae_vs_primary": mae_all_prim,
            "all_win_medae_vs_primary": medae_all_prim,
            "all_win_rmse_vs_primary": rmse_all_prim,
            "all_win_r_vs_primary": r_all_prim,
            "all_win_acc5_vs_primary": acc5_all_prim,
            # Reliable Windows Metrics vs Primary Ref
            "reliable_win_mae_vs_primary": mae_rel_prim,
            "reliable_win_medae_vs_primary": medae_rel_prim,
            "reliable_win_rmse_vs_primary": rmse_rel_prim,
            "reliable_win_r_vs_primary": r_rel_prim,
            "reliable_win_acc5_vs_primary": acc5_rel_prim,
            # All Windows Metrics vs Secondary Ref
            "all_win_mae_vs_secondary": mae_all_sec,
            "all_win_medae_vs_secondary": medae_all_sec,
            "all_win_rmse_vs_secondary": rmse_all_sec,
            "all_win_r_vs_secondary": r_all_sec,
            "all_win_acc5_vs_secondary": acc5_all_sec,
            # Reliable Windows Metrics vs Secondary Ref
            "reliable_win_mae_vs_secondary": mae_rel_sec,
            "reliable_win_medae_vs_secondary": medae_rel_sec,
            "reliable_win_rmse_vs_secondary": rmse_rel_sec,
            "reliable_win_r_vs_secondary": r_rel_sec,
            "reliable_win_acc5_vs_secondary": acc5_rel_sec,
        })

    df_ext = pd.DataFrame(extended_rows)
    ext_csv_out = PROJECT_ROOT / cfg.get("outputs", {}).get("spectrum_window_metrics_csv", "results/spectrum_and_window_metrics.csv")
    df_ext.to_csv(ext_csv_out, index=False)
    print(f"\nSaved extended spectrum and window metrics to {ext_csv_out}")

    print("\n=== All Tests Completed Successfully ===")


if __name__ == "__main__":
    main()
