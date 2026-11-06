"""Run FacePhys baseline evaluation on UBFC-RPPG Dataset 2 subjects.

Protocol and execution:
- Pretrained FacePhys from open-rppg (FacePhys.rlap).
- Indexes predicted BVP per frame, maps frame i to GT timestamp i,
  and resamples both prediction and reference to uniform 30 Hz grid.
- Primary Reference: Welch PPG-derived HR.
- Secondary Reference: Oximeter HR line.
- Diagnostic Only: Peak-detection HR (hr_from_peaks).
- Reports video-level HR error against both references.
- Reports windowed MAE, RMSE, Pearson r on all windows and on reference-reliable windows (|Primary - Secondary| <= 5 BPM).
- Records package versions, Welch parameters, video resolution, and face-detection success.
- Saves predictions to results/preds/ and writes results/ubfc_facephys.csv.
- Plots across-subject Bland-Altman (axes: 'model - reference').
"""

from importlib import metadata as importlib_metadata
from pathlib import Path
import subprocess
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.io import (
    find_ubfc_subjects,
    load_ubfc_subject,
    check_ubfc_alignment,
    resample_to_uniform,
)
from src.hr import (
    estimate_hr_from_bvp,
    estimate_hr_video,
    estimate_hr_windowed,
    hr_from_peaks,
)
from src.eval import calc_metrics, calc_bland_altman
from src.models import predict_bvp


def get_git_hash() -> str:
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
    return "unknown"


def get_package_versions() -> str:
    pkgs = ["open-rppg", "onnxruntime", "keras", "scipy", "numpy", "opencv-python", "av"]
    versions = []
    for p in pkgs:
        try:
            v = importlib_metadata.version(p)
            versions.append(f"{p}=={v}")
        except Exception:
            pass
    return "; ".join(versions)


def main():
    config_path = repo_root / "configs" / "facephys_baseline.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["experiment"]["seed"]
    np.random.seed(seed)
    git_hash = get_git_hash()
    pkg_versions_str = get_package_versions()

    print(f"=== Running FacePhys Baseline Evaluation ===")
    print(f"Git commit: {git_hash}")
    print(f"Package versions: {pkg_versions_str}\n")

    data_root = repo_root / cfg["dataset"]["root"]
    subjects_map = find_ubfc_subjects(data_root)
    target_subjects = cfg["dataset"]["subjects"]

    fs_uniform = float(cfg["parameters"]["fs_uniform"])
    win_sec = float(cfg["parameters"]["window_sec"])
    stride_sec = float(cfg["parameters"]["stride_sec"])
    lowcut = float(cfg["parameters"]["bandpass_low"])
    highcut = float(cfg["parameters"]["bandpass_high"])
    filter_order = int(cfg["parameters"]["filter_order"])
    nfft_welch = int(cfg["parameters"]["nfft_welch"])
    nfft_vid = int(cfg["parameters"]["nfft_video"])
    reliability_threshold = float(cfg["parameters"]["reliability_threshold_bpm"])

    welch_param_str = (
        f"fs={fs_uniform}Hz, win={win_sec}s, stride={stride_sec}s, "
        f"bandpass=[{lowcut}, {highcut}]Hz, order={filter_order}, "
        f"nfft_win={nfft_welch}, nfft_vid={nfft_vid}"
    )

    preds_dir = repo_root / cfg["outputs"]["preds_dir"]
    preds_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = repo_root / cfg["outputs"]["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for subj_id in target_subjects:
        if subj_id not in subjects_map:
            print(f"Warning: Subject {subj_id} not found in {data_root}")
            continue

        sdir = subjects_map[subj_id]
        print(f"--> Processing {subj_id} from {sdir}...")
        sdata = load_ubfc_subject(sdir, subject_id=subj_id)
        align = check_ubfc_alignment(sdata)

        res_str = (
            f"{sdata.video_metadata['width']}x{sdata.video_metadata['height']}"
            if sdata.video_metadata
            else "unknown"
        )

        # 1. Run FacePhys model inference
        # Maps frame i -> GT timestamp i, then resamples both to uniform 30 Hz
        bvp_uniform, t_uniform, meta = predict_bvp(
            video_path=sdata.video_path,
            timestamps=sdata.time,
            fs_target=fs_uniform,
            model_name=cfg["experiment"]["model"],
        )

        # 2. Resample ground-truth reference signals to identical uniform time grid
        _, ppg_uniform = resample_to_uniform(sdata.time, sdata.ppg, target_fs=fs_uniform)
        _, hr_ox_uniform = resample_to_uniform(
            sdata.time, sdata.hr, target_fs=fs_uniform, method="linear"
        )

        # Ensure prediction and reference match in length
        min_len = min(len(bvp_uniform), len(ppg_uniform), len(hr_ox_uniform), len(t_uniform))
        bvp_uniform = bvp_uniform[:min_len]
        ppg_uniform = ppg_uniform[:min_len]
        hr_ox_uniform = hr_ox_uniform[:min_len]
        t_uniform = t_uniform[:min_len]

        # 3. Save frame-level and uniform predictions to CSV
        pred_csv_path = preds_dir / f"{subj_id}_facephys_pred.csv"
        # Save raw frame-level array
        raw_bvp = meta["raw_bvp"]
        orig_t = meta["orig_timestamps"]
        df_raw = pd.DataFrame({
            "frame_idx": np.arange(len(raw_bvp)),
            "gt_timestamp_sec": orig_t[: len(raw_bvp)],
            "facephys_bvp_raw": raw_bvp,
        })
        df_raw.to_csv(pred_csv_path, index=False)
        print(f"    Saved raw predictions to {pred_csv_path}")

        # 4. Video-Level HR Estimation
        hr_pred_vid = estimate_hr_video(
            bvp_uniform, fs=fs_uniform, lowcut=lowcut, highcut=highcut, order=filter_order, nfft=nfft_vid
        )
        hr_primary_vid = estimate_hr_video(
            ppg_uniform, fs=fs_uniform, lowcut=lowcut, highcut=highcut, order=filter_order, nfft=nfft_vid
        )
        hr_secondary_vid = float(np.mean(sdata.hr))
        hr_peaks_vid = hr_from_peaks(ppg_uniform, fs=fs_uniform, lowcut=lowcut, highcut=highcut)

        err_vid_vs_primary = float(abs(hr_pred_vid - hr_primary_vid))
        err_vid_vs_secondary = float(abs(hr_pred_vid - hr_secondary_vid))

        # 5. Windowed HR Estimation (30s window, 1s stride)
        win_times, win_hrs_pred = estimate_hr_windowed(
            bvp_uniform,
            fs=fs_uniform,
            window_sec=win_sec,
            stride_sec=stride_sec,
            lowcut=lowcut,
            highcut=highcut,
            order=filter_order,
            nfft=nfft_welch,
        )
        _, win_hrs_primary = estimate_hr_windowed(
            ppg_uniform,
            fs=fs_uniform,
            window_sec=win_sec,
            stride_sec=stride_sec,
            lowcut=lowcut,
            highcut=highcut,
            order=filter_order,
            nfft=nfft_welch,
        )

        win_samples = int(round(win_sec * fs_uniform))
        stride_samples = int(round(stride_sec * fs_uniform))
        win_hrs_secondary = []
        for start in range(0, len(hr_ox_uniform) - win_samples + 1, stride_samples):
            win_hrs_secondary.append(float(np.mean(hr_ox_uniform[start : start + win_samples])))
        win_hrs_secondary = np.array(win_hrs_secondary, dtype=np.float64)

        # Match lengths if any edge truncation
        w_len = min(len(win_hrs_pred), len(win_hrs_primary), len(win_hrs_secondary))
        win_hrs_pred = win_hrs_pred[:w_len]
        win_hrs_primary = win_hrs_primary[:w_len]
        win_hrs_secondary = win_hrs_secondary[:w_len]
        win_times = win_times[:w_len]

        # (A) All windows metrics
        m_all_vs_primary = calc_metrics(win_hrs_primary, win_hrs_pred)
        m_all_vs_secondary = calc_metrics(win_hrs_secondary, win_hrs_pred)

        # (B) Reference-reliable windows (|Primary - Secondary| <= threshold)
        ref_discrepancy = np.abs(win_hrs_primary - win_hrs_secondary)
        reliable_mask = ref_discrepancy <= reliability_threshold
        reliable_count = int(np.sum(reliable_mask))

        if reliable_count > 0:
            m_rel_vs_primary = calc_metrics(win_hrs_primary[reliable_mask], win_hrs_pred[reliable_mask])
            m_rel_vs_secondary = calc_metrics(win_hrs_secondary[reliable_mask], win_hrs_pred[reliable_mask])
        else:
            m_rel_vs_primary = {k: np.nan for k in m_all_vs_primary}
            m_rel_vs_secondary = {k: np.nan for k in m_all_vs_secondary}

        print(
            f"    Video HR: Pred={hr_pred_vid:.2f} BPM, Primary={hr_primary_vid:.2f} BPM, Secondary={hr_secondary_vid:.2f} BPM\n"
            f"    All Windows (N={w_len}): vs Primary MAE={m_all_vs_primary['mae']:.2f} BPM, r={m_all_vs_primary['pearson_r']:.3f}\n"
            f"    Reliable Windows (N={reliable_count}/{w_len}): vs Primary MAE={m_rel_vs_primary['mae']:.2f} BPM, r={m_rel_vs_primary['pearson_r']:.3f}"
        )

        summary_rows.append({
            "git_hash": git_hash,
            "subject_id": subj_id,
            "model_name": cfg["experiment"]["model"],
            "resolution": res_str,
            "total_frames": meta["total_frames"],
            "decoded_frames": meta["decoded_frames"],
            "face_detected_ratio": meta["face_detected_ratio"],
            "face_null_frames": meta["face_null_frames"],
            "package_versions": pkg_versions_str,
            "welch_parameters": welch_param_str,
            # Video-level HR
            "video_hr_pred_bpm": hr_pred_vid,
            "video_hr_primary_ref_bpm": hr_primary_vid,
            "video_hr_secondary_ref_bpm": hr_secondary_vid,
            "video_hr_peaks_diagnostic_bpm": hr_peaks_vid,
            "video_error_vs_primary_bpm": err_vid_vs_primary,
            "video_error_vs_secondary_bpm": err_vid_vs_secondary,
            # All Windows Metrics
            "all_windows_count": w_len,
            "all_win_mae_vs_primary": m_all_vs_primary["mae"],
            "all_win_rmse_vs_primary": m_all_vs_primary["rmse"],
            "all_win_r_vs_primary": m_all_vs_primary["pearson_r"],
            "all_win_mae_vs_secondary": m_all_vs_secondary["mae"],
            "all_win_rmse_vs_secondary": m_all_vs_secondary["rmse"],
            "all_win_r_vs_secondary": m_all_vs_secondary["pearson_r"],
            # Reliable Windows Metrics
            "reliable_windows_count": reliable_count,
            "reliable_win_mae_vs_primary": m_rel_vs_primary["mae"],
            "reliable_win_rmse_vs_primary": m_rel_vs_primary["rmse"],
            "reliable_win_r_vs_primary": m_rel_vs_primary["pearson_r"],
            "reliable_win_mae_vs_secondary": m_rel_vs_secondary["mae"],
            "reliable_win_rmse_vs_secondary": m_rel_vs_secondary["rmse"],
            "reliable_win_r_vs_secondary": m_rel_vs_secondary["pearson_r"],
        })

    # Save summary dataframe to results/ubfc_facephys.csv
    df_results = pd.DataFrame(summary_rows)
    results_path = repo_root / cfg["outputs"]["results_csv"]
    df_results.to_csv(results_path, index=False)
    print(f"\nSaved results summary to {results_path}")

    # -------------------------------------------------------------
    # Plot Bland-Altman Across Subjects (axes: model - reference)
    # -------------------------------------------------------------
    ba_plot_path = repo_root / cfg["outputs"]["ba_plot"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Subplot 1: Model vs Primary Reference (Welch PPG HR)
    ref_col1 = "video_hr_primary_ref_bpm"
    pred_col = "video_hr_pred_bpm"
    y_ref1 = df_results[ref_col1].values
    y_pred = df_results[pred_col].values
    diff1 = y_pred - y_ref1  # Model - Reference
    mean1 = (y_pred + y_ref1) / 2.0
    ba1 = calc_bland_altman(y_ref1, y_pred)

    for i, row in df_results.iterrows():
        x = (row[pred_col] + row[ref_col1]) / 2.0
        y = row[pred_col] - row[ref_col1]
        lbl = f"{row['subject_id']} ({row['resolution']})"
        ax1.scatter([x], [y], s=90, zorder=5)
        ax1.annotate(lbl, (x, y), textcoords="offset points", xytext=(8, 5), fontsize=9)

    ax1.axhline(ba1["bias"], color="red", linestyle="--", label=f"Bias: {ba1['bias']:.2f} BPM")
    ax1.axhline(ba1["upper_loa"], color="gray", linestyle=":", label=f"+1.96 SD: {ba1['upper_loa']:.2f} BPM")
    ax1.axhline(ba1["lower_loa"], color="gray", linestyle=":", label=f"-1.96 SD: {ba1['lower_loa']:.2f} BPM")
    ax1.set_title("Across Subjects: FacePhys vs Primary Ref (Welch PPG HR)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("(Model + Primary Ref) / 2 (BPM)", fontsize=10)
    ax1.set_ylabel("Model − Reference (BPM)", fontsize=10)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Model vs Secondary Reference (Oximeter Line)
    ref_col2 = "video_hr_secondary_ref_bpm"
    y_ref2 = df_results[ref_col2].values
    diff2 = y_pred - y_ref2  # Model - Reference
    mean2 = (y_pred + y_ref2) / 2.0
    ba2 = calc_bland_altman(y_ref2, y_pred)

    for i, row in df_results.iterrows():
        x = (row[pred_col] + row[ref_col2]) / 2.0
        y = row[pred_col] - row[ref_col2]
        lbl = f"{row['subject_id']} ({row['resolution']})"
        ax2.scatter([x], [y], s=90, zorder=5)
        ax2.annotate(lbl, (x, y), textcoords="offset points", xytext=(8, 5), fontsize=9)

    ax2.axhline(ba2["bias"], color="red", linestyle="--", label=f"Bias: {ba2['bias']:.2f} BPM")
    ax2.axhline(ba2["upper_loa"], color="gray", linestyle=":", label=f"+1.96 SD: {ba2['upper_loa']:.2f} BPM")
    ax2.axhline(ba2["lower_loa"], color="gray", linestyle=":", label=f"-1.96 SD: {ba2['lower_loa']:.2f} BPM")
    ax2.set_title("Across Subjects: FacePhys vs Secondary Ref (Oximeter Line)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("(Model + Secondary Ref) / 2 (BPM)", fontsize=10)
    ax2.set_ylabel("Model − Reference (BPM)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("FacePhys Baseline Across-Subject Bland-Altman Evaluation (Axes: Model − Reference)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ba_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved across-subject Bland-Altman plot to {ba_plot_path}")

    # Print formatted summary table
    print("\n" + "=" * 125)
    print("FACEPHYS BASELINE EVALUATION SUMMARY (RESULTS/UBFC_FACEPHYS.CSV)")
    print("=" * 125)
    cols_to_print = [
        "subject_id",
        "resolution",
        "face_detected_ratio",
        "video_hr_pred_bpm",
        "video_hr_primary_ref_bpm",
        "video_error_vs_primary_bpm",
        "video_hr_secondary_ref_bpm",
        "video_error_vs_secondary_bpm",
        "all_win_mae_vs_primary",
        "reliable_win_mae_vs_primary",
        "reliable_windows_count",
    ]
    print(df_results[cols_to_print].to_string(index=False))


if __name__ == "__main__":
    main()
