"""Hypothesis Test: Effect of FFV1 Video Compression on FacePhys rPPG Performance.

Controlled Experiment on Subject 5:
1. Extract 242x242 center-face crop from subject5 (matching subject4's frame size).
2. Generate two versions:
   - Version A: Uncompressed raw crop (rawvideo, bgr24, 242x242).
   - Version B: Re-encoded with FFV1 lossless intra-frame compression (ffv1, bgr0, 242x242, matching subject4).
3. Evaluate both through baseline-v1 (timing_mode='gt_timestamp', synchronous=True).
4. Compare r, SNR, MedAE, Acc5, and HR error against each other and full-frame baseline.
5. Report result as an empirical hypothesis test, not a rule.
Saves results to results/subject5_compression_test.csv.
"""

import os
import shutil
import sys
from pathlib import Path
import av
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import (
    calc_bland_altman,
    calc_fraction_within_threshold,
    calc_mae,
    calc_medae,
    calc_pearson_r,
    calc_rmse,
    calculate_snr,
    max_lag_waveform_correlation,
)
from src.hr import (
    butterworth_bandpass_filter,
    estimate_hr_video,
    estimate_hr_welch,
    estimate_hr_windowed,
)
from src.io import find_ubfc_subjects, load_ubfc_subject, resample_to_uniform
from src.models import FacePhysWrapper


def encode_ffv1_video(frames_rgb: list, out_path: Path, fps: float = 30.0):
    """Encode RGB frames to AVI container using FFV1 codec and bgr0 pixel format (matching subject4)."""
    h, w, _ = frames_rgb[0].shape
    container = av.open(str(out_path), mode="w", format="avi")
    stream = container.add_stream("ffv1", rate=int(round(fps)))
    stream.width = w
    stream.height = h
    stream.pix_fmt = "bgr0"

    for frame in frames_rgb:
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        av_frame = av_frame.reformat(format="bgr0")
        for packet in stream.encode(av_frame):
            container.mux(packet)
    
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def encode_raw_video(frames_rgb: list, out_path: Path, fps: float = 30.0):
    """Encode RGB frames to AVI container using uncompressed rawvideo and bgr24 pixel format."""
    h, w, _ = frames_rgb[0].shape
    container = av.open(str(out_path), mode="w", format="avi")
    stream = container.add_stream("rawvideo", rate=int(round(fps)))
    stream.width = w
    stream.height = h
    stream.pix_fmt = "bgr24"

    for frame in frames_rgb:
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        av_frame = av_frame.reformat(format="bgr24")
        for packet in stream.encode(av_frame):
            container.mux(packet)
            
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def read_video_frames_rgb(video_path: Path):
    """Read video frames into a list of RGB numpy arrays."""
    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return frames


def find_face_center_crop_coords(frames_rgb: list, crop_size: int = 242):
    """Find coordinates for a crop_size x crop_size window centered on the face in subject5."""
    import rppg
    m = rppg.Model()
    centers_x, centers_y = [], []
    for frame in frames_rgb[:30]:
        res = m.detector.detect(frame)
        if len(res) > 0:
            box = res[0][0]
            # box is (x1, y1, x2, y2)
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            centers_x.append((x1 + x2) / 2.0)
            centers_y.append((y1 + y2) / 2.0)

    h_img, w_img, _ = frames_rgb[0].shape
    if centers_x and centers_y:
        cx = int(round(np.median(centers_x)))
        cy = int(round(np.median(centers_y)))
    else:
        cx, cy = w_img // 2, h_img // 2

    # Define crop bounding box
    half = crop_size // 2
    x1 = max(0, min(cx - half, w_img - crop_size))
    y1 = max(0, min(cy - half, h_img - crop_size))
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    return x1, y1, x2, y2


def evaluate_variant(frames: list, sdata, wrapper: FacePhysWrapper, variant_name: str, fs_target: float = 30.0):
    """Run baseline-v1 inference and metrics on a set of frames."""
    primary_ref_hr = estimate_hr_video(sdata.ppg, fs=fs_target, time=sdata.time)
    
    bvp_u, t_u, meta = wrapper.predict_from_frames(
        frames=frames,
        timestamps=sdata.time,
        fs_target=fs_target,
        timing_mode="gt_timestamp",
        synchronous=True,
    )
    raw_bvp = meta["raw_bvp"]
    raw_ts = meta.get("raw_timestamps", sdata.time[:len(raw_bvp)])

    # Resample GT to uniform grid
    t_u_gt, ppg_u = resample_to_uniform(sdata.time, sdata.ppg, target_fs=fs_target)
    _, ppg_hr_u = resample_to_uniform(sdata.time, sdata.hr, target_fs=fs_target)

    n_common = min(len(bvp_u), len(ppg_u))
    bvp_c = bvp_u[:n_common]
    ppg_c = ppg_u[:n_common]
    t_c = t_u[:n_common]
    hr_c = ppg_hr_u[:n_common]

    bvp_filt = butterworth_bandpass_filter(bvp_c, fs=fs_target, lowcut=0.7, highcut=3.0, order=3)
    ppg_filt = butterworth_bandpass_filter(ppg_c, fs=fs_target, lowcut=0.7, highcut=3.0, order=3)

    video_hr, f_bvp, psd_bvp = estimate_hr_welch(bvp_filt, fs=fs_target, low_bpm=42.0, high_bpm=180.0, nfft=16384)
    error_bpm = abs(video_hr - primary_ref_hr)

    direct_r = calc_pearson_r(bvp_c, ppg_c)
    lag_dict = max_lag_waveform_correlation(bvp_c, ppg_c, fs=fs_target, max_lag_sec=0.5)

    snr_db = calculate_snr(psd_bvp, f_bvp, ref_hr_bpm=primary_ref_hr, delta_hz=0.1, band=(0.7, 3.0))

    t_wins, win_preds = estimate_hr_windowed(bvp_c, fs=fs_target, window_sec=30.0, stride_sec=1.0)
    _, win_refs = estimate_hr_windowed(ppg_c, fs=fs_target, window_sec=30.0, stride_sec=1.0)
    min_wins = min(len(win_preds), len(win_refs))
    win_preds = win_preds[:min_wins]
    win_refs = win_refs[:min_wins]

    win_mae = calc_mae(win_preds, win_refs)
    win_medae = calc_medae(win_preds, win_refs)
    win_rmse = calc_rmse(win_preds, win_refs)
    win_acc5 = calc_fraction_within_threshold(win_preds, win_refs, threshold=5.0)

    return {
        "variant": variant_name,
        "video_hr_bpm": round(video_hr, 4),
        "primary_ref_hr_bpm": round(primary_ref_hr, 4),
        "error_bpm": round(error_bpm, 4),
        "waveform_r_direct": round(direct_r, 4),
        "waveform_r_max_lag_05s": round(lag_dict["max_r"], 4),
        "waveform_best_lag_sec": round(lag_dict.get("optimal_lag_sec", 0.0), 4),
        "snr_db": round(snr_db, 2),
        "windowed_mae_bpm": round(win_mae, 4),
        "windowed_medae_bpm": round(win_medae, 4),
        "windowed_rmse_bpm": round(win_rmse, 4),
        "windowed_acc5_pct": round(win_acc5, 2),
        "stat_null": meta.get("statistic", {}).get("null", 0),
        "stat_filled": meta.get("statistic", {}).get("filled", 0),
        "stat_skipped": meta.get("statistic", {}).get("skipped", 0),
    }


def main():
    print("=" * 95)
    print("=== CONTROLLED EXPERIMENT: SUBJECT 5 COMPRESSION HYPOTHESIS TEST ===")
    print("=" * 95)

    dataset_root = PROJECT_ROOT / "data" / "ubfc" / "Data Set 2"
    subjects_map = find_ubfc_subjects(dataset_root)
    sdir = subjects_map["subject5"]
    sdata = load_ubfc_subject(sdir, subject_id="subject5")

    scratch_dir = PROJECT_ROOT / "data" / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading original subject5 frames from: {sdata.video_path}")
    orig_frames = read_video_frames_rgb(sdata.video_path)
    print(f"Loaded {len(orig_frames)} frames (resolution: {orig_frames[0].shape[1]}x{orig_frames[0].shape[0]})")

    # Find center crop coordinates (242x242 matching subject4)
    x1, y1, x2, y2 = find_face_center_crop_coords(orig_frames, crop_size=242)
    print(f"Face center crop box (242x242): x=[{x1}, {x2}], y=[{y1}, {y2}]")

    cropped_frames = [f[y1:y2, x1:x2] for f in orig_frames]
    assert cropped_frames[0].shape[:2] == (242, 242), f"Crop shape mismatch: {cropped_frames[0].shape}"

    # Path to re-encoded files
    raw_crop_path = scratch_dir / "subject5_242x242_raw.avi"
    ffv1_crop_path = scratch_dir / "subject5_242x242_ffv1.avi"

    if not raw_crop_path.exists():
        print("\nEncoding Version A: Uncompressed 242x242 rawvideo...")
        encode_raw_video(cropped_frames, raw_crop_path, fps=30.0)
    raw_size_mb = os.path.getsize(str(raw_crop_path)) / (1024 * 1024)
    raw_bpp = os.path.getsize(str(raw_crop_path)) / (242 * 242 * len(cropped_frames))
    print(f"Version A ready: {raw_size_mb:.2f} MB, {raw_bpp:.4f} bytes/pixel")

    if not ffv1_crop_path.exists():
        print("\nEncoding Version B: FFV1 compressed 242x242 (matching subject4)...")
        encode_ffv1_video(cropped_frames, ffv1_crop_path, fps=30.0)
    ffv1_size_mb = os.path.getsize(str(ffv1_crop_path)) / (1024 * 1024)
    ffv1_bpp = os.path.getsize(str(ffv1_crop_path)) / (242 * 242 * len(cropped_frames))
    print(f"Version B ready: {ffv1_size_mb:.2f} MB, {ffv1_bpp:.4f} bytes/pixel")

    # Decode frames from the files to test true decode pipeline
    raw_decoded_frames = read_video_frames_rgb(raw_crop_path)
    ffv1_decoded_frames = read_video_frames_rgb(ffv1_crop_path)

    wrapper = FacePhysWrapper()
    results = []

    # 1. Full-frame original baseline (640x480 uncropped rawvideo)
    print("\nEvaluating Baseline: Full-frame original (640x480 uncompressed rawvideo)...")
    res_orig = evaluate_variant(orig_frames, sdata, wrapper, "original_640x480_raw")
    res_orig["resolution"] = "640x480"
    res_orig["codec"] = "rawvideo"
    res_orig["file_size_mb"] = round(os.path.getsize(str(sdata.video_path)) / (1024 * 1024), 2)
    res_orig["bytes_per_pixel"] = round(os.path.getsize(str(sdata.video_path)) / (640 * 480 * len(orig_frames)), 4)
    results.append(res_orig)

    # 2. Version A: 242x242 uncompressed raw crop
    print("Evaluating Version A: Center crop 242x242 uncompressed rawvideo...")
    res_raw = evaluate_variant(raw_decoded_frames, sdata, wrapper, "crop_242x242_raw")
    res_raw["resolution"] = "242x242"
    res_raw["codec"] = "rawvideo"
    res_raw["file_size_mb"] = round(raw_size_mb, 2)
    res_raw["bytes_per_pixel"] = round(raw_bpp, 4)
    results.append(res_raw)

    # 3. Version B: 242x242 FFV1 compressed crop
    print("Evaluating Version B: Center crop 242x242 FFV1 compressed...")
    res_ffv1 = evaluate_variant(ffv1_decoded_frames, sdata, wrapper, "crop_242x242_ffv1")
    res_ffv1["resolution"] = "242x242"
    res_ffv1["codec"] = "ffv1"
    res_ffv1["file_size_mb"] = round(ffv1_size_mb, 2)
    res_ffv1["bytes_per_pixel"] = round(ffv1_bpp, 4)
    results.append(res_ffv1)

    df_out = pd.DataFrame(results)
    out_csv = PROJECT_ROOT / "results" / "subject5_compression_test.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved hypothesis test results to: {out_csv}")

    print("\n" + "=" * 95)
    print("=== HYPOTHESIS TEST RESULTS: COMPRESSION VS SPATIAL FRAMING ===")
    print("=" * 95)
    cols = [
        "variant", "resolution", "codec", "bytes_per_pixel",
        "video_hr_bpm", "error_bpm", "waveform_r_direct", "waveform_r_max_lag_05s",
        "snr_db", "windowed_mae_bpm", "windowed_acc5_pct"
    ]
    print(df_out[cols].to_string(index=False))
    print("=" * 95)

    # Hypothesis Testing Interpretation
    delta_r = res_ffv1["waveform_r_max_lag_05s"] - res_raw["waveform_r_max_lag_05s"]
    delta_snr = res_ffv1["snr_db"] - res_raw["snr_db"]
    delta_err = res_ffv1["error_bpm"] - res_raw["error_bpm"]

    crop_delta_r = res_raw["waveform_r_max_lag_05s"] - res_orig["waveform_r_max_lag_05s"]
    crop_delta_snr = res_raw["snr_db"] - res_orig["snr_db"]

    print("\n=== HYPOTHESIS TEST SUMMARY & EFFECT DECOMPOSITION ===")
    print(f"1. Compression Effect (FFV1 vs Raw crop at 242x242):")
    print(f"   Delta r (lag +/-0.5s): {delta_r:+.4f}")
    print(f"   Delta SNR: {delta_snr:+.2f} dB")
    print(f"   Delta Video Error: {delta_err:+.4f} BPM")
    print(f"2. Spatial Cropping Effect (242x242 Raw crop vs 640x480 Full-frame):")
    print(f"   Delta r (lag +/-0.5s): {crop_delta_r:+.4f}")
    print(f"   Delta SNR: {crop_delta_snr:+.2f} dB")

    if abs(delta_r) < 0.02 and abs(delta_snr) < 1.0:
        print("\nHYPOTHESIS CONCLUSION: Failed to reject Null Hypothesis H0.")
        print("FFV1 lossless intra-frame compression does NOT meaningfully degrade rPPG signal quality.")
        print("The performance difference on subject4 is driven by spatial framing/boundary effects, not FFV1 compression.")
    else:
        print("\nHYPOTHESIS CONCLUSION: Significant compression effect observed.")


if __name__ == "__main__":
    main()
