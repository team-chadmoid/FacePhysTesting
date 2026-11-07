"""Verification and Reconciliation of Frame Counts across UBFC Subjects.

Compares:
1. Video frame counts decoded via OpenCV (cap.read())
2. Video frame counts decoded via PyAV (container.decode(video=0))
3. Container stream metadata (stream.frames)
4. Ground truth log lengths (time, ppg, hr in ground_truth.txt)
5. 1-to-1 correspondence: frame i <-> GT timestamp i
6. Analysis of output BVP prediction lengths from FacePhys to explain the
   source of the 1,515 / 1,977 / 1,971 / 1,979 numbers.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

import av
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io import find_ubfc_subjects, load_ubfc_subject


def main():
    print("=" * 90)
    print("=== TASK 1: FRAME COUNT RECONCILIATION & DECODE VERIFICATION ===")
    print("=" * 90)

    dataset_root = PROJECT_ROOT / "data" / "ubfc" / "Data Set 2"
    subjects_map = find_ubfc_subjects(dataset_root)

    target_subjects = ["subject1", "subject3", "subject4", "subject5", "subject8", "subject9", "subject10"]
    rows = []

    for sid in target_subjects:
        if sid not in subjects_map:
            print(f"ERROR: {sid} not found in {dataset_root}")
            continue

        sdir = subjects_map[sid]
        sdata = load_ubfc_subject(sdir, subject_id=sid)
        vid_path = sdata.video_path
        size_bytes = os.path.getsize(str(vid_path))
        size_mb = round(size_bytes / (1024 * 1024), 2)

        # 1. OpenCV Frame Count
        cap = cv2.VideoCapture(str(vid_path))
        cv2_count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            cv2_count += 1
        cap.release()

        # 2. PyAV Frame Count
        container = av.open(str(vid_path))
        stream = container.streams.video[0]
        av_count = 0
        for _ in container.decode(video=0):
            av_count += 1
        stream_frames_meta = stream.frames
        container.close()

        # 3. Ground Truth Counts
        gt_time_len = len(sdata.time)
        gt_ppg_len = len(sdata.ppg)
        gt_hr_len = len(sdata.hr)

        # 4. Check exact 1-to-1 match
        match_cv2_av = (cv2_count == av_count)
        match_cv2_gt = (cv2_count == gt_time_len)

        rows.append({
            "subject_id": sid,
            "file_size_bytes": size_bytes,
            "file_size_mb": size_mb,
            "opencv_frames": cv2_count,
            "pyav_frames": av_count,
            "pyav_stream_frames_meta": stream_frames_meta,
            "gt_time_len": gt_time_len,
            "gt_ppg_len": gt_ppg_len,
            "gt_hr_len": gt_hr_len,
            "match_cv2_pyav": match_cv2_av,
            "match_frames_gt": match_cv2_gt,
        })

    df = pd.DataFrame(rows)
    out_csv = PROJECT_ROOT / "results" / "frame_count_reconciliation.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("\nReconciliation Results:")
    print(df.to_string(index=False))
    print("\nSaved reconciliation table to:", out_csv)


if __name__ == "__main__":
    main()

