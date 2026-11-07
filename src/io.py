"""I/O and dataset loader for UBFC-RPPG Dataset 2.

Handles parsing ground truth files (PPG, HR, timestamps), RIFF AVI video metadata,
directory discovery, integrity checks, and alignment verification.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import interpolate


@dataclass
class UBFCSubjectData:
    """Container for UBFC-RPPG Dataset 2 subject data."""
    subject_id: str
    subject_dir: Path
    gt_path: Path
    video_path: Optional[Path]
    ppg: np.ndarray          # Line 1: PPG trace (N samples)
    hr: np.ndarray           # Line 2: Heart rate trace (N samples, in BPM)
    time: np.ndarray         # Line 3: Timestamps (N samples, in seconds)
    video_metadata: Optional[Dict[str, Any]] = None


def parse_avi_header(video_path: Union[str, Path]) -> Dict[str, Any]:
    """Parse RIFF AVI header to extract video metadata without heavy dependencies.

    Args:
        video_path: Path to AVI video file.

    Returns:
        Dictionary with:
            - microsec_per_frame: int
            - fps: float (nominal fps computed from header)
            - total_frames: int
            - width: int
            - height: int
            - streams: int
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    with open(path, "rb") as f:
        data = f.read(4096)

    riff = data[:4]
    avi_magic = data[8:12]
    if riff != b"RIFF" or avi_magic != b"AVI ":
        raise ValueError(f"File {path} is not a valid RIFF AVI file (found {riff}, {avi_magic})")

    idx = data.find(b"avih")
    if idx == -1:
        raise ValueError(f"Could not find 'avih' chunk in {path}")

    avih_size = struct.unpack("<I", data[idx + 4 : idx + 8])[0]
    avih_data = data[idx + 8 : idx + 8 + avih_size]

    if len(avih_data) < 40:
        raise ValueError(f"'avih' chunk too short in {path}")

    microsec_per_frame, max_bytes_per_sec, padding, flags, total_frames = struct.unpack(
        "<IIIII", avih_data[:20]
    )
    streams, buf_size, width, height = struct.unpack("<IIII", avih_data[24:40])

    fps = 1_000_000.0 / microsec_per_frame if microsec_per_frame > 0 else 0.0

    return {
        "microsec_per_frame": microsec_per_frame,
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "streams": streams,
    }


def load_ubfc_ground_truth(gt_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ground_truth.txt for UBFC Dataset 2.

    UBFC Dataset 2 format:
        Line 1: PPG signal (space-separated float strings)
        Line 2: Heart rate (HR) in BPM
        Line 3: Timestep in seconds (scientific notation or float)

    Args:
        gt_path: Path to ground_truth.txt.

    Returns:
        Tuple of (ppg, hr, time) as 1D numpy arrays of float64.
    """
    path = Path(gt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 3:
        raise ValueError(f"ground_truth.txt must contain at least 3 lines, found {len(lines)} in {path}")

    ppg = np.array([float(x) for x in lines[0].split()], dtype=np.float64)
    hr = np.array([float(x) for x in lines[1].split()], dtype=np.float64)
    time = np.array([float(x) for x in lines[2].split()], dtype=np.float64)

    if not (len(ppg) == len(hr) == len(time)):
        raise ValueError(
            f"Length mismatch in ground truth {path}: "
            f"ppg={len(ppg)}, hr={len(hr)}, time={len(time)}"
        )

    return ppg, hr, time


def find_ubfc_subjects(dataset_dir: Union[str, Path]) -> Dict[str, Path]:
    """Find all subject directories in UBFC Dataset 2 directory.

    Handles both direct subject directories (e.g., `data/ubfc/Data Set 2/subject1`)
    and extracted archive nesting (e.g., `data/ubfc/Data Set 2/subject1-.../subject1`).

    Args:
        dataset_dir: Root directory of UBFC Dataset 2.

    Returns:
        Sorted dictionary mapping subject_id (e.g. 'subject1') to Path of subject directory.
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    subjects: Dict[str, Path] = {}
    gt_files = list(root.glob("**/ground_truth.txt"))

    for gt_file in gt_files:
        subject_dir = gt_file.parent
        # Look for subject id in folder name or ancestor folder names
        match = re.search(r"subject\d+", str(subject_dir), re.IGNORECASE)
        if match:
            subj_id = match.group(0).lower()
        else:
            subj_id = subject_dir.name.lower()

        # If duplicate, prefer the one closest to ground_truth.txt
        if subj_id not in subjects:
            subjects[subj_id] = subject_dir
        else:
            # If current path is more specific / nested closer
            if len(subject_dir.parts) >= len(subjects[subj_id].parts):
                subjects[subj_id] = subject_dir

    # Sort by numeric subject number if possible
    def sort_key(s: str):
        nums = re.findall(r"\d+", s)
        return int(nums[0]) if nums else s

    return dict(sorted(subjects.items(), key=lambda item: sort_key(item[0])))


def load_ubfc_subject(
    subject_dir: Union[str, Path],
    subject_id: Optional[str] = None,
    load_video_meta: bool = True,
) -> UBFCSubjectData:
    """Load a single UBFC subject.

    Args:
        subject_dir: Path to directory containing ground_truth.txt and vid.avi.
        subject_id: Optional subject identifier; derived from folder name if None.
        load_video_meta: Whether to parse AVI video header.

    Returns:
        UBFCSubjectData instance.
    """
    sdir = Path(subject_dir)
    if not sdir.is_dir():
        raise FileNotFoundError(f"Subject directory not found: {sdir}")

    if subject_id is None:
        match = re.search(r"subject\d+", str(sdir), re.IGNORECASE)
        subject_id = match.group(0).lower() if match else sdir.name.lower()

    gt_path = sdir / "ground_truth.txt"
    ppg, hr, time = load_ubfc_ground_truth(gt_path)

    video_path: Optional[Path] = None
    for candidate in ["vid.avi", "vid.mp4"]:
        vp = sdir / candidate
        if vp.is_file():
            video_path = vp
            break

    video_metadata: Optional[Dict[str, Any]] = None
    if load_video_meta and video_path is not None and video_path.suffix.lower() == ".avi":
        try:
            video_metadata = parse_avi_header(video_path)
        except Exception:
            video_metadata = None

    return UBFCSubjectData(
        subject_id=subject_id,
        subject_dir=sdir,
        gt_path=gt_path,
        video_path=video_path,
        ppg=ppg,
        hr=hr,
        time=time,
        video_metadata=video_metadata,
    )


def resample_to_uniform(
    time: np.ndarray,
    signal: np.ndarray,
    target_fs: float = 30.0,
    method: str = "cubic",
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample non-uniformly sampled signal onto a uniform time grid.

    Deduplicates identical consecutive timestamps by slightly jittering or keeping unique,
    then applies interpolation.

    Args:
        time: 1D array of original timestamps (seconds).
        signal: 1D array of signal values.
        target_fs: Target uniform sampling rate in Hz (default: 30.0 Hz).
        method: Interpolation method ('cubic' or 'linear').

    Returns:
        Tuple of (uniform_time, uniform_signal).
    """
    time_arr = np.asarray(time, dtype=np.float64)
    sig_arr = np.asarray(signal, dtype=np.float64)

    if len(time_arr) != len(sig_arr):
        raise ValueError("time and signal must have identical lengths")
    if len(time_arr) < 4:
        raise ValueError("time series too short to resample")

    # Deduplicate non-strictly-increasing timestamps
    diffs = np.diff(time_arr)
    non_pos = np.where(diffs <= 0)[0]
    if len(non_pos) > 0:
        # Keep strictly increasing timestamps
        keep = np.ones(len(time_arr), dtype=bool)
        for idx in non_pos:
            keep[idx + 1] = False
        t_clean = time_arr[keep]
        s_clean = sig_arr[keep]
    else:
        t_clean = time_arr
        s_clean = sig_arr

    dt_target = 1.0 / target_fs
    n_steps = int(round((t_clean[-1] - t_clean[0]) * target_fs)) + 1
    t_uniform = t_clean[0] + np.arange(n_steps) * dt_target
    t_uniform = t_uniform[t_uniform <= t_clean[-1] + 1e-9]

    # Use cubic spline if enough points, fallback to linear
    if method == "cubic" and len(t_clean) >= 4:
        interp_fn = interpolate.interp1d(
            t_clean, s_clean, kind="cubic", fill_value="extrapolate"
        )
    else:
        interp_fn = interpolate.interp1d(
            t_clean, s_clean, kind="linear", fill_value="extrapolate"
        )

    s_uniform = interp_fn(t_uniform)
    return t_uniform, s_uniform


def check_ubfc_alignment(subject_data: UBFCSubjectData) -> Dict[str, Any]:
    """Check alignment and timestamp integrity for a UBFC subject.

    Checks:
        - Exact match between video frames and GT length.
        - Monotonicity of timestamps and duplicate counts.
        - Sampling rate statistics (mean, min, max, std).
        - Duration comparison.
        - Warnings and anomalies.

    Args:
        subject_data: UBFCSubjectData instance.

    Returns:
        Dictionary of diagnostic alignment statistics.
    """
    time = subject_data.time
    ppg = subject_data.ppg
    hr = subject_data.hr

    n_samples = len(ppg)
    diffs = np.diff(time) if n_samples > 1 else np.array([])

    non_positive = int(np.sum(diffs <= 0)) if len(diffs) > 0 else 0
    mean_dt = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
    min_dt = float(np.min(diffs)) if len(diffs) > 0 else 0.0
    max_dt = float(np.max(diffs)) if len(diffs) > 0 else 0.0
    std_dt = float(np.std(diffs)) if len(diffs) > 0 else 0.0

    nominal_fps = float(1.0 / mean_dt) if mean_dt > 0 else 0.0
    duration_sec = float(time[-1] - time[0]) if n_samples > 1 else 0.0

    large_gaps = int(np.sum(diffs > 2.0 * mean_dt)) if mean_dt > 0 else 0

    video_frames: Optional[int] = None
    video_fps: Optional[float] = None
    frames_match = False

    flags: List[str] = []

    if subject_data.video_metadata is not None:
        video_frames = subject_data.video_metadata.get("total_frames")
        video_fps = subject_data.video_metadata.get("fps")
        if video_frames is not None:
            frames_match = (video_frames == n_samples)
            if not frames_match:
                flags.append(
                    f"Frame count mismatch: video has {video_frames} frames but GT has {n_samples} samples"
                )
    else:
        flags.append("Video metadata unavailable")

    if non_positive > 0:
        flags.append(f"Non-positive timestamp diffs detected ({non_positive} occurrences)")

    if large_gaps > 0:
        flags.append(f"Large timestamp gaps (>2x mean dt) detected ({large_gaps} occurrences)")

    return {
        "subject_id": subject_data.subject_id,
        "n_samples": n_samples,
        "video_frames": video_frames,
        "frames_match": frames_match,
        "duration_sec": duration_sec,
        "mean_dt": mean_dt,
        "min_dt": min_dt,
        "max_dt": max_dt,
        "std_dt": std_dt,
        "nominal_fps": nominal_fps,
        "video_fps": video_fps,
        "non_positive_count": non_positive,
        "large_gap_count": large_gaps,
        "hr_min": float(np.min(hr)) if n_samples > 0 else 0.0,
        "hr_max": float(np.max(hr)) if n_samples > 0 else 0.0,
        "hr_mean": float(np.mean(hr)) if n_samples > 0 else 0.0,
        "flags": flags,
    }
