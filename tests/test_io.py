"""Unit tests for src/io.py."""

import struct
from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.io import (
    UBFCSubjectData,
    parse_avi_header,
    load_ubfc_ground_truth,
    find_ubfc_subjects,
    load_ubfc_subject,
    resample_to_uniform,
    check_ubfc_alignment,
)


def create_dummy_avi(file_path: Path, width=640, height=480, fps=30.0, total_frames=100):
    """Helper to write a minimal valid RIFF AVI header."""
    microsec_per_frame = int(1_000_000 / fps)
    avih_chunk = struct.pack(
        "<IIIIIIII",
        microsec_per_frame,
        0,  # max_bytes_per_sec
        0,  # padding
        0,  # flags
        total_frames,
        0,  # initial frames
        1,  # streams
        0,  # suggested buffer size
    ) + struct.pack("<II", width, height) + b"\x00" * 16

    avih_header = b"avih" + struct.pack("<I", len(avih_chunk)) + avih_chunk
    riff_size = 4 + 8 + len(avih_header)
    riff_header = b"RIFF" + struct.pack("<I", riff_size) + b"AVI " + avih_header

    with open(file_path, "wb") as f:
        f.write(riff_header)


def create_dummy_ubfc_gt(file_path: Path, n_samples=100, fs=30.0):
    """Helper to write dummy ground_truth.txt for UBFC Dataset 2."""
    t = np.arange(n_samples) / fs
    ppg = np.sin(2 * np.pi * 1.2 * t)
    hr = np.full(n_samples, 72.0)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(" ".join(f"{v:.6e}" for v in ppg) + "\n")
        f.write(" ".join(f"{v:.6e}" for v in hr) + "\n")
        f.write(" ".join(f"{v:.6e}" for v in t) + "\n")


def test_parse_avi_header():
    with tempfile.TemporaryDirectory() as tmpdir:
        avi_file = Path(tmpdir) / "test.avi"
        create_dummy_avi(avi_file, width=640, height=480, fps=25.0, total_frames=150)

        meta = parse_avi_header(avi_file)
        assert meta["width"] == 640
        assert meta["height"] == 480
        assert meta["total_frames"] == 150
        assert abs(meta["fps"] - 25.0) < 0.1


def test_load_ubfc_ground_truth():
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_file = Path(tmpdir) / "ground_truth.txt"
        create_dummy_ubfc_gt(gt_file, n_samples=50, fs=30.0)

        ppg, hr, t = load_ubfc_ground_truth(gt_file)
        assert len(ppg) == 50
        assert len(hr) == 50
        assert len(t) == 50
        assert np.isclose(t[0], 0.0)
        assert np.isclose(t[-1], 49 / 30.0)
        assert np.allclose(hr, 72.0)


def test_load_ubfc_ground_truth_mismatch():
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_file = Path(tmpdir) / "ground_truth.txt"
        with open(gt_file, "w") as f:
            f.write("1.0 2.0 3.0\n")
            f.write("60.0 60.0\n")  # Only 2 items
            f.write("0.0 0.033 0.066\n")

        with pytest.raises(ValueError, match="Length mismatch"):
            load_ubfc_ground_truth(gt_file)


def test_resample_to_uniform():
    # Create non-uniform timestamps with slight jitter and one duplicate
    t_non_uniform = np.array([0.0, 0.031, 0.065, 0.065, 0.100, 0.133, 0.168, 0.201, 0.233, 0.267])
    sig = np.sin(2 * np.pi * 2.0 * t_non_uniform)

    t_u, s_u = resample_to_uniform(t_non_uniform, sig, target_fs=30.0)
    assert len(t_u) > 0
    assert len(t_u) == len(s_u)
    # Check strictly monotonic uniform step
    dt = np.diff(t_u)
    assert np.allclose(dt, 1.0 / 30.0)


def test_check_ubfc_alignment():
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        subj_dir = td / "subject1"
        subj_dir.mkdir()
        gt_file = subj_dir / "ground_truth.txt"
        avi_file = subj_dir / "vid.avi"

        create_dummy_ubfc_gt(gt_file, n_samples=90, fs=30.0)
        create_dummy_avi(avi_file, total_frames=90, fps=30.0)

        data = load_ubfc_subject(subj_dir)
        diag = check_ubfc_alignment(data)

        assert diag["frames_match"] is True
        assert diag["n_samples"] == 90
        assert diag["video_frames"] == 90
        assert diag["non_positive_count"] == 0
        assert abs(diag["nominal_fps"] - 30.0) < 0.5


def test_find_and_load_real_dataset_if_present():
    data_dir = Path("data/ubfc/Data Set 2")
    if data_dir.is_dir():
        subjects = find_ubfc_subjects(data_dir)
        assert len(subjects) >= 3
        assert "subject1" in subjects
        assert "subject3" in subjects
        assert "subject4" in subjects

        for subj_id in ["subject1", "subject3", "subject4"]:
            sdata = load_ubfc_subject(subjects[subj_id])
            diag = check_ubfc_alignment(sdata)
            assert diag["frames_match"] is True
            assert diag["n_samples"] > 1000
