"""Unit tests for src/models.py (FacePhys baseline model wrapper)."""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from src.models import FacePhysWrapper, predict_bvp


def create_dummy_video(file_path: Path, width=64, height=64, n_frames=90, fps=30):
    """Create a minimal test video with synthetic color variation."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
    for i in range(n_frames):
        # Pulsing skin tone
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * (i / fps))
        frame = np.full((height, width, 3), (120, int(150 + 20 * pulse), 180), dtype=np.uint8)
        out.write(frame)
    out.release()


def test_facephys_wrapper_init():
    wrapper = FacePhysWrapper()
    assert wrapper.model_name == "FacePhys.rlap"


def test_predict_bvp_dummy_video():
    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = Path(tmpdir) / "test.avi"
        create_dummy_video(vid_path, width=64, height=64, n_frames=90, fps=30)

        timestamps = np.arange(90) / 30.0
        bvp, t, meta = predict_bvp(vid_path, timestamps=timestamps, fs_target=30.0)

        assert len(bvp) > 0
        assert len(bvp) == len(t)
        assert meta["decoded_frames"] == 90
        assert "face_detected_ratio" in meta
        assert "model_name" in meta
