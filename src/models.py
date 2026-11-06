"""Model wrappers for rPPG models (FacePhys baseline via open-rppg).

Implements:
- FacePhysWrapper: wrapper around open-rppg pretrained FacePhys model.
- predict_bvp(video_path) -> (bvp, t, metadata): functional interface that indexes
  predicted BVP per frame, maps frame i to GT timestamp i, and resamples to uniform 30 Hz.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.io import resample_to_uniform


class FacePhysWrapper:
    """Wrapper for open-rppg FacePhys baseline model.

    Utilizes:
    - rppg.Model('FacePhys.rlap'): state-space CDE neural rPPG model.
    - model.update_frame(img, ts): frame-by-frame streaming face detection & inference.
    - model.bvp(raw=True): per-frame raw BVP prediction extraction.
    """

    def __init__(self, model_name: str = "FacePhys.rlap"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            import rppg
            self._model = rppg.Model(self.model_name)
        return self._model

    def predict_video(
        self,
        video_path: Union[str, Path],
        timestamps: Optional[np.ndarray] = None,
        fs_target: float = 30.0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Predict BVP waveform from video file.

        Indexes predicted BVP per frame, maps frame i to timestamp i,
        and resamples to a uniform time grid at fs_target.

        Args:
            video_path: Path to AVI / MP4 video file.
            timestamps: Optional 1D array of GT timestamps in seconds (length = total frames).
            fs_target: Target uniform sampling rate in Hz (default: 30.0).

        Returns:
            Tuple of:
                - bvp_uniform: 1D array of predicted BVP on uniform time grid.
                - t_uniform: 1D array of uniform timestamps (seconds).
                - metadata: Dictionary of diagnostic info (face detection success, frame counts, etc.).
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB for rppg
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        n_frames = len(frames)
        if n_frames == 0:
            raise ValueError(f"No frames could be read from video {path}")

        model = self._get_model()

        # Stream frames through model
        with model:
            for idx, frame in enumerate(frames):
                # We feed frame index as timestamp to preserve 1-to-1 frame correspondence
                model.update_frame(frame, idx / fs_target)

        bvp_raw, _ = model.bvp(raw=True)
        bvp_raw = np.asarray(bvp_raw, dtype=np.float64)

        # Retrieve face detection diagnostics from model
        stats = dict(getattr(model, "statistic", {}))
        hasface_val = getattr(model, "hasface", 0.0)
        null_count = stats.get("null", 0)
        detected_frames = n_frames - null_count
        face_detect_ratio = float(detected_frames / n_frames) if n_frames > 0 else 0.0

        # Alignment and uniform resampling:
        # If lengths differ slightly due to warmup/padding, handle gracefully
        if len(bvp_raw) != n_frames:
            # If bvp_raw has fewer or more samples, pad or truncate to match frames exactly
            if len(bvp_raw) < n_frames:
                pad = np.full(n_frames - len(bvp_raw), bvp_raw[-1] if len(bvp_raw) > 0 else 0.0)
                bvp_raw = np.concatenate([bvp_raw, pad])
            else:
                bvp_raw = bvp_raw[:n_frames]

        if timestamps is not None:
            t_orig = np.asarray(timestamps, dtype=np.float64)
            if len(t_orig) != n_frames:
                # If mismatch between GT timestamps and frames, interpolate timestamps to frame count
                t_orig = np.linspace(t_orig[0], t_orig[-1], n_frames)
        else:
            t_orig = np.arange(n_frames) / fs_target

        # Map frame i -> timestamp i, then resample both to uniform grid
        t_uniform, bvp_uniform = resample_to_uniform(t_orig, bvp_raw, target_fs=fs_target)

        metadata = {
            "model_name": self.model_name,
            "video_path": str(path),
            "video_width": width,
            "video_height": height,
            "total_frames": total_frames,
            "decoded_frames": n_frames,
            "bvp_raw_length": len(bvp_raw),
            "face_detected_ratio": face_detect_ratio,
            "face_null_frames": null_count,
            "face_stats": stats,
            "raw_bvp": bvp_raw,
            "orig_timestamps": t_orig,
        }

        return bvp_uniform, t_uniform, metadata


def predict_bvp(
    video_path: Union[str, Path],
    timestamps: Optional[np.ndarray] = None,
    fs_target: float = 30.0,
    model_name: str = "FacePhys.rlap",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Predict BVP waveform from video using FacePhys baseline.

    Args:
        video_path: Path to video file.
        timestamps: Optional GT timestamps array (seconds).
        fs_target: Target uniform sampling rate in Hz (default: 30.0).
        model_name: Model identifier (default: 'FacePhys.rlap').

    Returns:
        Tuple of (bvp_uniform, t_uniform, metadata).
    """
    wrapper = FacePhysWrapper(model_name=model_name)
    return wrapper.predict_video(video_path, timestamps=timestamps, fs_target=fs_target)
