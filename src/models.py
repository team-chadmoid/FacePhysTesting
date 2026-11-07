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


"""Model wrappers for rPPG models (FacePhys baseline via open-rppg).

Implements:
- FacePhysWrapper: wrapper around open-rppg pretrained FacePhys model.
- Pre-registered baseline of record: GT-timestamp path (feeds true GT capture timestamps
  to model.update_frame, allowing the internal watchdog to fill gaps where camera frames
  were dropped during hardware acquisition).
- predict_bvp(video_path) -> (bvp, t, metadata): functional interface.
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

    Pre-registered Baseline-of-Record:
        timing_mode='gt_timestamp' (feeds hardware capture timestamps t_gt[i] directly
        to update_frame so the library watchdog forward-fills real acquisition dropouts).
    """

    def __init__(self, model_name: str = "FacePhys.rlap"):
        self.model_name = model_name

    def _create_model(self):
        """Create a fresh open-rppg model instance.

        Note: rppg.Model thread pool shuts down on context exit, so a new
        instance is instantiated per video/frame run to avoid deadlock.
        """
        import rppg
        return rppg.Model(self.model_name)

    def predict_video(
        self,
        video_path: Union[str, Path],
        timestamps: Optional[np.ndarray] = None,
        timing_mode: str = "gt_timestamp",
        fs_target: float = 30.0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Predict BVP waveform from video file.

        Args:
            video_path: Path to AVI / MP4 video file.
            timestamps: Optional 1D array of GT timestamps in seconds (length = total frames).
            timing_mode: Timing strategy:
                - 'gt_timestamp': Feed GT capture timestamps to update_frame (baseline of record).
                - 'uniform': Feed synthetic uniform timestamps (i / fs_target).
                - 'container': Feed container PTS timestamps decoded via PyAV.
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

        frames: List[np.ndarray] = []
        container_ts: List[float] = []

        if timing_mode == "container":
            try:
                import av
                container = av.open(str(path))
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                for frame in container.decode(stream):
                    container_ts.append(float(frame.time) if frame.time is not None else 0.0)
                    img = frame.to_ndarray(format="rgb24")
                    rotation = -frame.rotation % 360
                    if rotation == 90:
                        img = img.swapaxes(0, 1)[:, ::-1, :]
                    elif rotation == 180:
                        img = img[::-1, ::-1, :]
                    elif rotation == 270:
                        img = img.swapaxes(0, 1)[::-1, :, :]
                    frames.append(img)
                container.close()
                width = frames[0].shape[1] if frames else 0
                height = frames[0].shape[0] if frames else 0
                total_frames = len(frames)
            except Exception:
                # Fallback to OpenCV if PyAV fails
                frames = []
                container_ts = []
                timing_mode = "uniform"

        if not frames:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise IOError(f"Could not open video file: {path}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            cap.release()

        n_frames = len(frames)
        if n_frames == 0:
            raise ValueError(f"No frames could be read from video {path}")

        feed_ts = None
        if timing_mode == "gt_timestamp" and timestamps is not None:
            feed_ts = np.asarray(timestamps, dtype=np.float64)
        elif timing_mode == "container" and len(container_ts) == n_frames:
            feed_ts = np.asarray(container_ts, dtype=np.float64)

        bvp_uniform, t_uniform, metadata = self.predict_from_frames(
            frames=frames,
            timestamps=feed_ts,
            timing_mode=timing_mode,
            fs_target=fs_target,
        )
        metadata.update({
            "video_path": str(path),
            "video_width": width,
            "video_height": height,
            "total_frames": total_frames,
            "timing_mode": timing_mode,
        })
        return bvp_uniform, t_uniform, metadata

    def predict_from_frames(
        self,
        frames: List[np.ndarray],
        timestamps: Optional[np.ndarray] = None,
        timing_mode: str = "gt_timestamp",
        fs_target: float = 30.0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Predict BVP waveform directly from an in-memory list of RGB frames.

        Args:
            frames: List of 2D/3D uint8 RGB images.
            timestamps: Optional 1D array of capture timestamps in seconds.
            timing_mode: 'gt_timestamp' (pre-registered baseline of record), 'uniform', or 'container'.
            fs_target: Target uniform sampling rate in Hz (default: 30.0).

        Returns:
            Tuple of (bvp_uniform, t_uniform, metadata).
        """
        n_frames = len(frames)
        if n_frames == 0:
            raise ValueError("No frames provided")

        model = self._create_model()

        # Determine timestamp feed
        use_ts = False
        if timing_mode == "gt_timestamp" and timestamps is not None and len(timestamps) == n_frames:
            use_ts = True
            ts_to_feed = np.asarray(timestamps, dtype=np.float64)
        elif timing_mode == "container" and timestamps is not None and len(timestamps) == n_frames:
            use_ts = True
            ts_to_feed = np.asarray(timestamps, dtype=np.float64)
        else:
            use_ts = False
            ts_to_feed = np.arange(n_frames, dtype=np.float64) / fs_target

        # Stream frames through model
        with model:
            for idx, frame in enumerate(frames):
                if use_ts:
                    model.update_frame(frame, float(ts_to_feed[idx]))
                else:
                    model.update_frame(frame, float(idx / fs_target))

        bvp_raw, ts_raw = model.bvp(raw=True)
        bvp_raw = np.asarray(bvp_raw, dtype=np.float64)
        ts_raw = np.asarray(ts_raw, dtype=np.float64) if len(ts_raw) > 0 else np.array([])

        # Retrieve face detection & pipeline diagnostics from model
        stats = dict(getattr(model, "statistic", {}))
        null_count = stats.get("null", 0)
        filled_count = stats.get("filled", 0)
        skipped_count = stats.get("skipped", 0)
        detected_frames = n_frames - null_count
        face_detect_ratio = float(detected_frames / n_frames) if n_frames > 0 else 0.0

        # Uniform resampling using actual output timestamps:
        if len(bvp_raw) == 0:
            # Fallback for empty prediction
            bvp_uniform = np.zeros(n_frames, dtype=np.float64)
            t_uniform = np.arange(n_frames) / fs_target
        elif len(ts_raw) == len(bvp_raw) and len(ts_raw) > 1:
            t_uniform, bvp_uniform = resample_to_uniform(ts_raw, bvp_raw, target_fs=fs_target)
        elif timestamps is not None and len(timestamps) == len(bvp_raw):
            t_uniform, bvp_uniform = resample_to_uniform(np.asarray(timestamps), bvp_raw, target_fs=fs_target)
        else:
            t_orig = np.arange(len(bvp_raw)) / fs_target
            t_uniform, bvp_uniform = resample_to_uniform(t_orig, bvp_raw, target_fs=fs_target)

        metadata = {
            "model_name": self.model_name,
            "timing_mode": timing_mode,
            "decoded_frames": n_frames,
            "bvp_raw_length": len(bvp_raw),
            "face_detected_ratio": face_detect_ratio,
            "face_null_frames": null_count,
            "filled_frames": filled_count,
            "skipped_frames": skipped_count,
            "face_stats": stats,
            "raw_bvp": bvp_raw,
            "raw_timestamps": ts_raw,
        }

        return bvp_uniform, t_uniform, metadata


def predict_bvp(
    video_path: Union[str, Path],
    timestamps: Optional[np.ndarray] = None,
    timing_mode: str = "gt_timestamp",
    fs_target: float = 30.0,
    model_name: str = "FacePhys.rlap",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Predict BVP waveform from video using FacePhys baseline.

    Args:
        video_path: Path to video file.
        timestamps: Optional GT timestamps array (seconds).
        timing_mode: Timing mode ('gt_timestamp', 'uniform', or 'container').
        fs_target: Target uniform sampling rate in Hz (default: 30.0).
        model_name: Model identifier (default: 'FacePhys.rlap').

    Returns:
        Tuple of (bvp_uniform, t_uniform, metadata).
    """
    wrapper = FacePhysWrapper(model_name=model_name)
    return wrapper.predict_video(
        video_path, timestamps=timestamps, timing_mode=timing_mode, fs_target=fs_target
    )

