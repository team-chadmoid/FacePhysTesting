"""Model wrappers for rPPG models (FacePhys baseline via open-rppg).

Implements:
- FacePhysWrapper: wrapper around open-rppg pretrained FacePhys model.
- Pre-registered baseline of record: GT-timestamp path (feeds true GT capture timestamps
  to model.update_frame, allowing the internal watchdog to fill gaps where camera frames
  were dropped during hardware acquisition).
- predict_bvp(video_path) -> (bvp, t, metadata): functional interface.
- Memory-safe streaming: decodes video frame-by-frame instead of accumulating gigabytes
  of raw uncompressed frames in RAM, preventing glibc heap corruption on large videos.
"""

import os
# Configure single-threaded BLAS/OpenMP and bounded JAX memory before imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.20"

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
cv2.setNumThreads(0)
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

    def _cleanup_model_safe(self, model: Any) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Safely shut down the open-rppg Model and extract BVP, replacing `with model:`.

        open-rppg's __exit__ shuts down face_detection_pool but omits face_resampling_pool,
        leaving dangling ThreadPoolExecutor workers that corrupt the glibc heap
        (``corrupted double-linked list`` / ``double free or corruption (!prev)``).

        This method manually replicates __exit__'s teardown in the correct order:
        1. Wait for pending face_detection_task.
        2. Signal inference thread to stop (alive=False, sp.release).
        3. Join inference thread (which may do a final face_resampling_pool.map).
        4. Join any self.run thread (wait_completion).
        5. Shut down BOTH face_resampling_pool AND face_detection_pool.
        6. Extract BVP + timestamps while model data structures are still valid.
        """
        # Step 1: wait for in-flight face detection
        if hasattr(model, "face_detection_task") and model.face_detection_task is not None:
            try:
                model.face_detection_task.result()
            except Exception:
                pass

        # Step 2: signal inference thread to stop
        model.alive = False
        if hasattr(model, "sp"):
            try:
                model.sp.release()
            except Exception:
                pass

        # Step 3: join inference thread
        if hasattr(model, "ift") and model.ift is not None:
            try:
                model.ift.join(timeout=120)
            except Exception:
                pass

        # Step 4: join any run thread
        if hasattr(model, "wait_completion"):
            try:
                model.wait_completion()
            except Exception:
                pass

        # Step 5: shut down BOTH thread pools (the root cause of heap corruption)
        for pool_name in ("face_resampling_pool", "face_detection_pool"):
            pool = getattr(model, pool_name, None)
            if pool is not None:
                try:
                    pool.shutdown(wait=True)
                except Exception:
                    pass

        # Step 6: extract BVP while model data is still valid
        try:
            bvp_raw, ts_raw = model.bvp(raw=True)
            bvp_raw = np.asarray(bvp_raw, dtype=np.float64)
            ts_raw = np.asarray(ts_raw, dtype=np.float64) if len(ts_raw) > 0 else np.array([])
        except Exception:
            bvp_raw = np.array([], dtype=np.float64)
            ts_raw = np.array([], dtype=np.float64)

        stats = dict(getattr(model, "statistic", {}))

        return bvp_raw, ts_raw, stats

    def predict_video(
        self,
        video_path: Union[str, Path],
        timestamps: Optional[np.ndarray] = None,
        timing_mode: str = "gt_timestamp",
        synchronous: bool = True,
        fs_target: float = 30.0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Predict BVP waveform from video file with streaming frame ingestion.

        Args:
            video_path: Path to AVI / MP4 video file.
            timestamps: Optional 1D array of GT timestamps in seconds (length = total frames).
            timing_mode: Timing strategy:
                - 'gt_timestamp': Feed GT capture timestamps to update_frame (baseline of record).
                - 'uniform': Feed synthetic uniform timestamps (i / fs_target).
                - 'container': Feed container PTS timestamps decoded via PyAV.
            synchronous: If True, waits for each frame's face detection and preprocessing
                task to complete before submitting the next frame, ensuring 100% determinism.
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

        # Stream ingestion path: reads one frame at a time to keep RAM consumption <= 1 MB
        model = self._create_model()
        width = 0
        height = 0
        total_frames = 0
        n_frames = 0
        use_pyav = (timing_mode == "container")

        # Start model (replaces `with model:` — we handle teardown ourselves
        # via _cleanup_model_safe to avoid the heap corruption in __exit__)
        model.__enter__()

        if use_pyav:
            try:
                import av
                container = av.open(str(path))
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                width = stream.width or 0
                height = stream.height or 0
                total_frames = stream.frames or 0

                for idx, frame in enumerate(container.decode(stream)):
                    img = frame.to_ndarray(format="rgb24")
                    rotation = -frame.rotation % 360
                    if rotation == 90:
                        img = img.swapaxes(0, 1)[:, ::-1, :]
                    elif rotation == 180:
                        img = img[::-1, ::-1, :]
                    elif rotation == 270:
                        img = img.swapaxes(0, 1)[::-1, :, :]

                    pts_sec = float(frame.time) if frame.time is not None else float(idx / fs_target)
                    model.update_frame(img, pts_sec)
                    if synchronous and model.face_detection_task is not None:
                        model.face_detection_task.result()
                    n_frames += 1
                container.close()
            except Exception:
                # Fallback to OpenCV if PyAV fails
                use_pyav = False
                n_frames = 0

        if not use_pyav:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                # Must still tear down the model we __enter__'d
                self._cleanup_model_safe(model)
                raise IOError(f"Could not open video file: {path}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            use_ts = (timing_mode == "gt_timestamp" and timestamps is not None)
            feed_ts = np.asarray(timestamps, dtype=np.float64) if use_ts else None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if use_ts and feed_ts is not None and n_frames < len(feed_ts):
                    ts_val = float(feed_ts[n_frames])
                else:
                    ts_val = float(n_frames / fs_target)
                model.update_frame(frame_rgb, ts_val)
                if synchronous and model.face_detection_task is not None:
                    model.face_detection_task.result()
                n_frames += 1
            cap.release()

        # Safe teardown: shut down ALL pools + extract BVP before any state is freed
        bvp_raw, ts_raw, stats = self._cleanup_model_safe(model)

        if n_frames == 0:
            raise ValueError(f"No frames could be read from video {path}")

        null_count = stats.get("null", 0)
        filled_count = stats.get("filled", 0)
        skipped_count = stats.get("skipped", 0)
        detected_frames = n_frames - null_count
        face_detect_ratio = float(detected_frames / n_frames) if n_frames > 0 else 0.0

        if len(bvp_raw) == 0:
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
            "video_path": str(path),
            "video_width": width,
            "video_height": height,
            "total_frames": total_frames,
            "timing_mode": timing_mode,
            "synchronous": synchronous,
            "decoded_frames": n_frames,
            "bvp_raw_length": len(bvp_raw),
            "face_detected_ratio": face_detect_ratio,
            "face_null_frames": null_count,
            "filled_frames": filled_count,
            "skipped_frames": skipped_count,
            "face_stats": stats,
            "raw_bvp": bvp_raw,
            "raw_timestamps": ts_raw,
            "orig_timestamps": timestamps,
        }
        return bvp_uniform, t_uniform, metadata

    def predict_from_frames(
        self,
        frames: List[np.ndarray],
        timestamps: Optional[np.ndarray] = None,
        timing_mode: str = "gt_timestamp",
        synchronous: bool = True,
        fs_target: float = 30.0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Predict BVP waveform directly from an in-memory list of RGB frames.

        Args:
            frames: List of 2D/3D uint8 RGB images.
            timestamps: Optional 1D array of capture timestamps in seconds.
            timing_mode: 'gt_timestamp' (pre-registered baseline of record), 'uniform', or 'container'.
            synchronous: If True, waits for face detection future per frame for 100% deterministic runs.
            fs_target: Target uniform sampling rate in Hz (default: 30.0).

        Returns:
            Tuple of (bvp_uniform, t_uniform, metadata).
        """
        n_frames = len(frames)
        if n_frames == 0:
            raise ValueError("No frames provided")

        model = self._create_model()

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

        # Start model manually (no `with model:` — we do our own safe teardown)
        model.__enter__()

        for idx, frame in enumerate(frames):
            if use_ts:
                model.update_frame(frame, float(ts_to_feed[idx]))
            else:
                model.update_frame(frame, float(idx / fs_target))
            if synchronous and model.face_detection_task is not None:
                model.face_detection_task.result()

        # Safe teardown: shut down ALL pools + extract BVP before any state is freed
        bvp_raw, ts_raw, stats = self._cleanup_model_safe(model)

        null_count = stats.get("null", 0)
        filled_count = stats.get("filled", 0)
        skipped_count = stats.get("skipped", 0)
        detected_frames = n_frames - null_count
        face_detect_ratio = float(detected_frames / n_frames) if n_frames > 0 else 0.0

        if len(bvp_raw) == 0:
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
            "orig_timestamps": timestamps,
        }

        return bvp_uniform, t_uniform, metadata


def predict_bvp(
    video_path: Union[str, Path],
    timestamps: Optional[np.ndarray] = None,
    timing_mode: str = "gt_timestamp",
    synchronous: bool = True,
    fs_target: float = 30.0,
    model_name: str = "FacePhys.rlap",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Predict BVP waveform from video using FacePhys baseline.

    Args:
        video_path: Path to video file.
        timestamps: Optional GT timestamps array (seconds).
        timing_mode: Timing mode ('gt_timestamp', 'uniform', or 'container').
        synchronous: If True, waits for face detection per frame for deterministic execution.
        fs_target: Target uniform sampling rate in Hz (default: 30.0).
        model_name: Model identifier (default: 'FacePhys.rlap').

    Returns:
        Tuple of (bvp_uniform, t_uniform, metadata).
    """
    wrapper = FacePhysWrapper(model_name=model_name)
    return wrapper.predict_video(
        video_path,
        timestamps=timestamps,
        timing_mode=timing_mode,
        synchronous=synchronous,
        fs_target=fs_target,
    )
