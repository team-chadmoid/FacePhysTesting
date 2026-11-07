"""Inspect video format, codec, pix_fmt, bitrate, and bytes per pixel using PyAV.

Directly wraps FFmpeg's libavformat / libavcodec to extract:
- codec name & long name
- pixel format
- bitrate (container & stream)
- width, height (frame size)
- total frames & duration
- total file size (bytes & MB)
- bytes per pixel: total_file_size / (width * height * num_frames)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import av
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io import find_ubfc_subjects, load_ubfc_subject


def inspect_video_pyav(vid_path: Path) -> Dict[str, Any]:
    container = av.open(str(vid_path))
    stream = container.streams.video[0]

    codec = stream.codec_context.name
    codec_long = stream.codec_context.codec.long_name if stream.codec_context.codec else "unknown"
    pix_fmt = stream.format.name if stream.format else "unknown"
    w = stream.width
    h = stream.height
    frames = stream.frames

    # If stream.frames is 0, count packets/frames
    if frames == 0:
        frames = sum(1 for _ in container.decode(video=0))
        container.seek(0)

    duration_sec = float(stream.duration * stream.time_base) if stream.duration else (float(container.duration) / 1e6 if container.duration else 0.0)
    container_bitrate = container.bit_rate or 0
    stream_bitrate = stream.bit_rate or 0
    effective_bitrate = stream_bitrate if stream_bitrate > 0 else container_bitrate

    file_size_bytes = os.path.getsize(str(vid_path))
    file_size_mb = file_size_bytes / (1024.0 * 1024.0)

    # Bytes per pixel = file_size_bytes / (width * height * frames)
    # Uncompressed 8-bit BGR is exactly 3.0000 bytes/pixel (or slightly >3.0 due to AVI header).
    # Compressed video is typically << 1.0 byte/pixel.
    total_pixels = w * h * frames
    bytes_per_pixel = file_size_bytes / total_pixels if total_pixels > 0 else 0.0

    container.close()

    return {
        "video_path": str(vid_path),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 2),
        "codec": codec,
        "codec_long": codec_long,
        "pix_fmt": pix_fmt,
        "width": w,
        "height": h,
        "frame_size": f"{w}x{h}",
        "frames": frames,
        "duration_sec": round(duration_sec, 3),
        "bitrate_bps": effective_bitrate,
        "bitrate_kbps": round(effective_bitrate / 1000.0, 1) if effective_bitrate else round((file_size_bytes * 8.0) / (duration_sec * 1000.0), 1),
        "bytes_per_pixel": round(bytes_per_pixel, 4),
        "is_uncompressed": (bytes_per_pixel >= 2.5),
    }


def print_ffprobe_format(vid_path: Path, subject_id: str):
    container = av.open(str(vid_path))
    s = container.streams.video[0]
    dur = float(s.duration * s.time_base) if s.duration else (float(container.duration) / 1e6 if container.duration else 0.0)
    
    print(f"\n==================== FFPROBE -show_streams -show_format: {subject_id} ====================")
    print("[STREAM]")
    print(f"index={s.index}")
    print(f"codec_name={s.codec_context.name}")
    print(f"codec_long_name={s.codec_context.codec.long_name if s.codec_context.codec else 'N/A'}")
    print("codec_type=video")
    print(f"codec_tag_string={s.codec_context.codec_tag}")
    print(f"width={s.width}")
    print(f"height={s.height}")
    print(f"pix_fmt={s.format.name}")
    print(f"r_frame_rate={s.average_rate}")
    print(f"avg_frame_rate={s.average_rate}")
    print(f"time_base={s.time_base}")
    print(f"duration={dur:.6f}")
    print(f"bit_rate={container.bit_rate or s.bit_rate or 0}")
    print(f"nb_frames={s.frames}")
    print("[/STREAM]")
    print("[FORMAT]")
    print(f"filename={vid_path}")
    print(f"nb_streams={len(container.streams)}")
    print(f"format_name={container.format.name}")
    print(f"format_long_name={container.format.long_name}")
    print(f"duration={float(container.duration)/1e6 if container.duration else 0.0:.6f}")
    print(f"size={os.path.getsize(str(vid_path))}")
    print(f"bit_rate={container.bit_rate or 0}")
    print("[/FORMAT]\n")
    container.close()


def main():
    print("=" * 95)
    print("=== TASK 1: FFPROBE / PYAV STREAM & CODEC INSPECTION ===")
    print("=" * 95)

    dataset_root = PROJECT_ROOT / "data" / "ubfc" / "Data Set 2"
    subjects_map = find_ubfc_subjects(dataset_root)

    target_subjects = ["subject1", "subject3", "subject4", "subject5", "subject8", "subject9", "subject10"]
    rows = []

    for sid in target_subjects:
        sdir = subjects_map[sid]
        sdata = load_ubfc_subject(sdir, subject_id=sid)
        info = inspect_video_pyav(sdata.video_path)
        info["subject_id"] = sid
        rows.append(info)

        if sid in ["subject4", "subject5"]:
            print_ffprobe_format(sdata.video_path, sid)

    df = pd.DataFrame(rows)
    cols = [
        "subject_id", "codec", "pix_fmt", "frame_size", "file_size_mb",
        "frames", "bitrate_kbps", "bytes_per_pixel", "is_uncompressed"
    ]
    print("\n" + "=" * 95)
    print("=== VIDEO FORMAT & BYTES PER PIXEL ACROSS ALL 7 SUBJECTS ===")
    print("=" * 95)
    print(df[cols].to_string(index=False))
    print("=" * 95)

    out_csv = PROJECT_ROOT / "results" / "video_compression_inspection.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved inspection results to: {out_csv}")


if __name__ == "__main__":
    main()
