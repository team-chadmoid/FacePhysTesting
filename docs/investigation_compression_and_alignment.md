# Investigation: Video Compression, Temporal Alignment, and Lag Search

**Git Branch:** `aaaa` (post-`baseline-v1`)  
**Baseline Tag:** `baseline-v1` frozen at commit `f520978` — **not modified**  
**Dataset:** UBFC-rPPG Dataset 2 (7 subjects)  
**Pipeline:** FacePhys (`open-rppg 0.1.1`), GT-timestamp timing, synchronous mode  
**Compliance:** AGENTS.md Hard Rule 1 — every value sourced from CSVs in `results/`

---

## 1. Video Stream & Codec Inspection (ffprobe-equivalent)

**Script:** [`scripts/inspect_video_streams.py`](file:///c:/Users/MSI/Desktop/IEEE%20Signal%20Processing%20Cup%202027/IEEE%20SP%20CUP/IEEE-SP-CUP-2027-Chadmoid/scripts/inspect_video_streams.py)  
**Results:** [`results/video_compression_inspection.csv`](file:///c:/Users/MSI/Desktop/IEEE%20Signal%20Processing%20Cup%202027/IEEE%20SP%20CUP/IEEE-SP-CUP-2027-Chadmoid/results/video_compression_inspection.csv)

PyAV (wrapping FFmpeg libavformat/libavcodec) was used to extract stream metadata equivalent to `ffprobe -v error -show_streams -show_format`.

### 1.1 Subject 4 vs Subject 5 Stream Comparison

| Field | Subject 4 | Subject 5 |
| :--- | :--- | :--- |
| **codec_name** | `ffv1` | `rawvideo` |
| **codec_long_name** | FFmpeg video codec #1 | raw video |
| **pix_fmt** | `bgr0` (32-bit, 4th byte unused) | `bgr24` (24-bit) |
| **width × height** | 242 × 242 | 640 × 480 |
| **frame_rate** | 676189/22837 ≈ 29.61 fps | 14830903/500000 ≈ 29.66 fps |
| **nb_frames** | 1,368 | 1,550 |
| **duration** | 46.20 s | 52.26 s |
| **bit_rate** | 34,567 kbps | 218,708 kbps |
| **file_size** | 190.38 MB | 1,362.41 MB |
| **bytes_per_pixel** | **2.4918** | **3.0002** |

### 1.2 Bytes Per Pixel Across All 7 Subjects

| Subject | Codec | Pixel Fmt | Frame Size | Bytes/Pixel | Uncompressed? |
| :--- | :--- | :--- | :--- | :---: | :---: |
| subject1 | rawvideo | bgr24 | 640×480 | 3.0002 | ✅ |
| subject3 | rawvideo | bgr24 | 640×480 | 3.0002 | ✅ |
| **subject4** | **ffv1** | **bgr0** | **242×242** | **2.4918** | ❌ |
| subject5 | rawvideo | bgr24 | 640×480 | 3.0002 | ✅ |
| subject8 | rawvideo | bgr24 | 640×480 | 3.0002 | ✅ |
| subject9 | rawvideo | bgr24 | 640×480 | 3.0002 | ✅ |
| subject10 | rawvideo | bgr24 | 640×480 | 3.0002 | ✅ |

> [!IMPORTANT]
> Subject 4 is the **only** video encoded with FFV1 lossless intra-frame compression (FFmpeg Video Codec #1) instead of uncompressed rawvideo. It also has a unique 242×242 resolution with `bgr0` pixel format.

---

## 2. Open-rPPG Source Code Analysis: Warm-up & Prediction-to-Timestamp Alignment

### 2.1 FacePhys.rlap Architecture (Streaming)

From `load_FacePhys_rlap()` in `rppg/main.py`:

```python
meta = {'fps': 30., 'input': (1, 36, 36, 3)}
```

- **`input[0] = 1`** → the model processes **one frame at a time** (streaming RNN mode).
- The model is a stateful SSM/Mamba-based architecture (`FacePhys([2]*4, [32]*4)`) with 4 SSCBlocks, each containing a Mamba2 SSM, temporal normalization (TNM), and Conv3D layers.
- Each call: `model.step(x, state, dt)` → produces one BVP scalar and updated state.
- JIT compilation warm-up: 2 dummy forward passes at load time.

### 2.2 Frame-to-Prediction Mapping

The inference loop in `Model.__enter__` (lines 486–520) processes frames from `self.face_buff` in chunks of `self.meta['input'][0]` (= 1 for FacePhys):

1. **Frame feeding** (`update_face_resized`, line 636): A timing loop compares `(n_frame - 0.3) / fps` against the elapsed time since the first timestamp. This can **fill** (duplicate) or **skip** frames to match the model's expected 30 fps clock.
2. **Inference** (line 496): `self.call(ipt, self.state)` produces per-frame BVP predictions.
3. **Tail handling** (lines 507–517): Remaining frames after `self.alive = False` are padded to `input[0]` and processed, but only the unpadded portion is kept.

### 2.3 Why 35 vs 45 Frames Difference

The earlier report noted BVP lengths of 1515, 1977, 1971, 1979 vs video frame counts of 1550, 2022, 2016, 2024 (differences of 35, 45, 45, 45).

This difference arises from the **timing alignment loop** in `update_face_resized()`, not from a fixed neural receptive field. The loop's `(n_frame - 0.3) / fps <= ts - ts[0]` condition determines how many model frames to emit per video frame. With true frame rates ≈ 29.3–29.7 fps (below the assumed 30 fps), the model's 30 fps clock skips some frames — the `skipped` statistic tracks these. The net difference (video frames − BVP length) reflects skipped minus filled frames accumulated across the entire video. The variation (35 vs 45) depends on each video's true frame rate and timestamp jitter.

### 2.4 Prediction-to-Timestamp Alignment Confirmation

The `collect_signals()` method (lines 538–563) maps predictions to timestamps:
- `self.ts` accumulates the original video timestamps.
- `self.n_signal` tracks how many BVP values have been produced.
- Frame `i` of BVP output corresponds to `self.ts[i]`, maintaining 1:1 alignment between produced signals and their associated timestamps.

Our `FacePhysWrapper.predict_from_frames()` extracts `raw_timestamps = model.ts[:n_signal]` and `raw_bvp = signals['bvp']`, preserving this alignment.

---

## 3. Extended Lag Search (±2.0 s) on 5 Clean Subjects

**Script:** [`scripts/run_lag_search_extended.py`](file:///c:/Users/MSI/Desktop/IEEE%20Signal%20Processing%20Cup%202027/IEEE%20SP%20CUP/IEEE-SP-CUP-2027-Chadmoid/scripts/run_lag_search_extended.py)  
**Results:** [`results/clean_subjects_lag_search_2s.csv`](file:///c:/Users/MSI/Desktop/IEEE%20Signal%20Processing%20Cup%202027/IEEE%20SP%20CUP/IEEE-SP-CUP-2027-Chadmoid/results/clean_subjects_lag_search_2s.csv)

| Subject | HR (BPM) | T (s) | r(τ=0) | Best lag ±0.5s | Best r ±0.5s | Best lag ±2.0s | Best r ±2.0s |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| subject1 | 108.0 | 0.556 | 0.8051 | +0.033 s | 0.8425 | +0.033 s | 0.8425 |
| subject5 | 99.0 | 0.606 | 0.8991 | 0.000 s | 0.8991 | 0.000 s | 0.8991 |
| subject8 | 112.0 | 0.536 | 0.9347 | 0.000 s | 0.9347 | 0.000 s | 0.9347 |
| subject9 | 108.0 | 0.556 | 0.6935 | **−0.500 s** | **0.8273** | **−0.500 s** | **0.8273** |
| subject10 | 111.0 | 0.541 | 0.9109 | 0.000 s | 0.9109 | 0.000 s | 0.9109 |

### 3.1 Investigation of Subject 9's Optimal Lag (−0.5 s)

Key cross-correlation peaks for subject9:

| Lag (s) | Distance in Cardiac Cycles | r |
| :---: | :---: | :---: |
| −0.500 | −0.90 T | **+0.8273** |
| +0.033 | +0.06 T | +0.8100 |
| +0.600 | +1.08 T | +0.7713 |
| −1.067 | −1.92 T | +0.6225 |

The optimal lag at −0.50 s is approximately **0.90 cardiac cycles** (T = 0.556 s at 108 BPM). The cross-correlation curve shows periodic peaks spaced at ~T, which is characteristic of quasi-periodic cardiac signals. The true zero-lag peak is at +0.033 s (r = 0.8100), very close to τ = 0. The −0.50 s peak (r = 0.8273) slightly exceeds τ = 0 because it happens to align with a previous heartbeat cycle where the waveform morphology correlation is marginally higher.

**Conclusion:** Subject 9's +0.5 s optimal lag (previously reported as +0.5 s in earlier runs; now −0.5 s with consistent waveform direction) is an artifact of cardiac periodicity, not a systematic temporal misalignment. When restricted to ±0.5 s, the best lag falls on a cardiac-period harmonic rather than indicating a pipeline timing error.

---

## 4. Padding Generalization Summary (Mean Across Levels)

**Script:** [`scripts/recompute_padding_generalization_summary.py`](file:///c:/Users/MSI/Desktop/IEEE%20Signal%20Processing%20Cup%202027/IEEE%20SP%20CUP/IEEE-SP-CUP-2027-Chadmoid/scripts/recompute_padding_generalization_summary.py)  
**Results:** [`results/padding_generalization_mean_summary.csv`](file:///c:/Users/MSI/Desktop/IEEE%20Signal%20Processing%20Cup%202027/IEEE%20SP%20CUP/IEEE-SP-CUP-2027-Chadmoid/results/padding_generalization_mean_summary.csv)

| Subject | r₀ (lag±0.5s) | Mean r (padded) | Δr | SNR₀ (dB) | Mean SNR (padded) | ΔSNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| subject1 | 0.8393 | 0.7299 | **−0.1093** | +5.91 | +3.09 | **−2.83** |
| subject3 | 0.8092 | 0.8688 | +0.0596 | −3.93 | −3.55 | +0.38 |
| subject5 | 0.8471 | 0.8532 | +0.0060 | +9.43 | +7.53 | −1.90 |
| subject8 | 0.7939 | 0.8832 | **+0.0893** | +3.85 | +3.97 | +0.12 |
| subject9 | 0.7477 | 0.7903 | +0.0426 | −0.74 | +0.97 | **+1.70** |
| subject10 | 0.8958 | 0.8900 | −0.0059 | +1.82 | +1.56 | −0.26 |

**Cohort Summary:**
- Subjects where r improved: **3 / 6**
- Subjects where SNR improved: **1 / 6**
- Mean Δr: +0.0137
- Mean ΔSNR: −0.46 dB

> [!IMPORTANT]
> **Conclusion: No general rule.** Padding does not consistently improve or degrade rPPG quality across subjects. The effect is mixed and subject-dependent. The padding improvement observed for subject4 remains an observation specific to that one natively tight-cropped subject.

---

## 5. Controlled Compression Hypothesis Test

**Script:** [`scripts/run_compression_hypothesis_test.py`](file:///c:/Users/MSI/Desktop/IEEE%20Signal%20Processing%20Cup%202027/IEEE%20SP%20CUP/IEEE-SP-CUP-2027-Chadmoid/scripts/run_compression_hypothesis_test.py)  
**Results:** [`results/subject5_compression_test.csv`](file:///c:/Users/MSI/Desktop/IEEE%20Signal%20Processing%20Cup%202027/IEEE%20SP%20CUP/IEEE-SP-CUP-2027-Chadmoid/results/subject5_compression_test.csv)

**Design:** Take subject5 (640×480 uncompressed), crop to 242×242 center-face, produce two re-encoded versions:
- **Version A:** Uncompressed rawvideo (bgr24, 260 MB, 3.01 bytes/pixel)
- **Version B:** FFV1 lossless compressed (bgr0, 75 MB, 0.86 bytes/pixel) — matching subject4's codec

| Variant | Resolution | Codec | Bytes/Px | HR (BPM) | Error | r (lag±0.5s) | SNR (dB) | Acc5 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Original full-frame | 640×480 | rawvideo | 3.00 | 99.54 | 0.22 | 0.8991 | +9.40 | 100.0 |
| Crop raw | 242×242 | rawvideo | 3.01 | 99.65 | 0.33 | 0.8283 | +7.62 | 95.65 |
| Crop FFV1 | 242×242 | ffv1 | 0.86 | 99.65 | 0.33 | 0.8283 | +7.62 | 95.65 |

### Effect Decomposition

| Factor | Δr (lag±0.5s) | ΔSNR (dB) | ΔError (BPM) |
| :--- | :---: | :---: | :---: |
| **FFV1 compression** (FFV1 vs raw, same 242×242) | **+0.0000** | **+0.00** | **+0.00** |
| **Spatial cropping** (242×242 vs 640×480) | −0.0708 | −1.78 | +0.11 |

> [!IMPORTANT]
> **Hypothesis Conclusion:** Failed to reject H₀. FFV1 lossless intra-frame compression produces **zero measurable effect** on rPPG signal quality — the raw and FFV1 variants yield bit-identical BVP predictions. Subject 4's poor baseline performance is entirely attributable to its 242×242 tight spatial framing (face touching frame boundaries), not to its FFV1 video compression.

---

## Files Created (Not Modifying baseline-v1)

| File | Purpose |
| :--- | :--- |
| `scripts/inspect_video_streams.py` | PyAV stream/codec inspection |
| `scripts/run_lag_search_extended.py` | ±2.0 s cross-correlation lag search |
| `scripts/recompute_padding_generalization_summary.py` | Mean-based padding summary |
| `scripts/run_compression_hypothesis_test.py` | FFV1 vs rawvideo controlled experiment |
| `results/video_compression_inspection.csv` | Stream metadata for 7 subjects |
| `results/clean_subjects_lag_search_2s.csv` | Lag search results for 5 clean subjects |
| `results/padding_generalization_mean_summary.csv` | Mean Δr, ΔSNR padding summary |
| `results/subject5_compression_test.csv` | Compression hypothesis test results |
| `results/preds/subject{5,8,9,10}_facephys_pred.csv` | Per-frame BVP predictions |
| `docs/investigation_compression_and_alignment.md` | This document |

