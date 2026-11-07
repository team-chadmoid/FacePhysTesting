# Cohort Evaluation Report v2: Reconciled Timing & Dual-Variant Benchmarking

**Git Commit Reference:** Tracking post-`baseline-v1` evaluation on branch `aaaa`  
**Dataset:** UBFC-rPPG Dataset 2 (7 subjects: `subject1`, `subject3`, `subject4`, `subject5`, `subject8`, `subject9`, `subject10`)  
**Pipeline:** Frozen FacePhys (`open-rppg`), Path 3 (GT-timestamp timing), sequential determinism (`synchronous=True`)  
**Compliance:** Strictly adheres to AGENTS.md Hard Rule 1 — every figure, metric, and count is sourced directly from CSVs in `results/`.

---

## 1. Frame Count Reconciliation & Decode Verification

### 1.1 Resolution of the Frame Count Discrepancy
In preliminary experiment logs, frame counts were occasionally quoted as **1,515 / 1,977 / 1,971 / 1,979**, whereas file sizes and ground-truth text logs implied **1,550 / 2,022 / 2,016 / 2,024**.

A systematic decode verification script ([`scripts/verify_frame_counts.py`](file:///c:/Users/MSI/Desktop/IEEE%20Signal%20Processing%20Cup%202027/IEEE%20SP%20CUP/IEEE-SP-CUP-2027-Chadmoid/scripts/verify_frame_counts.py)) was executed, decoding each subject's full video with both OpenCV (`cv2.VideoCapture`) and PyAV (`container.decode(video=0)`), comparing them directly against container stream metadata and the length of ground-truth rows in `ground_truth.txt`.

### Table 1: Frame Count Reconciliation Matrix (`results/frame_count_reconciliation.csv`)
| Subject | File Size (Bytes) | File Size (MB) | OpenCV Decoded Frames | PyAV Decoded Frames | Container Stream Frames | GT Timestamp Count | GT PPG Count | Frames Match GT? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`subject1`** | 1,425,830,568 | 1359.78 | **1,547** | **1,547** | 1,547 | 1,547 | 1,547 | **TRUE (100% Match)** |
| **`subject3`** | 1,659,925,096 | 1583.03 | **1,801** | **1,801** | 1,801 | 1,801 | 1,801 | **TRUE (100% Match)** |
| **`subject4`** | 199,630,138 | 190.38 | **1,368** | **1,368** | 1,368 | 1,368 | 1,368 | **TRUE (100% Match)** |
| **`subject5`** | 1,428,595,464 | 1362.41 | **1,550** | **1,550** | 1,550 | 1,550 | 1,550 | **TRUE (100% Match)** |
| **`subject8`** | 1,863,605,768 | 1777.27 | **2,022** | **2,022** | 2,022 | 2,022 | 2,022 | **TRUE (100% Match)** |
| **`subject9`** | 1,858,075,976 | 1772.00 | **2,016** | **2,016** | 2,016 | 2,016 | 2,016 | **TRUE (100% Match)** |
| **`subject10`** | 1,865,449,032 | 1779.03 | **2,024** | **2,024** | 2,024 | 2,024 | 2,024 | **TRUE (100% Match)** |

### 1.2 Origin of the Preliminary Numbers
- **Zero Decoded Frames Lost:** OpenCV and PyAV decoded counts match the container header and GT timestamp array with $100\%$ bit-exact equality.
- **The Receptive-Field Artifact:** In open-rppg's FacePhys implementation, the internal neural temporal receptive field / warm-up buffer requires 35 frames (for single-rate) or 45 frames (for multi-rate). 
  - `subject5`: $1550 - 35 = \mathbf{1515}$
  - `subject8`: $2022 - 45 = \mathbf{1977}$
  - `subject9`: $2016 - 45 = \mathbf{1971}$
  - `subject10`: $2024 - 45 = \mathbf{1979}$
  
  The numbers 1,515 / 1,977 / 1,971 / 1,979 represented the length of the raw unpadded BVP prediction array before temporal alignment, **not** the video frame count. The verified video frame counts are definitively **1,550 / 2,022 / 2,016 / 2,024**.

---

## 2. Dual-Variant Cohort Benchmark Table

To provide complete transparency without masking `subject4`'s native edge-to-edge cropping behavior, both **unpadded** (native 242×242) and **padded** (+60px reflect border) configurations are reported side-by-side in [`results/cohort_v2_7_subjects.csv`](file:///c:/Users/MSI/Desktop/IEEE%20Signal%20Processing%20Cup%202027/IEEE%20SP%20CUP/IEEE-SP-CUP-2027-Chadmoid/results/cohort_v2_7_subjects.csv).

### Table 2: Complete Per-Subject Benchmark Metrics
| Subject | Verified Video Frames | Effective Resolution | Framing Condition | Video HR (BPM) | Ref GT HR (BPM) | Video Error (BPM) | Direct $r$ | Max-Lag $r$ ($\pm 0.5\text{s}$) | Windowed MedAE (BPM) | Windowed Acc5 (%) | SNR (dB) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`subject1`** | 1,547 | 640×480 | Uncropped natural framing | 107.01 | 107.12 | 0.11 | 0.6639 | 0.8757 | 0.22 | 100.0% | +6.27 |
| **`subject3`** | 1,801 | 640×480 | Uncropped natural framing | 88.88 | 101.07 | 12.19 | 0.5378 | 0.8841 | 3.30 | 58.1% | -3.12 |
| **`subject4 (unpadded)`** | 1,368 | 242×242 | Native tight crop (0px border) | 62.40 | 111.18 | **48.78** | 0.0261 | 0.0168 | 48.34 | 35.3% | -9.14 |
| **`subject4 (padded)`** | 1,368 | 362×362 | Native crop + 60px reflect | 111.18 | 111.18 | **0.00** | 0.4018 | 0.5745 | 5.05 | 47.1% | -4.62 |
| **`subject5`** | 1,550 | 640×480 | Uncropped natural framing | 99.54 | 99.32 | 0.22 | 0.7861 | 0.8989 | 0.00 | 100.0% | +9.40 |
| **`subject8`** | 2,022 | 640×480 | Uncropped natural framing | 113.71 | 113.38 | 0.33 | 0.8598 | 0.9347 | 0.22 | 97.4% | +4.47 |
| **`subject9`** | 2,016 | 640×480 | Uncropped natural framing | 107.89 | 108.33 | 0.44 | 0.3938 | 0.8313 | 0.22 | 94.9% | +0.24 |
| **`subject10`** | 2,024 | 640×480 | Uncropped natural framing | 118.54 | 118.65 | 0.11 | 0.7513 | 0.9112 | 0.22 | 92.1% | +2.16 |

---

## 3. Cohort Aggregate Statistics Comparison

All values below are sourced directly from [`results/cohort_v2_summary.csv`](file:///c:/Users/MSI/Desktop/IEEE%20Signal%20Processing%20Cup%202027/IEEE%20SP%20CUP/IEEE-SP-CUP-2027-Chadmoid/results/cohort_v2_summary.csv):

### Table 3: Summary Comparison Between Padded and Unpadded Cohort Variants
| Benchmark Metric | Variant A: `subject4` Padded (+60px reflect) | Variant B: `subject4` Unpadded (Native 242×242) | Pilot Gate Requirement | Gate Status (Both Variants) |
| :--- | :---: | :---: | :---: | :---: |
| **Cohort Median Error (MedAE)** | **0.2197 BPM** | **0.3296 BPM** | $\le 2.0\text{ BPM}$ | **PASSED (Both)** |
| **Fraction with Waveform $r \ge 0.7$** | **85.7%** (6/7 subjects) | **85.7%** (6/7 subjects) | $\ge 70.0\%$ | **PASSED (Both)** |
| **Cohort Mean Absolute Error (MAE)** | **1.9148 BPM** | **8.8832 BPM** | — | — |
| **Cohort Root Mean Square Error (RMSE)**| **4.6150 BPM** | **19.0057 BPM** | — | — |
| **Cohort Pearson Correlation ($r$)** | **0.9106** | **0.2967** | — | — |
| **Bland-Altman Mean Bias** | **-1.7578 BPM** | **-8.7263 BPM** | — | — |
| **Bland-Altman 95% LoA** | **[-10.79, +7.28] BPM** | **[-44.47, +27.02] BPM** | — | — |

### Key Takeaways
1. **Gate Robustness:** Even when `subject4` is evaluated completely unpadded (yielding a 48.78 BPM error), the cohort **Median Absolute Error is 0.3296 BPM**, comfortably passing the $\le 2.0\text{ BPM}$ gate, and **85.7%** of subjects retain waveform $r \ge 0.7$ (passing the $\ge 70\%$ gate).
2. **Impact of Native Face Boundary Touching:** `subject4`'s native tight crop causes a catastrophic failure if unpadded (48.78 BPM error, $r=0.0168$), heavily skewing cohort MAE ($1.91 \to 8.88\text{ BPM}$) and cohort Pearson $r$ ($0.91 \to 0.30$).

---

## 4. Golden Check Script Enhancements (`scripts/golden_check.py`)

Per task requirements:
1. **Tolerance Mode:** Added `--tolerance-mode` (`-t`) to [`scripts/golden_check.py`](file:///c:/Users/MSI/Desktop/IEEE%20Signal%20Processing%20Cup%202027/IEEE%20SP%20CUP/IEEE-SP-CUP-2027-Chadmoid/scripts/golden_check.py) enabling relaxed cross-architecture verification (HR $\pm 0.5\text{ BPM}$, $r \pm 0.05$) while preserving strict bit-exact verification as the default.
2. **Hardware Environment Metadata:** Every execution now logs the exact host system specs:
   - Operating System & Kernel Version: `Windows 11 (10.0.26200)`
   - CPU / Processor identification string: `Intel64 Family 6 Model 154 Stepping 4, GenuineIntel`
   - Logical CPU Thread Count: `8`
   - OpenCV execution threads: `8`
   - Active Python version: `3.13.5`

---

## 5. Padding Generalization Sweep & Hypothesis Testing (Task 3)

### 5.1 Protocol & Decision Rule
To determine whether border padding should be adopted generally as a function of face-to-frame ratio, artificial tight crops (face touching boundaries, 0px margin) were created for all 6 naturally framed subjects (`subject1`, `subject3`, `subject5`, `subject8`, `subject9`, `subject10`). Symmetric reflect padding (`cv2.BORDER_REFLECT`) was swept across fractions of crop width: **0% (unpadded tight), 15%, 30%, 45%, 60%**.

**Pre-registered Decision Rule:**
Only propose a general rule in terms of face-to-frame ratio if the padding trend replicates on at least **4 of 6 subjects**; otherwise, maintain it strictly as an observation for `subject4` only.

### Table 4: Padding Sweep Summary across 6 Subjects (`results/padding_generalization_sweep.csv`)
| Subject | Tight Crop Box | 0% Pad (Tight) Err | Best Padded Err | 0% Pad $r$ | Best Padded $r$ | 0% Pad SNR | Best Padded SNR | Best Fraction | Padding Meaningfully Helps? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`subject1`** | 135×165 | **0.00 BPM** | 0.11 BPM | 0.8393 | 0.8482 | +5.91 dB | +5.50 dB | 30% | **False** (Baseline already perfect) |
| **`subject3`** | 138×168 | 12.19 BPM | **8.68 BPM** | 0.8092 | 0.8418 | -3.93 dB | -3.82 dB | 45% | **True** (Slight error reduction) |
| **`subject5`** | 150×183 | 0.22 BPM | **0.11 BPM** | 0.8471 | 0.8496 | +9.43 dB | +6.48 dB | 30% | **False** (Baseline already <0.3 BPM) |
| **`subject8`** | 170×208 | 0.55 BPM | **0.44 BPM** | 0.7939 | 0.9191 | +3.85 dB | +4.42 dB | 15% | **False** (Baseline already <0.6 BPM) |
| **`subject9`** | 126×153 | 0.77 BPM | **0.11 BPM** | 0.7477 | 0.7957 | -0.74 dB | +1.01 dB | 60% | **False** (Baseline already <0.8 BPM) |
| **`subject10`** | 139×169 | **0.00 BPM** | 0.00 BPM | 0.8958 | 0.8951 | +1.82 dB | +1.71 dB | 30% | **False** (Baseline already perfect) |

### 5.2 Formal Decision
- **Subjects where padding meaningfully helped:** **1 of 6 subjects** (`subject3`).
- On subjects 1, 5, 8, 9, and 10, the unpadded tight crop (face touching boundaries, face-to-frame ratio = 1.0) already yielded video errors $\le 0.77\text{ BPM}$ and waveform correlation $r \ge 0.75 - 0.90$.
- **Conclusion:** The trend **DOES NOT REPLICATE** across the cohort ($1/6 < 4/6$).
- **Formal Action:** We **do not propose a general padding rule in terms of face-to-frame ratio**. The edge-border degradation effect is maintained strictly as an isolated observation for `subject4` only.


