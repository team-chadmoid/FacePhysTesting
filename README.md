# FacePhys Baseline Evaluation — UBFC Low-Light (C1 Condition)

> **NOTE**
> Git hash: `ee993a4` (lux conditions) + `2758b56` (original_baseline via ffmpeg re-encode)
> Source: [`results/ubfc_lowlight_conditions.csv`](results/ubfc_lowlight_conditions.csv)
> Total evaluated: **240 runs (42 subjects × 6 conditions, minus 12 missing from lux50)**

The **FacePhys** baseline was evaluated across the **UBFC_lowlight** dataset variations and the
original baseline directly from Google Drive, per
[`notebooks/run_ubfc_lowlight_colab.ipynb`](notebooks/run_ubfc_lowlight_colab.ipynb)
(clone branch `aaaa`, run `scripts/run_colab_lowlight.py`).

All metrics are in **BPM** (beats per minute). MAE / RMSE / MedAE = video-level HR error vs
primary GT (Welch PPG). Pearson r = correlation across subjects within condition.
Face% = mean face detection ratio. SNR = spectral signal-to-noise ratio around the GT HR peak.
Bland–Altman bias = mean(pred − ref), LoA = bias ± 1.96 × SD of differences (sample SD,
`ddof=1`, matching `src/eval.py::calc_bland_altman`).

---

## Condition-Level Summary

| Condition | n | Video MAE | Video RMSE | Video MedAE | Pearson r | Acc ≤ 5 BPM | Acc ≤ 10 BPM | Face% | SNR (dB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| original_baseline | 42 | 1.19 | 2.82 | 0.22 | 0.990 | 92.9% | 95.2% | 100% | 2.2 |
| lux50 | 30 | 12.64 | 18.03 | 7.64 | 0.538 | 40.0% | 56.7% | 100% | −6.2 |
| lux30 | 42 | 23.31 | 28.65 | 19.17 | 0.019 | 16.7% | 26.2% | 100% | −7.8 |
| lux10 | 42 | 36.34 | 43.46 | 33.29 | 0.069 | 11.9% | 19.0% | 52% | −22.1 |
| lux5 | 42 | 56.32 | 59.55 | 58.12 | 0.023 | 0.0% | 0.0% | 3% | −47.0 |
| lux1 | 42 | 57.88 | 60.81 | 58.12 | N/A | 0.0% | 0.0% | 0% | −50.0 |

At ≤5 lux the prediction is a constant ~42.08 BPM fallback (42 sentinel videos at lux1,
39 at lux5, 14 at lux10), so the Pearson r across subjects is undefined (N/A) at lux1.

## Window-Level Summary (30 s windows, 1 s stride)

Per-video windowed metrics, averaged across videos per condition (`all_win_*` columns).

| Condition | n | Win MAE | Win MedAE | Win RMSE | Win r (mean) | Win Acc ≤ 5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| original_baseline | 42 | 1.02 | 0.25 | 2.19 | 0.850 | 94.5% |
| lux50 | 30 | 16.23 | 14.40 | 19.87 | 0.142 | 33.4% |
| lux30 | 42 | 21.96 | 21.73 | 24.94 | −0.023 | 17.5% |
| lux10 | 42 | 43.59 | 43.71 | 45.23 | 0.173 | 4.7% |
| lux5 | 42 | 56.62 | 57.09 | 56.87 | −0.036 | 0.0% |
| lux1 | 42 | 57.25 | 57.59 | 57.41 | 0.000 | 0.0% |

## Bland–Altman Limits of Agreement

Bias/Sd/LoA of video-level predicted HR vs primary reference HR
(differences = pred − ref; 95% LoA = bias ± 1.96 × sample SD, per `src/eval.py`).

| Condition | Bias (BPM) | SD (BPM) | 95% LoA |
| :--- | :---: | :---: | :---: |
| original_baseline | −0.69 | 2.76 | [−6.10, +4.73] |
| lux50 | −6.66 | 17.04 | [−40.06, +26.75] |
| lux30 | −15.39 | 24.47 | [−63.34, +32.57] |
| lux10 | −26.48 | 34.87 | [−94.83, +41.87] |
| lux5 | −56.32 | 19.60 | [−94.74, −17.89] |
| lux1 | −57.88 | 18.88 | [−94.89, −20.88] |

---

## Key Observations

### Baseline Performance (original_baseline)
FacePhys achieves excellent performance on well-lit UBFC data: MAE 1.19 BPM, r = 0.990,
92.9% of videos within 5 BPM. Bland–Altman bias is −0.69 BPM with tight LoA [−6.10, +4.73].

### Degradation Under Low Light (C1 Challenge)
- **lux50 (50 lux):** Still usable — MAE 12.6 BPM, 40% within 5 BPM, face detection 100%.
- **lux30 (30 lux):** Severe degradation — MAE 23.3 BPM, face detection still 100% but
  correlation near zero.
- **lux10 (10 lux):** Near failure — MAE 36.3 BPM, face detection drops to 52%.
- **lux5 / lux1 (≤5 lux):** Total failure — face detection ≤3%, model outputs a constant
  ~42 BPM fallback.

### Face Detection as the Primary Bottleneck
At ≤5 lux, the BlazeFace ONNX detector finds zero faces → the model falls back to
zero-filled inputs → constant 42.08 BPM prediction regardless of true HR.
At 10 lux, detection is intermittent (52%) → noisy BVP signal → poor HR estimation.
At ≥30 lux, detection is 100% but the dark image still degrades the rPPG signal quality.

> [!IMPORTANT]
> The primary opportunity for improvement under C1 is a more robust face detector
> (or a detector-free approach) at ≤10 lux, followed by low-light-adapted rPPG signal
> extraction at 30–50 lux.

## Cohort Notes
- **lux50:** Only 30 of 42 subjects discovered (12 missing — subject3, 4, 8–17 — due to
  Google Drive shortcut resolution / FUSE listing incompleteness). Of the 30 lux50 rows,
  28 ran at `ee993a4c` and 2 (subject1, subject5) at `97aad6b`.
- **original_baseline:** Processed via ffmpeg H.264 re-encoding (CRF 18) because Colab's
  Python runtime segfaults reading 1.4 GB rawvideo files from Drive FUSE; ffmpeg's C-level
  I/O handles the files correctly.
- `results/exclusions.csv` is empty: no videos were silently dropped; the 12 undiscovered
  lux50 subjects are logged here instead, per AGENTS.md rule 4.

## Reproduction

1. Add Drive shortcuts for `UBFC_lowlight` (and optionally `Data Set 2`) to My Drive,
   and set the Colab runtime to **T4 GPU**.
2. Open [`notebooks/run_ubfc_lowlight_colab.ipynb`](notebooks/run_ubfc_lowlight_colab.ipynb)
   or run its kernel steps locally: mount Drive → clone branch `aaaa` → install pinned deps
   (`open-rppg`, `onnxruntime`, ...) → `python scripts/run_colab_lowlight.py --data-dir
   /content/drive/MyDrive/UBFC_lowlight`.
3. Results land in `results/ubfc_lowlight_conditions.csv`; the notebook's final cell
   re-derives the condition-level aggregates via a plain `groupby('condition').mean()`.
