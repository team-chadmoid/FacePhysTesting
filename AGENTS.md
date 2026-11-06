# AGENTS.md: IEEE SP Cup 2027 (FaceVitals), Round 1: HR from RGB face video

## Goal
Estimate heart rate (BPM) from RGB face video. Round 1 due 30 Nov 2026.
Baseline = pretrained FacePhys. Novelty = robustness under at least 2 of C1-C7
(C1 low light <=50 lux, C4 low resolution, C5 <=10 fps / <=500 kbps, C7 mask, ...).
Metrics: MAE, RMSE, Pearson r, Bland-Altman limits of agreement (all BPM).

## Hard rules
1. Never fabricate numbers. Every figure or table value comes from a CSV in results/
   written by a script in this repo, with the git hash recorded.
2. FacePhys weights are licensed for LOCAL inference only. No cloud notebooks,
   no uploading videos, no server-side runs. Read the LICENSE before any deployment change.
3. data/ is gitignored. Never commit or upload organizer/hospital data.
4. Do not silently drop videos or "correct" ground-truth offsets. Log every exclusion
   with a reason in results/exclusions.csv.
5. Fixed seeds. Each experiment = one config file + one script.
6. One task per commit. Run `pytest -q` before committing.
7. Ask before adding heavy dependencies or changing the evaluation protocol.

## Layout
docs/      official SP Cup PDF, FacePhys paper, CanonicalPhys paper (read-only reference)
data/      ubfc/, aipex/, organizer/ (gitignored)
src/       io.py (dataset loaders), hr.py (HR estimator), eval.py (metrics),
           models.py (open-rppg wrapper, FacePhys-Release ONNX wrapper, POS, CHROM),
           degrade.py (low light, resolution, fps, bitrate, mask), export_mat.py
configs/   one YAML per experiment
results/   CSVs, plots, exclusions.csv
tests/     pytest

## Evaluation protocol (single source of truth: src/eval.py)
- HR from BVP: 3rd-order Butterworth bandpass 0.7-3.0 Hz, Welch PSD, peak frequency x 60.
- Windows: 30 s, 1 s stride. Also report one video-level HR per video.
- Ground truth: HR derived from the GT PPG with the same estimator; also compare
  to the provided HR line. Flag videos where the two disagree by more than 5 BPM.
- Report per-dataset and per-condition tables, plus Bland-Altman plots.

## Conditions to simulate (src/degrade.py, applied to well-lit baseline video)
- C1: multiplicative gain <1 plus Poisson/Gaussian noise
- C4: downscale to 640x480 and 320x240 (bicubic)
- C5: 10 fps decimation; re-encode at <=500 kbps with ffmpeg
- C7: occlude lower 40% of the face using landmarks

## Round 1 deliverables (match naming exactly)
Round1_Baseline_None.mat with variables: facial_video, roi_data, reference_signal,
processed_signal, metrics, params. Also: process_round1.py, target_video.mp4,
reference_signal.csv, processed_output.csv, and an IEEE-format technical report.

## Workflow for the agent
Plan first, then implement. Start each task by stating what files you will touch.
Test the HR estimator on synthetic sinusoids before using real data.
When a result looks too good, check for leakage or a metrics bug before reporting it.