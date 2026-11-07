"""Unit tests for src/eval.py."""

from pathlib import Path
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.eval import (
    calc_mae,
    calc_medae,
    calc_fraction_within_threshold,
    calc_rmse,
    calc_pearson_r,
    calc_bland_altman,
    calc_metrics,
    calculate_snr,
    get_top_spectral_peaks,
    max_lag_waveform_correlation,
    evaluate_ground_truth_consistency,
    log_exclusion,
    plot_bland_altman,
    plot_bland_altman_across_subjects,
)


def test_metric_calculations():
    y_true = np.array([60.0, 70.0, 80.0, 90.0, 100.0])
    y_pred = np.array([62.0, 69.0, 83.0, 88.0, 101.0])

    mae = calc_mae(y_true, y_pred)
    # diffs: +2, -1, +3, -2, +1 -> abs diffs: 2, 1, 3, 2, 1 -> sum=9 / 5 = 1.8
    assert np.isclose(mae, 1.8)

    rmse = calc_rmse(y_true, y_pred)
    # sq diffs: 4, 1, 9, 4, 1 = 19 / 5 = 3.8 -> sqrt(3.8) = 1.94935886896
    assert np.isclose(rmse, np.sqrt(3.8))

    r = calc_pearson_r(y_true, y_pred)
    assert r > 0.98

    ba = calc_bland_altman(y_true, y_pred)
    # diffs: [2, -1, 3, -2, 1], mean = 1.0 / 1 = 0.6
    assert np.isclose(ba["bias"], 0.6)
    assert ba["upper_loa"] > ba["bias"]
    assert ba["lower_loa"] < ba["bias"]

    metrics = calc_metrics(y_true, y_pred)
    assert metrics["mae"] == mae
    assert metrics["rmse"] == rmse
    assert metrics["pearson_r"] == r
    assert metrics["bias"] == ba["bias"]


def test_gt_consistency():
    fs = 30.0
    duration = 45.0
    t = np.arange(0, duration, 1.0 / fs)
    target_bpm = 75.0
    # Perfect synthetic pulse
    ppg = np.sin(2 * np.pi * (target_bpm / 60.0) * t)
    hr_line = np.full_like(t, target_bpm)

    res = evaluate_ground_truth_consistency(ppg, hr_line, time=t, fs=fs, threshold_bpm=5.0)

    # Discrepancy should be tiny (< 0.2 BPM)
    assert res["video_level"]["discrepancy_bpm"] < 0.2
    assert res["video_level"]["flagged"] is False
    assert res["windowed"]["flagged"] is False


def test_gt_consistency_flagged():
    fs = 30.0
    duration = 45.0
    t = np.arange(0, duration, 1.0 / fs)
    # Pulse is at 80 BPM, but HR line reports 100 BPM (discrepancy = 20 BPM)
    ppg = np.sin(2 * np.pi * (80.0 / 60.0) * t)
    hr_line = np.full_like(t, 100.0)

    res = evaluate_ground_truth_consistency(ppg, hr_line, time=t, fs=fs, threshold_bpm=5.0)

    assert res["video_level"]["flagged"] is True
    assert res["windowed"]["flagged"] is True
    assert res["video_level"]["discrepancy_bpm"] > 15.0


def test_log_exclusion():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "exclusions.csv"
        log_exclusion("subject_test", "Bad motion artifact", exclusions_file=csv_path)

        assert csv_path.is_file()
        content = csv_path.read_text(encoding="utf-8")
        assert "subject_test" in content
        assert "Bad motion artifact" in content

        # Append another
        log_exclusion("subject_test2", "Sensor disconnect", exclusions_file=csv_path)
        content = csv_path.read_text(encoding="utf-8")
        assert "subject_test2" in content


def test_plot_bland_altman():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "ba_test.png"
        y_true = np.array([60.0, 70.0, 80.0])
        y_pred = np.array([61.0, 69.0, 82.0])

        fig = plot_bland_altman(y_true, y_pred, title="Test", save_path=save_path)
        assert save_path.is_file()
        assert save_path.stat().st_size > 0


def test_plot_bland_altman_across_subjects():
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "ba_subjects.png"
        df = pd.DataFrame([
            {"subject_id": "subject1", "resolution": "640x480", "hr_oximeter": 106.7, "hr_ppg": 107.1},
            {"subject_id": "subject3", "resolution": "640x480", "hr_oximeter": 101.2, "hr_ppg": 101.1},
            {"subject_id": "subject4", "resolution": "242x242", "hr_oximeter": 112.3, "hr_ppg": 111.2},
        ])
        fig = plot_bland_altman_across_subjects(
            df,
            true_col="hr_oximeter",
            pred_col="hr_ppg",
            save_path=save_path,
        )
        assert save_path.is_file()
        assert save_path.stat().st_size > 0


def test_medae_and_fraction_within_threshold():
    y_true = np.array([60.0, 70.0, 80.0, 90.0, 100.0])
    # Diffs: [1.0, 4.0, 6.0, 2.0, 10.0] -> sorted: [1.0, 2.0, 4.0, 6.0, 10.0] -> median is 4.0
    y_pred = np.array([61.0, 74.0, 86.0, 92.0, 110.0])

    medae = calc_medae(y_true, y_pred)
    assert np.isclose(medae, 4.0)

    # Within 5.0 BPM: 1.0, 4.0, 2.0 (3 out of 5 = 60.0%)
    frac = calc_fraction_within_threshold(y_true, y_pred, threshold=5.0)
    assert np.isclose(frac, 60.0)

    # Check calc_metrics returns them
    metrics = calc_metrics(y_true, y_pred)
    assert "medae" in metrics
    assert "fraction_within_5bpm" in metrics
    assert np.isclose(metrics["medae"], 4.0)
    assert np.isclose(metrics["fraction_within_5bpm"], 60.0)


def test_calculate_snr_synthetic():
    fs = 30.0
    t = np.arange(0, 30.0, 1.0 / fs)
    target_bpm = 90.0  # 1.5 Hz
    f0 = target_bpm / 60.0

    # Clean signal + weak white noise
    sig = np.sin(2 * np.pi * f0 * t) + 0.1 * np.sin(2 * np.pi * (2 * f0) * t)
    noise = 0.05 * np.cos(2 * np.pi * 0.8 * t)

    from scipy import signal as sp_signal
    f, psd = sp_signal.welch(sig + noise, fs=fs, nperseg=int(fs * 30), nfft=8192)

    snr = calculate_snr(psd, f, ref_hr_bpm=target_bpm, delta_hz=0.1, harmonic=True)
    # With clean dominant sinusoid, SNR should be positive and high (> 10 dB)
    assert snr > 10.0

    # With completely uncorrelated frequency reference (e.g. 50 BPM when pulse is at 90 BPM)
    snr_wrong = calculate_snr(psd, f, ref_hr_bpm=50.0, delta_hz=0.05, harmonic=False)
    assert snr_wrong < 0.0


def test_top_spectral_peaks_synthetic():
    fs = 30.0
    t = np.arange(0, 30.0, 1.0 / fs)
    # Three frequencies: 72 BPM (1.2 Hz, large), 120 BPM (2.0 Hz, medium), 90 BPM (1.5 Hz, small)
    sig = (
        1.0 * np.sin(2 * np.pi * 1.2 * t)
        + 0.5 * np.sin(2 * np.pi * 2.0 * t)
        + 0.25 * np.sin(2 * np.pi * 1.5 * t)
    )

    from scipy import signal as sp_signal
    f, psd = sp_signal.welch(sig, fs=fs, nperseg=int(fs * 30), nfft=8192)

    peaks = get_top_spectral_peaks(psd, f, n_peaks=3)
    assert len(peaks) == 3
    # Top peak should be near 72 BPM
    assert np.isclose(peaks[0]["freq_bpm"], 72.0, atol=1.0)
    # Second peak should be near 120 BPM
    assert np.isclose(peaks[1]["freq_bpm"], 120.0, atol=1.0)
    # Third peak should be near 90 BPM
    assert np.isclose(peaks[2]["freq_bpm"], 90.0, atol=1.0)
    assert peaks[0]["rel_power_max"] == 1.0
    assert peaks[1]["rel_power_max"] < 1.0


def test_max_lag_waveform_correlation_synthetic():
    fs = 30.0
    t = np.arange(0, 30.0, 1.0 / fs)
    # Sinusoid at 1.0 Hz (60 BPM)
    sig1 = np.sin(2 * np.pi * 1.0 * t)
    # Shifted by 6 samples (0.2 s)
    shift_samples = 6
    sig2 = np.sin(2 * np.pi * 1.0 * (t - shift_samples / fs))

    res = max_lag_waveform_correlation(sig2, sig1, fs=fs, max_lag_sec=1.0)
    # Optimal lag should match the planted shift
    assert abs(res["optimal_lag_frames"] - shift_samples) <= 1
    assert abs(res["optimal_lag_sec"] - 0.2) <= 0.05
    assert res["max_r"] > 0.99



