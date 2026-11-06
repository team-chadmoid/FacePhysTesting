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
    calc_rmse,
    calc_pearson_r,
    calc_bland_altman,
    calc_metrics,
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


