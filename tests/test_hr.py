"""Unit tests for src/hr.py (HR estimation on synthetic sinusoids and edge cases)."""

import numpy as np
import pytest

from src.hr import (
    butterworth_bandpass_filter,
    estimate_hr_welch,
    estimate_hr_from_bvp,
    estimate_hr_windowed,
    estimate_hr_video,
)


def generate_synthetic_sinusoid(
    bpm: float,
    fs: float = 30.0,
    duration_sec: float = 60.0,
    snr_db: float = 30.0,
    harmonics: bool = True,
):
    """Generate a synthetic pulse sinusoid with realistic physiological harmonics."""
    t = np.arange(0, duration_sec, 1.0 / fs)
    f0 = bpm / 60.0
    sig = np.sin(2 * np.pi * f0 * t)

    if harmonics:
        # 2nd harmonic (dicrotic notch feature)
        sig += 0.3 * np.sin(4 * np.pi * f0 * t + 0.5)

    # Add Gaussian noise
    noise = np.random.randn(len(t))
    sig_power = np.mean(sig**2)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    sig = sig + np.sqrt(noise_power) * noise

    return t, sig


@pytest.mark.parametrize("target_bpm", [50.0, 60.0, 72.0, 85.0, 100.0, 120.0, 140.0, 160.0])
def test_hr_synthetic_sinusoids(target_bpm):
    """Verify HR estimator achieves < 0.2 BPM error across human HR range on synthetic sinusoids."""
    fs = 30.0
    np.random.seed(42)
    t, sig = generate_synthetic_sinusoid(bpm=target_bpm, fs=fs, duration_sec=40.0, snr_db=25.0)

    est_bpm = estimate_hr_from_bvp(sig, fs=fs, time=t)
    error = abs(est_bpm - target_bpm)
    assert error < 0.2, f"Target {target_bpm} BPM failed with error {error:.3f} (est: {est_bpm:.3f})"


def test_butterworth_bandpass_attenuation():
    """Verify that frequencies outside [0.7, 3.0] Hz are attenuated."""
    fs = 30.0
    t = np.arange(0, 30.0, 1.0 / fs)

    # In-band component: 1.5 Hz (90 BPM)
    # Low out-of-band: 0.2 Hz (respiration artifact)
    # High out-of-band: 6.0 Hz (high frequency noise)
    sig_in = np.sin(2 * np.pi * 1.5 * t)
    sig_low = np.sin(2 * np.pi * 0.2 * t)
    sig_high = np.sin(2 * np.pi * 6.0 * t)
    composite = sig_in + sig_low + sig_high

    filtered = butterworth_bandpass_filter(composite, fs=fs, lowcut=0.7, highcut=3.0, order=3)

    # The in-band frequency should dominate
    hr_bpm, _, _ = estimate_hr_welch(filtered, fs=fs)
    assert abs(hr_bpm - 90.0) < 0.5


def test_estimate_hr_windowed():
    """Verify windowed HR estimation on time-varying synthetic signal."""
    fs = 30.0
    duration = 60.0
    t = np.arange(0, duration, 1.0 / fs)

    # Linear chirp from 60 BPM (1.0 Hz) to 120 BPM (2.0 Hz)
    f0 = 60.0 / 60.0
    f1 = 120.0 / 60.0
    k = (f1 - f0) / duration
    sig = np.sin(2 * np.pi * (f0 * t + 0.5 * k * t**2))

    win_times, win_hrs = estimate_hr_windowed(
        sig, fs=fs, time=t, window_sec=30.0, stride_sec=1.0
    )

    # For 60s signal with 30s window and 1s stride: 31 windows
    expected_windows = int((duration - 30.0) / 1.0) + 1
    assert len(win_times) == expected_windows
    assert len(win_hrs) == expected_windows

    # Each window should reflect the increasing HR
    assert win_hrs[-1] > win_hrs[0]
    # Check that estimated HRs are within plausible chirp range [60, 120]
    assert np.all(win_hrs >= 60.0)
    assert np.all(win_hrs <= 120.0)


def test_estimate_hr_video():
    """Verify video-level HR returns single float."""
    fs = 30.0
    t, sig = generate_synthetic_sinusoid(bpm=78.5, fs=fs, duration_sec=45.0, snr_db=30.0)
    hr_vid = estimate_hr_video(sig, fs=fs, time=t)
    assert isinstance(hr_vid, float)
    assert abs(hr_vid - 78.5) < 0.2
