"""Heart rate (HR) estimation from BVP / PPG signals.

Evaluation protocol implementation:
- 3rd-order Butterworth bandpass [0.7, 3.0] Hz (42 to 180 BPM).
- Welch Power Spectral Density (PSD).
- Peak frequency in [0.7, 3.0] Hz * 60 = HR (BPM).
- 30-second windows with 1-second stride.
- Video-level HR estimator.
"""

from typing import Optional, Tuple, Union

import numpy as np
from scipy import signal as sp_signal

from src.io import resample_to_uniform


def butterworth_bandpass_filter(
    data: np.ndarray,
    fs: float,
    lowcut: float = 0.7,
    highcut: float = 3.0,
    order: int = 3,
) -> np.ndarray:
    """Apply a 3rd-order zero-phase Butterworth bandpass filter.

    Default bandpass is [0.7, 3.0] Hz (42 to 180 BPM).

    Args:
        data: 1D signal array.
        fs: Sampling rate in Hz.
        lowcut: Lower cutoff frequency in Hz (default: 0.7 Hz).
        highcut: Upper cutoff frequency in Hz (default: 3.0 Hz).
        order: Filter order (default: 3).

    Returns:
        Filtered 1D signal.
    """
    nyq = 0.5 * fs
    if highcut >= nyq:
        highcut = nyq - 0.05
    if lowcut <= 0 or lowcut >= highcut:
        raise ValueError(f"Invalid cutoff frequencies: low={lowcut}, high={highcut}, nyq={nyq}")

    b, a = sp_signal.butter(order, [lowcut, highcut], btype="bandpass", fs=fs)

    padlen = 3 * max(len(a), len(b))
    if len(data) <= padlen:
        # Not enough samples for standard filtfilt padding
        return sp_signal.lfilter(b, a, data)

    return sp_signal.filtfilt(b, a, data)


def estimate_hr_welch(
    signal: np.ndarray,
    fs: float,
    low_bpm: float = 42.0,
    high_bpm: float = 180.0,
    nperseg: Optional[int] = None,
    nfft: int = 8192,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Estimate HR from filtered signal using Welch PSD.

    Args:
        signal: Filtered 1D BVP or PPG signal.
        fs: Sampling frequency in Hz.
        low_bpm: Lower bound for HR search in BPM (default: 42.0).
        high_bpm: Upper bound for HR search in BPM (default: 180.0).
        nperseg: Length of each Welch segment (defaults to min(len(signal), int(fs * 30))).
        nfft: Number of FFT points for fine frequency resolution (default: 8192).

    Returns:
        Tuple of (estimated_bpm, frequencies, psd).
    """
    if len(signal) < 2:
        raise ValueError("Signal too short for PSD estimation")

    if nperseg is None:
        nperseg = min(len(signal), int(fs * 30))

    f, psd = sp_signal.welch(signal, fs=fs, nperseg=nperseg, nfft=nfft, window="hann")

    low_hz = low_bpm / 60.0
    high_hz = high_bpm / 60.0

    mask = (f >= low_hz) & (f <= high_hz)
    if not np.any(mask):
        raise ValueError(f"No frequency bins found in [{low_hz:.2f}, {high_hz:.2f}] Hz")

    peak_freq = f[mask][np.argmax(psd[mask])]
    hr_bpm = float(peak_freq * 60.0)

    return hr_bpm, f, psd


def estimate_hr_from_bvp(
    bvp: np.ndarray,
    fs: float = 30.0,
    time: Optional[np.ndarray] = None,
    lowcut: float = 0.7,
    highcut: float = 3.0,
    order: int = 3,
    nfft: int = 8192,
) -> float:
    """Full single-segment HR estimation pipeline.

    1. Optional resampling onto uniform grid if non-uniform timestamps provided.
    2. 3rd-order Butterworth bandpass [0.7, 3.0] Hz.
    3. Welch PSD peak detection in [0.7, 3.0] Hz * 60.

    Args:
        bvp: 1D signal.
        fs: Sampling rate in Hz (default: 30.0).
        time: Optional timestamps for non-uniform time series.
        lowcut: Filter lower cutoff in Hz (0.7 Hz = 42 BPM).
        highcut: Filter upper cutoff in Hz (3.0 Hz = 180 BPM).
        order: Filter order (default: 3).
        nfft: FFT points.

    Returns:
        Estimated HR in BPM.
    """
    if time is not None:
        _, sig_uniform = resample_to_uniform(time, bvp, target_fs=fs)
    else:
        sig_uniform = bvp

    sig_filt = butterworth_bandpass_filter(
        sig_uniform, fs=fs, lowcut=lowcut, highcut=highcut, order=order
    )

    hr_bpm, _, _ = estimate_hr_welch(
        sig_filt, fs=fs, low_bpm=lowcut * 60.0, high_bpm=highcut * 60.0, nfft=nfft
    )
    return hr_bpm


def estimate_hr_windowed(
    signal: np.ndarray,
    fs: float = 30.0,
    time: Optional[np.ndarray] = None,
    window_sec: float = 30.0,
    stride_sec: float = 1.0,
    lowcut: float = 0.7,
    highcut: float = 3.0,
    order: int = 3,
    nfft: int = 8192,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate HR in sliding windows (default: 30 s window, 1 s stride).

    Args:
        signal: 1D BVP or PPG signal.
        fs: Sampling frequency in Hz.
        time: Optional 1D timestamps.
        window_sec: Window length in seconds (default: 30.0).
        stride_sec: Stride length in seconds (default: 1.0).
        lowcut: Filter lower cutoff in Hz (default: 0.7).
        highcut: Filter upper cutoff in Hz (default: 3.0).
        order: Filter order (default: 3).
        nfft: FFT points.

    Returns:
        Tuple of (window_center_times, window_hrs_bpm).
    """
    if time is not None:
        t_uniform, sig_uniform = resample_to_uniform(time, signal, target_fs=fs)
    else:
        sig_uniform = signal
        t_uniform = np.arange(len(signal)) / fs

    # Pre-filter the entire continuous trace to minimize boundary transients
    sig_filt = butterworth_bandpass_filter(
        sig_uniform, fs=fs, lowcut=lowcut, highcut=highcut, order=order
    )

    win_samples = int(round(window_sec * fs))
    stride_samples = int(round(stride_sec * fs))

    total_samples = len(sig_filt)
    if total_samples < win_samples:
        # Fall back to single segment estimate if shorter than window_sec
        hr = estimate_hr_from_bvp(
            sig_filt, fs=fs, lowcut=lowcut, highcut=highcut, order=order, nfft=nfft
        )
        t_mid = (t_uniform[0] + t_uniform[-1]) / 2.0
        return np.array([t_mid]), np.array([hr])

    window_times = []
    window_hrs = []

    for start in range(0, total_samples - win_samples + 1, stride_samples):
        end = start + win_samples
        seg = sig_filt[start:end]
        t_center = float(t_uniform[start] + window_sec / 2.0)

        hr, _, _ = estimate_hr_welch(
            seg,
            fs=fs,
            low_bpm=lowcut * 60.0,
            high_bpm=highcut * 60.0,
            nperseg=win_samples,
            nfft=nfft,
        )
        window_times.append(t_center)
        window_hrs.append(hr)

    return np.array(window_times, dtype=np.float64), np.array(window_hrs, dtype=np.float64)


def estimate_hr_video(
    signal: np.ndarray,
    fs: float = 30.0,
    time: Optional[np.ndarray] = None,
    lowcut: float = 0.7,
    highcut: float = 3.0,
    order: int = 3,
    nfft: int = 16384,
) -> float:
    """Estimate a single video-level HR over the entire duration."""
    return estimate_hr_from_bvp(
        signal,
        fs=fs,
        time=time,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
        nfft=nfft,
    )
