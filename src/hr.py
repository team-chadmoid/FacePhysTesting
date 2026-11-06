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


def detect_peaks_ppg(
    signal: np.ndarray,
    fs: float,
    min_bpm: float = 42.0,
    max_bpm: float = 180.0,
    prominence: Optional[float] = None,
    refine_subsample: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect systolic peaks in a filtered PPG or BVP waveform.

    Args:
        signal: Filtered PPG signal.
        fs: Sampling rate in Hz.
        min_bpm: Minimum physiological HR in BPM (default: 42.0).
        max_bpm: Maximum physiological HR in BPM (default: 180.0).
        prominence: Peak prominence threshold.
        refine_subsample: Use parabolic interpolation around peaks for sub-sample accuracy.

    Returns:
        Tuple of (peak_indices, refined_peak_times_sec).
    """
    min_distance = int(round(fs * (60.0 / max_bpm)))
    if min_distance < 1:
        min_distance = 1

    peaks, _ = sp_signal.find_peaks(
        signal,
        distance=min_distance,
        prominence=prominence,
    )

    t_peaks = peaks / fs
    if refine_subsample and len(peaks) > 0:
        refined_times = []
        for p in peaks:
            if 0 < p < len(signal) - 1:
                y0 = signal[p]
                ym1 = signal[p - 1]
                yp1 = signal[p + 1]
                denom = ym1 - 2.0 * y0 + yp1
                if denom < 0:  # Concave down peak
                    delta = 0.5 * (ym1 - yp1) / denom
                    # Keep delta within [-0.5, 0.5] samples
                    delta = np.clip(delta, -0.5, 0.5)
                    refined_times.append((p + delta) / fs)
                else:
                    refined_times.append(p / fs)
            else:
                refined_times.append(p / fs)
        t_peaks = np.array(refined_times, dtype=np.float64)

    return peaks, t_peaks


def hr_from_peaks(
    signal: np.ndarray,
    fs: float = 30.0,
    time: Optional[np.ndarray] = None,
    min_bpm: float = 42.0,
    max_bpm: float = 180.0,
    lowcut: float = 0.7,
    highcut: float = 3.0,
    order: int = 3,
    refine_subsample: bool = True,
) -> float:
    """Estimate HR from peak detection using median inter-beat interval (IBI).

    1. Optional resampling onto uniform grid if non-uniform timestamps provided.
    2. 3rd-order Butterworth bandpass filtering.
    3. scipy.signal.find_peaks with sub-sample parabolic refinement.
    4. HR = 60.0 / median(IBI).

    Args:
        signal: 1D signal.
        fs: Sampling rate in Hz.
        time: Optional timestamps for non-uniform time series.
        min_bpm: Minimum physiological HR in BPM.
        max_bpm: Maximum physiological HR in BPM.
        lowcut: Filter lower cutoff in Hz.
        highcut: Filter upper cutoff in Hz.
        order: Filter order.
        refine_subsample: Refine peak times using parabolic interpolation.

    Returns:
        Estimated HR in BPM, or NaN if fewer than 2 peaks detected.
    """
    if time is not None:
        t_uniform, sig_uniform = resample_to_uniform(time, signal, target_fs=fs)
    else:
        sig_uniform = signal
        t_uniform = np.arange(len(signal)) / fs

    sig_filt = butterworth_bandpass_filter(
        sig_uniform, fs=fs, lowcut=lowcut, highcut=highcut, order=order
    )

    peaks, t_peaks_rel = detect_peaks_ppg(
        sig_filt, fs=fs, min_bpm=min_bpm, max_bpm=max_bpm, refine_subsample=refine_subsample
    )
    if len(peaks) < 2:
        return float("nan")

    t_peaks = t_uniform[0] + t_peaks_rel
    ibis = np.diff(t_peaks)
    median_ibi = float(np.median(ibis))
    if median_ibi <= 0:
        return float("nan")

    return float(60.0 / median_ibi)


def estimate_hr_peaks_windowed(
    signal: np.ndarray,
    fs: float = 30.0,
    time: Optional[np.ndarray] = None,
    window_sec: float = 30.0,
    stride_sec: float = 1.0,
    min_bpm: float = 42.0,
    max_bpm: float = 180.0,
    lowcut: float = 0.7,
    highcut: float = 3.0,
    order: int = 3,
    refine_subsample: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate HR in sliding windows using peak detection (median IBI per window).

    Args:
        signal: 1D signal.
        fs: Sampling frequency in Hz.
        time: Optional timestamps.
        window_sec: Window length in seconds (default: 30.0).
        stride_sec: Stride length in seconds (default: 1.0).
        min_bpm: Minimum HR in BPM.
        max_bpm: Maximum HR in BPM.
        lowcut: Filter lower cutoff.
        highcut: Filter upper cutoff.
        order: Filter order.
        refine_subsample: Sub-sample peak refinement.

    Returns:
        Tuple of (window_center_times, window_hrs_bpm).
    """
    if time is not None:
        t_uniform, sig_uniform = resample_to_uniform(time, signal, target_fs=fs)
    else:
        sig_uniform = signal
        t_uniform = np.arange(len(signal)) / fs

    sig_filt = butterworth_bandpass_filter(
        sig_uniform, fs=fs, lowcut=lowcut, highcut=highcut, order=order
    )

    peaks, t_peaks_rel = detect_peaks_ppg(
        sig_filt, fs=fs, min_bpm=min_bpm, max_bpm=max_bpm, refine_subsample=refine_subsample
    )
    t_peaks = t_uniform[0] + t_peaks_rel

    win_samples = int(round(window_sec * fs))
    stride_samples = int(round(stride_sec * fs))
    total_samples = len(sig_filt)

    if total_samples < win_samples:
        hr = hr_from_peaks(
            sig_filt,
            fs=fs,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            lowcut=lowcut,
            highcut=highcut,
            refine_subsample=refine_subsample,
        )
        t_mid = (t_uniform[0] + t_uniform[-1]) / 2.0
        return np.array([t_mid]), np.array([hr])

    window_times = []
    window_hrs = []

    for start in range(0, total_samples - win_samples + 1, stride_samples):
        end = start + win_samples
        t_start = t_uniform[start]
        t_end = t_uniform[end - 1]
        t_center = float(t_start + window_sec / 2.0)

        # Peaks falling within current window
        mask = (t_peaks >= t_start) & (t_peaks <= t_end)
        w_peaks = t_peaks[mask]

        if len(w_peaks) >= 2:
            w_ibis = np.diff(w_peaks)
            hr_val = float(60.0 / np.median(w_ibis))
        else:
            hr_val = float("nan")

        window_times.append(t_center)
        window_hrs.append(hr_val)

    return np.array(window_times, dtype=np.float64), np.array(window_hrs, dtype=np.float64)



def estimate_hr_raw_frames(
    signal: np.ndarray,
    fs_nominal: float = 30.0,
    lowcut: float = 0.7,
    highcut: float = 3.0,
    order: int = 3,
    nfft: int = 16384,
) -> float:
    """Estimate video-level HR on raw frame-index signal without timestamp resampling."""
    return estimate_hr_from_bvp(
        signal,
        fs=fs_nominal,
        time=None,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
        nfft=nfft,
    )


def estimate_hr_windowed_raw_frames(
    signal: np.ndarray,
    fs_nominal: float = 30.0,
    window_sec: float = 30.0,
    stride_sec: float = 1.0,
    lowcut: float = 0.7,
    highcut: float = 3.0,
    order: int = 3,
    nfft: int = 8192,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate windowed HR on raw frame-index signal without timestamp resampling."""
    return estimate_hr_windowed(
        signal,
        fs=fs_nominal,
        time=None,
        window_sec=window_sec,
        stride_sec=stride_sec,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
        nfft=nfft,
    )


