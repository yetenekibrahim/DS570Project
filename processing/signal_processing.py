import numpy as np
import pandas as pd

def compute_fft_spectrum(signal: np.ndarray, fs: float = 1.0):
    """
    Compute one-sided FFT of a signal.
    signal: 1D array (e.g. velocity or altitude time series)
    fs: sampling frequency
    """
    N = len(signal)
    fft_vals = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_vals[:N // 2])
    freqs = np.fft.fftfreq(N, d=1/fs)[:N // 2]
    return freqs, fft_magnitude

def compute_psd(signal: np.ndarray, fs: float = 1.0):
    """
    Power Spectral Density estimate via periodogram method.
    PSD shows how signal power is distributed across frequencies.
    """
    freqs, magnitude = compute_fft_spectrum(signal, fs)
    psd = (magnitude ** 2) / len(signal)
    psd_db = 10 * np.log10(psd + 1e-10)  # convert to dB
    return freqs, psd, psd_db

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for anomaly detection.
    Domain-motivated features based on flight physics.
    """
    df = df.copy()

    # Ratio of altitude to velocity: unusual if very high/low
    df["altitude_velocity_ratio"] = df["baro_altitude"] / (df["velocity"] + 1e-6)

    # Altitude bands (flight phases)
    df["flight_phase"] = pd.cut(
        df["baro_altitude"],
        bins=[0, 3000, 10000, 13000, 99999],
        labels=["takeoff_landing", "climb_descent", "cruise", "high_altitude"]
    )

    # Speed bands
    df["speed_category"] = pd.cut(
        df["velocity"],
        bins=[0, 100, 200, 300, 400, 1000],
        labels=["very_slow", "slow", "medium", "fast", "very_fast"]
    )

    return df

def compute_spectral_entropy(signal: np.ndarray) -> float:
    """
    Spectral entropy: measures how spread out the frequency content is.
    Low entropy = dominated by few frequencies (structured signal).
    High entropy = spread across many frequencies (noise-like).
    """
    _, psd, _ = compute_psd(signal)
    psd_norm = psd / (psd.sum() + 1e-10)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    return entropy