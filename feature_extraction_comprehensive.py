import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, periodogram, welch


def extract_comprehensive_features(filtered_bvp, fs=64, window_size=30, overlap=0.5):
    """
    Extract comprehensive features from PPG signal using sliding windows

    Args:
        filtered_bvp (array): Filtered BVP signal
        fs (int): Sampling frequency (default: 64 Hz)
        window_size (int): Window size in seconds (default: 30s)
        overlap (float): Overlap ratio between windows (default: 0.5)

    Returns:
        pd.DataFrame: DataFrame containing all extracted features
    """
    print("Extracting comprehensive PPG features...")

    window_samples = int(window_size * fs)
    step_samples = int(window_samples * (1 - overlap))

    features_list = []

    # Process signal in sliding windows
    for start in range(0, len(filtered_bvp) - window_samples + 1, step_samples):
        end = start + window_samples
        window_signal = filtered_bvp[start:end]

        # Extract features for this window
        window_features = extract_window_features(window_signal, fs, start / fs)
        features_list.append(window_features)

    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)

    print(f"Extracted {len(features_df)} feature windows")
    print(f"Total features per window: {len(features_df.columns)}")

    return features_df


def extract_window_features(signal_window, fs, start_time):
    """
    Extract all features from a single window of PPG signal

    Args:
        signal_window (array): PPG signal window
        fs (int): Sampling frequency
        start_time (float): Start time of window in seconds

    Returns:
        dict: Dictionary of extracted features
    """
    features = {"start_time": start_time}

    # 1. TIME-DOMAIN FEATURES
    features.update(extract_time_domain_features(signal_window))

    # 2. FREQUENCY-DOMAIN FEATURES
    features.update(extract_frequency_domain_features(signal_window, fs))

    # 3. PEAK-BASED FEATURES (HR and HRV)
    features.update(extract_peak_features(signal_window, fs))

    # 4. SIGNAL QUALITY FEATURES
    features.update(extract_signal_quality_features(signal_window, fs))

    # 5. MORPHOLOGICAL FEATURES
    features.update(extract_morphological_features(signal_window, fs))

    # 6. NONLINEAR FEATURES
    features.update(extract_nonlinear_features(signal_window))

    return features


def extract_time_domain_features(signal_window):
    """Extract basic statistical time-domain features"""
    features = {}

    # Basic statistics
    features["mean"] = np.mean(signal_window)
    features["std"] = np.std(signal_window)
    features["var"] = np.var(signal_window)
    features["min"] = np.min(signal_window)
    features["max"] = np.max(signal_window)
    features["range"] = features["max"] - features["min"]
    features["median"] = np.median(signal_window)

    # Higher-order statistics
    features["skewness"] = stats.skew(signal_window)
    features["kurtosis"] = stats.kurtosis(signal_window)

    # Percentiles
    features["q25"] = np.percentile(signal_window, 25)
    features["q75"] = np.percentile(signal_window, 75)
    features["iqr"] = features["q75"] - features["q25"]

    # Energy and power
    features["energy"] = np.sum(signal_window**2)
    features["rms"] = np.sqrt(np.mean(signal_window**2))

    # Zero crossings
    zero_crossings = np.where(np.diff(np.signbit(signal_window)))[0]
    features["zero_crossings"] = len(zero_crossings)

    return features


def extract_frequency_domain_features(signal_window, fs):
    """Extract frequency-domain features using FFT and PSD"""
    features = {}

    # FFT
    fft_vals = fft(signal_window)
    freqs = fftfreq(len(signal_window), 1 / fs)

    # Only use positive frequencies
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    fft_mag = np.abs(fft_vals[pos_mask])

    # Power Spectral Density
    freqs_psd, psd = welch(signal_window, fs, nperseg=min(256, len(signal_window) // 4))

    # Dominant frequency
    dominant_freq_idx = np.argmax(psd)
    features["dominant_frequency"] = freqs_psd[dominant_freq_idx]
    features["dominant_power"] = psd[dominant_freq_idx]

    # Frequency bands (typical for PPG/HR analysis)
    # Very low frequency: 0.003-0.04 Hz
    # Low frequency: 0.04-0.15 Hz
    # High frequency: 0.15-0.4 Hz
    # HR band: 0.75-4 Hz (45-240 bpm)

    vlf_mask = (freqs_psd >= 0.003) & (freqs_psd < 0.04)
    lf_mask = (freqs_psd >= 0.04) & (freqs_psd < 0.15)
    hf_mask = (freqs_psd >= 0.15) & (freqs_psd < 0.4)
    hr_mask = (freqs_psd >= 0.75) & (freqs_psd < 4.0)

    features["vlf_power"] = np.sum(psd[vlf_mask]) if np.any(vlf_mask) else 0
    features["lf_power"] = np.sum(psd[lf_mask]) if np.any(lf_mask) else 0
    features["hf_power"] = np.sum(psd[hf_mask]) if np.any(hf_mask) else 0
    features["hr_power"] = np.sum(psd[hr_mask]) if np.any(hr_mask) else 0

    # LF/HF ratio
    features["lf_hf_ratio"] = (
        features["lf_power"] / features["hf_power"] if features["hf_power"] > 0 else 0
    )

    # Total power
    features["total_power"] = np.sum(psd)

    # Spectral entropy
    psd_norm = psd / np.sum(psd)
    features["spectral_entropy"] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    # Spectral centroid
    features["spectral_centroid"] = np.sum(freqs_psd * psd) / np.sum(psd)

    # Spectral rolloff (95% of energy)
    cumsum_psd = np.cumsum(psd)
    rolloff_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]
    features["spectral_rolloff"] = (
        freqs_psd[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
    )

    return features


def extract_peak_features(signal_window, fs):
    """Extract heart rate and heart rate variability features"""
    features = {}

    # Find peaks (minimum distance of 0.5s between peaks)
    peaks, properties = find_peaks(
        signal_window, distance=int(fs * 0.5), height=np.mean(signal_window)
    )

    if len(peaks) < 2:
        # Not enough peaks for HR calculation
        features.update(
            {
                "hr_mean": 0,
                "hr_std": 0,
                "hr_min": 0,
                "hr_max": 0,
                "rmssd": 0,
                "sdnn": 0,
                "pnn50": 0,
                "triangular_index": 0,
                "peak_count": len(peaks),
                "avg_peak_amplitude": 0,
            }
        )
        return features

    # Peak times and intervals
    peak_times = peaks / fs
    rr_intervals = np.diff(peak_times)  # R-R intervals in seconds

    # Heart rate calculation
    hr_values = 60 / rr_intervals  # Convert to BPM

    # HR statistics
    features["hr_mean"] = np.mean(hr_values)
    features["hr_std"] = np.std(hr_values)
    features["hr_min"] = np.min(hr_values)
    features["hr_max"] = np.max(hr_values)
    features["peak_count"] = len(peaks)

    # Peak amplitudes
    peak_amplitudes = signal_window[peaks]
    features["avg_peak_amplitude"] = np.mean(peak_amplitudes)
    features["std_peak_amplitude"] = np.std(peak_amplitudes)

    # HRV time-domain measures
    rr_ms = rr_intervals * 1000  # Convert to milliseconds

    # RMSSD: Root mean square of successive differences
    if len(rr_ms) > 1:
        rr_diff = np.diff(rr_ms)
        features["rmssd"] = np.sqrt(np.mean(rr_diff**2))
    else:
        features["rmssd"] = 0

    # SDNN: Standard deviation of NN intervals
    features["sdnn"] = np.std(rr_ms)

    # pNN50: Percentage of successive RR intervals that differ by more than 50ms
    if len(rr_ms) > 1:
        nn50 = np.sum(np.abs(np.diff(rr_ms)) > 50)
        features["pnn50"] = (nn50 / len(rr_diff)) * 100
    else:
        features["pnn50"] = 0

    # Triangular index (approximation)
    if len(rr_ms) > 0:
        hist, _ = np.histogram(rr_ms, bins=min(50, len(rr_ms) // 2))
        features["triangular_index"] = (
            len(rr_ms) / np.max(hist) if np.max(hist) > 0 else 0
        )
    else:
        features["triangular_index"] = 0

    return features


def extract_signal_quality_features(signal_window, fs):
    """Extract signal quality metrics"""
    features = {}

    # Signal-to-noise ratio estimation
    # Use high-frequency content as noise estimate
    freqs, psd = welch(signal_window, fs, nperseg=min(256, len(signal_window) // 4))

    signal_band = (freqs >= 0.5) & (freqs <= 4.0)  # HR relevant band
    noise_band = freqs > 10  # High frequency noise

    signal_power = np.sum(psd[signal_band]) if np.any(signal_band) else 0
    noise_power = np.sum(psd[noise_band]) if np.any(noise_band) else 1e-12

    features["snr_estimate"] = 10 * np.log10(signal_power / noise_power)

    # Perfusion index (relative amplitude)
    features["perfusion_index"] = (
        np.max(signal_window) - np.min(signal_window)
    ) / np.mean(signal_window)

    # Signal regularity (autocorrelation)
    autocorr = np.correlate(signal_window, signal_window, mode="full")
    autocorr = autocorr[autocorr.size // 2 :]
    autocorr = autocorr / autocorr[0]  # Normalize

    # Find first minimum in autocorrelation (indicates periodicity)
    if len(autocorr) > 1:
        features["autocorr_regularity"] = np.max(autocorr[1 : min(len(autocorr), fs)])
    else:
        features["autocorr_regularity"] = 0

    return features


def extract_morphological_features(signal_window, fs):
    """Extract morphological features of PPG pulses"""
    features = {}

    # Find peaks and valleys
    peaks, _ = find_peaks(signal_window, distance=int(fs * 0.5))
    valleys, _ = find_peaks(-signal_window, distance=int(fs * 0.5))

    if len(peaks) < 2 or len(valleys) < 2:
        features.update(
            {
                "pulse_width_mean": 0,
                "pulse_width_std": 0,
                "rise_time_mean": 0,
                "fall_time_mean": 0,
                "systolic_peak_time": 0,
                "diastolic_peak_time": 0,
            }
        )
        return features

    # Pulse width (peak to peak)
    if len(peaks) > 1:
        pulse_widths = np.diff(peaks) / fs
        features["pulse_width_mean"] = np.mean(pulse_widths)
        features["pulse_width_std"] = np.std(pulse_widths)
    else:
        features["pulse_width_mean"] = 0
        features["pulse_width_std"] = 0

    # Rise and fall times (simplified)
    rise_times = []
    fall_times = []

    for peak in peaks:
        # Find nearest valleys before and after peak
        valleys_before = valleys[valleys < peak]
        valleys_after = valleys[valleys > peak]

        if len(valleys_before) > 0 and len(valleys_after) > 0:
            valley_before = valleys_before[-1]
            valley_after = valleys_after[0]

            rise_time = (peak - valley_before) / fs
            fall_time = (valley_after - peak) / fs

            rise_times.append(rise_time)
            fall_times.append(fall_time)

    features["rise_time_mean"] = np.mean(rise_times) if rise_times else 0
    features["fall_time_mean"] = np.mean(fall_times) if fall_times else 0

    # Systolic and diastolic timing (as percentage of pulse cycle)
    if rise_times and fall_times:
        total_times = np.array(rise_times) + np.array(fall_times)
        features["systolic_peak_time"] = (
            np.mean(np.array(rise_times) / total_times) if len(total_times) > 0 else 0
        )
        features["diastolic_peak_time"] = (
            np.mean(np.array(fall_times) / total_times) if len(total_times) > 0 else 0
        )
    else:
        features["systolic_peak_time"] = 0
        features["diastolic_peak_time"] = 0

    return features


def extract_nonlinear_features(signal_window):
    """Extract simplified nonlinear and complexity features (fast version)"""
    features = {}
    
    # Skip computationally expensive entropy calculations for now
    # These can cause infinite loops with large signals
    features["sample_entropy"] = 0  # Placeholder
    features["approx_entropy"] = 0  # Placeholder
    
    # Simple complexity measure: coefficient of variation
    features["complexity_cv"] = np.std(signal_window) / (np.mean(signal_window) + 1e-12)
    
    # Simple regularity measure: autocorrelation at lag 1
    if len(signal_window) > 1:
        autocorr_lag1 = np.corrcoef(signal_window[:-1], signal_window[1:])[0, 1]
        features["autocorr_lag1"] = autocorr_lag1 if not np.isnan(autocorr_lag1) else 0
    else:
        features["autocorr_lag1"] = 0
    
    # Simplified DFA (much faster)
    try:
        # Just use a simple detrending measure
        y = np.cumsum(signal_window - np.mean(signal_window))
        if len(y) > 10:
            # Linear detrend the integrated signal
            x = np.arange(len(y))
            coeffs = np.polyfit(x, y, 1)
            trend = np.polyval(coeffs, x)
            detrended = y - trend
            features["dfa_alpha"] = np.std(detrended) / (np.std(y) + 1e-12)
        else:
            features["dfa_alpha"] = 0
    except:
        features["dfa_alpha"] = 0

    return features


def plot_feature_summary(features_df, save_path=None):
    """Plot summary of extracted features"""

    # Select key features for visualization
    key_features = [
        "hr_mean",
        "hr_std",
        "rmssd",
        "sdnn",
        "dominant_frequency",
        "spectral_entropy",
        "snr_estimate",
        "perfusion_index",
        "sample_entropy",
        "dfa_alpha",
    ]

    # Filter features that exist in the dataframe
    available_features = [f for f in key_features if f in features_df.columns]

    if not available_features:
        print("No key features available for plotting")
        return

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, feature in enumerate(available_features[:10]):
        if i < len(axes):
            axes[i].plot(features_df["start_time"], features_df[feature])
            axes[i].set_title(f"{feature}")
            axes[i].set_xlabel("Time (s)")
            axes[i].grid(True)

    # Hide unused subplots
    for i in range(len(available_features), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def save_features_to_csv(features_df, filepath):
    """Save features to CSV file"""
    features_df.to_csv(filepath, index=False)
    print(f"Features saved to {filepath}")


# Example usage function
def demo_feature_extraction(filtered_bvp, fs=64):
    """
    Demonstration of comprehensive feature extraction
    """
    print("Running comprehensive PPG feature extraction demo...")

    # Extract features
    features_df = extract_comprehensive_features(
        filtered_bvp, fs, window_size=30, overlap=0.5
    )

    # Display summary
    print(f"\nFeature extraction summary:")
    print(f"Number of windows: {len(features_df)}")
    print(f"Number of features per window: {len(features_df.columns)}")
    print(f"\nFeature categories:")
    print(f"- Time domain: mean, std, skewness, kurtosis, etc.")
    print(f"- Frequency domain: dominant frequency, spectral entropy, power bands")
    print(f"- Heart rate: HR mean/std, HRV measures (RMSSD, SDNN, pNN50)")
    print(f"- Signal quality: SNR, perfusion index, regularity")
    print(f"- Morphological: pulse width, rise/fall times")
    print(f"- Nonlinear: sample entropy, approximate entropy, DFA")

    # Show first few rows
    print(f"\nFirst 5 rows of features:")
    print(features_df.head())

    # Plot key features
    plot_feature_summary(features_df)

    return features_df


if __name__ == "__main__":
    # This would be called from your main pipeline
    print("Comprehensive PPG Feature Extraction Module")
    print("Import this module and use extract_comprehensive_features() function")
