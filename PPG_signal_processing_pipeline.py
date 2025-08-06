import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

# Import comprehensive feature extraction
from feature_extraction_comprehensive import (
    extract_comprehensive_features,
    plot_feature_summary,
    save_features_to_csv,
)
from scipy.signal import butter, filtfilt, find_peaks


def load_data(data_folder):
    """
    Step 1: Load PPG data from pickle file

    Args:
        data_folder (str): Path to the folder containing the S1.pkl file

    Returns:
        dict: Loaded data dictionary
    """
    print("Step 1: Loading data...")

    # Load S1.pkl
    with open(os.path.join(data_folder, "S1.pkl"), "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # Explore main keys and basic structure
    print("Main keys:", data.keys())
    print("Activity:", type(data["activity"]), ", shape:", data["activity"].shape)
    print("Label (e.g., HR):", type(data["label"]), ", shape:", data["label"].shape)
    print("Questionnaire fields:", data["questionnaire"].keys())
    print("R-peak count:", len(data["rpeaks"]))
    print("Signal keys:", data["signal"].keys())
    print("Chest modalities:", data["signal"]["chest"].keys())
    print("Wrist modalities:", data["signal"]["wrist"].keys())
    print("Subject label:", data["subject"])
    print("Data loaded successfully!\n")

    return data


def preprocess_data(data, fs=64, lowcut=0.5, highcut=10, order=3, plot=True):
    """
    Step 2: Preprocess PPG data with bandpass filtering

    Args:
        data (dict): Data dictionary from load_data
        fs (int): Sampling frequency (default: 64 Hz)
        lowcut (float): Low cutoff frequency for bandpass filter (default: 0.5 Hz)
        highcut (float): High cutoff frequency for bandpass filter (default: 10 Hz)
        order (int): Filter order (default: 3)
        plot (bool): Whether to plot comparison of raw vs filtered signal

    Returns:
        tuple: (raw_bvp, filtered_bvp)
    """
    print("Step 2: Preprocessing data...")

    def bandpass_filter(signal, lowcut=lowcut, highcut=highcut, fs=fs, order=order):
        """Bandpass filter for PPG signal"""
        # Check minimum signal length required for filtfilt
        min_length = 3 * max(order, 1) * 2 + 1  # Conservative estimate
        if len(signal) < min_length:
            print(
                f"Warning: Signal too short ({len(signal)} samples) for filtering. Minimum required: {min_length}"
            )
            return signal  # Return unfiltered signal

        nyq_freq = fs / 2
        low = lowcut / nyq_freq
        high = highcut / nyq_freq
        b, a = butter(order, [low, high], btype="band")
        return filtfilt(b, a, signal)

    # Extract raw BVP signal
    bvp_raw = data["signal"]["wrist"]["BVP"]

    print(f"Raw BVP signal shape: {bvp_raw.shape}")
    print(f"Raw BVP signal length: {len(bvp_raw)} samples")
    print(f"Raw BVP signal type: {type(bvp_raw)}")

    # Flatten the signal if it's 2D (remove extra dimensions)
    if bvp_raw.ndim > 1:
        bvp_raw = bvp_raw.flatten()
        print(f"Flattened BVP signal shape: {bvp_raw.shape}")

    # Apply bandpass filter
    bvp_filtered = bandpass_filter(bvp_raw)

    print(f"Filtered BVP signal shape: {bvp_filtered.shape}")

    # Plot comparison if requested
    if plot and len(bvp_raw) > 0:
        plot_samples = min(1000, len(bvp_raw))
        plt.figure(figsize=(12, 4))
        plt.plot(bvp_raw[:plot_samples], label="Raw PPG")
        plt.plot(bvp_filtered[:plot_samples], label="Filtered PPG")
        plt.legend()
        plt.xlabel("Sample")
        plt.ylabel("PPG")
        plt.title(f"Raw vs. Filtered Wrist BVP Signal (first {plot_samples} samples)")
        plt.show()
    elif len(bvp_raw) == 0:
        print("Warning: BVP signal is empty, skipping plot.")

    print("Data preprocessing completed!\n")

    return bvp_raw, bvp_filtered


def extract_HR(filtered_bvp, fs=64, plot=True):
    """
    Step 3: Extract heart rate from filtered PPG signal

    Args:
        filtered_bvp (array): Filtered BVP signal
        fs (int): Sampling frequency (default: 64 Hz)
        plot (bool): Whether to plot heart rate over time

    Returns:
        dict: Dictionary containing heart rate information
    """
    print("Step 3: Extracting heart rate...")

    # Find peaks in the filtered signal
    # Use minimum distance between peaks (0.5 seconds = 120 BPM max)
    min_distance = int(fs * 0.5)
    peaks, properties = find_peaks(
        filtered_bvp, distance=min_distance, height=np.mean(filtered_bvp)
    )

    if len(peaks) < 2:
        print("Warning: Not enough peaks found for heart rate calculation")
        return {"peaks": peaks, "hr_values": [], "hr_mean": 0, "hr_std": 0}

    # Calculate time intervals between peaks
    peak_times = peaks / fs  # Convert to seconds
    rr_intervals = np.diff(peak_times)  # R-R intervals in seconds

    # Convert to heart rate (beats per minute)
    hr_values = 60 / rr_intervals

    # Calculate statistics
    hr_mean = np.mean(hr_values)
    hr_std = np.std(hr_values)

    print(f"Found {len(peaks)} peaks")
    print(f"Mean heart rate: {hr_mean:.1f} ± {hr_std:.1f} BPM")
    print(f"HR range: {np.min(hr_values):.1f} - {np.max(hr_values):.1f} BPM")

    # Plot if requested
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot signal with detected peaks
        time_axis = np.arange(len(filtered_bvp)) / fs
        ax1.plot(time_axis, filtered_bvp, label="Filtered PPG")
        ax1.plot(peak_times, filtered_bvp[peaks], "ro", label="Detected Peaks")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("PPG Amplitude")
        ax1.set_title("PPG Signal with Detected Peaks")
        ax1.legend()
        ax1.grid(True)

        # Plot heart rate over time
        hr_times = peak_times[1:]  # HR values correspond to intervals between peaks
        ax2.plot(hr_times, hr_values, "b-o", markersize=3)
        ax2.axhline(
            y=hr_mean, color="r", linestyle="--", label=f"Mean: {hr_mean:.1f} BPM"
        )
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Heart Rate (BPM)")
        ax2.set_title("Heart Rate Over Time")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    hr_results = {
        "peaks": peaks,
        "peak_times": peak_times,
        "rr_intervals": rr_intervals,
        "hr_values": hr_values,
        "hr_mean": hr_mean,
        "hr_std": hr_std,
    }

    print("Heart rate extraction completed!\n")

    return hr_results


def extract_comprehensive_features_step(
    filtered_bvp,
    fs=64,
    window_size=30,
    overlap=0.5,
    plot=True,
    save_csv=False,
    output_dir=None,
):
    """
    Step 4: Extract comprehensive features from filtered PPG signal

    Args:
        filtered_bvp (array): Filtered BVP signal
        fs (int): Sampling frequency (default: 64 Hz)
        window_size (int): Window size in seconds (default: 30s)
        overlap (float): Overlap ratio between windows (default: 0.5)
        plot (bool): Whether to plot feature summary
        save_csv (bool): Whether to save features to CSV
        output_dir (str): Directory to save outputs (default: same as data folder)

    Returns:
        pd.DataFrame: DataFrame containing all extracted features
    """
    print("Step 4: Extracting comprehensive features...")

    # Extract comprehensive features
    features_df = extract_comprehensive_features(
        filtered_bvp, fs=fs, window_size=window_size, overlap=overlap
    )

    print(f"Extracted {len(features_df)} feature windows")
    print(f"Features per window: {len(features_df.columns)}")
    print(f"Total recording time: {len(filtered_bvp)/fs:.1f} seconds")
    print(f"Window size: {window_size}s, Overlap: {overlap*100:.0f}%")

    # Display feature categories
    print("\nFeature categories extracted:")
    print("- Time domain: mean, std, skewness, kurtosis, energy, etc.")
    print("- Frequency domain: dominant frequency, spectral entropy, power bands")
    print("- Heart rate & HRV: HR statistics, RMSSD, SDNN, pNN50")
    print("- Signal quality: SNR estimate, perfusion index, regularity")
    print("- Morphological: pulse width, rise/fall times, peak characteristics")
    print("- Nonlinear: sample entropy, approximate entropy, DFA")

    # Show sample of features
    print(f"\nSample features (first 3 windows):")
    key_features = [
        "start_time",
        "hr_mean",
        "hr_std",
        "rmssd",
        "dominant_frequency",
        "spectral_entropy",
        "snr_estimate",
        "perfusion_index",
    ]
    available_key_features = [f for f in key_features if f in features_df.columns]
    if available_key_features:
        print(features_df[available_key_features].head(3).to_string(index=False))

    # Plot feature summary if requested
    if plot:
        plot_feature_summary(features_df)

    # Save to CSV if requested
    if save_csv and output_dir:
        csv_path = os.path.join(output_dir, "ppg_features.csv")
        save_features_to_csv(features_df, csv_path)

    print("Comprehensive feature extraction completed!\n")

    return features_df


def run_pipeline(
    data_folder, plot_results=True, extract_features=True, save_features=False
):
    """
    Main pipeline function that runs all processing steps

    Args:
        data_folder (str): Path to the folder containing the S1.pkl file
        plot_results (bool): Whether to plot results during processing
        extract_features (bool): Whether to extract comprehensive features
        save_features (bool): Whether to save features to CSV

    Returns:
        dict: Dictionary containing all processed data
    """
    print("Starting PPG Signal Processing Pipeline...")
    print("=" * 50)

    # Step 1: Load data
    data = load_data(data_folder)

    # Step 2: Preprocess data
    bvp_raw, bvp_filtered = preprocess_data(data, plot=plot_results)

    # Step 3: Extract heart rate
    hr_bvp = extract_HR(bvp_filtered, plot=plot_results)

    # Package basic results
    results = {
        "raw_data": data,
        "bvp_raw": bvp_raw,
        "bvp_filtered": bvp_filtered,
        "hr_bvp": hr_bvp,
        "sampling_rate": 64,
    }

    # Step 4: Extract comprehensive features (optional)
    if extract_features:
        try:
            features_df = extract_comprehensive_features_step(
                bvp_filtered,
                fs=64,
                plot=plot_results,
                save_csv=save_features,
                output_dir=data_folder if save_features else None,
            )
            results["features"] = features_df
        except Exception as e:
            print(f"Warning: Feature extraction failed: {e}")
            print("Continuing with basic pipeline...")

    print("=" * 50)
    print("Pipeline completed successfully!")
    print(f"Results contain: {list(results.keys())}")

    return results


if __name__ == "__main__":
    # Configuration
    data_folder = (
        "/Users/yuannie/Downloads/PPG_signal_processing/ppg_dalia/PPG_FieldStudy/S1"
    )

    # Run the pipeline with comprehensive features
    results = run_pipeline(
        data_folder, plot_results=True, extract_features=True, save_features=True
    )

    # Example of accessing results
    print("\nExample usage of results:")
    print(f"Raw BVP signal length: {len(results['bvp_raw'])} samples")
    print(f"Filtered BVP signal length: {len(results['bvp_filtered'])} samples")
    print(f"Sampling rate: {results['sampling_rate']} Hz")
    print(f"Mean heart rate: {results['hr_bvp']['hr_mean']:.1f} BPM")

    if "features" in results:
        print(f"Number of feature windows: {len(results['features'])}")
        print(f"Features per window: {len(results['features'].columns)}")
