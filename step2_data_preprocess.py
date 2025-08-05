import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

# Typical PPG sampling rate in PPG-DaLiA is 64 Hz for wrist BVP
FS = 64


# Bandpass filter: [0.5 Hz, 10 Hz] for PPG signal
def bandpass_filter(signal, lowcut=0.5, highcut=10, fs=64, order=3):
    nyq_freq = fs / 2
    low = lowcut / nyq_freq
    high = highcut / nyq_freq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


bvp_raw = data["signal"]["wrist"]["BVP"]
bvp_filtered = bandpass_filter(bvp_raw)

# Compare raw and filtered PPG signals (first 1000 samples)
plt.figure(figsize=(12, 4))
plt.plot(bvp_raw[:1000], label="Raw PPG")
plt.plot(bvp_filtered[:1000], label="Filtered PPG")
plt.legend()
plt.xlabel("Sample")
plt.ylabel("PPG")
plt.title("Raw vs. Filtered Wrist BVP Signal (first 1000 samples)")
plt.show()
