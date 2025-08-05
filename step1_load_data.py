import os
import pickle

# Path to subject S1 folder
data_folder = (
    "/Users/yuannie/Downloads/PPG_signal_processing/ppg_dalia/PPG_FieldStudy/S1"
)

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
