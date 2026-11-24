from pathlib import Path
import pandas as pd
import numpy as np
import torch

# 1. Verify PyTorch & GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device:      {torch.cuda.get_device_name(0)}")
print("-" * 50)

# 2. Locate all Signals CSV files inside data/raw
raw_dir = Path("data/raw")
signal_files = sorted(raw_dir.rglob("bidmc_*_Signals.csv"))
print(f"Found {len(signal_files)} signal files.")

if not signal_files:
    raise FileNotFoundError("No bidmc_*_Signals.csv files found inside data/raw/")

# 3. Inspect columns, sampling rate, duration, and NaNs across all patients
lead_counts = {}
missing_lead_ii = []
files_with_nans = []
durations = []
fs_values = []

for f in signal_files:
    df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]

    # Check sampling rate from Time [s]
    time_col = [c for c in df.columns if "time" in c.lower()][0]
    dt = np.median(np.diff(df[time_col].values))
    fs = round(1.0 / dt)
    fs_values.append(fs)
    durations.append(df[time_col].iloc[-1] - df[time_col].iloc[0])

    # Track available columns
    for col in df.columns:
        lead_counts[col] = lead_counts.get(col, 0) + 1

    # Check specifically for Lead II ECG
    if "II" not in df.columns:
        missing_lead_ii.append((f.name, list(df.columns)))
    elif df["II"].isna().any():
        files_with_nans.append((f.name, int(df["II"].isna().sum())))

# 4. Print Summary
print(f"Sampling rates found (Hz): {set(fs_values)}")
print(f"Duration range (seconds):  {min(durations):.1f}s to {max(durations):.1f}s")
print("\nColumn presence across files:")
for col, count in sorted(lead_counts.items(), key=lambda x: -x[1]):
    print(f"  {col:<12}: {count}/{len(signal_files)} files")

print(f"\nFiles missing Lead 'II':   {len(missing_lead_ii)}")
for name, cols in missing_lead_ii:
    print(f"  - {name}: {cols}")

print(f"\nFiles with NaNs in 'II':   {len(files_with_nans)}")
for name, n_nans in files_with_nans:
    print(f"  - {name}: {n_nans} NaNs")