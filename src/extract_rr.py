from pathlib import Path
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
RES_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FS = 500  # Upsample target (2 ms resolution)


def clean_rr_artifacts(rr_ms: np.ndarray, min_rr=300.0, max_rr=2000.0, dev_thresh=0.20):
    """Detect non-physiological or ectopic RR intervals and replace via linear interpolation."""
    local_med = median_filter(rr_ms, size=11, mode="nearest")
    bad_mask = (
        (rr_ms < min_rr)
        | (rr_ms > max_rr)
        | (np.abs(rr_ms - local_med) / np.maximum(local_med, 1e-6) > dev_thresh)
    )

    n_bad = int(np.sum(bad_mask))
    valid_idx = np.where(~bad_mask)[0]

    if len(valid_idx) < 50:
        return rr_ms.copy(), n_bad, True  # Flag as corrupted recording

    rr_clean = rr_ms.copy()
    if n_bad > 0:
        all_idx = np.arange(len(rr_ms))
        rr_clean[bad_mask] = np.interp(all_idx[bad_mask], valid_idx, rr_ms[valid_idx])

    return rr_clean, n_bad, False


def main():
    signal_files = sorted(RAW_DIR.rglob("bidmc_*_Signals.csv"))
    if not signal_files:
        raise FileNotFoundError("No bidmc_*_Signals.csv files found in data/raw/")

    records = []
    plotted_sample = False

    for f in tqdm(signal_files, desc="Extracting RR intervals"):
        patient_id = f.name.replace("_Signals.csv", "")
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]

        time_col = [c for c in df.columns if "time" in c.lower()][0]
        t_raw = df[time_col].values.astype(np.float64)
        ecg_raw = df["II"].values.astype(np.float64)

        # True sampling rate from total samples and duration
        duration = t_raw[-1] - t_raw[0]
        true_fs = round((len(df) - 1) / duration)

        # Construct uniform original time grid and upsample to 500 Hz
        t_orig = np.arange(len(ecg_raw)) / true_fs
        n_target = int(round(len(ecg_raw) * (TARGET_FS / true_fs)))
        t_new = np.arange(n_target) / TARGET_FS
        ecg_up = CubicSpline(t_orig, ecg_raw)(t_new)

        # Handle inverted ICU ECG leads automatically if present
        try:
            ecg_up, was_inverted = nk.ecg_invert(ecg_up, sampling_rate=TARGET_FS)
        except Exception:
            was_inverted = False

        # Clean ECG and detect R-peaks
        ecg_clean = nk.ecg_clean(ecg_up, sampling_rate=TARGET_FS, method="neurokit")
        _, info = nk.ecg_peaks(ecg_clean, sampling_rate=TARGET_FS, method="neurokit")
        r_peaks = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)

        # Raw RR intervals in milliseconds
        rr_raw = np.diff(r_peaks) / TARGET_FS * 1000.0
        rr_clean, n_artifacts, failed = clean_rr_artifacts(rr_raw)

        # Save cleaned RR sequence
        np.save(OUT_DIR / f"{patient_id}_rr.npy", rr_clean.astype(np.float32))

        artifact_pct = (n_artifacts / max(len(rr_raw), 1)) * 100.0
        mean_rr = float(np.mean(rr_clean))
        sdnn = float(np.std(rr_clean, ddof=1))
        rmssd = float(np.sqrt(np.mean(np.diff(rr_clean) ** 2)))

        records.append({
            "patient_id": patient_id,
            "true_fs": true_fs,
            "inverted": bool(was_inverted),
            "n_beats": len(rr_clean),
            "artifacts_fixed": n_artifacts,
            "artifact_pct": round(artifact_pct, 2),
            "mean_rr_ms": round(mean_rr, 2),
            "mean_hr_bpm": round(60000.0 / mean_rr, 1),
            "sdnn_ms": round(sdnn, 2),
            "rmssd_ms": round(rmssd, 2),
            "failed": failed,
        })

        # Save a visual sanity check for the first patient
        if not plotted_sample:
            fig, axes = plt.subplots(2, 1, figsize=(12, 6))

            # Top: first 10 seconds of cleaned ECG + detected R-peaks
            win = 10 * TARGET_FS
            peaks_10s = r_peaks[r_peaks < win]
            axes[0].plot(t_new[:win], ecg_clean[:win], lw=1.0, label="Cleaned ECG (500 Hz)")
            axes[0].scatter(t_new[peaks_10s], ecg_clean[peaks_10s], color="red", s=25, zorder=5, label="R-peaks")
            axes[0].set_title(f"{patient_id} — First 10s ECG & R-Peak Detection")
            axes[0].set_xlabel("Time (s)")
            axes[0].set_ylabel("Amplitude")
            axes[0].legend()

            # Bottom: Raw vs cleaned RR intervals across full 8 mins
            axes[1].plot(rr_raw, alpha=0.45, label="Raw RR")
            axes[1].plot(rr_clean, lw=1.2, label="Cleaned RR")
            axes[1].set_title(f"{patient_id} — Extracted RR Intervals ({len(rr_clean)} beats, {artifact_pct:.1f}% corrected)")
            axes[1].set_xlabel("Beat Index")
            axes[1].set_ylabel("RR Interval (ms)")
            axes[1].legend()

            plt.tight_layout()
            plt.savefig(RES_DIR / "rr_extraction_check.png", dpi=150)
            plt.close()
            plotted_sample = True

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(OUT_DIR / "rr_summary.csv", index=False)

    print("\n" + "=" * 60)
    print(f"Processed patients:    {len(summary_df)}")
    print(f"Verified sampling Hz:  {summary_df['true_fs'].unique().tolist()}")
    print(f"Total RR intervals:    {summary_df['n_beats'].sum():,}")
    print(f"Beats/patient range:   {summary_df['n_beats'].min()} to {summary_df['n_beats'].max()}")
    print(f"Failed extractions:    {int(summary_df['failed'].sum())}")

    high_art = summary_df[summary_df["artifact_pct"] > 10.0]
    print(f"High-artifact (>10%):  {len(high_art)} patients")
    if len(high_art) > 0:
        print(high_art[["patient_id", "n_beats", "artifact_pct", "mean_hr_bpm", "sdnn_ms"]].to_string(index=False))

    print("\nSaved cleaned .npy files to data/processed/")
    print("Saved diagnostic plot to results/rr_extraction_check.png")


if __name__ == "__main__":
    main()