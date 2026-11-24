import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import welch

from dataset import PROCESSED_DIR, get_dataloaders
from models import build_model, count_parameters

CKPT_DIR = Path("checkpoints")
RES_DIR = Path("results")
RES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# HRV Metric Helpers
# ---------------------------------------------------------------------------
def compute_window_hrv(rr_windows_ms: np.ndarray) -> dict[str, np.ndarray]:
    """Compute vectorized time-domain HRV metrics for shape (N, H) RR windows in ms."""
    mean_rr = np.mean(rr_windows_ms, axis=1)
    mean_hr = 60000.0 / np.clip(mean_rr, 250.0, 2500.0)
    sdnn = np.std(rr_windows_ms, axis=1, ddof=1)
    diff_rr = np.diff(rr_windows_ms, axis=1)
    rmssd = np.sqrt(np.mean(diff_rr ** 2, axis=1))
    pnn50 = np.mean(np.abs(diff_rr) > 50.0, axis=1) * 100.0
    return {
        "mean_hr_bpm": mean_hr,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "pnn50_pct": pnn50,
    }


def compute_lf_hf_ratio(rr_ms: np.ndarray, fs_interp: float = 4.0) -> float:
    """Compute frequency-domain LF/HF ratio from a multi-minute RR series (ms)."""
    if len(rr_ms) < 60:
        return np.nan

    t = np.cumsum(rr_ms) / 1000.0
    t = t - t[0]
    if t[-1] < 60.0:
        return np.nan

    t_uni = np.arange(0.0, t[-1], 1.0 / fs_interp)
    rr_uni = np.interp(t_uni, t, rr_ms)
    rr_uni = rr_uni - np.mean(rr_uni)

    nperseg = min(256, len(rr_uni))
    freqs, psd = welch(rr_uni, fs=fs_interp, nperseg=nperseg)

    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)

    lf_power = np.trapezoid(psd[lf_mask], freqs[lf_mask])
    hf_power = np.trapezoid(psd[hf_mask], freqs[hf_mask])

    if hf_power < 1e-9:
        return np.nan
    return float(lf_power / hf_power)


def compute_full_recording_hrv(rr_ms: np.ndarray) -> dict[str, float]:
    """Time + frequency domain HRV on a stitched multi-minute RR tachogram."""
    diff_rr = np.diff(rr_ms)
    return {
        "sdnn_ms": float(np.std(rr_ms, ddof=1)),
        "rmssd_ms": float(np.sqrt(np.mean(diff_rr ** 2))),
        "pnn50_pct": float(np.mean(np.abs(diff_rr) > 50.0) * 100.0),
        "lf_hf": compute_lf_hf_ratio(rr_ms),
    }


# ---------------------------------------------------------------------------
# Inference Helpers
# ---------------------------------------------------------------------------
def predict_dl_model(model, loader, device):
    """Run batched inference and measure latency."""
    model.eval()
    preds = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            out = model(xb)
            preds.append(out.cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return np.concatenate(preds, axis=0), elapsed_ms


def predict_stitched_patient(model_or_name, rr_ms, mean_ms, std_ms, lookback, horizon, device):
    """Non-overlapping 30-beat rolling forecast across a full patient recording."""
    rr_norm = (rr_ms - mean_ms) / std_ms
    starts = list(range(0, len(rr_norm) - lookback - horizon + 1, horizon))

    X_win = np.array([rr_norm[s : s + lookback] for s in starts], dtype=np.float32)
    y_true_blocks = [rr_ms[s + lookback : s + lookback + horizon] for s in starts]
    y_true_stitched = np.concatenate(y_true_blocks)

    if model_or_name == "persistence":
        pred_norm = np.repeat(X_win[:, -1:], horizon, axis=1)
    elif model_or_name == "ma10":
        pred_norm = np.repeat(X_win[:, -10:].mean(axis=1, keepdims=True), horizon, axis=1)
    else:
        xb = torch.tensor(X_win, dtype=torch.float32, device=device).unsqueeze(-1)
        with torch.no_grad():
            pred_norm = model_or_name(xb).cpu().numpy()

    y_pred_stitched = (pred_norm * std_ms + mean_ms).reshape(-1)
    return y_true_stitched, y_pred_stitched


# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------
def plot_training_curves(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {"lstm": "#2563EB", "gru": "#10B981", "transformer": "#F59E0B"}

    for name in ["lstm", "gru", "transformer"]:
        h = history[name]
        epochs = np.arange(1, len(h["train_loss"]) + 1)
        c = colors[name]

        axes[0].plot(epochs, h["train_loss"], linestyle="--", alpha=0.55, color=c, label=f"{name.upper()} Train")
        axes[0].plot(epochs, h["val_loss"], lw=2.0, color=c, label=f"{name.upper()} Val")
        axes[1].plot(epochs, h["val_mae_ms"], lw=2.0, marker="o", ms=4, color=c, label=f"{name.upper()} (best {h['best_val_mae_ms']:.2f} ms)")

    axes[0].set_title("Training vs. Validation Huber Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_title("Validation RR MAE Across Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (ms)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(RES_DIR / "01_training_curves.png", dpi=150)
    plt.close()


def plot_horizon_error(horizon_maes: dict):
    fig, ax = plt.subplots(figsize=(10, 5))
    styles = {
        "Persistence": ("#9CA3AF", ":", 1.8),
        "MA(10)": ("#6B7280", "--", 1.8),
        "LSTM": ("#2563EB", "-", 2.2),
        "GRU": ("#10B981", "-", 2.2),
        "Transformer": ("#F59E0B", "-", 2.2),
    }
    steps = np.arange(1, len(next(iter(horizon_maes.values()))) + 1)

    for label, mae_curve in horizon_maes.items():
        c, ls, lw = styles[label]
        ax.plot(steps, mae_curve, color=c, linestyle=ls, lw=lw, label=label)

    ax.set_title("Test-Set RR Forecasting Error by Horizon Step (Beats 1–30)")
    ax.set_xlabel("Forecast Horizon (Beats Ahead)")
    ax.set_ylabel("MAE (ms)")
    ax.set_xticks(range(1, len(steps) + 1, 2))
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(RES_DIR / "02_horizon_error.png", dpi=150)
    plt.close()


def plot_sample_forecasts(X_ms, y_ms, preds_ms, patient_tags, test_ids):
    chosen_patients = test_ids[:3]
    fig, axes = plt.subplots(len(chosen_patients), 1, figsize=(12, 8.5), sharex=True)
    lookback = X_ms.shape[1]
    horizon = y_ms.shape[1]
    t_hist = np.arange(-lookback, 0)
    t_fut = np.arange(0, horizon)

    for ax, pid in zip(axes, chosen_patients):
        idxs = [i for i, p in enumerate(patient_tags) if p == pid]
        idx = idxs[len(idxs) // 2]  # Middle window of recording

        ax.plot(t_hist, X_ms[idx], color="#374151", lw=1.4, label="History (60 beats)")
        ax.plot(t_fut, y_ms[idx], color="#111827", lw=2.0, label="Ground Truth (30 beats)")
        ax.plot(t_fut, preds_ms["LSTM"][idx], color="#2563EB", lw=1.7, label="LSTM")
        ax.plot(t_fut, preds_ms["GRU"][idx], color="#10B981", lw=1.7, linestyle="--", label="GRU")
        ax.plot(t_fut, preds_ms["Transformer"][idx], color="#F59E0B", lw=1.7, linestyle="-.", label="Transformer")
        ax.axvline(0, color="#EF4444", linestyle=":", lw=1.2)
        ax.set_title(f"Test Patient {pid} — 60-Beat History & 30-Beat RR Forecast")
        ax.set_ylabel("RR Interval (ms)")
        ax.grid(alpha=0.3)

    axes[0].legend(loc="upper left", ncol=5, fontsize=8.5)
    axes[-1].set_xlabel("Relative Beat Index (0 = Forecast Start)")
    plt.tight_layout()
    plt.savefig(RES_DIR / "03_sample_forecasts.png", dpi=150)
    plt.close()


def plot_comparison_bars(rr_df: pd.DataFrame, hrv_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    models = rr_df["Model"].tolist()
    x = np.arange(len(models))
    w = 0.35

    # Left: RR MAE & RMSE
    b1 = axes[0].bar(x - w / 2, rr_df["RR_MAE_ms"], width=w, color="#2563EB", label="RR MAE (ms)")
    b2 = axes[0].bar(x + w / 2, rr_df["RR_RMSE_ms"], width=w, color="#60A5FA", label="RR RMSE (ms)")
    axes[0].set_title("Test-Set RR Forecasting Errors (Lower is Better)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15)
    axes[0].set_ylabel("Error (ms)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    # Right: Derived Window HRV Errors (SDNN & RMSSD MAE)
    b3 = axes[1].bar(x - w / 2, hrv_df["Win_SDNN_MAE_ms"], width=w, color="#10B981", label="SDNN MAE (ms)")
    b4 = axes[1].bar(x + w / 2, hrv_df["Win_RMSSD_MAE_ms"], width=w, color="#F59E0B", label="RMSSD MAE (ms)")
    axes[1].set_title("Derived Window-Level HRV Errors (Lower is Better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15)
    axes[1].set_ylabel("MAE (ms)")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()
    for bar in list(b3) + list(b4):
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.25, f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(RES_DIR / "04_model_comparison.png", dpi=150)
    plt.close()


def plot_stitched_tachogram(pid, y_true, stitched_preds):
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(y_true, color="#111827", lw=1.4, alpha=0.85, label="Ground Truth RR")
    ax.plot(stitched_preds["LSTM"], color="#2563EB", lw=1.3, alpha=0.9, label="LSTM (stitched 30-beat forecasts)")
    ax.plot(stitched_preds["GRU"], color="#10B981", lw=1.2, alpha=0.85, linestyle="--", label="GRU")
    ax.plot(stitched_preds["Transformer"], color="#F59E0B", lw=1.2, alpha=0.85, linestyle="-.", label="Transformer")
    ax.set_title(f"Full-Recording Stitched RR Forecast vs. Ground Truth — Test Patient {pid}")
    ax.set_xlabel("Beat Index")
    ax.set_ylabel("RR Interval (ms)")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(RES_DIR / "05_stitched_tachogram.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main Evaluation Pipeline
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, cfg = get_dataloaders(batch_size=64)

    mean_ms = cfg["train_mean_ms"]
    std_ms = cfg["train_std_ms"]
    lookback = cfg["lookback"]
    horizon = cfg["horizon"]

    with open(RES_DIR / "training_history.json", "r") as f:
        history = json.load(f)

    # Inverse-transform test X and y to milliseconds
    X_norm = test_loader.dataset.X.squeeze(-1).numpy()
    y_norm = test_loader.dataset.y.numpy()
    X_ms = X_norm * std_ms + mean_ms
    y_ms = y_norm * std_ms + mean_ms
    patient_tags = test_loader.dataset.patient_tags

    # Collect predictions (in ms) for baselines + all 3 DL models
    preds_ms = {}
    infer_ms = {}
    params_map = {"Persistence": 0, "MA(10)": 0}
    train_time_map = {"Persistence": 0.0, "MA(10)": 0.0}
    loaded_models = {}

    # Naive Baselines
    t0 = time.perf_counter()
    preds_ms["Persistence"] = np.repeat(X_ms[:, -1:], horizon, axis=1)
    infer_ms["Persistence"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    preds_ms["MA(10)"] = np.repeat(X_ms[:, -10:].mean(axis=1, keepdims=True), horizon, axis=1)
    infer_ms["MA(10)"] = (time.perf_counter() - t0) * 1000.0

    # Deep Learning Models
    for key, label in [("lstm", "LSTM"), ("gru", "GRU"), ("transformer", "Transformer")]:
        model = build_model(key, horizon=horizon).to(device)
        ckpt_path = CKPT_DIR / f"{key}_best.pt"
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        loaded_models[label] = model

        pred_norm, elapsed_ms = predict_dl_model(model, test_loader, device)
        preds_ms[label] = pred_norm * std_ms + mean_ms
        infer_ms[label] = elapsed_ms
        params_map[label] = count_parameters(model)
        train_time_map[label] = history[key]["train_time_sec"]

    # 1. Window-Level RR & HRV Metrics
    true_win_hrv = compute_window_hrv(y_ms)
    rr_rows = []
    hrv_rows = []
    horizon_maes = {}

    model_order = ["Persistence", "MA(10)", "LSTM", "GRU", "Transformer"]

    for label in model_order:
        yp = preds_ms[label]
        err = yp - y_ms

        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mape = float(np.mean(np.abs(err) / np.clip(y_ms, 1e-6, None)) * 100.0)
        horizon_maes[label] = np.mean(np.abs(err), axis=0)

        rr_rows.append({
            "Model": label,
            "Params": params_map[label],
            "Train_Time_s": train_time_map[label],
            "Infer_Time_ms": round(infer_ms[label], 2),
            "RR_MAE_ms": round(mae, 2),
            "RR_RMSE_ms": round(rmse, 2),
            "RR_MAPE_pct": round(mape, 2),
        })

        # Window-level HRV error
        pred_win_hrv = compute_window_hrv(yp)
        win_hr_mae = float(np.mean(np.abs(pred_win_hrv["mean_hr_bpm"] - true_win_hrv["mean_hr_bpm"])))
        win_sdnn_mae = float(np.mean(np.abs(pred_win_hrv["sdnn_ms"] - true_win_hrv["sdnn_ms"])))
        win_rmssd_mae = float(np.mean(np.abs(pred_win_hrv["rmssd_ms"] - true_win_hrv["rmssd_ms"])))
        win_pnn50_mae = float(np.mean(np.abs(pred_win_hrv["pnn50_pct"] - true_win_hrv["pnn50_pct"])))

        # Patient-level stitched recording HRV error (includes LF/HF)
        full_sdnn_errs, full_rmssd_errs, full_lfhf_errs = [], [], []
        model_ref = label.lower() if label in ("Persistence",) else ("ma10" if label == "MA(10)" else loaded_models[label])

        for pid in cfg["test_ids"]:
            rr_full = np.load(PROCESSED_DIR / f"{pid}_rr.npy").astype(np.float64)
            yt_s, yp_s = predict_stitched_patient(model_ref, rr_full, mean_ms, std_ms, lookback, horizon, device)
            h_true = compute_full_recording_hrv(yt_s)
            h_pred = compute_full_recording_hrv(yp_s)

            full_sdnn_errs.append(abs(h_pred["sdnn_ms"] - h_true["sdnn_ms"]))
            full_rmssd_errs.append(abs(h_pred["rmssd_ms"] - h_true["rmssd_ms"]))
            if not (np.isnan(h_true["lf_hf"]) or np.isnan(h_pred["lf_hf"])):
                full_lfhf_errs.append(abs(h_pred["lf_hf"] - h_true["lf_hf"]))

        hrv_rows.append({
            "Model": label,
            "Win_HR_MAE_bpm": round(win_hr_mae, 2),
            "Win_SDNN_MAE_ms": round(win_sdnn_mae, 2),
            "Win_RMSSD_MAE_ms": round(win_rmssd_mae, 2),
            "Win_pNN50_MAE_pct": round(win_pnn50_mae, 2),
            "Full_SDNN_MAE_ms": round(float(np.mean(full_sdnn_errs)), 2),
            "Full_RMSSD_MAE_ms": round(float(np.mean(full_rmssd_errs)), 2),
            "Full_LFHF_MAE": round(float(np.mean(full_lfhf_errs)), 2) if full_lfhf_errs else np.nan,
        })

    rr_df = pd.DataFrame(rr_rows)
    hrv_df = pd.DataFrame(hrv_rows)
    rr_df.to_csv(RES_DIR / "rr_comparison.csv", index=False)
    hrv_df.to_csv(RES_DIR / "hrv_comparison.csv", index=False)

    # Generate all 5 plots
    plot_training_curves(history)
    plot_horizon_error(horizon_maes)
    plot_sample_forecasts(X_ms, y_ms, preds_ms, patient_tags, cfg["test_ids"])
    plot_comparison_bars(rr_df, hrv_df)

    # Stitched tachogram for the first test patient
    sample_pid = cfg["test_ids"][0]
    rr_sample = np.load(PROCESSED_DIR / f"{sample_pid}_rr.npy").astype(np.float64)
    stitched_dict = {}
    for label in ["LSTM", "GRU", "Transformer"]:
        yt_s, yp_s = predict_stitched_patient(loaded_models[label], rr_sample, mean_ms, std_ms, lookback, horizon, device)
        stitched_dict[label] = yp_s
    plot_stitched_tachogram(sample_pid, yt_s, stitched_dict)

    # Print final comparative tables
    print("\n" + "=" * 82)
    print("TABLE 1: TEST-SET RR INTERVAL FORECASTING COMPARISON (7 Unseen Patients, 4,136 Windows)")
    print("=" * 82)
    print(rr_df.to_string(index=False))

    print("\n" + "=" * 95)
    print("TABLE 2: DERIVED HRV ACCURACY (Window-Level 30-Beat & Full-Recording Stitched HRV)")
    print("=" * 95)
    print(hrv_df.to_string(index=False))

    print("\nSaved CSV tables and 5 comparison plots to results/")


if __name__ == "__main__":
    main()