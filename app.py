import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.models import build_model, count_parameters

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CKPT_DIR = BASE_DIR / "checkpoints"
RES_DIR = BASE_DIR / "results"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="RR-HRV Forecast Studio")

if RES_DIR.exists():
    app.mount("/results", StaticFiles(directory=str(RES_DIR)), name="results")

with open(PROCESSED_DIR / "split_config.json", "r") as f:
    SPLIT_CFG = json.load(f)

SUMMARY_DF = pd.read_csv(PROCESSED_DIR / "rr_summary.csv").set_index("patient_id")

LOOKBACK = int(SPLIT_CFG["lookback"])
HORIZON = int(SPLIT_CFG["horizon"])
MEAN_MS = float(SPLIT_CFG["train_mean_ms"])
STD_MS = float(SPLIT_CFG["train_std_ms"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

MODELS = {}
MODEL_PARAMS = {}
for key, label in [("lstm", "LSTM"), ("gru", "GRU"), ("transformer", "Transformer")]:
    m = build_model(key, horizon=HORIZON).to(DEVICE)
    ckpt = CKPT_DIR / f"{key}_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    m.eval()
    MODELS[label] = m
    MODEL_PARAMS[label] = count_parameters(m)


def compute_hrv(rr_ms: np.ndarray) -> dict:
    mean_rr = float(np.mean(rr_ms))
    mean_hr = float(60000.0 / np.clip(mean_rr, 250.0, 2500.0))
    sdnn = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0
    diff_rr = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) > 0 else 0.0
    pnn50 = float(np.mean(np.abs(diff_rr) > 50.0) * 100.0) if len(diff_rr) > 0 else 0.0
    return {
        "mean_rr_ms": round(mean_rr, 2),
        "mean_hr_bpm": round(mean_hr, 2),
        "sdnn_ms": round(sdnn, 2),
        "rmssd_ms": round(rmssd, 2),
        "pnn50_pct": round(pnn50, 2),
    }


def run_all_models(history_ms: np.ndarray):
    hist_norm = (history_ms - MEAN_MS) / STD_MS
    x_tensor = torch.tensor(hist_norm, dtype=torch.float32, device=DEVICE).view(1, LOOKBACK, 1)

    preds = {}
    timings = {}

    t0 = time.perf_counter()
    preds["Persistence"] = np.full(HORIZON, history_ms[-1], dtype=np.float64)
    timings["Persistence"] = round((time.perf_counter() - t0) * 1000.0, 3)

    t0 = time.perf_counter()
    preds["MA(10)"] = np.full(HORIZON, float(np.mean(history_ms[-10:])), dtype=np.float64)
    timings["MA(10)"] = round((time.perf_counter() - t0) * 1000.0, 3)

    with torch.no_grad():
        for label, model in MODELS.items():
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out_norm = model(x_tensor).squeeze(0).cpu().numpy()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            timings[label] = round((time.perf_counter() - t0) * 1000.0, 3)
            preds[label] = (out_norm * STD_MS + MEAN_MS).astype(np.float64)

    return preds, timings


@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="static/index.html not found")
    return FileResponse(index_file)


@app.get("/api/meta")
def get_meta():
    patients = []
    for split_name, id_list in [
        ("test", SPLIT_CFG["test_ids"]),
        ("val", SPLIT_CFG["val_ids"]),
        ("train", SPLIT_CFG["train_ids"]),
    ]:
        for pid in id_list:
            rr_path = PROCESSED_DIR / f"{pid}_rr.npy"
            if not rr_path.exists():
                continue
            n_beats = int(len(np.load(rr_path)))
            row = SUMMARY_DF.loc[pid] if pid in SUMMARY_DF.index else None
            patients.append({
                "patient_id": pid,
                "split": split_name,
                "n_beats": n_beats,
                "max_start": max(0, n_beats - LOOKBACK - HORIZON),
                "mean_hr_bpm": float(row["mean_hr_bpm"]) if row is not None else None,
                "artifact_pct": float(row["artifact_pct"]) if row is not None else None,
            })

    return {
        "device": DEVICE_NAME,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "train_mean_ms": round(MEAN_MS, 2),
        "train_std_ms": round(STD_MS, 2),
        "model_params": MODEL_PARAMS,
        "patients": patients,
    }


@app.get("/api/predict_patient")
def predict_patient(
    patient_id: str = Query(...),
    start_idx: int = Query(0, ge=0),
):
    rr_path = PROCESSED_DIR / f"{patient_id}_rr.npy"
    if not rr_path.exists():
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    rr = np.load(rr_path).astype(np.float64)
    max_start = len(rr) - LOOKBACK - HORIZON
    if max_start < 0:
        raise HTTPException(status_code=400, detail="Patient sequence is shorter than lookback + horizon")

    start_idx = min(start_idx, max_start)
    history = rr[start_idx : start_idx + LOOKBACK]
    truth = rr[start_idx + LOOKBACK : start_idx + LOOKBACK + HORIZON]

    preds, timings = run_all_models(history)
    truth_hrv = compute_hrv(truth)

    model_metrics = []
    forecasts_out = {}
    for label in ["LSTM", "GRU", "Transformer", "MA(10)", "Persistence"]:
        yp = preds[label]
        err = yp - truth
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        hrv = compute_hrv(yp)
        forecasts_out[label] = [round(float(v), 2) for v in yp]
        model_metrics.append({
            "model": label,
            "mae_ms": round(mae, 2),
            "rmse_ms": round(rmse, 2),
            "infer_ms": timings[label],
            "hrv": hrv,
            "sdnn_err_ms": round(abs(hrv["sdnn_ms"] - truth_hrv["sdnn_ms"]), 2),
            "rmssd_err_ms": round(abs(hrv["rmssd_ms"] - truth_hrv["rmssd_ms"]), 2),
        })

    return {
        "patient_id": patient_id,
        "start_idx": start_idx,
        "max_start": max_start,
        "n_beats": len(rr),
        "history_ms": [round(float(v), 2) for v in history],
        "truth_ms": [round(float(v), 2) for v in truth],
        "truth_hrv": truth_hrv,
        "forecasts_ms": forecasts_out,
        "metrics": model_metrics,
    }


class CustomRRRequest(BaseModel):
    rr_values: list[float]


@app.post("/api/predict_custom")
def predict_custom(req: CustomRRRequest):
    vals = [float(v) for v in req.rr_values if 250.0 <= float(v) <= 2500.0]
    if len(vals) < LOOKBACK:
        raise HTTPException(
            status_code=400,
            detail=f"Provide at least {LOOKBACK} valid RR intervals (between 250 and 2500 ms). Got {len(vals)}.",
        )

    history = np.array(vals[-LOOKBACK:], dtype=np.float64)
    preds, timings = run_all_models(history)

    results = []
    forecasts_out = {}
    for label in ["LSTM", "GRU", "Transformer"]:
        yp = preds[label]
        forecasts_out[label] = [round(float(v), 2) for v in yp]
        results.append({
            "model": label,
            "infer_ms": timings[label],
            "hrv": compute_hrv(yp),
        })

    return {
        "history_ms": [round(float(v), 2) for v in history],
        "history_hrv": compute_hrv(history),
        "forecasts_ms": forecasts_out,
        "results": results,
    }


@app.get("/api/benchmarks")
def get_benchmarks():
    rr_csv = RES_DIR / "rr_comparison.csv"
    hrv_csv = RES_DIR / "hrv_comparison.csv"
    hist_json = RES_DIR / "training_history.json"

    rr_table = (
        pd.read_csv(rr_csv).replace({np.nan: None}).to_dict(orient="records")
        if rr_csv.exists()
        else []
    )
    hrv_table = (
        pd.read_csv(hrv_csv).replace({np.nan: None}).to_dict(orient="records")
        if hrv_csv.exists()
        else []
    )
    training_history = json.loads(hist_json.read_text()) if hist_json.exists() else {}

    plot_titles = {
        "01_training_curves.png": "Training vs. Validation Loss & MAE Curves",
        "02_horizon_error.png": "Forecasting Error by Horizon Step (Beats 1–30)",
        "03_sample_forecasts.png": "Sample 30-Beat Forecasts on Unseen Test Patients",
        "04_model_comparison.png": "RR & Derived HRV Error Comparison",
        "05_stitched_tachogram.png": "Full-Recording Stitched RR Tachogram",
        "rr_extraction_check.png": "ECG R-Peak Detection & Artifact Cleaning Check",
    }
    plots = []
    for p in sorted(RES_DIR.glob("*.png")):
        plots.append({
            "filename": p.name,
            "url": f"/results/{p.name}",
            "title": plot_titles.get(p.name, p.stem.replace("_", " ").title()),
        })

    return {
        "rr_comparison": rr_table,
        "hrv_comparison": hrv_table,
        "training_history": training_history,
        "plots": plots,
    }


if __name__ == "__main__":
    print(f"Starting RR-HRV Forecast Studio on http://127.0.0.1:8000 (Device: {DEVICE_NAME})")
    uvicorn.run(app, host="127.0.0.1", port=8000)
