import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders
from models import build_model, count_parameters

CKPT_DIR = Path("checkpoints")
RES_DIR = Path("results")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 8


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_naive_baselines(val_loader, criterion):
    """Reference losses for simple non-DL baselines on the validation set."""
    last_losses, ma10_losses = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            horizon = yb.size(1)
            pred_last = xb[:, -1, 0].unsqueeze(1).expand(-1, horizon)
            pred_ma10 = xb[:, -10:, 0].mean(dim=1, keepdim=True).expand(-1, horizon)
            last_losses.append(criterion(pred_last, yb).item() * len(xb))
            ma10_losses.append(criterion(pred_ma10, yb).item() * len(xb))

    n = len(val_loader.dataset)
    return sum(last_losses) / n, sum(ma10_losses) / n


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_mae_norm = 0.0
    n_samples = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            preds = model(xb)
            loss = criterion(preds, yb)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_mae_norm += torch.mean(torch.abs(preds - yb)).item() * bs
            n_samples += bs

    return total_loss / n_samples, total_mae_norm / n_samples


def train_single_model(model_name, train_loader, val_loader, cfg, device):
    set_seed(SEED)
    model = build_model(model_name, horizon=cfg["horizon"]).to(device)
    n_params = count_parameters(model)

    criterion = nn.HuberLoss(delta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_ctr = 0
    ckpt_path = CKPT_DIR / f"{model_name}_best.pt"
    std_ms = cfg["train_std_ms"]

    history = {
        "model": model_name,
        "params": n_params,
        "train_loss": [],
        "val_loss": [],
        "val_mae_ms": [],
        "lr": [],
    }

    print("\n" + "=" * 65)
    print(f"Training {model_name.upper():<12} | Params: {n_params:,} | Device: {device}")
    print("=" * 65)

    t0 = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, _ = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_mae_norm = run_epoch(model, val_loader, criterion, device, optimizer=None)
        val_mae_ms = val_mae_norm * std_ms

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_ms"].append(val_mae_ms)
        history["lr"].append(current_lr)

        marker = ""
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
            marker = " *"
        else:
            patience_ctr += 1

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {tr_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val MAE: {val_mae_ms:6.2f} ms | "
            f"LR: {current_lr:.1e}{marker}"
        )

        if patience_ctr >= EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
            break

    elapsed = time.perf_counter() - t0
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    history["best_val_mae_ms"] = history["val_mae_ms"][best_epoch - 1]
    history["train_time_sec"] = round(elapsed, 2)

    print(
        f"Done {model_name.upper()} in {elapsed:.1f}s | "
        f"Best Val Loss: {best_val_loss:.5f} | "
        f"Best Val MAE: {history['best_val_mae_ms']:.2f} ms"
    )
    return history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _, cfg = get_dataloaders(batch_size=BATCH_SIZE)

    criterion = nn.HuberLoss(delta=0.5)
    last_loss, ma10_loss = evaluate_naive_baselines(val_loader, criterion)
    print(f"Validation Reference -> Persistence Loss: {last_loss:.5f} | MA(10) Loss: {ma10_loss:.5f}")

    all_histories = {}
    for name in ["lstm", "gru", "transformer"]:
        all_histories[name] = train_single_model(name, train_loader, val_loader, cfg, device)

    out_path = RES_DIR / "training_history.json"
    with open(out_path, "w") as f:
        json.dump(all_histories, f, indent=2)

    print("\n" + "=" * 65)
    print("TRAINING SUMMARY (Validation Set)")
    print("=" * 65)
    print(f"{'Model':<14} {'Params':>10} {'Best Ep':>8} {'Val Loss':>10} {'Val MAE (ms)':>14} {'Time (s)':>10}")
    print("-" * 65)
    for name, h in all_histories.items():
        print(
            f"{name.upper():<14} {h['params']:>10,} {h['best_epoch']:>8} "
            f"{h['best_val_loss']:>10.5f} {h['best_val_mae_ms']:>14.2f} {h['train_time_sec']:>10.1f}"
        )
    print(f"\nSaved checkpoints to {CKPT_DIR}/ and history to {out_path}")


if __name__ == "__main__":
    main()