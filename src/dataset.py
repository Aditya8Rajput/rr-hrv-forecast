import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

PROCESSED_DIR = Path("data/processed")
CONFIG_PATH = PROCESSED_DIR / "split_config.json"

LOOKBACK = 60          # Past 60 beats as input
HORIZON = 30           # Next 30 beats (~25-30s) as forecast target
MAX_ARTIFACT_PCT = 20.0
SEED = 42


def prepare_splits(
    max_artifact_pct: float = MAX_ARTIFACT_PCT,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
    seed: int = SEED,
    force: bool = False,
) -> dict:
    """Create reproducible patient-wise splits and compute train-only normalization stats."""
    if CONFIG_PATH.exists() and not force:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    summary_path = PROCESSED_DIR / "rr_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError("Run src/extract_rr.py first to generate rr_summary.csv.")

    df = pd.read_csv(summary_path)
    min_required_beats = lookback + horizon + 10

    valid_mask = (
        (~df["failed"])
        & (df["artifact_pct"] <= max_artifact_pct)
        & (df["n_beats"] >= min_required_beats)
    )

    valid_ids = sorted(df.loc[valid_mask, "patient_id"].tolist())
    excluded_ids = sorted(df.loc[~valid_mask, "patient_id"].tolist())

    rng = np.random.default_rng(seed)
    shuffled = valid_ids.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(round(0.70 * n_total))
    n_val = int(round(0.15 * n_total))

    train_ids = sorted(shuffled[:n_train])
    val_ids = sorted(shuffled[n_train : n_train + n_val])
    test_ids = sorted(shuffled[n_train + n_val :])

    # Compute standardization stats ONLY on training patients (zero data leakage)
    train_rr = np.concatenate([
        np.load(PROCESSED_DIR / f"{pid}_rr.npy") for pid in train_ids
    ])
    mean_rr = float(np.mean(train_rr))
    std_rr = float(np.std(train_rr))

    config = {
        "seed": seed,
        "lookback": lookback,
        "horizon": horizon,
        "max_artifact_pct": max_artifact_pct,
        "train_mean_ms": mean_rr,
        "train_std_ms": std_rr,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "excluded_ids": excluded_ids,
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    return config


class RRForecastDataset(Dataset):
    """Sliding-window RR-interval dataset returning (X, y) normalized tensors."""

    def __init__(
        self,
        patient_ids: list[str],
        mean: float,
        std: float,
        lookback: int = LOOKBACK,
        horizon: int = HORIZON,
        stride: int = 1,
    ):
        self.X_list = []
        self.y_list = []
        self.patient_tags = []

        for pid in patient_ids:
            rr = np.load(PROCESSED_DIR / f"{pid}_rr.npy").astype(np.float32)
            rr_norm = (rr - mean) / std

            max_start = len(rr_norm) - lookback - horizon + 1
            for start in range(0, max_start, stride):
                end_x = start + lookback
                end_y = end_x + horizon
                self.X_list.append(rr_norm[start:end_x])
                self.y_list.append(rr_norm[end_x:end_y])
                self.patient_tags.append(pid)

        # Shapes: X -> (N, lookback, 1), y -> (N, horizon)
        self.X = torch.tensor(np.array(self.X_list), dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(np.array(self.y_list), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def get_dataloaders(batch_size: int = 64, force_resplit: bool = False):
    """Build train, val, and test DataLoaders along with the split config."""
    config = prepare_splits(force=force_resplit)

    train_ds = RRForecastDataset(
        config["train_ids"],
        config["train_mean_ms"],
        config["train_std_ms"],
        config["lookback"],
        config["horizon"],
        stride=1,
    )
    val_ds = RRForecastDataset(
        config["val_ids"],
        config["train_mean_ms"],
        config["train_std_ms"],
        config["lookback"],
        config["horizon"],
        stride=1,
    )
    test_ds = RRForecastDataset(
        config["test_ids"],
        config["train_mean_ms"],
        config["train_std_ms"],
        config["lookback"],
        config["horizon"],
        stride=1,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, config


if __name__ == "__main__":
    train_loader, val_loader, test_loader, cfg = get_dataloaders(batch_size=64, force_resplit=True)
    xb, yb = next(iter(train_loader))

    print("=" * 60)
    print(f"Kept patients:     {len(cfg['train_ids']) + len(cfg['val_ids']) + len(cfg['test_ids'])} / 53")
    print(f"Excluded (>20%):   {len(cfg['excluded_ids'])} -> {cfg['excluded_ids']}")
    print(f"Train / Val / Test patients: {len(cfg['train_ids'])} / {len(cfg['val_ids'])} / {len(cfg['test_ids'])}")
    print(f"Train scaler:      mean = {cfg['train_mean_ms']:.2f} ms, std = {cfg['train_std_ms']:.2f} ms")
    print("-" * 60)
    print(f"Train windows:     {len(train_loader.dataset):,}")
    print(f"Val windows:       {len(val_loader.dataset):,}")
    print(f"Test windows:      {len(test_loader.dataset):,}")
    print(f"Batch X shape:     {tuple(xb.shape)}  (batch, lookback, features)")
    print(f"Batch y shape:     {tuple(yb.shape)}  (batch, horizon)")
    print(f"\nSaved split config to {CONFIG_PATH}")