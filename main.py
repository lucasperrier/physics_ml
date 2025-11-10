import datetime
import json
from pathlib import Path
from typing import Dict, List
# from transformers import set_seed
import os
import uuid
import numpy as np
import torch
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from scipy.signal import decimate

from data_processing.datasets import TrajectoryWindowDataset
from data_processing.generate_data import read_trajectories_parquet_as_dicts, as_torch as sample_to_torch
from models.MLP import WindowMLP

def load_data(path: str | Path, *, to_torch: bool = True) -> List[Dict]:
    """
    Load all trajectory records stored as Parquet shards.
    """
    dataset_path = Path(os.fspath(path))
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    samples: List[Dict] = []
    for raw_sample in read_trajectories_parquet_as_dicts(dataset_path):
        samples.append(sample_to_torch(raw_sample) if to_torch else raw_sample)
    return samples

def split_train_val(samples: List[Dict], val_ratio: float, seed: int) -> tuple[List[Dict], List[Dict]]:
    n = len(samples)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(round(val_ratio * n))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    arr = np.array(samples, dtype=object)
    return arr[train_idx].tolist(), arr[val_idx].tolist()

def build_loader(samples: List[Dict], data_cfg: Dict, train: bool, batch_size: int, num_workers: int):
    dataset = TrajectoryWindowDataset(
        samples,
        input_length=data_cfg["input_length"],
        target_length=data_cfg["target_length"],
        step=data_cfg["step"],
        decimation=data_cfg["decimation"],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )


def preprocess_full_sequence(sample: Dict, data_cfg: Dict) -> torch.Tensor:
    """
    Replicates TrajectoryWindowDataset preprocessing for an entire trajectory.
    Returns tensor shaped [state_dim, T_decimated].
    """
    seq = sample["y"]
    if not isinstance(seq, torch.Tensor):
        seq = torch.tensor(seq, dtype=torch.float32)
    else:
        seq = seq.to(torch.float32)
    if seq.ndim != 2:
        raise ValueError("sample['y'] must be [time, states].")
    q = int(data_cfg.get("decimation", 10))
    if q > 1:
        arr = decimate(seq.cpu().numpy(), q, axis=0, zero_phase=True)
        seq = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    # transpose to [states, time]
    return seq.transpose(0, 1).contiguous()

def forecast_full_trajectory(model: WindowMLP, sample: Dict, data_cfg: Dict) -> dict:
    """
    Produce full-horizon forecast on one preprocessed (decimated) trajectory.
    """
    input_len = int(data_cfg["input_length"])
    target_len = int(data_cfg["target_length"])
    full_seq = preprocess_full_sequence(sample, data_cfg)  # [S, T_dec]
    T_dec = full_seq.size(1)
    horizon = T_dec - input_len
    if horizon <= 0:
        return {"run_id": sample.get("run_id"), "rmse": float("nan"), "forecast": None}
    seed = full_seq[:, :input_len]
    with torch.no_grad():
        full_generated, forecast_tail = model.autoregressive_forecast(
            seed,
            forecast_horizon=horizon,
            input_len=input_len,
            target_len=target_len,
            device=model.device,
        )
    # Ground truth tail (same decimated grid)
    target_tail = full_seq[:, input_len:]
    rmse = torch.sqrt(torch.mean((forecast_tail - target_tail) ** 2)).item()
    return {
        "run_id": sample.get("run_id"),
        "rmse": rmse,
        "forecast": forecast_tail.cpu(),
        "target": target_tail.cpu(),
    }


def eval_from_config(config: Dict) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(config["data"]["root"])
    test_dir = root / config["data"]["test_subdir"]
    test_series = load_data(test_dir)  # raw dict samples

    model = WindowMLP.load_from_checkpoint(config["model"]["checkpoint_path"])
    model.to(device)
    model.eval()

    per_sample = []
    results_cache = []
    for sample in test_series:
        res = forecast_full_trajectory(model, sample, config["data"])
        results_cache.append(res)
        if not np.isnan(res["rmse"]):
            per_sample.append(res["rmse"])
    overall_rmse = float(np.mean(per_sample)) if per_sample else float("nan")

    print(f"Eval (decimated) RMSE: {overall_rmse:.6f}")

    out_cfg = config.get("output", {})
    metrics_path = out_cfg.get("metrics_path")
    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump({"rmse": overall_rmse}, f, indent=2)

    forecast_dir = out_cfg.get("forecast_dir")
    if forecast_dir:
        fd = Path(forecast_dir)
        fd.mkdir(parents=True, exist_ok=True)
        for res in results_cache:
            if res["forecast"] is not None:
                torch.save(
                    {
                        "run_id": res.get("run_id"),
                        "forecast": res["forecast"],
                        "target": res["target"],
                        "rmse": res["rmse"],
                    },
                    fd / f"{res.get('run_id', 'series')}.pt",
                )
    return {"rmse": overall_rmse}

def train_from_config(config):
    run_id = uuid.uuid4().hex[:8]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("artifacts") / f"{config['experiment']['name'].replace(' ', '_')}-{timestamp}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    root = Path(config["data"]["root"])
    train_val_dir = root / config["data"]["train_val_subdir"]
    test_dir = root / config["data"]["test_subdir"]

    # Load series (rows) from parquet shards
    tfain_val_samples = load_data(train_val_dir)
    test_series = load_data(test_dir)

    # Random split at series level (not windows)
    train_series, val_series = split_train_val(
        tfain_val_samples,
        val_ratio=config["data"]["val_ratio"],
        seed=config["data"]["seed"],
    )
    # Dataloaders
    train_loader = build_loader(
        train_series, config["data"], train=True,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )
    val_loader = build_loader(
        val_series, config["data"], train=False,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )
    test_loader = build_loader(
        test_series, config["data"], train=False,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )

    # Model
    model = WindowMLP(
        state_dim=config["model"]["state_dim"],
        input_len=config["data"]["input_length"],
        target_len=config["data"]["target_length"],
        hidden_sizes=tuple(config["model"]["hidden_sizes"]),
        lr=config["model"]["lr"],
    )

    loaders_cfg = {
        "input_length": config["data"]["input_length"],
        "target_length": config["data"]["target_length"],
        "step": config["data"]["step"],
        "decimation": config["data"]["decimation"],
        "batch_size": config["training"]["batch_size"],
        "shuffle": config["training"].get("shuffle", True),
        "num_workers": config["training"].get("num_workers", os.cpu_count() - 1),
    }

    checkpoint_cb = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="epoch{epoch:02d}-val_loss{val_loss:.4f}",
        save_top_k=-1,
        every_n_epochs=1,
    )

    # MLflow logger
    mlflow_logger = MLFlowLogger(
        experiment_name=config["experiment"]["name"],
        tracking_uri=config["experiment"]["tracking_uri"],
    )
    # Also log non-model data/training params
    mlflow_logger.log_hyperparams({
        **dict(model.hparams),  # from save_hyperparameters
        "step": config["data"]["step"],
        "decimation": config["data"]["decimation"],
        "batch_size": config["training"]["batch_size"],
        "num_workers": config["training"]["num_workers"],
        "val_ratio": config["data"]["val_ratio"],
        "seed": config["data"]["seed"],
    })

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        logger=mlflow_logger,
        callbacks=[checkpoint_cb],
        deterministic=True,
        default_root_dir=run_dir,
    )
    trainer.fit(model, train_loader, val_loader)

    final_ckpt_path = Path("artifacts") / "final_model.ckpt"
    final_ckpt_path.parent.mkdir(exist_ok=True)
    trainer.save_checkpoint(str(final_ckpt_path))
    print(f"Saved final checkpoint to {final_ckpt_path}")

if __name__ == "__main__":
    with open("config/train.yaml", "r") as fh:
        cfg = yaml.safe_load(fh)
    train_from_config(cfg)
    # with open("config/eval.yaml", "r") as fh:
    #     cfg = yaml.safe_load(fh)
    # eval_from_config(cfg)
