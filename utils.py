
from email.headerregistry import DateHeader
import os
from pathlib import Path
from typing import Dict, List
import uuid
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
import argparse
import pytorch_lightning as pl
import matplotlib.pyplot as plt


from data_processing.datasets import TrajectoryWindowDataset
from data_processing.generate_data import read_trajectories_parquet_as_dicts, as_torch as sample_to_torch
from models.MLP import WindowMLP


def plot_test_series_prediction(
    model: WindowMLP,
    sample: Dict,
    data_cfg: Dict,
    *,
    forecast_result: Dict | None = None,
    state_indices: List[int] | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot ground-truth vs autoregressive forecast for a single test trajectory.
    """
    if forecast_result is None:
        forecast_result = forecast_full_trajectory(model, sample, data_cfg)
    forecast = forecast_result.get("forecast")
    target = forecast_result.get("target")
    if forecast is None or target is None:
        raise RuntimeError("Sample too short for the configured window; no forecast produced.")

    full_seq = preprocess_full_sequence(sample, data_cfg)  # [S, T_decimated]
    input_len = int(data_cfg["input_length"])
    time_raw = sample["t"]
    if isinstance(time_raw, torch.Tensor):
        time_raw = time_raw.detach().cpu().numpy()
    else:
        time_raw = np.asarray(time_raw, dtype=np.float64)
    time_dec = np.linspace(time_raw[0], time_raw[-1], full_seq.size(1), dtype=np.float64)
    forecast_len = forecast.size(1)
    tail_time = time_dec[input_len: input_len + forecast_len]

    forecast_np = forecast.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    if state_indices is None:
        state_indices = list(range(target_np.shape[0]))

    fig, axes = plt.subplots(len(state_indices), 1, figsize=(8, 2.4 * len(state_indices)), sharex=True, squeeze=False)
    axes = axes.flatten()

    for ax, idx in zip(axes, state_indices):
        ax.plot(tail_time, target_np[idx], label="ground truth", linewidth=1.4)
        ax.plot(tail_time, forecast_np[idx], label="prediction", linestyle="--", linewidth=1.4)
        ax.set_ylabel(f"state {idx}")
        ax.legend(loc="best")

    axes[-1].set_xlabel("time")
    fig.suptitle(f"Forecast vs Actual — run_id={sample.get('run_id')}", fontsize=12)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig

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
    decimation = int(data_cfg.get("decimation", 1))
    seq_time_states = TrajectoryWindowDataset.preprocess_series(sample, decimation=decimation)  # [time, states]
    return seq_time_states.transpose(0, 1).contiguous()

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
