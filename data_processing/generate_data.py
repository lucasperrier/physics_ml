import os
from pathlib import Path
from typing import Callable, Dict, List, Iterable
import uuid

from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import matplotlib.pyplot as plt

def simulate_ode(
    rhs: Callable[[float, np.ndarray, Dict[str, float]], np.ndarray],
    y0: np.ndarray,
    t0: float,
    t1: float,
    dt: float,
    params: Dict[str, float],
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = np.inf,
) -> Dict:
    """
    Simulate a system dy/dt = rhs(t, y, params) on a fixed time grid.

    Returns dict with keys: t (T,), y (T, D), dt, params
    """
    t_eval = np.arange(t0 + 1e-12, t1, dt, dtype=np.float64)
    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, params),
        t_span=(t0, t1),
        y0=np.asarray(y0, dtype=np.float64),
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        method="Radau",
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    y = sol.y.T
    return {"t": t_eval, "y": y, "dt": float(dt), "params": params}

def trajectories_to_table(rows: List[Dict]) -> pa.Table:
    """
    rows: list of dicts with keys:
      - system: str
      - run_id: str
      - dt: float
      - t: np.ndarray (T,)
      - y: np.ndarray (T, D)
      - param_keys: List[str]
      - param_values: List[float]
    """
    # Convert numpy arrays to Arrow list arrays
    t_col = pa.array([r["t"] for r in rows], type=pa.list_(pa.float64()))
    # y is list<list<float64>>
    y_col = pa.array([r["y"].tolist() for r in rows], type=pa.list_(pa.list_(pa.float64())))
    system_col = pa.array([r["system"] for r in rows], type=pa.string())
    run_id_col = pa.array([r["run_id"] for r in rows], type=pa.string())
    dt_col = pa.array([float(r["dt"]) for r in rows], type=pa.float64())
    pkeys_col = pa.array([r["param_keys"] for r in rows], type=pa.list_(pa.string()))
    pvals_col = pa.array([r["param_values"] for r in rows], type=pa.list_(pa.float64()))
    return pa.Table.from_arrays(
        [system_col, run_id_col, dt_col, t_col, y_col, pkeys_col, pvals_col],
        names=["system", "run_id", "dt", "t", "y", "param_keys", "param_values"],
    )

def save_trajectories_parquet(file_path: str | Path, rows: List[Dict], overwrite: bool = True) -> None:
    file_path = Path(file_path)
    table = trajectories_to_table(rows)
    if overwrite and file_path.exists():
        file_path.unlink()
    pq.write_table(table, str(file_path))

def read_trajectories_parquet_as_dicts(file_path: str | Path) -> Iterable[Dict]:
    file_path = Path(file_path)
    if file_path.is_dir():
        batches = ds.dataset(str(file_path), format="parquet").to_batches()
    else:
        batches = pq.ParquetFile(str(file_path)).iter_batches()
    for batch in batches:
        df = batch.to_pandas()
        for _, row in df.iterrows():
            params = dict(zip(list(row["param_keys"]), list(row["param_values"])))
            yield {
                "system": str(row["system"]),
                "run_id": str(row["run_id"]),
                "dt": float(row["dt"]),
                "t": np.asarray(list(row["t"]), dtype=np.float64),
                "y": np.asarray([list(state) for state in row["y"]], dtype=np.float64),
                "params": params,
            }

def as_torch(sample: Dict):
    return {
        "system": sample["system"],
        "run_id": sample["run_id"],
        "dt": torch.tensor(sample["dt"], dtype=torch.float32),
        "t": torch.tensor(sample["t"], dtype=torch.float32),
        "y": torch.tensor(sample["y"], dtype=torch.float32),  # [T, D]
        "params": {k: torch.tensor(v, dtype=torch.float32) for k, v in sample["params"].items()},
    }

def plot_sample(sample: Dict, show: bool = True, save_dir: str | Path | None = None) -> None:
    """
    Plot one trajectory (time series for each state dimension).
    """
    t = sample["t"]
    y = sample["y"]
    D = y.shape[1]
    state_handles = []

    fig, ax = plt.subplots(figsize=(8, 3.5))
    cmap = plt.cm.get_cmap("tab10" if D <= 10 else "tab20")
    for i in range(D):
        color = cmap(i % cmap.N)
        (line,) = ax.plot(t, y[:, i], linewidth=1.2, label=f"y[{i}]", color=color)
        state_handles.append(line)

    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.set_title(f"{sample['system']} | run_id={sample['run_id']}")
    if D > 1:
        ax.legend(loc="best")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"{sample['system']}_{sample['run_id']}.png", dpi=150, bbox_inches="tight")

    state_legend = None
    if D > 1:
        state_legend = ax.legend(handles=state_handles, loc="upper left", title="states")
    if state_legend is not None:
        ax.add_artist(state_legend)

    params = sample.get("params", {})
    if params:
        param_handles = [
            Line2D([], [], linestyle="", marker="", color="none", label=f"{name} = {float(value):.4g}")
            for name, value in params.items()
        ]
        ax.legend(handles=param_handles, loc="upper right", title="params", frameon=False)
        
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_all_trajectories(file_path: str | Path, show: bool = True, save_dir: str | Path | None = None, limit: int | None = None) -> None:
    """
    Iterate through all trajectories in a Parquet file and plot them.
    limit: optional maximum number of trajectories to plot.
    """
    count = 0
    for sample in read_trajectories_parquet_as_dicts(file_path):
        plot_sample(sample, show=show, save_dir=save_dir)
        count += 1
        if limit is not None and count >= limit:
            break

def append_trajectories_dataset(dataset_dir: str | Path, rows: List[Dict]) -> None:
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    table = trajectories_to_table(rows)
    shard_path = dataset_dir / f"part-{uuid.uuid4()}.parquet"
    pq.write_table(table, shard_path)

def collect_start_and_end_values(
    file_path: str | Path,
    system: str,
    start_var_idx: int,
    end_var_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    starts: list[float] = []
    ends: list[float] = []
    for sample in read_trajectories_parquet_as_dicts(file_path):
        if sample["system"] != system:
            continue
        y = sample["y"]
        if start_var_idx >= y.shape[1] or end_var_idx >= y.shape[1]:
            raise IndexError("state variable index out of range")
        starts.append(float(y[0, start_var_idx]))
        ends.append(float(y[-1, end_var_idx]))
    return np.asarray(starts, dtype=np.float64), np.asarray(ends, dtype=np.float64)

def plot_start_vs_end(
    start_values: np.ndarray,
    end_values: np.ndarray,
    start_label: str,
    end_label: str,
    show: bool = True,
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if start_values.shape != end_values.shape:
        raise ValueError("start_values and end_values must have the same shape")
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    ax.scatter(start_values, end_values, s=12, alpha=0.7)
    ax.set_xlabel(f"Initial {start_label}")
    ax.set_ylabel(f"Final {end_label}")
    ax.set_title(f"{end_label} vs {start_label}")
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return ax

# ------ Plotting --------

def collect_param_and_end_values(
    file_path: str | Path,
    system: str,
    param_name: str,
    end_var_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    starts: list[float] = []
    ends: list[float] = []
    for sample in read_trajectories_parquet_as_dicts(file_path):
        if sample["system"] != system:
            continue
        if param_name not in sample["params"]:
            raise KeyError(f"parameter '{param_name}' missing in run {sample['run_id']}")
        y = sample["y"]
        if end_var_idx >= y.shape[1]:
            raise IndexError("state variable index out of range")
        starts.append(float(sample["params"][param_name]))
        ends.append(float(y[-1, end_var_idx]))
    return np.asarray(starts, dtype=np.float64), np.asarray(ends, dtype=np.float64)

# ------ ODE Systems ------

def lorenz(t: float, y: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    sigma = p.get("sigma", 10.0)
    beta = p.get("beta", 8.0 / 3.0)
    rho = p.get("rho", 28.0)
    x, v, z = y
    return np.array([
        sigma * (v - x),
        x * (rho - z) - v,
        x * v - beta * z,
    ], dtype=np.float64)

def lotka_volterra(t: float, y: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    alpha = p.get("alpha", 2)
    beta = p.get("beta", 2)
    gamma = p.get("gamma", 2)
    delta = p.get("delta", 2)
    x, y = y
    return np.array([
        alpha * x - beta * x * y,
        -gamma * y + delta * x * y,
    ], dtype=np.float64)

def chen(t: float, y: np.ndarray, parameters: Dict[str, float]) -> np.ndarray:
    GAMMA_d = parameters.get("GAMMA_d", 2)
    gamma_z = parameters.get("gamma_z", 2)
    delta = parameters.get("delta", 2)
    p, s, z, sigma = y
    return np.array([
         p - 2 * z * s * np.cos(sigma),
        -GAMMA_d * s + z * p * np.cos(sigma),
        -gamma_z * z + 2 * p * s * np.cos(sigma),
        -delta - (p * z * np.sin(sigma)) / s,
    ], dtype=np.float64)

def plot_param_vs_endpoint(
    dataset_dir: str | Path,
    system: str,
    param_name: str,
    end_var_idx: int,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Axes:
    dataset_dir = Path(dataset_dir)
    starts, ends = collect_param_and_end_values(
        file_path=dataset_dir,
        system=system,
        param_name=param_name,
        end_var_idx=end_var_idx,
    )
    return plot_start_vs_end(
        start_values=starts,
        end_values=ends,
        start_label=param_name,
        end_label=f"final y[{end_var_idx}]",
        show=show,
        save_path=save_path,
    )

if __name__ == "__main__":

    # system = "lotka_volterra"
    # dt = 1e-3
    # t0, t1 = 0.0, 10.0
    # params = {"sigma": 10.0, "beta": 8.0 / 3.0, "rho": 28.0}

    # system = "lotka_volterra"
    # dt = 1e-3
    # t0, t1 = 0.0, 10.0
    # params = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0}

    system = "chen"
    dt = 1e-3
    t0, t1 = 0.0, 500.0
    state_low, state_high = 2.0, 2.0
    param_ranges = {
        "GAMMA_d": (2.0, 2.0),
        "gamma_z": (0.1, 1.0),
        "delta": (2.0, 2.0),
    }

    rng_seed = None
    rng = np.random.default_rng(rng_seed)

    N = 1
    trajectories = []
    for _ in range(N):
        params = {
            key: float(rng.uniform(low, high))
            for key, (low, high) in param_ranges.items()
        }
        y0 = np.array([
            2.0,      # p
            2.0,      # s
            rng.uniform(state_low, state_high),      # z
            rng.uniform(0.0, 2.0 * np.pi),           # sigma
        ], dtype=np.float64)

        sim = simulate_ode(chen, y0=y0, t0=t0, t1=t1, dt=dt, params=params)
        run_id = str(uuid.uuid4())
        trajectories.append({
            "system": system,
            "run_id": run_id,
            "dt": sim["dt"],
            "t": sim["t"],
            "y": sim["y"],
            "param_keys": list(sim["params"].keys()),
            "param_values": [float(v) for v in sim["params"].values()],
        })

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = out_dir / f"{system}_trajectories"
    append_trajectories_dataset(dataset_dir, trajectories)
    print(f"Appended {len(trajectories)} trajectories to {dataset_dir}")

    # for sample in read_trajectories_parquet_as_dicts(dataset_dir):
    #     print(sample["system"], sample["run_id"], sample["t"].shape, sample["y"].shape)

    # plot_all_trajectories(dataset_dir, show=True)

    file_path = dataset_dir
    param_name="gamma_z"

    initial, end = collect_param_and_end_values(
        file_path=file_path,
        system="chen",
        param_name=param_name,
        end_var_idx=2,
    )
    plot_start_vs_end(initial, end, start_label=param_name, end_label="final y[0]")