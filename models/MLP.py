
import torch
from torch import nn

from models.seq2seqmodule import Seq2SeqForecastingModule

class WindowMLP(Seq2SeqForecastingModule):
    def __init__(
        self,
        state_dim: int,
        input_len: int,
        target_len: int,
        hidden_sizes: tuple[int, ...],
        lr: float = 1e-3,
    ):
        super().__init__(lr=lr)
        self.save_hyperparameters({
            "state_dim": state_dim,
            "input_len": input_len,
            "target_len": target_len,
            "hidden_sizes": list(hidden_sizes),
            "lr": lr,
        })

        layers = []
        in_dim = state_dim * input_len
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, state_dim * target_len))
        self.net = nn.Sequential(*layers)
        self.state_dim = state_dim
        self.target_len = target_len

    def forward(self, past: torch.Tensor, future_len: int) -> torch.Tensor:  # type: ignore[override]
        if future_len != self.target_len:
            raise ValueError(f"WindowMLP configured for target_len={self.target_len}, got {future_len}.")
        batch = past.size(0)
        x = past.view(batch, -1)
        y = self.net(x)
        return y.view(batch, self.state_dim, self.target_len)
        
    def autoregressive_forecast(
        model: nn.Module,
        initial_sequence: torch.Tensor,
        forecast_horizon: int,
        input_len: int,
        target_len: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if initial_sequence.ndim != 2:
            raise ValueError("initial_sequence must be shaped [states, time].")
        if initial_sequence.size(1) < input_len:
            raise ValueError("initial_sequence must contain at least input_len steps.")
        device = device or next(model.parameters()).device
        history = initial_sequence.to(device)
        generated: list[torch.Tensor] = []
        produced = 0
        while produced < forecast_horizon:
            window = history[:, -input_len:].contiguous()
            window_batch = window.unsqueeze(0)  # [1, S, input_len]
            with torch.no_grad():
                step_pred = model(window_batch, target_len)  # type: ignore[call-arg]
            step_pred = step_pred.squeeze(0)                 # [S, target_len]
            need = min(step_pred.size(1), forecast_horizon - produced)
            step_pred = step_pred[:, :need]
            generated.append(step_pred)
            history = torch.cat([history, step_pred], dim=1)
            produced += need
        forecast = torch.cat(generated, dim=1) if generated else torch.empty(
            history.size(0), 0, device=history.device
        )
        full_sequence = torch.cat([initial_sequence.to(device), forecast], dim=1)
        return full_sequence, forecast