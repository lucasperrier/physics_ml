
import torch
from torch import nn
from models.seq2seqmodule import Seq2SeqForecastingModule


class EncoderDecoderRNN(Seq2SeqForecastingModule):
    def __init__(self,
                state_dim: int,
                input_length: int,
                target_length: int,
                hidden_dim: int,
                num_layers: int = 1,
                lr: float = 1e-3,
                ):
        super().__init__(lr=lr)

        self.state_dim = state_dim 
        self.target_length = target_length

        self.save_hyperparameters({
            "state_dim": state_dim,
            "target_length": input_length,
            "target_length": target_length,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "lr": lr,
        })

        self.encoder = nn.RNN(
            input_size=state_dim,      
            hidden_size=hidden_dim,    # latent size 
            num_layers=num_layers,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, target_length * state_dim),
        )

    def decoder(self, h0):
        h_last = h0[-1]                     # (N, hidden_dim)
        flat = self.mlp(h_last)             # (N, target_length*state_dim)
        return flat.view(-1, self.target_length, self.state_dim)
    
    def forward(self, past: torch.Tensor, future_len: int) -> torch.Tensor:  # type: ignore[override]
        if future_len != self.target_len:
            raise ValueError(f"WindowMLP configured for target_len={self.target_len}, got {future_len}.")
        
        past_seq = past.permute(0, 2, 1).contiguous()   # Expect past as [batch, state_dim, input_len] -> convert to [batch, input_len, state_dim]
        _, h_n = self.encoder(past_seq)                 # h_n: [num_layers, batch, hidden_dim]
        dec_seq = self.decoder(h_n)                     # [batch, target_length, state_dim]
        return dec_seq.permute(0, 2, 1).contiguous()    # [batch, state_dim, target_length]
    
    @torch.no_grad()
    def autoregressive_forecast(
        self,
        initial_sequence: torch.Tensor,
        forecast_horizon: int,
        input_len: int,
        target_len: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Since forward is now parallel, use it directly for full horizon if target_len == forecast_horizon
        # Otherwise, fall back to chunking (but parallel per chunk)
        if forecast_horizon <= target_len:
            full_generated = initial_sequence
            forecast_tail = self(initial_sequence.unsqueeze(0), forecast_horizon).squeeze(0)
            return full_generated, forecast_tail
        else:
            # Chunk into target_len blocks (parallel per block)
            generated = []
            history = initial_sequence
            produced = 0
            while produced < forecast_horizon:
                remaining = min(target_len, forecast_horizon - produced)
                window = history[:, -input_len:].unsqueeze(0)
                preds = self(window, target_len).squeeze(0)[:, :remaining]
                generated.append(preds)
                history = torch.cat([history, preds], dim=1)
                produced += remaining
            forecast = torch.cat(generated, dim=1)
            return history, forecast