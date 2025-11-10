
import torch
from torch import nn
from data_processing.datasets import Seq2SeqForecastingModule

class AutoencoderRNN(Seq2SeqForecastingModule):
    def __init__(self, state_dim: int, hidden_dim: int, num_layers: int = 1, lr: float = 1e-3):
        super().__init__(lr=lr)
        self.encoder = nn.GRU(state_dim, hidden_dim, num_layers)
        self.decoder = nn.GRU(state_dim, hidden_dim, num_layers)
        self.output = nn.Linear(hidden_dim, state_dim)

    def forward(self, past: torch.Tensor, future_len: int) -> torch.Tensor:  # type: ignore[override]
        past_seq = past.permute(2, 0, 1).contiguous()   # [T1, batch, S]
        _, hidden = self.encoder(past_seq)
        decoder_input = past_seq[-1:].contiguous()      # [1, batch, S]
        outputs = []
        for _ in range(future_len):
            dec_out, hidden = self.decoder(decoder_input, hidden)
            step = self.output(dec_out[-1])             # [batch, S]
            outputs.append(step)
            decoder_input = step.unsqueeze(0)           # [1, batch, S]
        preds = torch.stack(outputs, dim=0)             # [T2, batch, S]
        return preds.permute(1, 2, 0).contiguous()      # [batch, S, T2]
