import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Seq2SeqForecastingModule(pl.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = float(lr)

    def forward(self, past: torch.Tensor, future_len: int) -> torch.Tensor:  # type: ignore[override]
        raise NotImplementedError

    def _shared_step(self, batch):
        past, future = batch
        preds = self.forward(past, future.size(-1))
        loss = F.mse_loss(preds, future)
        return loss, preds

    def training_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds = self._shared_step(batch)
        mae = F.l1_loss(preds, batch[1])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae", mae, prog_bar=True, on_step=False, on_epoch=True)

    def on_fit_start(self):
        # Push all saved hparams to MLflow (requires child to call save_hyperparameters)
        if getattr(self, "logger", None) and hasattr(self.logger, "log_hyperparams"):
            try:
                self.logger.log_hyperparams(dict(self.hparams))
            except Exception:
                pass
            
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)