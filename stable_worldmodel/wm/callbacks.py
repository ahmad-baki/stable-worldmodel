import math

import wandb
from lightning.pytorch.callbacks import Callback
from loguru import logger as logging

from stable_worldmodel.wm.utils import save_pretrained


class SaveCkptCallback(Callback):
    """Save model checkpoint after each epoch via save_pretrained and log a wandb artifact reference."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1, val_metric_key: str = 'val/loss'):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval
        self.val_metric_key = val_metric_key
        self.best_val_loss = math.inf

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if not trainer.is_global_zero:
            return

        epoch = trainer.current_epoch + 1
        is_interval = epoch % self.epoch_interval == 0
        is_final = epoch == trainer.max_epochs

        if is_interval or is_final:
            val_loss = self._get_val_loss(trainer)
            ckpt_path = self._save(pl_module.model, epoch)
            if ckpt_path is not None:
                self._log_artifact(ckpt_path, epoch, val_loss)

    def _save(self, model, epoch):
        return save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )

    def _get_val_loss(self, trainer):
        metric = trainer.callback_metrics.get(self.val_metric_key)
        if metric is None:
            return None
        try:
            return float(metric)
        except (TypeError, ValueError):
            return None

    def _log_artifact(self, ckpt_path, epoch, val_loss):
        run = wandb.run
        if run is None:
            return

        metadata = {'epoch': epoch}
        if val_loss is not None:
            metadata['val_loss'] = val_loss

        artifact = wandb.Artifact(
            name=self.run_name,
            type='world-model',
            metadata=metadata,
        )
        artifact.add_reference(f'file://{ckpt_path.absolute()}')

        aliases = [f'epoch_{epoch:04d}', 'latest']
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            aliases.append('best')

        try:
            run.log_artifact(artifact, aliases=aliases)
        except Exception as e:
            logging.warning(f'Failed to log wandb artifact for epoch {epoch}: {e}')
