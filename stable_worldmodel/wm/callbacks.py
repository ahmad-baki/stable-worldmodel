import math

import torch
import wandb
from lightning.pytorch.callbacks import Callback
from loguru import logger as logging

from stable_worldmodel.wm.utils import save_pretrained


class SaveCkptCallback(Callback):
    """Save model checkpoint after each epoch via save_pretrained and log a wandb artifact reference."""

    def __init__(
        self,
        run_name,
        cfg,
        epoch_interval: int = 1,
        val_metric_key: str = 'val/loss',
        save_subdir: str | None = None,
    ):
        super().__init__()
        self.run_name = run_name
        self.save_subdir = save_subdir or run_name
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
            ckpt_path, object_path = self._save(pl_module.model, epoch)
            # Prefer referencing the full-module object so the artifact loads
            # directly (torch.load / AutoCostModel) with no architecture rebuild;
            # fall back to the state-dict .pt when dump_object is off.
            ref_path = object_path or ckpt_path
            if ref_path is not None:
                self._log_artifact(ref_path, epoch, val_loss)

    def _save(self, model, epoch):
        ckpt_path = save_pretrained(
            model,
            run_name=self.save_subdir,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )
        # Dump the full module object *per epoch* so the wandb artifact can
        # reference a directly-loadable module (torch.load / AutoCostModel) with
        # no architecture rebuild. The '_object.ckpt' suffix keeps
        # swm.policy.AutoCostModel's '*_object.ckpt' glob working; the epoch in
        # the name avoids the single-file overwrite so every version is kept.
        object_path = None
        if ckpt_path is not None and self.cfg.get('dump_object', False):
            object_path = (
                ckpt_path.parent
                / f'{self.run_name}_epoch_{epoch:04d}_object.ckpt'
            )
            torch.save(model, object_path)
            logging.info(f'📦 Saved model object to {object_path}')
        return ckpt_path, object_path

    def _get_val_loss(self, trainer):
        metric = trainer.callback_metrics.get(self.val_metric_key)
        if metric is None:
            return None
        try:
            return float(metric)
        except (TypeError, ValueError):
            return None

    def _log_artifact(self, ref_path, epoch, val_loss):
        run = wandb.run
        if run is None:
            return

        metadata = {'epoch': epoch}
        if val_loss is not None:
            metadata['val_loss'] = val_loss

        artifact = wandb.Artifact(
            name=self.run_name,
            type='model',
            metadata=metadata,
        )
        # Reference the on-disk file (full module object when dump_object is on,
        # else the state-dict .pt). file:// refs resolve only on the filesystem
        # that logged them.
        artifact.add_reference(f'file://{ref_path.absolute()}')

        aliases = [f'epoch_{epoch:04d}', 'latest']
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            aliases.append('best')

        try:
            run.log_artifact(artifact, aliases=aliases)
        except Exception as e:
            logging.warning(f'Failed to log wandb artifact for epoch {epoch}: {e}')
