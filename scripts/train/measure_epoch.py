"""Measure per-epoch wall time for (cached) GCBC training.

Reuses the real ``get_data`` / ``get_gcbc_policy`` pipeline and just adds a
per-epoch timer callback, so the number reflects the actual training loop
(cached embeddings + real disk I/O on full data). Val is disabled to isolate
train-epoch time.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import hydra
import lightning as pl
from lightning.pytorch.callbacks import Callback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.train.gcbc import get_data, get_gcbc_policy


class EpochTimer(Callback):
    def on_train_epoch_start(self, *_):
        self._t = time.time()
        self._nb = 0

    def on_train_batch_end(self, *_):
        self._nb += 1

    def on_train_epoch_end(self, trainer, pl_module):
        dt = time.time() - self._t
        bps = self._nb / dt if dt else 0
        print(f'[TIMING] epoch {trainer.current_epoch}: {dt:.1f}s '
              f'({self._nb} batches, {bps:.2f} batch/s)', flush=True)


@hydra.main(version_base=None, config_path='./config', config_name='gcbc')
def run(cfg):
    data = get_data(cfg)
    module = get_gcbc_policy(cfg)
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[EpochTimer()],
        num_sanity_val_steps=0,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(module, datamodule=data)


if __name__ == '__main__':
    run()
