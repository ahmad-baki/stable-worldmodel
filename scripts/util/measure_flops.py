"""Measure real training FLOPs per optimizer step for GCBC, then extrapolate.

Reuses the real ``get_data`` / ``get_gcbc_policy`` pipeline and counts actual
ATen FLOPs (forward + backward) for ONE batch with torch's built-in
``FlopCounterMode``. Multiply the printed per-step number by the number of
optimizer steps a run actually completed (pull that from wandb, key
``trainer/global_step``) to get total training FLOPs for that run.

Run just like training (same overrides pick the same architecture):
    python scripts/train/measure_flops.py
    python scripts/train/measure_flops.py cache_embeddings=true embed_dataset_name=...

NOTE: with ``cache_embeddings=true`` the frozen DINO forward is skipped during
training (it's a one-time precompute cost), so the per-step FLOPs then reflect
only the trainable predictor. Measure with the SAME flags the run used.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import hydra
import torch
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.train.gcbc import get_data, get_gcbc_policy


@hydra.main(version_base=None, config_path='./config', config_name='gcbc')
def run(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = get_data(cfg)
    module = get_gcbc_policy(cfg).to(device)
    module.train()
    model = module.model

    try:
        loader = data.train_dataloader()
    except Exception:
        loader = data.train
    batch = next(iter(loader))
    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }

    hist = cfg.dinowm.history_size
    px_key = 'pixels_embed' if cfg.get('cache_embeddings', False) else 'pixels'

    def one_step():
        b = dict(batch)
        if 'proprio' in b:
            b['proprio'] = torch.nan_to_num(b['proprio'], 0.0)
        b['action'] = torch.nan_to_num(b['action'], 0.0)
        b = model.encode(b, target='embed', pixels_key=px_key)
        b = model.encode(
            b, target='goal_embed', pixels_key=f'goal_{px_key}', prefix='goal_'
        )
        emb = b['embed'][:, :hist, :, :]
        goal_emb = b['goal_embed']
        action_pred, _ = model.predict_actions(emb, goal_emb)
        action_target = b['action'][:, :hist, :]
        loss = F.mse_loss(action_pred, action_target)
        loss.backward()

    # Warm up once (build the autograd graph / lazy buffers) before counting.
    one_step()
    module.zero_grad(set_to_none=True)

    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        one_step()
    total = flop_counter.get_total_flops()  # forward + backward, one batch

    bs = cfg.batch_size
    spe = len(loader)  # optimizer steps per epoch (drop_last=True)
    print(f'[FLOPS] cache_embeddings      : {cfg.get("cache_embeddings", False)}')
    print(f'[FLOPS] batch_size            : {bs}')
    print(f'[FLOPS] steps / epoch         : {spe}')
    print(f'[FLOPS] FLOPs / optimizer step: {total:.4e}   (fwd+bwd)')
    print(f'[FLOPS] FLOPs / sample        : {total / bs:.4e}')
    print(f'[FLOPS] ---')
    print(f'[FLOPS] total = (FLOPs/step) x (num optimizer steps from wandb).')
    print(
        f'[FLOPS] e.g. full {cfg.trainer.max_epochs} epochs '
        f'= {cfg.trainer.max_epochs * spe} steps '
        f'-> {total * cfg.trainer.max_epochs * spe:.4e} FLOPs'
    )


if __name__ == '__main__':
    run()
