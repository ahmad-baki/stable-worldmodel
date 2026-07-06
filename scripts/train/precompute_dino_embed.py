"""Precompute frozen DINOv2 patch embeddings for a pixel dataset.

The GCBC encoder (``facebook/dinov2-small``) is frozen, so a frame's patch
embedding is identical every epoch. This script encodes every frame once and
writes a self-contained HDF5 dataset with a ``pixels_embed`` column (plus
copied ``action``/``proprio``/``ep_len``/``ep_offset``) that ``gcbc.py`` can
train on directly with ``cache_embeddings=true`` — removing ~99% of the
per-epoch compute.

Embeddings are produced with the EXACT training image pipeline
(``get_img_pipeline``) and the same ``last_hidden_state[:, 1:, :]`` slice as
``GCRL._encode_image``, so cached features match on-the-fly encoding.

Usage:
    python scripts/train/precompute_dino_embed.py \
        --input pusht_expert_train.h5 \
        --output /.../datasets/pusht_expert_train_dinoembed.h5 \
        [--num-episodes M]   # subset of the first M episodes (for testing)
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import h5py
import numpy as np
import stable_pretraining as spt
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import stable_worldmodel as swm
from scripts.train.gcbc import get_img_pipeline

PATCHES, DIM = 256, 384  # dinov2-small @ 224/14


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='pusht_expert_train.h5')
    ap.add_argument('--output', required=True)
    ap.add_argument('--num-episodes', type=int, default=None,
                    help='only encode the first M episodes (contiguous); '
                         'default = all')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--image-size', type=int, default=224)
    args = ap.parse_args()

    # Raw pixel dataset with the exact training image pipeline, one frame/clip.
    ds = swm.data.load_dataset(
        args.input, num_steps=1, frameskip=1, transform=None,
        keys_to_load=['pixels'],
    )
    ds.transform = spt.data.transforms.Compose(
        get_img_pipeline('pixels', 'pixels', args.image_size)
    )
    lengths = np.asarray(ds.lengths)
    offsets = np.asarray(ds.offsets)
    E, N = len(lengths), int(lengths.sum())

    M = args.num_episodes if args.num_episodes else E
    M = min(M, E)
    F = int(offsets[M]) if M < E else N  # frames in first M episodes
    print(f'dataset={args.input} episodes={E} frames={N} | '
          f'encoding first M={M} episodes -> F={F} frames')
    # For num_steps=1/frameskip=1 and no num_traj, clip index == flat frame
    # index, so DataLoader(shuffle=False) over Subset(range(F)) yields frames
    # 0..F-1 in flat order.
    assert len(ds) == N, f'expected {N} clips, got {len(ds)}'

    enc = AutoModel.from_pretrained('facebook/dinov2-small')
    enc = enc.to('cuda').eval().requires_grad_(False)

    # Read the small columns into memory and CLOSE the raw file *before* the
    # DataLoader forks workers. Each worker reopens the raw dataset with
    # swmr=True; an inherited non-SWMR handle to the same file (kept open in the
    # parent) makes HDF5 raise "SWMR read access flag not the same".
    with h5py.File(ds.h5_path, 'r') as raw:
        action = raw['action'][:F]
        proprio = raw['proprio'][:F] if 'proprio' in raw else None

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out = h5py.File(args.output, 'w', libver='latest')
    emb = out.create_dataset(
        'pixels_embed', shape=(F, PATCHES, DIM), dtype='f2',
        chunks=(1, PATCHES, DIM),
    )
    out.create_dataset('action', data=action)
    if proprio is not None:
        out.create_dataset('proprio', data=proprio)
    out.create_dataset('ep_len', data=lengths[:M])
    out.create_dataset('ep_offset', data=offsets[:M])
    est_gb = F * PATCHES * DIM * 2 / 1e9
    print(f'writing {args.output} | pixels_embed ~{est_gb:.1f} GB (fp16)')

    loader = DataLoader(
        Subset(ds, range(F)), batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, prefetch_factor=4,
    )

    idx, t0, last = 0, time.time(), time.time()
    for batch in loader:
        x = batch['pixels'].squeeze(1).float().to('cuda', non_blocking=True)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            out_emb = enc(x, interpolate_pos_encoding=True).last_hidden_state
            out_emb = out_emb[:, 1:, :]  # drop CLS -> (b, 256, 384)
        b = out_emb.shape[0]
        emb[idx:idx + b] = out_emb.float().half().cpu().numpy()
        idx += b
        if time.time() - last > 30:
            rate = idx / (time.time() - t0)
            print(f'  {idx}/{F} frames  ({rate:.0f} f/s, '
                  f'ETA {(F - idx) / max(rate, 1) / 60:.1f} min)', flush=True)
            last = time.time()

    assert idx == F, f'wrote {idx} != {F}'
    out.close()
    print(f'DONE: {idx} frames in {(time.time() - t0) / 60:.1f} min -> '
          f'{args.output}')


if __name__ == '__main__':
    main()
