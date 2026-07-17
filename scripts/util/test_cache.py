"""Correctness gate for precomputed DINO embeddings.

Verifies that (1) cached ``pixels_embed`` values equal on-the-fly DINO encoding
of the same raw frames (alignment + fp16 storage), and (2) the cached dataset
loads through the training pipeline with the right shapes/keys.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import h5py
import numpy as np
import stable_pretraining as spt
import torch
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import stable_worldmodel as swm
from scripts.train.gcbc import get_img_pipeline

RAW = os.environ.get('RAW', 'pusht_expert_train.h5')
EMBED = os.environ['EMBED']  # embed h5 name or path


def transform_pixels(raw_frames_hwc):
    """Apply the exact training image pipeline to (T,H,W,C) uint8 frames."""
    pipe = spt.data.transforms.Compose(get_img_pipeline('pixels', 'pixels'))
    t = torch.from_numpy(raw_frames_hwc).permute(0, 3, 1, 2)  # -> (T,C,H,W)
    return pipe({'pixels': t})['pixels']


def main():
    enc = AutoModel.from_pretrained('facebook/dinov2-small')
    enc = enc.to('cuda').eval().requires_grad_(False)

    raw_ds = swm.data.load_dataset(RAW, num_steps=1, keys_to_load=['pixels'])
    emb_ds = swm.data.load_dataset(EMBED, num_steps=1,
                                   keys_to_load=['pixels_embed', 'action'])
    # Match HDF5Dataset's swmr=True so the later emb_ds[0] (which reopens the
    # embed file with swmr=True in this same process) doesn't clash.
    raw_h5 = h5py.File(raw_ds.h5_path, 'r', swmr=True)
    emb_h5 = h5py.File(emb_ds.h5_path, 'r', swmr=True)

    F = emb_h5['pixels_embed'].shape[0]
    print(f'embed frames F={F} | shape={emb_h5["pixels_embed"].shape} '
          f'dtype={emb_h5["pixels_embed"].dtype}')

    # ---- (1) numerical gate on random frames ----
    rng = np.random.default_rng(0)
    idxs = sorted(rng.choice(F, size=16, replace=False).tolist())
    raw_px = raw_h5['pixels'][idxs]                       # (16,H,W,C)
    x = transform_pixels(raw_px).to('cuda')               # (16,C,H,W)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        live = enc(x, interpolate_pos_encoding=True).last_hidden_state[:, 1:, :]
    live = live.float().cpu().numpy()
    cached = emb_h5['pixels_embed'][idxs].astype(np.float32)

    abs_err = np.abs(live - cached)
    denom = np.abs(live).mean() + 1e-6
    aligned_rel = abs_err.mean() / denom
    print(f'cached vs live: max|err|={abs_err.max():.4f} '
          f'mean|err|={abs_err.mean():.5f} rel={aligned_rel:.4%} '
          f'|feat|={np.abs(live).mean():.3f}')
    # fp16 + bf16-encode tolerance; a misalignment would blow this up massively
    assert aligned_rel < 0.02, 'cached embeddings diverge from live!'
    # Confirm it is NOT accidentally matching a shifted frame. Adjacent PushT
    # frames are very similar, so use a relative bar: the shifted-frame error
    # must dwarf the aligned error (misalignment would make them comparable).
    shifted = emb_h5['pixels_embed'][[(i + 1) % F for i in idxs]].astype('f4')
    shift_err = np.abs(live - shifted).mean() / denom
    print(f'sanity: rel err vs SHIFTED-by-1 frame = {shift_err:.2%} '
          f'({shift_err/aligned_rel:.0f}x the aligned error)')
    assert shift_err > 20 * aligned_rel, 'embeddings look frame-shifted!'

    # ---- (2) pipeline shape check via the training-style loaders ----
    row = emb_ds[0]
    pe = row['pixels_embed']
    print(f'pipeline: pixels_embed shape={tuple(pe.shape)} dtype={pe.dtype} '
          f"action shape={tuple(row['action'].shape)}")
    assert pe.shape[-2:] == (256, 384), 'unexpected embed shape'

    print('OK: cached embeddings verified.')


if __name__ == '__main__':
    main()
