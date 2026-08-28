#!/usr/bin/env python3
"""Convert a re-rendered PushT expert ``.npz`` into our HDF5 schema.

The npz is the Diffusion-Policy PushT demo set re-rendered at 224px by
``offline-rl-with-le-wm-dev/scripts/regenerate_pusht_expert.py``: keys
``states(N,5)``, ``actions(N,2)``, ``images(N,224,224,3) uint8`` and
``episode_ends(206,)``.

Two action layouts ship under that name and ``--action auto`` tells them apart
by the ``action_space`` marker the relative export writes:

* ``pusht_expert_224.npz`` -- **absolute**: DP's raw target positions in the
  512-px world. Our env expects a normalised delta
  ``(target - agent) / action_scale`` with ``action_scale=100``, clipped to
  [-1, 1], which is what ``convert_dp_pusht.py`` applies and what this script
  reproduces.
* ``pusht_expert_224_relative.npz`` -- **relative** (``action_space =
  'swm_relative'``, ``action_scale = 100``, box [-1, 1]): that same transform
  has already been applied at export time, so the actions are copied straight
  through. Verified: the stored relative actions equal
  ``clip((absolute - agent) / 100, -1, 1)`` to 2.9e-8.

This is otherwise ``convert_dp_pusht.py`` with the rendering step removed. The
``states``/``actions``/``episode_ends`` of the absolute npz are byte-identical
to ``pusht_dp_raw/pusht_dp_source.npz``, so the state convention is reconciled
exactly as that script does: **state** is widened 5 -> 7 by appending an agent
velocity, giving ``[agent_xy, block_xy, angle, agent_vx, agent_vy]``, and
``proprio`` is ``[agent_xy, agent_vx, agent_vy]``.

The only thing that is *not* reproducible from this source is velocity. The
expert set (``pusht_expert_success.h5``) stores the true pymunk agent velocity,
which is not a finite difference of consecutive positions -- checked on the
first three frames, the implied dt is not constant. Both options here are
therefore approximations. ``--velocity forward`` (the default) reproduces
``pusht_dp.h5`` exactly, so the two conversions of this same source agree;
``--velocity backward`` instead matches the expert set's visible convention of
a zero velocity on the first frame of each episode. Velocity is unused by the
LeWM GCBC recipe (``use_proprio_encoder: false``) and PushT runs with
``damping=0``, so the choice only matters for exact state replay.

Images are copied through untouched; they are streamed straight out of the npz
zip member so the 3.9 GB array is never materialised.

Output columns match ``pusht_expert_success.h5`` exactly: action, pixels,
state, proprio, episode_idx, step_idx (+ ep_len/ep_offset written by
HDF5Writer).

Usage::

    python scripts/data/convert_pusht_expert_npz.py \
        --npz <datasets>/pusht_expert_224.npz \
        --out <datasets>/pusht_expert_224.h5
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from stable_worldmodel.data.formats.hdf5 import HDF5Writer


ACTION_SCALE = 100.0
CONTROL_DT = 0.1  # DP demos are logged at 10 Hz; used only for the velocity finite-diff.

DEFAULT_NPZ = Path(
    '/hkfs/work/workspace/scratch/usjuy-worldbenchdata/datasets/pusht_expert_224.npz'
)
DEFAULT_OUT = Path(
    '/hkfs/work/workspace/scratch/usjuy-worldbenchdata/datasets/pusht_expert_224.h5'
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--npz', type=Path, default=DEFAULT_NPZ)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--mode', choices=('overwrite', 'error', 'append'), default='error')
    p.add_argument(
        '--velocity',
        choices=('forward', 'backward'),
        default='forward',
        help='forward = match pusht_dp.h5; backward = zero velocity on the first frame.',
    )
    p.add_argument(
        '--action',
        choices=('auto', 'absolute', 'relative'),
        default='auto',
        help='absolute = DP target positions, converted; relative = already '
        'normalised, copied through; auto = decide from the npz metadata.',
    )
    p.add_argument('--max-episodes', type=int, default=None, help='Debug cap on #episodes.')
    return p.parse_args()


def _read_npy_header(stream):
    version = np.lib.format.read_magic(stream)
    shape, fortran_order, dtype = np.lib.format._read_array_header(stream, version)
    return tuple(shape), bool(fortran_order), np.dtype(dtype)


class NpzImageStream:
    """Sequential reader for an NPZ image member, yielding one episode at a time.

    Episodes are contiguous and in order in the source, so a single forward pass
    over the zip member is enough. This avoids materialising the whole (N, 224,
    224, 3) uint8 array, which is 3.9 GB.
    """

    def __init__(self, path: Path, key: str = 'images'):
        self._archive = zipfile.ZipFile(path)
        self._stream = self._archive.open(f'{key}.npy')
        shape, fortran_order, dtype = _read_npy_header(self._stream)
        if fortran_order:
            raise ValueError('Fortran-ordered image arrays are not supported')
        if len(shape) != 4 or shape[-1] != 3:
            raise ValueError(f'expected NHWC RGB images, got shape {shape}')
        self.shape = shape
        self.dtype = dtype
        self._frame_shape = shape[1:]
        self._values_per_frame = int(np.prod(self._frame_shape))

    def read(self, count: int) -> np.ndarray:
        byte_count = count * self._values_per_frame * self.dtype.itemsize
        payload = self._stream.read(byte_count)
        if len(payload) != byte_count:
            raise ValueError('truncated images array')
        return np.frombuffer(payload, dtype=self.dtype).reshape(
            (count, *self._frame_shape)
        )

    def close(self) -> None:
        self._stream.close()
        self._archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def episode_slices(episode_ends: np.ndarray):
    starts = np.concatenate([[0], episode_ends[:-1]])
    for ep, (s, e) in enumerate(zip(starts, episode_ends, strict=True)):
        yield ep, int(s), int(e)


def resolve_action_mode(npz, requested: str) -> str:
    """Decide whether the npz stores absolute targets or normalised deltas.

    The relative export tags itself with ``action_space``; the original DP
    layout has no such key. When the tag is present its ``action_scale`` and
    box bounds are checked against ours, because silently converting an
    already-converted action would halve the effective action scale.
    """
    space = str(npz['action_space']) if 'action_space' in npz else None
    detected = 'relative' if space == 'swm_relative' else 'absolute'

    if detected == 'relative':
        scale = float(npz['action_scale']) if 'action_scale' in npz else ACTION_SCALE
        if scale != ACTION_SCALE:
            raise ValueError(
                f'npz action_scale={scale} disagrees with ACTION_SCALE={ACTION_SCALE}'
            )
        if 'action_low' in npz and 'action_high' in npz:
            low, high = np.asarray(npz['action_low']), np.asarray(npz['action_high'])
            if not (np.all(low == -1.0) and np.all(high == 1.0)):
                raise ValueError(f'expected a [-1,1] action box, got [{low}, {high}]')
    elif space is not None:
        raise ValueError(f'unrecognised action_space {space!r}')

    if requested == 'auto':
        return detected
    if requested != detected:
        print(
            f'WARNING: --action {requested} overrides detected {detected!r} '
            f'(action_space={space!r})'
        )
    return requested


def agent_velocity(agent: np.ndarray, how: str) -> np.ndarray:
    """Finite-difference agent velocity, (L, 2)."""
    length = len(agent)
    vel = np.zeros((length, 2), dtype=np.float64)
    if length > 1:
        if how == 'forward':
            vel[:-1] = (agent[1:] - agent[:-1]) / CONTROL_DT
            vel[-1] = vel[-2]
        else:  # backward: first frame of the episode has zero velocity
            vel[1:] = (agent[1:] - agent[:-1]) / CONTROL_DT
    return vel


def main() -> None:
    args = parse_args()

    with np.load(args.npz, allow_pickle=False) as d:
        for key in ('states', 'actions', 'episode_ends'):
            if key not in d:
                raise KeyError(f'{args.npz} missing required key {key!r}')
        action_mode = resolve_action_mode(d, args.action)
        states = d['states'].astype(np.float64)        # (N,5) agent_xy, block_xy, angle
        actions_raw = d['actions'].astype(np.float64)  # (N,2) absolute target, or delta
        episode_ends = d['episode_ends'].astype(np.int64)

    n_frames, n_eps = len(states), len(episode_ends)
    if states.ndim != 2 or states.shape[1] != 5:
        raise ValueError(f'expected states shaped (N, 5), got {states.shape}')
    if len(actions_raw) != n_frames:
        raise ValueError(f'states/actions length mismatch: {n_frames} vs {len(actions_raw)}')
    if action_mode == 'relative' and np.abs(actions_raw).max() > 1.0:
        raise ValueError(
            'actions declared relative but fall outside [-1,1] '
            f'(max |a| = {np.abs(actions_raw).max()}); the npz is mislabelled'
        )
    if not n_eps or int(episode_ends[-1]) != n_frames:
        raise ValueError('episode_ends must be non-empty and end at the dataset length')

    with NpzImageStream(args.npz) as images:
        if images.shape[0] != n_frames:
            raise ValueError(
                f'images/states length mismatch: {images.shape[0]} vs {n_frames}'
            )
        if images.dtype != np.uint8:
            raise ValueError(f'expected uint8 images, got {images.dtype}')
        print(
            f'loaded {n_frames} frames / {n_eps} episodes from {args.npz} '
            f'| pixels {images.shape[1:]} | actions={action_mode} '
            f'| velocity={args.velocity}'
        )

        written = 0
        with HDF5Writer(str(args.out), mode=args.mode) as w:
            for ep, s, e in tqdm(list(episode_slices(episode_ends)), desc='episodes'):
                if args.max_episodes is not None and ep >= args.max_episodes:
                    break
                st = states[s:e]            # (L,5)
                act_raw = actions_raw[s:e]  # (L,2)
                length = len(st)
                agent = st[:, :2]

                if action_mode == 'absolute':
                    # normalised delta action, clipped to the policy's [-1,1] box.
                    action = np.clip(
                        (act_raw - agent) / ACTION_SCALE, -1.0, 1.0
                    ).astype(np.float32)
                else:
                    action = act_raw.astype(np.float32)

                vel = agent_velocity(agent, args.velocity)
                state7 = np.concatenate([st, vel], axis=1).astype(np.float32)      # (L,7)
                proprio = np.concatenate([agent, vel], axis=1).astype(np.float32)  # (L,4)

                # np.frombuffer gives a read-only view; HDF5 wants a plain array.
                pixels = np.array(images.read(length), dtype=np.uint8)

                w.write_episode(
                    {
                        'action': action,
                        'pixels': pixels,
                        'state': state7,
                        'proprio': proprio,
                        'episode_idx': np.full(length, ep, dtype=np.int64),
                        'step_idx': np.arange(length, dtype=np.int64),
                    }
                )
                written += 1

    print(f'wrote {written} episodes -> {args.out}')


if __name__ == '__main__':
    main()
