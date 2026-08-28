#!/usr/bin/env python3
"""Convert the Diffusion-Policy PushT demo set into our HDF5 schema.

Source: ``pusht_cchi_v7_replay.zarr`` (206 human demos, 25650 steps), exported
to a plain ``.npz`` with keys ``states(N,5)``, ``actions(N,2)``,
``images(N,96,96,3) uint8`` and ``episode_ends(206,)``.

Two conventions differ from our ``pusht_expert_train`` set and are reconciled
here:

* **Images** are re-rendered from state through ``swm/PushT-v1`` at 224px
  (native, in-distribution for the LeWM/DINO encoder) rather than upsampled
  from the shipped 96px frames. The green goal-T is pinned to the fixed
  DP target pose ``[256, 256, pi/4]`` (matches the expert set's visuals).
* **Actions** in DP are absolute target positions in the 512-px world; our env
  expects a normalised delta ``(target - agent) / action_scale`` with
  ``action_scale=100``, clipped to [-1, 1] (matches our expert action range).

State is widened 5 -> 7 by appending a finite-difference agent velocity
(``[agent_xy, block_xy, angle, agent_vx, agent_vy]``); ``proprio`` is
``[agent_xy, agent_vx, agent_vy]``. Velocity is approximate and unused by the
LeWM GCBC recipe (``use_proprio_encoder: false``) -- it only keeps the schema
identical to the expert set. Output columns match
``pusht_expert_train.h5`` exactly: action, pixels, state, proprio,
episode_idx, step_idx (+ ep_len/ep_offset written by HDF5Writer).

Usage::

    MUJOCO_GL=egl SDL_VIDEODRIVER=dummy python scripts/data/convert_dp_pusht.py \
        --npz  <datasets>/pusht_dp_raw/pusht_dp_source.npz \
        --out  <datasets>/pusht_dp.h5
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
from tqdm import tqdm

import stable_worldmodel  # noqa: F401  registers swm/ envs
from stable_worldmodel.data.formats.hdf5 import HDF5Writer


ACTION_SCALE = 100.0
GOAL_POSE = np.array([256.0, 256.0, np.pi / 4], dtype=np.float64)
CONTROL_DT = 0.1  # DP demos are logged at 10 Hz; used only for velocity finite-diff.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--npz', required=True, help='DP source npz (states/actions/images/episode_ends).')
    p.add_argument('--out', required=True, help='Destination .h5 path.')
    p.add_argument('--resolution', type=int, default=224)
    p.add_argument('--mode', choices=('overwrite', 'error', 'append'), default='overwrite')
    p.add_argument('--max-episodes', type=int, default=None, help='Debug cap on #episodes.')
    return p.parse_args()


def episode_slices(episode_ends: np.ndarray):
    starts = np.concatenate([[0], episode_ends[:-1]])
    for ep, (s, e) in enumerate(zip(starts, episode_ends, strict=True)):
        yield ep, int(s), int(e)


def main() -> None:
    args = parse_args()
    d = np.load(args.npz)
    states = d['states'].astype(np.float64)          # (N,5) agent_xy, block_xy, angle
    actions_abs = d['actions'].astype(np.float64)    # (N,2) absolute target in 512 space
    episode_ends = d['episode_ends'].astype(np.int64)
    n_frames, n_eps = len(states), len(episode_ends)
    print(f'loaded {n_frames} frames / {n_eps} episodes from {args.npz}')

    goal_state = np.array(
        [0.0, 0.0, GOAL_POSE[0], GOAL_POSE[1], GOAL_POSE[2], 0.0, 0.0], dtype=np.float64
    )
    env = gym.make(
        'swm/PushT-v1', resolution=args.resolution, with_target=True, render_mode='rgb_array'
    )
    env.reset(options={'state': np.concatenate([states[0], [0, 0]]), 'goal_state': goal_state})
    u = env.unwrapped
    u._set_goal_state(goal_state)

    written = 0
    with HDF5Writer(args.out, mode=args.mode) as w:
        for ep, s, e in tqdm(list(episode_slices(episode_ends)), desc='episodes'):
            if args.max_episodes is not None and ep >= args.max_episodes:
                break
            st = states[s:e]                 # (L,5)
            act_abs = actions_abs[s:e]       # (L,2)
            L = len(st)
            agent = st[:, :2]

            # normalised delta action, clipped to the policy's [-1,1] box.
            action = np.clip((act_abs - agent) / ACTION_SCALE, -1.0, 1.0).astype(np.float32)

            # agent velocity by forward finite difference (last step repeats).
            vel = np.zeros((L, 2), dtype=np.float64)
            if L > 1:
                vel[:-1] = (agent[1:] - agent[:-1]) / CONTROL_DT
                vel[-1] = vel[-2]

            state7 = np.concatenate([st, vel], axis=1).astype(np.float32)   # (L,7)
            proprio = np.concatenate([agent, vel], axis=1).astype(np.float32)  # (L,4)

            pixels = np.empty((L, args.resolution, args.resolution, 3), dtype=np.uint8)
            for j in range(L):
                u._set_state(np.concatenate([st[j], vel[j]]))
                u.goal_pose = GOAL_POSE  # keep the green T pinned
                pixels[j] = np.asarray(u.render(), dtype=np.uint8)

            ep_data = {
                'action': action,
                'pixels': pixels,
                'state': state7,
                'proprio': proprio,
                'episode_idx': np.full(L, ep, dtype=np.int64),
                'step_idx': np.arange(L, dtype=np.int64),
            }
            w.write_episode(ep_data)
            written += 1

    env.close()
    print(f'wrote {written} episodes -> {args.out}')


if __name__ == '__main__':
    main()
