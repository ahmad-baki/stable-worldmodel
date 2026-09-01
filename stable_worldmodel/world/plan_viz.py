"""Render what the planner is thinking into the evaluation video.

An MPC eval video normally shows only the executed rollout, which hides the
part that actually explains a failure: the optimization that chose the actions.
This module splices that back in. At every replan the env is held still while
the solver's candidate cloud is animated collapsing onto the elite mean, then
the video resumes and plays the executed sequence with that plan drawn on top,
then holds again at the next replan.

It is driven by the per-iteration state a
:class:`~stable_worldmodel.solver.callbacks.PlanRecorder` captures inside the
solver and :class:`~stable_worldmodel.policy.WorldModelPolicy` stashes for each
env that replanned.

Where the env's actions are spatially interpretable (PushT: a relative agent
delta scaled by ``action_scale`` in a ``window_size`` world) each action
sequence is integrated into a path and drawn over the observation. Otherwise
the panel falls back to per-dimension action-vs-time curves, which is
env-agnostic.
"""

from typing import Any

import numpy as np


__all__ = ['PlanVideoMuxer', 'make_path_mapper']


def _agent_state(state: dict):
    """Plan-time agent position and velocity in world coords, or ``(None, None)``.

    Prefers ``proprio`` (``[x, y, vx, vy]`` for PushT) over ``pos_agent``.
    That is not a stylistic choice: in dataset-driven eval the start state is
    applied by teleporting the env *after* reset, and only the keys present in
    the dataset get pinned into ``infos``. ``proprio`` is a dataset column and
    is therefore correct on the very first plan; ``pos_agent`` is not, and
    still holds the pre-teleport reset position until the first env step.
    """
    proprio = state.get('proprio')
    if proprio is not None:
        arr = np.asarray(proprio, dtype=np.float64)
        arr = arr.reshape(-1, arr.shape[-1])[-1]
        if arr.shape[-1] >= 4:
            return arr[:2].copy(), arr[2:4].copy()
        if arr.shape[-1] >= 2:
            return arr[:2].copy(), np.zeros(2)

    pos = state.get('pos_agent')
    if pos is None:
        return None, None
    arr = np.asarray(pos, dtype=np.float64)
    arr = arr.reshape(-1, arr.shape[-1])[-1]
    vel = state.get('vel_agent')
    if vel is not None:
        vel = np.asarray(vel, dtype=np.float64)
        vel = vel.reshape(-1, vel.shape[-1])[-1][:2].copy()
    return arr[:2].copy(), (vel if vel is not None else np.zeros(2))


def make_path_mapper(env: Any, img_size: int):
    """Build ``f(state, actions) -> (T+1, 2)`` pixel-space path, or ``None``.

    Returns ``None`` when the env's actions don't describe motion in the image
    plane, which tells the muxer to fall back to action-vs-time curves.

    When the env exposes its PD gains, the path is produced by integrating the
    *same* controller the env steps with, rather than by accumulating the raw
    action deltas. PushT treats an action as a target offset
    (``pos + a * action_scale``) that the agent only partially reaches within
    one control step, so accumulating deltas compounds that shortfall and draws
    a path several times longer than the one the agent actually walks.
    """
    unwrapped = getattr(env, 'unwrapped', env)
    scale = getattr(unwrapped, 'action_scale', None)
    window = getattr(unwrapped, 'window_size', None)
    relative = bool(getattr(unwrapped, 'relative', False))

    if scale is None or window is None:
        return None

    k_p = getattr(unwrapped, 'k_p', None)
    k_v = getattr(unwrapped, 'k_v', None)
    dt = getattr(unwrapped, 'dt', None)
    control_hz = getattr(unwrapped, 'control_hz', None)
    simulate = relative and None not in (k_p, k_v, dt, control_hz)
    substeps = (
        max(1, int(round(1.0 / (dt * control_hz)))) if simulate else 0
    )

    def mapper(state: dict, actions: np.ndarray) -> np.ndarray | None:
        pos, vel = _agent_state(state)
        if pos is None:
            return None
        actions = np.asarray(actions, dtype=np.float64)[:, :2]

        if simulate:
            # Mirror PushT.step: hold `pos + a * scale` as the PD target for
            # one control step, integrating velocity then position each
            # substep. Contact with the block is not modelled, so the path
            # drifts from reality once the agent starts pushing.
            pts = [pos.copy()]
            for a in actions:
                target = pos + a * scale
                for _ in range(substeps):
                    acc = k_p * (target - pos) + k_v * (-vel)
                    vel = vel + acc * dt
                    pos = pos + vel * dt
                pts.append(pos.copy())
            pts = np.asarray(pts)
        elif relative:
            pts = np.concatenate(
                [pos[None], pos + np.cumsum(actions * scale, axis=0)], axis=0
            )
        else:
            pts = np.concatenate([pos[None], actions * scale], axis=0)

        return pts * (img_size / float(window))

    return mapper


def _last(frame: np.ndarray) -> np.ndarray:
    """Drop a leading time axis from a stacked observation, as a copy.

    The copy is not optional. ``EnvPool`` writes each env's info into
    pre-allocated stacked arrays in-place (``_write_env_info``), so a view into
    ``world.infos`` is only valid until the next step -- and the goal image is
    retained across a whole receding-horizon execution segment.
    """
    frame = np.asarray(frame)
    return np.array(frame[-1] if frame.ndim > 3 else frame, copy=True)


class PlanVideoMuxer:
    """Splice CEM optimization panels into a per-env eval frame stream.

    Args:
        env_ids: Env indices to render panels for. Rendering is the expensive
            part of this, so keep it to a handful.
        path_mapper: Result of :func:`make_path_mapper`, or ``None`` to use
            the env-agnostic action-vs-time fallback.
        cell: Side length in pixels of each of the three panels.
        iter_hold: Frames to hold each recorded CEM iteration.
        final_hold: Frames to hold the converged plan before execution.
        elite_cap: Max elite paths drawn per iteration.
    """

    def __init__(
        self,
        env_ids,
        path_mapper=None,
        cell: int = 448,
        iter_hold: int = 1,
        final_hold: int = 10,
        elite_cap: int = 24,
    ) -> None:
        self.env_ids = set(int(i) for i in env_ids)
        self.path_mapper = path_mapper
        self.cell = int(cell)
        self.iter_hold = max(1, int(iter_hold))
        self.final_hold = max(0, int(final_hold))
        self.elite_cap = int(elite_cap)

        self._pending: dict[int, list[np.ndarray]] = {}
        self._active: dict[int, dict] = {}
        self._exec_step: dict[int, int] = {}
        self._fig = None

    # -- frame stream ------------------------------------------------------

    def consume(self, world: Any) -> None:
        """Pull the latest plans off the policy and render their panels.

        Must be called *after* the policy planned and *before* the env steps,
        so the panels are drawn on the observation the planner actually saw.
        """
        policy = world.policy
        if not hasattr(policy, 'pop_plan_records'):
            return
        records = policy.pop_plan_records()
        if not records:
            return

        pixels = world.infos.get('pixels')
        goals = world.infos.get('goal')

        for env_i, record in records.items():
            if env_i not in self.env_ids:
                continue
            obs = _last(pixels[env_i]) if pixels is not None else None
            goal = _last(goals[env_i]) if goals is not None else None
            self._pending[env_i] = self._render_plan(obs, goal, record)
            self._active[env_i] = record
            self._exec_step[env_i] = 0

    def frames_for(self, env_i: int, frame: np.ndarray) -> list[np.ndarray]:
        """Frames to emit for ``env_i`` this step: any panels, then the rollout."""
        if env_i not in self.env_ids:
            return [frame]

        out = self._pending.pop(env_i, [])
        record = self._active.get(env_i)
        if record is None:
            return out + [self._compose(frame, None, None, None, '')]

        step = self._exec_step.get(env_i, 0)
        self._exec_step[env_i] = step + 1
        goal = record.get('_goal')
        out.append(
            self._compose(
                frame,
                goal,
                self._exec_overlay(record, step),
                record['_costs'],
                f'executing step {step + 1}/{len(record["plan"])}',
                cost_marker=None,
            )
        )
        return out

    # -- rendering ---------------------------------------------------------

    def _render_plan(
        self, obs: np.ndarray | None, goal: np.ndarray | None, record: dict
    ) -> list[np.ndarray]:
        """Animate the recorded CEM iterations, then hold the chosen plan."""
        iters = record['iters']
        costs = {
            'step': [it['step'] for it in iters],
            'elite_min': [it['elite_cost_min'] for it in iters],
            'elite_mean': [it['elite_cost_mean'] for it in iters],
            'pop_mean': [it['pop_cost_mean'] for it in iters],
        }
        record['_costs'] = costs
        record['_goal'] = goal

        state = record['state']
        frames: list[np.ndarray] = []
        n = len(iters)

        for k, it in enumerate(iters):
            overlay = {
                'elites': [
                    self._to_path(state, a)
                    for a in it['elites'][: self.elite_cap]
                ],
                'mean': self._to_path(state, it['mean']),
                'label': 'distribution mean',
            }
            title = f'CEM iteration {it["step"] + 1}  ({k + 1}/{n} recorded)'
            frame = self._compose(obs, goal, overlay, costs, title, k)
            frames.extend([frame] * self.iter_hold)

        if self.final_hold and iters:
            overlay = {
                'elites': [],
                'mean': self._to_path(state, record['plan']),
                'label': 'chosen plan',
            }
            frame = self._compose(
                obs, goal, overlay, costs, 'chosen plan -> executing', n - 1
            )
            frames.extend([frame] * self.final_hold)

        return frames

    def _exec_overlay(self, record: dict, step: int) -> dict:
        """The executed plan, with the part already played marked off.

        The path is fixed for the whole segment -- it is the plan as decided,
        anchored at the agent's position when it was decided -- so it is
        simulated once and cached rather than re-integrated per frame.
        """
        path = record.get('_exec_path')
        if path is None:
            path = self._to_path(record['state'], record['plan'])
            record['_exec_path'] = path
        return {'elites': [], 'mean': path, 'label': 'executing', 'at': step + 1}

    def _to_path(self, state: dict, actions: np.ndarray):
        """Action sequence -> drawable, as a pixel path or as raw actions."""
        if self.path_mapper is not None:
            path = self.path_mapper(state, actions)
            if path is not None:
                return path
        return np.asarray(actions, dtype=np.float64)

    def _compose(
        self,
        obs: np.ndarray | None,
        goal: np.ndarray | None,
        overlay: dict | None,
        costs: dict | None,
        title: str,
        cost_marker: int | None = None,
    ) -> np.ndarray:
        """Draw one video frame: observation+plan | goal | cost history."""
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        spatial = self.path_mapper is not None
        dpi = 112
        if self._fig is None:
            self._fig = plt.figure(
                figsize=(3 * self.cell / dpi, self.cell / dpi), dpi=dpi
            )
        fig = self._fig
        fig.clf()
        fig.patch.set_facecolor('white')
        axes = fig.subplots(1, 3)

        # -- panel 1: observation with the plan drawn on top
        ax = axes[0]
        size = obs.shape[0] if obs is not None else 1
        if obs is not None and (spatial or overlay is None):
            ax.imshow(obs)

        if overlay is not None and spatial:
            elites = [e for e in overlay['elites'] if e is not None]
            if elites:
                ax.add_collection(
                    LineCollection(
                        elites, colors='#f2c14e', linewidths=0.8, alpha=0.45
                    )
                )
            if elites:
                ax.plot([], [], '-', color='#f2c14e', lw=1.2, label='elites')
            mean = overlay['mean']
            if mean is not None:
                ax.plot(
                    mean[:, 0],
                    mean[:, 1],
                    '-',
                    color='#e63946',
                    linewidth=2.2,
                    label=overlay['label'],
                )
                ax.plot(
                    mean[0, 0],
                    mean[0, 1],
                    'o',
                    color='#1d3557',
                    ms=6,
                    label='plan start',
                )
                at = overlay.get('at')
                if at is not None and at < len(mean):
                    ax.plot(
                        mean[at, 0],
                        mean[at, 1],
                        'o',
                        color='#2a9d8f',
                        ms=8,
                        label='now',
                    )
                else:
                    ax.plot(
                        mean[-1, 0],
                        mean[-1, 1],
                        '*',
                        color='#e63946',
                        ms=11,
                        label='plan end',
                    )
            ax.set_xlim(0, size)
            ax.set_ylim(size, 0)
            ax.legend(fontsize=6, loc='upper right', framealpha=0.75)
        elif overlay is not None:
            # env-agnostic fallback: action value vs plan step, per dimension
            mean = np.asarray(overlay['mean'])
            for d in range(mean.shape[-1]):
                for e in overlay['elites']:
                    ax.plot(
                        np.asarray(e)[:, d],
                        color=f'C{d}',
                        alpha=0.15,
                        linewidth=0.7,
                    )
                ax.plot(
                    mean[:, d], color=f'C{d}', linewidth=2.0, label=f'a[{d}]'
                )
            ax.set_xlabel('plan step')
            ax.legend(fontsize=6, loc='upper right')
        if spatial or overlay is None:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=9)

        # -- panel 2: the goal the planner is optimizing toward
        ax = axes[1]
        if goal is not None:
            ax.imshow(goal)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('goal', fontsize=9)

        # -- panel 3: how the cost fell across CEM iterations
        ax = axes[2]
        if costs is not None and costs['step']:
            x = costs['step']
            ax.plot(x, costs['pop_mean'], color='#adb5bd', label='pop mean')
            ax.plot(x, costs['elite_mean'], color='#f2c14e', label='elite mean')
            ax.plot(x, costs['elite_min'], color='#e63946', label='elite best')
            if cost_marker is not None and cost_marker < len(x):
                ax.axvline(x[cost_marker], color='#1d3557', lw=1.0, ls='--')
            ax.legend(fontsize=6)
        ax.set_xlabel('CEM iteration', fontsize=8)
        ax.set_ylabel('cost', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title('optimization', fontsize=9)

        fig.tight_layout(pad=0.6)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        return buf[..., :3].copy()

    def close(self) -> None:
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
