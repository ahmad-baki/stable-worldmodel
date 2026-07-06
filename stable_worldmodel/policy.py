from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from collections.abc import Callable

import numpy as np
import torch
from loguru import logger as logging
from torchvision import tv_tensors

import stable_worldmodel as swm
from stable_worldmodel.solver import Solver


@dataclass(frozen=True)
class PlanConfig:
    """Configuration for the MPC planning loop.

    Attributes:
        horizon: Planning horizon in number of steps.
        receding_horizon: Number of steps to execute before re-planning.
        history_len: Number of past observations to consider.
        action_block: Number of times each action is repeated (frameskip).
        warm_start: Whether to use the previous plan to initialize the next one.
    """

    horizon: int
    receding_horizon: int
    history_len: int = 1
    action_block: int = 1
    warm_start: bool = True

    @property
    def plan_len(self) -> int:
        """Total plan length in environment steps."""
        return self.horizon * self.action_block


class Transformable(Protocol):
    """Protocol for reversible data transformations (e.g., normalizers, scalers)."""

    def transform(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Apply preprocessing to input data.

        Args:
            x: Input data as a numpy array.

        Returns:
            Preprocessed data as a numpy array.
        """
        ...

    def inverse_transform(
        self, x: np.ndarray
    ) -> np.ndarray:  # pragma: no cover
        """Reverse the preprocessing transformation.

        Args:
            x: Preprocessed data as a numpy array.

        Returns:
            Original data as a numpy array.
        """
        ...


class Actionable(Protocol):
    """Protocol for model action computation."""

    def get_action(info) -> torch.Tensor:  # pragma: no cover
        """Compute action from observation and goal"""
        ...


class BasePolicy:
    """Base class for agent policies.

    Attributes:
        env: The environment the policy is associated with.
        type: A string identifier for the policy type.
    """

    env: Any
    type: str

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the base policy.

        Args:
            **kwargs: Additional configuration parameters.
        """
        self.env = None
        self.type = 'base'
        for arg, value in kwargs.items():
            setattr(self, arg, value)

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        """Get action from the policy given the observation.

        Args:
            obs: The current observation from the environment.
            **kwargs: Additional parameters for action selection.

        Returns:
            Selected action as a numpy array.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    def set_env(self, env: Any) -> None:
        """Associate this policy with an environment.

        Args:
            env: The environment to associate.
        """
        self.env = env

    def _prepare_info(self, info_dict: dict) -> dict[str, torch.Tensor]:
        """Pre-process and transform observations.

        Applies preprocessing (via `self.process`) and transformations (via `self.transform`)
        to observation data. Used by subclasses like FeedForwardPolicy and WorldModelPolicy.
        Returns a new dict; the input is not mutated.

        Args:
            info_dict: Raw observation dictionary from the environment.

        Returns:
            A dictionary of processed tensors.

        Raises:
            ValueError: If an expected numpy array is missing for processing.
        """
        out = {}
        for k, v in info_dict.items():
            is_numpy = isinstance(v, (np.ndarray | np.generic))

            if hasattr(self, 'process') and k in self.process:
                if not is_numpy:
                    raise ValueError(
                        f"Expected numpy array for key '{k}' in process, got {type(v)}"
                    )

                # flatten extra dimensions if needed
                shape = v.shape
                if len(shape) > 2:
                    v = v.reshape(-1, *shape[2:])

                # process and reshape back
                v = self.process[k].transform(v)
                v = v.reshape(shape)

            # collapse env and time dimensions for transform (e, t, ...) -> (e * t, ...)
            # then restore after transform
            if hasattr(self, 'transform') and k in self.transform:
                shape = None
                if is_numpy or torch.is_tensor(v):
                    if v.ndim > 2:
                        shape = v.shape
                        v = v.reshape(-1, *shape[2:])
                if k.startswith('pixels') or k.startswith('goal'):
                    # permute channel first for transform
                    if is_numpy:
                        v = np.transpose(v, (0, 3, 1, 2))
                    else:
                        v = v.permute(0, 3, 1, 2)
                v = torch.stack(
                    [self.transform[k](tv_tensors.Image(x)) for x in v]
                )
                is_numpy = isinstance(v, (np.ndarray | np.generic))

                if shape is not None:
                    v = v.reshape(*shape[:2], *v.shape[1:])

            if is_numpy and v.dtype.kind not in 'USO':
                v = torch.from_numpy(v)

            out[k] = v

        return out


class RandomPolicy(BasePolicy):
    """Policy that samples random actions from the action space."""

    def __init__(self, seed: int | None = None, **kwargs: Any) -> None:
        """Initialize the random policy.

        Args:
            seed: Optional random seed for the action space.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.type = 'random'
        self.seed = seed

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        """Get a random action from the environment's action space.

        Args:
            obs: The current observation (ignored).
            **kwargs: Additional parameters (ignored).

        Returns:
            A randomly sampled action.
        """
        return self.env.action_space.sample()

    def set_seed(self, seed: int) -> None:
        """Set the random seed for action sampling.

        Args:
            seed: The seed value.
        """
        if self.env is not None:
            self.env.action_space.seed(seed)


class NoOpPolicy(BasePolicy):
    """Policy that always emits a zero action (a "do nothing" baseline).

    For relative-action envs (e.g. PushT) a zero action targets the current
    position, so the agent stays put. Useful for measuring the success rate
    attributable purely to start states that already satisfy the goal.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.type = 'noop'

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        """Return an all-zero action shaped like the action space."""
        return np.zeros_like(self.env.action_space.sample())


class ExpertPolicy(BasePolicy):
    """Policy using expert demonstrations or heuristics."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the expert policy.

        Args:
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.type = 'expert'

    def get_action(
        self, obs: Any, goal_obs: Any, **kwargs: Any
    ) -> np.ndarray | None:
        """Get action from the expert policy.

        Args:
            obs: The current observation.
            goal_obs: The goal observation.
            **kwargs: Additional parameters.

        Returns:
            The expert action, or None if not available.
        """
        # Implement expert policy logic here
        pass


class FeedForwardPolicy(BasePolicy):
    """Feed-Forward Policy using a neural network model.

    Actions are computed via a single forward pass through the model.
    Useful for imitation learning policies like Goal-Conditioned Behavioral Cloning (GCBC).

    Attributes:
        model: Neural network model implementing the Actionable protocol.
        process: Dictionary of data preprocessors for specific keys.
        transform: Dictionary of tensor transformations (e.g., image transforms).
    """

    def __init__(
        self,
        model: Actionable,
        process: dict[str, Transformable] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]]
        | None = None,
        history_len: int | None = None,
        history_keys: tuple[str, ...] = ('pixels', 'proprio'),
        **kwargs: Any,
    ) -> None:
        """Initialize the feed-forward policy.

        Args:
            model: Neural network model with a `get_action` method.
            process: Dictionary of data preprocessors for specific keys.
            transform: Dictionary of tensor transformations (e.g., image transforms).
            history_len: Number of past observations to stack along the time
                axis before the forward pass. Defaults to the model's own
                ``history_size`` attribute (``1`` disables stacking, matching a
                stateless single-frame policy).
            history_keys: Observation keys to accumulate into the temporal
                window (goal keys are intentionally excluded).
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.type = 'feed_forward'
        self.model = model.eval()
        self.process = process or {}
        self.transform = transform or {}
        # GCBC-style models are trained on a fixed-length observation history
        # (``history_size``), but the env only ever emits the current frame
        # (T=1). Keep a per-env rolling buffer and stack the last
        # ``history_len`` observations so the model runs in the regime it was
        # trained on. ``history_len == 1`` preserves the old single-frame
        # behavior for stateless models.
        if history_len is None:
            history_len = int(getattr(model, 'history_size', 1) or 1)
        self.history_len = max(1, int(history_len))
        self.history_keys = tuple(history_keys)
        self._hist: dict[str, list[deque]] = {}
        # Per-env buffer of pending sub-actions. Models trained with action
        # chunking (frameskip>1) output ``chunk_size * action_dim`` values; we
        # split each chunk into ``chunk_size`` env-steps and only re-query the
        # model when an env's buffer drains. Replanning every ``chunk_size``
        # steps also spaces the stacked history frames ``chunk_size`` steps
        # apart — matching the frameskip-subsampled window the model saw in
        # training. chunk_size == 1 reproduces the old replan-every-step policy.
        self._action_buffer: list[deque] | None = None
        self._action_dim: int | None = None

    def set_env(self, env: Any) -> None:
        """Attach the env and (re)initialize per-env action/history buffers."""
        super().set_env(env)
        n = getattr(env, 'num_envs', 1)
        self._action_buffer = [deque() for _ in range(n)]
        self._hist = {}
        space = getattr(env, 'single_action_space', None)
        if space is None:
            space = getattr(env, 'action_space', None)
        self._action_dim = (
            int(np.asarray(space.shape)[-1]) if space is not None else None
        )

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        """Get an action, unrolling chunked model outputs across env steps.

        The model returns ``chunk_size * action_dim`` values per env
        (``chunk_size == 1`` for non-chunked models). Each chunk is split into
        ``chunk_size`` sub-actions, buffered per env, and emitted one per step;
        the model is only re-queried when an env's buffer empties (or the env
        just reset), which keeps the stacked history frames ``chunk_size`` steps
        apart to match the frameskip window seen in training.

        Args:
            info_dict: Current state info; must contain 'goal'. ``_needs_flush``
                (per-env reset flags) is consumed if present.

        Returns:
            One action per env, shape ``(num_envs, action_dim)``.
        """
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert 'goal' in info_dict, "'goal' must be provided in info_dict"

        needs_flush = info_dict.pop('_needs_flush', None)
        n = getattr(self.env, 'num_envs', 1)
        if self._action_buffer is None or len(self._action_buffer) != n:
            self._action_buffer = [deque() for _ in range(n)]
            self._hist = {}

        # A reset env abandons its remaining chunk (its history is re-primed on
        # the next replan below).
        if needs_flush is not None:
            for i in range(n):
                if bool(needs_flush[i]):
                    self._action_buffer[i].clear()

        replan_idx = [i for i in range(n) if len(self._action_buffer[i]) == 0]
        if replan_idx:
            self._replan(info_dict, replan_idx, needs_flush)

        # Emit one buffered sub-action per env this step.
        return np.stack([self._action_buffer[i].popleft() for i in range(n)])

    def _replan(
        self,
        info_dict: dict,
        replan_idx: list[int],
        needs_flush: np.ndarray | None,
    ) -> None:
        """Query the model for the envs in ``replan_idx`` and refill their
        action buffers with the un-normalized chunk of sub-actions."""
        info = self._history_info(info_dict, replan_idx, needs_flush)
        info = self._prepare_info(info)
        if 'goal' in info:
            info['goal_pixels'] = info['goal']
        device = next(self.model.parameters()).device
        for k, v in info.items():
            if torch.is_tensor(v):
                info[k] = v.to(device)

        with torch.no_grad():
            out = self.model.get_action(info)
        if torch.is_tensor(out):
            out = out.cpu().detach().numpy()
        out = np.asarray(out)  # (m, chunk_size * action_dim)

        m = out.shape[0]
        adim = self._action_dim or out.shape[-1]
        chunk_size = max(1, out.shape[-1] // adim)
        # Un-normalize in action_dim-sized blocks, then split into sub-actions.
        flat = out.reshape(m * chunk_size, adim)
        if 'action' in self.process:
            flat = self.process['action'].inverse_transform(flat)
        chunk = np.asarray(flat).reshape(m, chunk_size, adim)

        for row, i in enumerate(replan_idx):
            for a in chunk[row]:
                self._action_buffer[i].append(np.asarray(a))

    def _history_info(
        self,
        info_dict: dict,
        replan_idx: list[int],
        needs_flush: np.ndarray | None,
    ) -> dict:
        """Slice ``info_dict`` to ``replan_idx`` and replace history keys with
        their stacked ``(len(replan_idx), history_len, ...)`` windows.

        History is appended only for the replanning envs, so an env mid-chunk
        keeps its window untouched; because replans occur every ``chunk_size``
        steps, its frames stay ``chunk_size`` steps apart. On the first step of
        an episode (empty buffer or ``needs_flush[i]``) the window is primed by
        repeating the current frame.
        """
        n = getattr(self.env, 'num_envs', 1)
        out: dict = {}
        for k, v in info_dict.items():
            if k in self.history_keys:
                continue
            if isinstance(v, np.ndarray):
                out[k] = v[replan_idx]
            elif torch.is_tensor(v):
                out[k] = v[torch.as_tensor(replan_idx)]
            elif isinstance(v, list):
                out[k] = [v[i] for i in replan_idx]
            else:
                out[k] = v

        for key in self.history_keys:
            if key not in info_dict:
                continue
            v = np.asarray(info_dict[key])
            # drop the env-provided singleton time dim: (n, 1, ...) -> (n, ...)
            cur = v[:, 0] if v.ndim >= 2 and v.shape[1] == 1 else v
            buffers = self._hist.get(key)
            if buffers is None or len(buffers) != n:
                buffers = [deque(maxlen=self.history_len) for _ in range(n)]
                self._hist[key] = buffers
            frames = []
            for i in replan_idx:
                buf = buffers[i]
                if needs_flush is not None and bool(needs_flush[i]):
                    buf.clear()
                if len(buf) == 0:
                    for _ in range(self.history_len):
                        buf.append(cur[i])
                else:
                    buf.append(cur[i])
                frames.append(np.stack(list(buf), axis=0))  # (T, ...)
            out[key] = np.stack(frames, axis=0)  # (m, T, ...)
        return out


class WorldModelPolicy(BasePolicy):
    """Policy using a world model and planning solver for action selection."""

    def __init__(
        self,
        solver: Solver,
        config: PlanConfig,
        process: dict[str, Transformable] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]]
        | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the world model policy.

        Args:
            solver: The planning solver to use.
            config: MPC planning configuration.
            process: Dictionary of data preprocessors for specific keys.
            transform: Dictionary of tensor transformations (e.g., image transforms).
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)

        self.type = 'world_model'
        self.cfg = config
        self.solver = solver
        self.process = process or {}
        self.transform = transform or {}
        self._action_buffer: list[deque[torch.Tensor]] | None = None
        self._next_init: torch.Tensor | None = None

    @property
    def flatten_receding_horizon(self) -> int:
        """Receding horizon in environment steps (with frameskip)."""
        return self.cfg.receding_horizon * self.cfg.action_block

    def set_env(self, env: Any) -> None:
        """Configure the policy and solver for the given environment.

        Args:
            env: The environment to associate with the policy.
        """
        self.env = env
        n_envs = getattr(env, 'num_envs', 1)
        self.solver.configure(
            action_space=env.action_space, n_envs=n_envs, config=self.cfg
        )
        self._action_buffer = [
            deque(maxlen=self.flatten_receding_horizon) for _ in range(n_envs)
        ]

        assert isinstance(self.solver, Solver), (
            'Solver must implement the Solver protocol'
        )

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        """Get action via planning with the world model.

        Args:
            info_dict: Current state information from the environment.
            **kwargs: Additional parameters for planning.

        Returns:
            The selected action(s) as a numpy array.
        """
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert 'pixels' in info_dict, "'pixels' must be provided in info_dict"
        assert 'goal' in info_dict, "'goal' must be provided in info_dict"

        info_dict = self._prepare_info(info_dict)
        n_envs = self.env.num_envs

        needs_flush = info_dict.pop('_needs_flush', None)
        if needs_flush is not None:
            for i in range(n_envs):
                if needs_flush[i]:
                    self._action_buffer[i].clear()
                    if self._next_init is not None:
                        self._next_init[i] = 0

        terminated = info_dict.get('terminated')
        dead = (
            np.asarray(terminated, dtype=bool)
            if terminated is not None
            else np.zeros(n_envs, dtype=bool)
        )

        replan_idx = [
            i
            for i in range(n_envs)
            if len(self._action_buffer[i]) == 0 and not dead[i]
        ]

        if replan_idx:
            idx_tensor = torch.as_tensor(replan_idx, dtype=torch.long)
            sliced = {}
            for k, v in info_dict.items():
                if torch.is_tensor(v):
                    sliced[k] = v[idx_tensor]
                elif isinstance(v, np.ndarray):
                    sliced[k] = v[replan_idx]
                elif isinstance(v, list):
                    sliced[k] = [v[i] for i in replan_idx]
                else:
                    sliced[k] = v

            sliced_init = (
                self._next_init[idx_tensor]
                if self._next_init is not None
                else None
            )

            outputs = self.solver(sliced, init_action=sliced_init)

            actions = outputs['actions']
            keep_horizon = self.cfg.receding_horizon
            plan = actions[:, :keep_horizon]
            rest = actions[:, keep_horizon:]

            if self.cfg.warm_start and rest.shape[1] > 0:
                if self._next_init is None:
                    self._next_init = torch.zeros(
                        n_envs, rest.shape[1], rest.shape[2], dtype=rest.dtype
                    )
                self._next_init[idx_tensor] = rest
            elif not self.cfg.warm_start:
                self._next_init = None

            plan = plan.reshape(
                len(replan_idx), self.flatten_receding_horizon, -1
            )

            for row, env_i in enumerate(replan_idx):
                self._action_buffer[env_i].extend(plan[row])

        action_dim = self.env.single_action_space.shape[-1]
        action = torch.full((n_envs, action_dim), float('nan'))
        for i in range(n_envs):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()

        action = action.reshape(*self.env.action_space.shape)
        action = action.float().numpy()

        if 'action' in self.process:
            action = self.process['action'].inverse_transform(action)

        return action


def _load_model_with_attribute(run_name, attribute_name, cache_dir=None):
    """Helper function to load a model checkpoint and find a module with the specified attribute.

    Args:
        run_name: Path or name of the model run
        attribute_name: Name of the attribute to look for in the module (e.g., 'get_action', 'get_cost')
        cache_dir: Optional cache directory path

    Returns:
        The module with the specified attribute

    Raises:
        RuntimeError: If no module with the specified attribute is found
    """
    if Path(run_name).exists():
        run_path = Path(run_name)
    else:
        run_path = Path(
            cache_dir
            or swm.data.utils.get_cache_dir(sub_folder='checkpoints'),
            run_name,
        )

    if run_path.is_dir():
        ckpt_files = list(run_path.glob('*_object.ckpt'))
        ckpt_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        path = ckpt_files[0]
        logging.info(f'Loading model from checkpoint: {path}')
    elif run_path.is_file():
        # Direct path to an *_object.ckpt file (e.g. resolved from a wandb
        # reference artifact). Use it as-is instead of appending the suffix.
        path = run_path
        logging.info(f'Loading model from checkpoint: {path}')
    else:
        path = Path(f'{run_path}_object.ckpt')
        assert path.exists(), (
            f'Checkpoint path does not exist: {path}. Launch pretraining first.'
        )

    spt_module = torch.load(path, weights_only=False, map_location='cpu')

    def scan_module(module):
        if hasattr(module, attribute_name):
            if isinstance(module, torch.nn.Module):
                module = module.eval()
            return module
        for child in module.children():
            result = scan_module(child)
            if result is not None:
                return result
        return None

    result = scan_module(spt_module)
    if result is not None:
        return result

    raise RuntimeError(
        f"No module with '{attribute_name}' found in the loaded world model."
    )


def AutoActionableModel(
    run_name: str, cache_dir: str | Path | None = None
) -> torch.nn.Module:
    """Load a model checkpoint and return the module with a `get_action` method.

    Automatically scans the checkpoint for a module implementing the Actionable
    protocol (i.e., has a `get_action` method).

    Args:
        run_name: Path or name of the model run/checkpoint.
        cache_dir: Optional cache directory path. Defaults to STABLEWM_HOME.

    Returns:
        The module with a `get_action` method, set to eval mode.

    Raises:
        RuntimeError: If no module with `get_action` is found in the checkpoint.
    """
    return _load_model_with_attribute(run_name, 'get_action', cache_dir)


def AutoCostModel(
    run_name: str, cache_dir: str | Path | None = None
) -> torch.nn.Module:
    """Load a model checkpoint and return the module with a `get_cost` method.

    Automatically scans the checkpoint for a module implementing a cost function
    (i.e., has a `get_cost` method) for use with planning solvers.

    Args:
        run_name: Path or name of the model run/checkpoint.
        cache_dir: Optional cache directory path. Defaults to STABLEWM_HOME.

    Returns:
        The module with a `get_cost` method, set to eval mode.

    Raises:
        RuntimeError: If no module with `get_cost` is found in the checkpoint.
    """
    return _load_model_with_attribute(run_name, 'get_cost', cache_dir)


# Alias for backward compatibility and type hinting
Policy = BasePolicy
