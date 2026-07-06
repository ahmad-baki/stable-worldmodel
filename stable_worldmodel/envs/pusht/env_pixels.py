import numpy as np
from gymnasium import spaces

from .env import PushT


class PushTPixels(PushT):
    """PushT variant that exposes rendered pixels in the Gym observation.

    The base PushT env keeps rendered observations in ``info`` for the
    stable-worldmodel evaluation stack. SB3 only passes ``obs`` to policies, so
    this wrapper-style subclass moves the current render and rendered goal into
    the observation dict while keeping proprioception available.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        proprio_space = self.observation_space['proprio']
        proprio_space = spaces.Box(
            low=proprio_space.low.astype(np.float32),
            high=proprio_space.high.astype(np.float32),
            shape=proprio_space.shape,
            dtype=np.float32,
        )
        image_shape = (self.render_size, self.render_size, 3)
        image_space = spaces.Box(
            low=0,
            high=255,
            shape=image_shape,
            dtype=np.uint8,
        )
        self.observation_space = spaces.Dict(
            {
                'pixels': image_space,
                'goal': image_space,
                'proprio': proprio_space,
                'goal_proprio': proprio_space,
            }
        )

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        return self._pixel_observation(observation, info), info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return (
            self._pixel_observation(observation, info),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def _pixel_observation(self, observation, info):
        return {
            'pixels': np.asarray(self.render(), dtype=np.uint8),
            'goal': np.asarray(self._goal, dtype=np.uint8),
            'proprio': np.asarray(observation['proprio'], dtype=np.float32),
            'goal_proprio': np.asarray(info['goal_proprio'], dtype=np.float32),
        }
