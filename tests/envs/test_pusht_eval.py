"""Tests for the PushT success / orientation condition (``eval_state``)."""

import os

import numpy as np
import pytest


# Use a headless SDL driver so importing/constructing the env never needs a
# display (eval_state itself does not render, but this keeps CI robust).
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from stable_worldmodel.envs.pusht.env import PushT


# state / goal layout: [agent_x, agent_y, block_x, block_y, block_angle,
#                       agent_vx, agent_vy]
POS_THRESHOLD = 20  # eval_state: pos_diff < 20
ANGLE_THRESHOLD = np.pi / 9  # eval_state: angle_diff < pi/9


@pytest.fixture(scope='module')
def env():
    """A PushT instance; __init__ does not start pygame so this is cheap."""
    return PushT()


def make_state(block_xy=(256, 256), angle=0.0, agent_xy=(256, 400)):
    return np.array(
        [agent_xy[0], agent_xy[1], block_xy[0], block_xy[1], angle, 0.0, 0.0],
        dtype=np.float64,
    )


def test_matching_pose_is_success(env):
    """Same block position and angle -> success."""
    goal = make_state(block_xy=(256, 256), angle=0.5)
    cur = make_state(block_xy=(256, 256), angle=0.5)
    success, _ = env.eval_state(goal, cur)
    assert bool(success) is True


def test_wrong_orientation_negative_goal_is_failure(env):
    """Regression: goal angle sampled negative (unwrapped) while the current
    angle is wrapped to [0, 2*pi). A block flipped ~180 degrees must NOT count
    as success. Previously |goal - cur| > 2*pi made ``2*pi - angle_diff``
    negative and np.minimum returned it, so angle_diff < pi/9 was always true.
    """
    # goal == -2*pi (equivalent to 0), block actually at pi (upside-down T)
    goal = make_state(block_xy=(256, 256), angle=-2 * np.pi)
    cur = make_state(block_xy=(256, 256), angle=np.pi)
    success, _ = env.eval_state(goal, cur)
    assert bool(success) is False


@pytest.mark.parametrize('goal_angle, cur_angle', [(-6.0, 3.3), (-5.5, 2.9)])
def test_wrong_orientation_negative_goal_param(env, goal_angle, cur_angle):
    """More negative-goal cases where the true orientation is far off."""
    goal = make_state(block_xy=(256, 256), angle=goal_angle)
    cur = make_state(block_xy=(256, 256), angle=cur_angle)
    success, _ = env.eval_state(goal, cur)
    assert bool(success) is False


def test_angle_wraps_around_2pi(env):
    """Angles that differ by exactly 2*pi are the same orientation."""
    goal = make_state(block_xy=(256, 256), angle=-np.pi / 6)  # ~ -30 deg
    cur = make_state(block_xy=(256, 256), angle=-np.pi / 6 + 2 * np.pi)
    success, _ = env.eval_state(goal, cur)
    assert bool(success) is True


def test_angle_just_below_threshold_is_success(env):
    goal = make_state(angle=0.0)
    cur = make_state(angle=ANGLE_THRESHOLD - 1e-3)
    success, _ = env.eval_state(goal, cur)
    assert bool(success) is True


def test_angle_just_above_threshold_is_failure(env):
    goal = make_state(angle=0.0)
    cur = make_state(angle=ANGLE_THRESHOLD + 1e-3)
    success, _ = env.eval_state(goal, cur)
    assert bool(success) is False


def test_position_just_below_threshold_is_success(env):
    goal = make_state(block_xy=(256, 256), angle=0.0)
    cur = make_state(block_xy=(256 + (POS_THRESHOLD - 1), 256), angle=0.0)
    success, _ = env.eval_state(goal, cur)
    assert bool(success) is True


def test_position_just_above_threshold_is_failure(env):
    goal = make_state(block_xy=(256, 256), angle=0.0)
    cur = make_state(block_xy=(256 + (POS_THRESHOLD + 1), 256), angle=0.0)
    success, _ = env.eval_state(goal, cur)
    assert bool(success) is False


def test_position_ignores_agent_and_velocity(env):
    """Only the block pose (indices 2:5) drives success, not agent/velocity."""
    goal = make_state(block_xy=(256, 256), angle=0.0, agent_xy=(256, 400))
    cur = make_state(block_xy=(256, 256), angle=0.0, agent_xy=(50, 50))
    cur[5:7] = [12.3, -7.0]  # nonzero agent velocity
    success, _ = env.eval_state(goal, cur)
    assert bool(success) is True
