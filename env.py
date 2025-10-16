# env.py

from __future__ import annotations
import numpy as np

MIN_POSITION = -1.2
MAX_POSITION =  0.5
MIN_VELOCITY = -0.07
MAX_VELOCITY =  0.07
TIMEOUT      = 1000

class MountainCarEnv:
    """
    Minimal deterministic Mountain Car environment.

    State: s = (x, v)
    Action: a in {-1, 0, +1}  (reverse, neutral, forward)
    Dynamics:
        v' = clip(v + 0.001*a - 0.0025*cos(3*x), [-0.07, 0.07])
        x' = clip(x + v', [-1.2, 0.5])
        if x' hits either boundary, set v' := 0
    Terminal:
        success if x' >= 0.5, or timeout at 1000 steps
    Reward:
        -1 per step (the episode ends at goal with 0 additional reward)
    Discount:
        gamma = 1.0 (we'll just sum rewards)
    Initial:
        x0 ~ Uniform[-0.6, -0.4], v0 = 0
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.state = None
        self.steps = 0

    def reset(self) -> np.ndarray:
        x0 = self.rng.uniform(-0.6, -0.4)
        v0 = 0.0
        self.state = np.array([x0, v0], dtype=np.float64)
        self.steps = 0
        return self.state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        action: -1, 0, +1
        returns: (next_state, reward, done, info)
        """
        assert action in (-1, 0, 1), "Action must be -1, 0, or +1."

        x, v = self.state
        v_next = v + 0.001 * action - 0.0025 * np.cos(3.0 * x)
        v_next = np.clip(v_next, MIN_VELOCITY, MAX_VELOCITY)
        x_next = x + v_next
        x_next = np.clip(x_next, MIN_POSITION, MAX_POSITION)

        # Wall reset rule
        if x_next <= MIN_POSITION or x_next >= MAX_POSITION:
            v_next = 0.0

        self.state = np.array([x_next, v_next], dtype=np.float64)
        self.steps += 1

        # Terminal conditions
        done = bool(x_next >= MAX_POSITION or self.steps >= TIMEOUT)

        # Reward per spec: -1 each step until terminal. If we hit goal now, episode ends.
        reward = -1.0
        return self.state.copy(), reward, done, {}
