from __future__ import annotations

import gymnasium as gym
import numpy as np


# ------------- TODO: Implement the following environment -------------
class MyEnv(gym.Env):
    """
    Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        """Initializes the observation and action space for the environment."""
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)

        self.reset()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Resets the environment to the initial state."""
        self.state = 0
        return self.state, {}

    def step(self, action: int):
        reward = int(action)
        done = False
        if action == 0:
            self.state = 0
        elif action == 1:
            self.state = 1
        else:
            raise RuntimeError(f"Invalid action: {action}, action must be 0 or 1.")

        return self.state, reward, done, False, {}

    def get_reward_per_action(self) -> np.ndarray:
        """Returns the reward for each (state, action) pair."""
        return np.array([[0, 1], [0, 1]])

    def get_transition_matrix(
        self,
        S: np.ndarray | None = None,
        A: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Construct a deterministic transition matrix T[s, a, s'].

        Parameters
        ----------
        S : np.ndarray, optional
            Array of states. Uses internal states if None.
        A : np.ndarray, optional
            Array of actions. Uses internal actions if None.
        P : np.ndarray, optional
            Action success probabilities. Uses internal P if None.

        Returns
        -------
        T : np.ndarray
            A (num_states, num_actions, num_states) tensor where
            T[s, a, s'] = probability of transitioning to s' from s via a.
        """
        if S is None:
            S = np.arange(self.observation_space.n)
        if A is None:
            A = np.arange(self.action_space.n)
        if P is None:
            P = np.ones((self.observation_space.n, self.action_space.n))

        T = np.zeros((len(S), len(A), len(S)))
        for s in S:
            for a in A:
                s_next = a
                T[s, a, s_next] = P[s, a]
        return T


class PartialObsWrapper(gym.Wrapper):
    """Wrapper that makes the underlying env partially observable by injecting
    observation noise: with probability `noise`, the true state is replaced by
    a random (incorrect) observation.

    Parameters
    ----------
    env : gym.Env
        The fully observable base environment.
    noise : float, default=0.1
        Probability in [0,1] of seeing a random wrong observation instead
        of the true one.
    seed : int | None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: MyEnv, fault_prob: float = 0.1, seed: int | None = None):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.env = env

        assert 0 <= fault_prob <= 1, "fault_prob must be in [0, 1]"
        self.fault_prob = fault_prob

        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        true_obs, info = self.env.reset(seed=seed, options=options)
        return self._inject_fault(true_obs), info

    def step(self, action: int):
        true_obs, reward, done, truncated, info = self.env.step(action)
        return self._inject_fault(true_obs), reward, done, truncated, info

    def _inject_fault(self, true_obs: int) -> int:
        """Injects a fault into the observation with probability `fault_prob`."""
        if self._rng.random() < self.fault_prob:
            if true_obs == 0:
                return 1
            else:
                return 0
        return true_obs
