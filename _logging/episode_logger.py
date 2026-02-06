import numpy as np
from _logging.metrics import (
    compute_energy,
    detect_fall_from_env,
    classify_fall_from_env,
    get_torso_height_from_env,
)
from _logging.summary_schema import make_episode_summary


class EpisodeLogger:

    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.total_reward = 0.0
        self.energy = 0.0
        self.heights = []
        self.fell = False
        self.fall_type = "none"
        self.length = 0

    def step(self, obs, action, reward):
        self.total_reward += reward
        self.energy += compute_energy(action)

        height = get_torso_height_from_env(self.env)
        height = get_torso_height_from_env(self.env)
        self.heights.append(height)

        self.length += 1

        if detect_fall_from_env(self.env) and not self.fell:
            self.fell = True
            self.fall_type = classify_fall_from_env(self.env)

    def summary(self):
        return make_episode_summary(
            total_reward=self.total_reward,
            episode_length=self.length,
            energy=self.energy,
            fell=self.fell,
            fall_type=self.fall_type,
            mean_height=float(np.mean(self.heights)) if self.heights else 0.0,
        )
