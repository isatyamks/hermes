import numpy as np
from skills.base import Skill
from _logging.metrics import get_torso_height, detect_fall


class WalkSkill(Skill):
    def __init__(self):
        super().__init__("walk")

    def reward(self, obs, action, env_reward):
        height = get_torso_height(obs)
        forward_velocity = obs[0]   # x-velocity (approx)
        energy_penalty = 0.003 * np.sum(action ** 2)

        walk_reward = forward_velocity + 0.5 * height
        return walk_reward - energy_penalty

    def termination(self, obs):
        return detect_fall(obs)
