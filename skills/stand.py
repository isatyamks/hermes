import numpy as np
from skills.base import Skill
from _logging.metrics import get_torso_height, detect_fall


class StandSkill(Skill):
    def __init__(self):
        super().__init__("stand")

    def reward(self, obs, action, env_reward):
        height = get_torso_height(obs)
        energy_penalty = 0.001 * np.sum(action ** 2)

        stand_reward = np.clip(height, 0.0, 2.0)
        #added for report b
        target_height = 1.3
        height_error = abs(height - target_height)
        stability_penalty = 0.5 * height_error



        return stand_reward - energy_penalty

    def termination(self, obs):
        return detect_fall(obs)
