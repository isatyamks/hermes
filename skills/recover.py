import numpy as np
from skills.base import Skill
from _logging.metrics import get_torso_height, detect_fall


class RecoverSkill(Skill):
    def __init__(self):
        super().__init__("recover")

    def reward(self, obs, action, env_reward):
        height = get_torso_height(obs)
        energy_penalty = 0.001 * np.sum(action ** 2)

        return height - energy_penalty

    def termination(self, obs):
        return not detect_fall(obs)
