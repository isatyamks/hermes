class Skill:

    def __init__(self, name: str):
        self.name = name

    def reward(self, obs, action, env_reward):
        raise NotImplementedError

    def termination(self, obs) -> bool:
        return False
