import gymnasium as gym


class SkillRewardWrapper(gym.Wrapper):

    def __init__(self, env, skill_manager):
        super().__init__(env)
        self.skill_manager = skill_manager

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        skill_reward = self.skill_manager.compute_reward(
            obs=obs,
            action=action,
            env_reward=env_reward,
        )

        info["env_reward"] = env_reward
        info["skill_reward"] = skill_reward
        info["active_skill"] = self.skill_manager.name

        return obs, skill_reward, terminated, truncated, info
