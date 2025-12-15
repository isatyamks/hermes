import gymnasium as gym
from env.reward_wrapper import SkillRewardWrapper


def make_env(render_mode=None, skill_manager=None):
    env = gym.make("Humanoid-v5", render_mode=render_mode)

    if skill_manager is not None:
        env = SkillRewardWrapper(env, skill_manager)

    return env
