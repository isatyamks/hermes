from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env.humanoid_env import make_env
# from rl.callbacks import EpisodeStatsCallback
from rl.callbacks import HermesLoggingCallback

from skills.stand import StandSkill
from skills.walk import WalkSkill
from skills.recover import RecoverSkill
from skills.manager import SkillManager




skills = [
    StandSkill(),
    WalkSkill(),
    RecoverSkill()
]

skill_manager = SkillManager(
    skills=skills,
    initial_skill="stand"
)








def train():
    # env = make_vec_env(
    #     make_env,
    #     n_envs=4,
    # )
    env = make_vec_env(
    lambda: make_env(skill_manager=skill_manager),
    n_envs=4
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        device="auto",
    )
    # callback = EpisodeStatsCallback()
    callback = HermesLoggingCallback()

    model.learn(
        total_timesteps=10_000,
        callback=callback,
    )
    model.save("experiments/base_ppo")

    env.close()
    print("Training finished")


if __name__ == "__main__":
    train()
