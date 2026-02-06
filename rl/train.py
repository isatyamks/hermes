from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envi.humanoid_env import make_env
from rl.callbacks import HermesLoggingCallback
from skills.switcher import SkillSwitcher
from skills.stand import StandSkill
from skills.walk import WalkSkill
from skills.recover import RecoverSkill
from skills.manager import SkillManager
from supervisor.data_build import EpisodeDatasetBuilder
import os




skills = [
    StandSkill(),
    WalkSkill(),
    RecoverSkill()
]

skill_manager = SkillManager(
    skills=skills,
    initial_skill="stand"
)
skill_switcher = SkillSwitcher(skill_manager)



MODEL_PATH = "experiments/base_ppo_500k.zip"

def train():
    env = make_vec_env(
        lambda: make_env(skill_manager=skill_manager),
        n_envs=4
    )

    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = PPO.load(MODEL_PATH, env=env, device="cpu")
    else:
        print("Training new model...")
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
            device="cpu",
        )

    callback = HermesLoggingCallback(
        skill_manager=skill_manager,
        skill_switcher=skill_switcher,
        env=env,
    )

    model.learn(
        total_timesteps=400_000,
        callback=callback,
        reset_num_timesteps=False,  # VERY IMPORTANT
    )

    model.save("experiments/base_ppo_500k")
    env.close() 
if __name__ == "__main__":
    train()