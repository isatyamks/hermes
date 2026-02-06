from stable_baselines3.common.callbacks import BaseCallback
from _logging.episode_logger import EpisodeLogger
import numpy as np
 


class EpisodeStatsCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None:
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done:
                    ep_info = info.get("episode")
                    if ep_info:
                        self.episode_rewards.append(ep_info["r"])
                        self.episode_lengths.append(ep_info["l"])
        return True

    def _on_training_end(self) -> None:
        if self.episode_rewards:
            print("\nTraining Summary")
            print(f"Mean Reward: {np.mean(self.episode_rewards):.2f}")
            print(f"Mean Episode Length: {np.mean(self.episode_lengths):.2f}")


class HermesLoggingCallback(BaseCallback):
    def __init__(self, skill_manager, skill_switcher, env=None, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.ep_logger = None
        self.episode_summaries = []
        self.skill_manager = skill_manager
        self.skill_switcher = skill_switcher
        self.training_analyzer = None

    def _on_step(self) -> bool:
        if self.ep_logger is None:
            if self.env is not None:
                env0 = self.env.envs[0] if hasattr(self.env, 'envs') else self.env
            else:
                env0 = self.training_env.envs[0]
            self.ep_logger = EpisodeLogger(env0)
        
        obs = self.locals["new_obs"]
        actions = self.locals["actions"]
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for i in range(len(dones)):
            self.ep_logger.step(obs[i], actions[i], rewards[i])

            if dones[i]:
                summary = self.ep_logger.summary()
                summary["skill"] = infos[i].get(
                    "active_skill", self.skill_manager.name
                )

                next_skill = self.skill_switcher.decide(summary)

                if next_skill != self.skill_manager.name:
                    print(
                        f"Skill switch: "
                        f"{self.skill_manager.name} → {next_skill}"
                    )
                    self.skill_manager.set_skill(next_skill)

                self.episode_summaries.append(summary)
                self.ep_logger.reset()

        return True

    def _on_training_end(self):
        print("\nSample Episode Summaries:")
        for s in self.episode_summaries[:5]:
            print(s)

        return