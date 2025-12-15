from stable_baselines3.common.callbacks import BaseCallback

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
