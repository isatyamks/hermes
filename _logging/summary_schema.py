from typing import Dict


def make_episode_summary(
    total_reward: float,
    episode_length: int,
    energy: float,
    fell: bool,
    fall_type: str,
    mean_height: float,
) -> Dict:
    return {
        "total_reward": round(total_reward, 2),
        "episode_length": episode_length,
        "energy": round(energy, 2),
        "fell": fell,
        "fall_type": fall_type,
        "mean_height": round(mean_height, 3),
    }
