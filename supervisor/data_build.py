import json
import os
from datetime import datetime
from collections import defaultdict
import numpy as np


class EpisodeDatasetBuilder:

    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def build(self, episode_summaries):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        report = {
            "run_metadata": self._run_metadata(episode_summaries),
            "overall_stats": self._overall_stats(episode_summaries),
            "skill_stats": self._skill_stats(episode_summaries),
            "failure_modes": self._failure_modes(episode_summaries),
            "episodes": episode_summaries,
        }

        path = os.path.join(
            self.output_dir, f"run_{timestamp}.json"
        )

        report = self._convert_numpy_types(report)

        try:
            with open(path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Successfully saved report to: {os.path.abspath(path)}")
        except Exception as e:
            print(f"Error saving report to {path}: {e}")
            raise

        return path

    def _convert_numpy_types(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        elif isinstance(obj, (bool, str, type(None))):
            return obj
        elif isinstance(obj, (int, float)):
            return obj
        else:
            try:
                if hasattr(obj, 'item'):
                    return obj.item()
                return obj
            except (ValueError, TypeError):
                return str(obj)


    def _run_metadata(self, episodes):
        return {
            "timestamp": datetime.now().isoformat(),
            "total_episodes": len(episodes),
            "skills_seen": sorted(
                list({ep["skill"] for ep in episodes})
            ),
        }

    def _overall_stats(self, episodes):
        return {
            "mean_reward": float(
                np.mean([ep["total_reward"] for ep in episodes])
            ),
            "mean_episode_length": float(
                np.mean([ep["episode_length"] for ep in episodes])
            ),
            "mean_height": float(
                np.mean([ep["mean_height"] for ep in episodes])
            ),
            "fall_rate": float(
                np.mean([ep["fell"] for ep in episodes])
            ),
        }

    def _skill_stats(self, episodes):
        skill_groups = defaultdict(list)

        for ep in episodes:
            skill_groups[ep["skill"]].append(ep)

        stats = {}
        for skill, eps in skill_groups.items():
            stats[skill] = {
                "episodes": len(eps),
                "mean_reward": float(
                    np.mean([e["total_reward"] for e in eps])
                ),
                "mean_height": float(
                    np.mean([e["mean_height"] for e in eps])
                ),
                "fall_rate": float(
                    np.mean([e["fell"] for e in eps])
                ),
            }

        return stats

    def _failure_modes(self, episodes):
        failures = defaultdict(int)

        for ep in episodes:
            if ep["fell"]:
                failures[ep["fall_type"]] += 1

        return dict(failures)
