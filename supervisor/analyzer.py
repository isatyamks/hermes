import json
from supervisor.prompts import SYSTEM_PROMPT, build_user_prompt
from supervisor.parser import parse_llm_response


class OfflineAnalyzer:
    def __init__(self, llm_client):
        self.llm = llm_client

    def analyze(self, report_path: str):
        with open(report_path, "r") as f:
            report = json.load(f)

        episodes = report["episodes"]

        summary = {
            "num_episodes": len(episodes),
            "mean_reward": sum(e["total_reward"] for e in episodes) / len(episodes),
            "mean_energy": sum(e["energy"] for e in episodes) / len(episodes),
            "fall_rate": sum(1 for e in episodes if e["fell"]) / len(episodes),
            "mean_height": sum(e["mean_height"] for e in episodes) / len(episodes),
            "skill_distribution": {
                s: sum(1 for e in episodes if e["skill"] == s)
                for s in set(e["skill"] for e in episodes)
            }
        }

        user_prompt = build_user_prompt(
            json.dumps(summary, indent=2)
        )

        raw_response = self.llm.chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        return parse_llm_response(raw_response)
