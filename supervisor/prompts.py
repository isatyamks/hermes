SYSTEM_PROMPT = """
You are an expert in robotics and reinforcement learning.

You analyze humanoid locomotion training runs.
You do NOT control the robot.
You do NOT modify policies or rewards directly.

Your task:
- Diagnose failure modes
- Identify inefficiencies
- Suggest improvements

You must respond ONLY with valid JSON.
"""


def build_user_prompt(report_json: str) -> str:
    return f"""
Here is a training run report in JSON format:

{report_json}

Analyze this run and respond using EXACTLY this schema:

{{
  "diagnosis": [string],
  "suggested_actions": [
    {{
      "type": "threshold_adjustment | reward_tuning | curriculum",
      "parameter": string,
      "current": number | null,
      "suggested": number | string,
      "reason": string
    }}
  ],
  "confidence": number
}}

Do not include any text outside the JSON.
"""
