Hierarchical Embodied Reinforcement Learning with LLM Supervision (HERMES)





#phase 1 : Pure RL Baseline (No Hierarchy, No LLM)


HERMES SYSTEM
│
├── base_ppo          ← this (low-level motor cortex)
│
├── skill_ppo         ← same control, skill rewards
│
├── curriculum_ppo    ← staged learning
│
└── hermes_ppo        ← RL + LLM supervision



🔧 Step 3.1 — Fix Height Measurement

Update hermes_logging/metrics.py:
def get_torso_height(obs):
    """
    MuJoCo Humanoid-v5 torso height.
    Observation layout:
    qpos starts at index 0
    z position is qpos[2]
    """
    return float(obs[2])


Then update episode_logger.py:

Replace this:
self.heights.append(obs[2])

With this:
from hermes_logging.metrics import get_torso_height
self.heights.append(get_torso_height(obs))


And in detect_fall() also use get_torso_height(obs).