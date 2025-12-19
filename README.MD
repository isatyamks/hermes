# HERMES: Hierarchical Embodied Reinforcement Learning with LLM Supervision

A sophisticated reinforcement learning framework for training humanoid locomotion using hierarchical skills and LLM-based supervision. HERMES combines traditional RL (PPO) with a skill-based architecture and leverages Large Language Models for offline analysis and improvement suggestions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [Workflow](#workflow)
- [Usage Examples](#usage-examples)
- [Development Roadmap](#development-roadmap)
- [Contributing](#contributing)
- [License](#license)

## Overview

HERMES is designed to train a humanoid robot (MuJoCo Humanoid-v5) to perform complex locomotion tasks through a hierarchical approach:

1. **Base PPO**: Low-level motor control (Phase 1 - Baseline)
2. **Skill PPO**: Same control architecture with skill-specific reward shaping
3. **Curriculum PPO**: Staged learning progression
4. **HERMES PPO**: Full RL + LLM supervision system

The system uses a skill-based hierarchy where the agent learns distinct skills (Stand, Walk, Recover) and switches between them based on performance metrics. An LLM supervisor analyzes training runs offline and provides actionable insights for improvement.

## Features

- 🎯 **Hierarchical Skill System**: Modular skills (Stand, Walk, Recover) with independent reward shaping
- 🤖 **PPO-based RL**: Stable-Baselines3 implementation for robust training
- 🧠 **LLM Supervision**: Offline analysis using local LLM (Ollama) for training diagnostics
- 📊 **Comprehensive Logging**: Episode-level metrics including energy, height, falls, and skill performance
- 🔄 **Automatic Skill Switching**: Rule-based supervisor switches skills based on performance
- 📈 **Training Reports**: JSON-based reports with aggregated statistics for analysis
- 🎮 **MuJoCo Integration**: Full support for Humanoid-v5 environment

## Architecture

```
HERMES SYSTEM
│
├── base_ppo          ← Phase 1: Pure RL Baseline (low-level motor cortex)
│
├── skill_ppo         ← Phase 2: Same control, skill-specific rewards
│
├── curriculum_ppo    ← Phase 3: Staged learning progression
│
└── hermes_ppo        ← Phase 4: RL + LLM supervision (full system)
```

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop (PPO)                      │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Environment │───▶│  Skill       │───▶│   Episode    │ │
│  │  (Humanoid-v5)│    │  Manager     │    │   Logger     │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                    │                    │         │
│         └────────────────────┼────────────────────┘         │
│                              │                              │
│                    ┌─────────▼─────────┐                   │
│                    │  Skill Switcher    │                   │
│                    │  (Rule-based)      │                   │
│                    └────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Episode Summaries  │
                    │  (JSON Reports)     │
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  LLM Analyzer        │
                    │  (Offline Analysis)  │
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Improvement        │
                    │  Suggestions        │
                    └─────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- MuJoCo (for Humanoid-v5 environment)
- Ollama (for LLM supervision - optional)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd hermes
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

The requirements include:
- `gymnasium`: OpenAI Gym-compatible environments
- `mujoco`: MuJoCo physics simulator
- `stable-baselines3`: PPO implementation
- `torch`: Deep learning framework
- `numpy`: Numerical computing
- `pyyaml`: YAML parsing

### Step 3: Install Ollama (Optional, for LLM Supervision)

If you want to use LLM-based analysis:

1. Install Ollama from [https://ollama.ai](https://ollama.ai)
2. Pull a model (e.g., `ollama pull llama3`)

## Quick Start

### 1. Train a Model

Train a PPO agent with hierarchical skills:

```bash
python -m rl.train
```

This will:
- Create a vectorized environment (4 parallel environments)
- Initialize skills: Stand, Walk, Recover
- Train for 10,000 timesteps
- Save the model to `experiments/base_ppo`
- Generate a training report in `supervisor/reports/`

### 2. Evaluate a Trained Model

```bash
python -m rl.evaluate
```

Note: `rl.evaluate` currently does not parse CLI arguments. To change the model path or episode count, edit the defaults in `rl/evaluate.py` (the `evaluate(model_path=..., episodes=...)` call under the `if __name__ == "__main__"` block), or import and call `evaluate()` from a Python session.

### 3. Analyze Training with LLM

After training, analyze the generated report with the offline LLM analyzer:

1) Find the report path printed at the end of training (saved by default under `supervisor/reports/`).
2) Update `report_path` in `tr.py` to that file path.
3) Run:

```bash
python tr.py
```

This will:
- Load the specified training report
- Send it to the Ollama LLM for analysis
- Save the analysis JSON to `supervisor/analyses/`

## Project Structure

```
hermes/
│
├── _logging/              # Episode logging and metrics
│   ├── __init__.py
│   ├── episode_logger.py  # Collects episode-level statistics
│   ├── metrics.py         # Physics and observation-based metrics
│   └── summary_schema.py  # Structured episode summaries
│
├── env/                   # Environment setup
│   ├── __init__.py
│   ├── humanoid_env.py    # MuJoCo Humanoid-v5 wrapper
│   └── reward_wrapper.py  # Skill-specific reward shaping
│
├── rl/                    # Reinforcement learning
│   ├── train.py           # Main training script
│   ├── evaluate.py        # Model evaluation
│   └── callbacks.py       # Training callbacks (logging, skill switching)
│
├── skills/                # Hierarchical skills
│   ├── base.py            # Base Skill interface
│   ├── manager.py         # Skill manager (active skill routing)
│   ├── switcher.py        # Rule-based skill switching logic
│   ├── stand.py           # Stand skill implementation
│   ├── walk.py            # Walk skill implementation
│   └── recover.py         # Recovery skill implementation
│
├── supervisor/            # LLM supervision system
│   ├── analyzer.py        # Offline LLM analyzer
│   ├── data_build.py      # Report generation from episodes (writes to supervisor/reports)
│   ├── ollama_client.py   # Ollama LLM client
│   ├── parser.py          # LLM response parser
│   ├── prompts.py         # LLM prompts (system/user)
│   ├── save_analysis.py   # Save analysis results (writes to supervisor/analyses)
│   ├── reports/           # Generated training reports (JSON)
│   └── analyses/          # LLM analysis results (JSON)
│
├── experiments/           # Saved models
│   └── base_ppo.zip       # Trained PPO model
│
├── tests/                 # Test suite
│   ├── sanity.py          # Basic sanity checks
│   └── test_env.py        # Environment tests
│
├── tr.py                  # Training report analysis script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Key Components

### 1. Skills System

Skills are modular behaviors with independent reward functions:

#### Stand Skill (`skills/stand.py`)
- **Goal**: Maintain upright posture
- **Reward**: Height-based reward with energy penalty
- **Termination**: Triggers when humanoid falls (height < 0.8m)

#### Walk Skill (`skills/walk.py`)
- **Goal**: Forward locomotion while maintaining balance
- **Reward**: Forward velocity + height maintenance - energy penalty
- **Termination**: Triggers on fall

#### Recover Skill (`skills/recover.py`)
- **Goal**: Recover from fall to upright position
- **Reward**: Height increase - energy penalty
- **Termination**: Triggers when upright again (height >= 0.8m)

### 2. Skill Manager (`skills/manager.py`)

Manages the active skill and routes reward computation:

```python
skill_manager = SkillManager(
    skills=[StandSkill(), WalkSkill(), RecoverSkill()],
    initial_skill="stand"
)
```

### 3. Skill Switcher (`skills/switcher.py`)

Rule-based supervisor that switches skills based on episode summaries:

- **Stand → Walk**: When mean height > 1.2m and episode length > 50
- **Walk → Recover**: When fall detected
- **Recover → Stand**: When upright (height > 1.0m) and no fall

### 4. Episode Logger (`_logging/episode_logger.py`)

Collects comprehensive episode statistics:
- Total reward
- Energy consumption (sum of squared actions)
- Height trajectory
- Fall detection and classification
- Episode length

### 5. Metrics (`_logging/metrics.py`)

Provides both observation-based and physics-based metrics:

- **Observation-based**: Fast, uses observation array
- **Physics-based**: Accurate, uses MuJoCo physics state (`env.data.qpos[2]`)

### 6. LLM Supervisor (`supervisor/`)

Offline analysis system:
- **EpisodeDatasetBuilder**: Aggregates episode summaries into reports
- **OfflineAnalyzer**: Sends reports to LLM for analysis
- **OllamaClient**: Local LLM integration (no API keys needed)

## Workflow

### Training Workflow

1. **Environment Setup**
   ```python
   env = make_vec_env(lambda: make_env(skill_manager=skill_manager), n_envs=4)
   ```

2. **Model Initialization**
   ```python
   model = PPO(policy="MlpPolicy", env=env, ...)
   ```

3. **Callback Setup**
   ```python
   callback = HermesLoggingCallback(
       skill_manager=skill_manager,
       skill_switcher=skill_switcher,
       env=env
   )
   ```

4. **Training**
   ```python
   model.learn(total_timesteps=10_000, callback=callback)
   ```

5. **Report Generation**
   ```python
   builder = EpisodeDatasetBuilder()
   report_path = builder.build(callback.episode_summaries)
   ```

### Analysis Workflow

1. **Load Report**
   ```python
   analyzer = OfflineAnalyzer(llm_client)
   analysis = analyzer.analyze("supervisor/reports/run_YYYY-MM-DD_HH-MM-SS.json")
   ```

2. **Save Analysis**
   ```python
   save_analysis(analysis)
   ```

3. **Review Suggestions**
   - Check `supervisor/analyses/analysis_*.json`
   - Review diagnosis and suggested actions
   - Implement improvements in skill rewards or switching logic

## Usage Examples

### Customizing Training

The provided entrypoint `rl/train.py` uses a default skill set (Stand, Walk, Recover) and `total_timesteps=10_000`. To customize:

- Edit `skills` and `initial_skill` near the top of `rl/train.py`.
- Adjust PPO hyperparameters and `model.learn(total_timesteps=...)` as needed.

Then run:

```bash
python -m rl.train
```

### Adding a New Skill

```python
from skills.base import Skill
from _logging.metrics import get_torso_height

class JumpSkill(Skill):
    def __init__(self):
        super().__init__("jump")
    
    def reward(self, obs, action, env_reward):
        height = get_torso_height(obs)
        vertical_velocity = obs[1]  # y-velocity
        return height + vertical_velocity - 0.001 * np.sum(action ** 2)
    
    def termination(self, obs):
        return get_torso_height(obs) < 1.0  # Terminate when landing
```

### Custom Skill Switching Logic

```python
from skills.switcher import SkillSwitcher

class CustomSkillSwitcher(SkillSwitcher):
    def decide(self, episode_summary):
        current = self.skill_manager.name
        
        # Custom logic here
        if current == "stand" and episode_summary["mean_height"] > 1.5:
            return "walk"
        
        return current
```

### Environment Testing

```bash
# Test environment setup
python -m tests.test_env

# Sanity checks
python -m tests.sanity
```

## Development Roadmap

### Phase 1: Pure RL Baseline ✅
- [x] Base PPO implementation
- [x] Environment setup
- [x] Basic logging

### Phase 2: Skill PPO (Current)
- [x] Skill-based reward shaping
- [x] Skill manager
- [x] Skill switching logic
- [x] Episode logging

### Phase 3: Curriculum PPO (Planned)
- [ ] Staged learning progression
- [ ] Curriculum scheduling
- [ ] Progressive difficulty

### Phase 4: HERMES PPO (Planned)
- [ ] Online LLM supervision
- [ ] Dynamic reward adjustment
- [ ] Real-time skill switching based on LLM
- [ ] Adaptive curriculum

## Technical Details

### Observation Space
- MuJoCo Humanoid-v5 provides observations including:
  - Position (qpos): Root position, joint angles
  - Velocity (qvel): Root velocity, joint velocities
  - Height: Torso z-position (index 2 in qpos)

### Action Space
- Continuous control: 17-dimensional action space
- Torque control for all joints

### Reward Structure
- **Environment Reward**: MuJoCo's default reward (survival, control, impact)
- **Skill Reward**: Skill-specific shaping (height, velocity, energy)

### Metrics
- **Torso Height**: `env.data.qpos[2]` (physics-based, accurate)
- **Energy**: Sum of squared actions (proxy for energy consumption)
- **Fall Detection**: Height < 0.8m threshold
- **Fall Classification**: Forward/backward/collapse based on velocity

## Troubleshooting

### Common Issues

1. **MuJoCo Not Found**
   ```bash
   pip install mujoco
   ```

2. **Ollama Not Available**
   - Install Ollama from https://ollama.ai
   - Or modify `supervisor/ollama_client.py` to use a different LLM API

3. **CUDA/GPU Issues**
   - Set `device="cpu"` in PPO initialization if GPU not available

4. **Import Errors**
   - Ensure you're running from the repository root
   - Check that all dependencies are installed: `pip install -r requirements.txt`

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Specify your license here]

## Citation

If you use HERMES in your research, please cite:

```bibtex
@software{hermes2024,
  title={HERMES: Hierarchical Embodied Reinforcement Learning with LLM Supervision},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

## Acknowledgments

- MuJoCo physics simulator
- Stable-Baselines3 team
- OpenAI Gym/Gymnasium
- Ollama for local LLM support

---

**Note**: This project is currently in active development. The current implementation focuses on Phase 2 (Skill PPO). Future phases will add curriculum learning and online LLM supervision.
