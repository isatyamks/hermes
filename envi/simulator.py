import gymnasium as gym
from stable_baselines3 import PPO


def run_simulation():
    # Create environment with rendering
    env = gym.make(
        "Humanoid-v5",
        render_mode="human"
    )

    # Load trained PPO model
    model = PPO.load(
        "experiments/base_ppo_500k.zip",
        device="cpu"   # MuJoCo physics runs on CPU
    )

    obs, info = env.reset()

    while True:
        # Get action from trained policy
        action, _ = model.predict(obs, deterministic=True)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Reset if episode ends
        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    run_simulation()
