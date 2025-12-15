import time
from stable_baselines3 import PPO

from env.humanoid_env import make_env


def evaluate(model_path="experiments/base_ppo", episodes=3):
    env = make_env(render_mode="human")
    model = PPO.load(model_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            time.sleep(0.01)

        print(f"episode {ep + 1} | total reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    evaluate()
