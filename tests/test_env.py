from envi.humanoid_env import make_env


def test_env_step():
    env = make_env()
    obs, info = env.reset()
    assert obs is not None, "Observation is None"
    print("Observation shape:", obs.shape)
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None, "Observation became None"
        assert isinstance(reward, (int, float)), "Reward is not numeric"

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print("Environment test passed")


if __name__ == "__main__":
    test_env_step()
