import gymnasium as gym


def make_env(render_mode=None):
    return gym.make(
        "Humanoid-v5",
        render_mode=render_mode
    )
