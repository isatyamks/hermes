import numpy as np



def get_torso_height(obs):
    return float(obs[2])


def detect_fall(obs, height_threshold=0.75):
    return get_torso_height(obs) < height_threshold



def get_torso_height_from_env(env):
    return float(env.unwrapped.data.qpos[2])


def compute_energy(action):
    return float(np.sum(np.square(action)))


def detect_fall_from_env(env, height_threshold=0.8):
    return get_torso_height_from_env(env) < height_threshold


def classify_fall_from_env(env):
    x_vel = float(env.unwrapped.data.qvel[0])

    if x_vel > 0.5:
        return "forward_fall"
    elif x_vel < -0.5:
        return "backward_fall"
    else:
        return "collapse"
