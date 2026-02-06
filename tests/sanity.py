from envi.humanoid_env import make_env
from _logging.metrics import get_torso_height_from_env

env = make_env()
obs, _ = env.reset()

print("Initial torso height:", get_torso_height_from_env(env))
