import numpy as np
from dm_control import viewer
from robopianist import suite

env = suite.load(
    "RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
    seed=0,
)
action_spec = env.action_spec()

def policy(time_step):
    return np.zeros(action_spec.shape, dtype=action_spec.dtype)

viewer.launch(env, policy)