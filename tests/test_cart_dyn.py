import numpy as np

from sindy_rl.environment import CartSurrogate, CartPoleEnv
from sindy_rl.dynamics import BaseDynModel, SINDyDynamics, CartPoleGymDynamics


def test_dynamics_equality():
    dyn_model = CartPoleGymDynamics()

    env_config = {'dyn_model': dyn_model}
    env = CartSurrogate(env_config)
    env2 = CartPoleEnv()

    observation = env.reset(seed=42)
    obs2 = env2.reset(seed=42)

    n_steps = 1000
    for i in range(n_steps):
        action = env.action_space.sample()
        observation, reward, terminated, info = env.step(action)
        obs2, rew2, term2, info2 = env2.step(action)

        assert np.all(observation == obs2)
        assert reward == rew2
        assert terminated == term2
        assert info == info2
       
        if terminated:
            observation = env.reset()
        if term2:
            obs2 = env2.reset()
