import numpy as np

from sindy_rl.envs.cartpole import CartSurrogate, CartPoleEnv
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
        obs2, rew2, terminated2, truncated,  info2 = env2.step(action)

        assert np.all(observation == obs2)
        assert reward == rew2
        assert terminated == terminated2 or truncated
        assert info == info2
       
        if terminated:
            observation = env.reset()
        if terminated2 or truncated:
            obs2 = env2.reset()

if __name__ == '__main__': 
    test_dynamics_equality()