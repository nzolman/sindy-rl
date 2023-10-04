from ray.rllib.algorithms.mbmpo import MBMPOConfig
from ray.rllib.algorithms.mbmpo.model_ensemble import DynamicsEnsembleCustomModel
from gymnasium.envs.classic_control import PendulumEnv
import gymnasium as gym

import numpy as np

from sindy_rl.refactor.registry import  DMCEnvWrapper

import numpy as np
from dm_control.utils.rewards import tolerance

class PendulumWrapper(gym.Wrapper):
    """Wrapper for the Pendulum-v1 environment.

    Adds an additional `reward` method for some model-based RL algos (e.g.
    MB-MPO).
    """

    # This is required by MB-MPO's model vector env so that it knows how many
    # steps to simulate the env for.
    _max_episode_steps = 200

    def __init__(self,env_config={}, **kwargs):
        env = gym.make("Pendulum-v1", **kwargs)
        gym.Wrapper.__init__(self, env)

    def reward(self, obs, action, obs_next):
        # obs = [cos(theta), sin(theta), dtheta/dt]
        # To get the angle back from obs: atan2(sin(theta), cos(theta)).
        theta = np.arctan2(np.clip(obs[:, 1], -1.0, 1.0), np.clip(obs[:, 0], -1.0, 1.0))
        # Do everything in (B,) space (single theta-, action- and
        # reward values).
        a = np.clip(action, -self.max_torque, self.max_torque)[0]
        costs = (
            self.angle_normalize(theta) ** 2 + 0.1 * obs[:, 2] ** 2 + 0.001 * (a**2)
        )
        print('obs', obs)
        print('action', action)
        print('obs_next', obs_next)
        print('costs', costs)
        return -costs

    @staticmethod
    def angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi


# def cart_reward(z, u):
#     '''dmc reward for cartpole'''
#     cart_pos, cos_th, sin_th, dx, dth = z
    
#     upright = (cos_th + 1.0)/2.0
#     centered = tolerance(cart_pos, margin=2)
#     centered = (1.0 + centered)/2.0
#     small_control = tolerance(u, margin=1, 
#                               value_at_margin=0, 
#                               sigmoid='quadratic')[0]
#     small_control = (4.0 + small_control)/5.0
#     small_velocity = tolerance(dth, margin=5)
#     small_velocity = (1.0 + small_velocity)/2.0
#     return upright * small_control * small_velocity * centered


# class TestCart(DMCEnvWrapper):
#     def __init__(self, env_config):
#         super().__init__(env_config)
#         self.step_counter = 0
#     def reward(self, obs, action, obs_next):
#         return np.zeros(obs.shape[0]) #cart_reward(obs_next, action)
    
#     def step(self, action):
#         obs, rew, term, trunc, info = super().step(action)
#         done = term or trunc
#         return obs, rew, done, done, info
    

# env_config = {      
#         'domain_name': "cartpole",
#       'task_name': "swingup",
#       'frame_skip': 1,
#       'from_pixels': False
#       }


dynamics_model = {
            "custom_model": DynamicsEnsembleCustomModel,
            # Number of Transition-Dynamics (TD) models in the ensemble.
            "ensemble_size": 5,
            # Hidden layers for each model in the TD-model ensemble.
            "fcnet_hiddens": [64, 64],
            # Model learning rate.
            "lr": 1e-3,
            # Max number of training epochs per MBMPO iter.
            "train_epochs": 500,
            # Model batch size.
            "batch_size": 500,
            # Training/validation split.
            "valid_split_ratio": 0.2,
            # Normalize data (obs, action, and deltas).
            "normalize_data": True,
        }


config = MBMPOConfig()
config = (config.training(lr=0.0003, dynamics_model=dynamics_model)
                # .environment(env_config=env_config)
                .rollouts(num_rollout_workers=16)
                .resources(num_cpus_per_worker=3)
            )
algo = config.build(env=PendulumWrapper)
res = algo.train()

print(res)

# if __name__ == "__main__":
#     env = PendulumWrapper()
#     env.reset()
#     for _ in range(1000):
#         obs, rew, term, trunc, info = env.step(env.action_space.sample())
#         if term or trunc:
#             print('TERM!')
#     print('done.')