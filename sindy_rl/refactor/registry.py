import warnings
from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv
from ray.rllib.env.wrappers.dm_control_wrapper import DMCEnv

from sindy_rl.refactor.swimmer import SwimmerWithBounds, SwimmerWithBoundsClassic
from sindy_rl.refactor.reward_fns import cart_reward

try: 
    from sindy_rl.refactor.hydroenv import CylinderLiftEnv
except ImportError:
    warnings.warn('Hydrogym is not installed!')
    

class DMCEnvWrapper(DMCEnv):
    def __init__(self, config=None):
        env_config = config or {}
        super().__init__(**env_config)
        
    # # needed for MBMPO!
    # def reward(self, obs, action, obs_next):
    #     rew = cart_reward(obs_next.T, action)
    #     return rew
        
    
    # def reset(self,seed=None):
    #     '''Note, the seed actually does nothing right now. Just to conform to our method'''
    #     return super().reset()

class SwimmerWrapper(SwimmerEnv):
    def __init__(self, config=None):
        env_config = config or {}
        super().__init__(**env_config)