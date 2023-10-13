import warnings
import numpy as np
from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv
from ray.rllib.env.wrappers.dm_control_wrapper import DMCEnv

from sindy_rl.refactor.swimmer import SwimmerWithBounds, SwimmerWithBoundsClassic
from sindy_rl.refactor.reward_fns import cart_reward

try: 
    from sindy_rl.refactor.hydroenv import CylinderLiftEnv
except ImportError:
    warnings.warn('Hydrogym is not installed!')
    

class DMCEnvWrapper(DMCEnv):
    '''
    A wrapper for all dm-control environments using RLlib's 
    DMCEnv wrapper. 
    '''
    # need to wrap with config dict instead of just passing kwargs
    def __init__(self, config=None):
        env_config = config or {}
        super().__init__(**env_config)

class SwimmerWrapper(SwimmerEnv):
    def __init__(self, config=None):
        env_config = config or {}
        super().__init__(**env_config)