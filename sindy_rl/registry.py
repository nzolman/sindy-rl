# --------------------------------------------------------
# intention to provide a convenient place to load custom/wrapped environments from
# when loading classes/configs from strings.
# --------------------------------------------------------

import warnings
import numpy as np
from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv
from ray.rllib.env.wrappers.dm_control_wrapper import DMCEnv

from sindy_rl.swimmer import SwimmerWithBounds, SwimmerWithBoundsClassic
from sindy_rl.reward_fns import cart_reward

# Don't require that user needs hydrogym installed
try: 
    from sindy_rl.hydroenv import CylinderLiftEnv, PinballLiftEnv, CylinderWrapper
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
    '''Wrapper to ensure compliant with RLib's config requirements'''
    def __init__(self, config=None):
        env_config = config or {}
        super().__init__(**env_config)
        
