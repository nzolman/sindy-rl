# --------------------------------------------------------
# intention to provide a convenient place to load custom/wrapped environments from
# when loading classes/configs from strings.
# --------------------------------------------------------

import warnings
import pickle
import numpy as np
from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv
from gymnasium.spaces import Box
from ray.rllib.env.wrappers.dm_control_wrapper import DMCEnv

from sindy_rl.swimmer import SwimmerWithBounds, SwimmerWithBoundsClassic
from sindy_rl.reward_fns import cart_reward
# from sindy_rl.env import safe_reset, safe_step

# Don't require that user needs hydrogym installed
try:
    # Used for old version of the code
    # from sindy_rl.hydroenv import CylinderLiftEnv, PinballLiftEnv, CylinderWrapper
    
    from hydrogym import FlowEnv
    from hydrogym import firedrake as hgym
except ImportError:
    warnings.warn('Hydrogym is not installed!')

def safe_reset(res):
    '''A ''safe'' wrapper for dealing with OpenAI gym's refactor.'''
    if isinstance(res[-1], dict):
        return res[0]
    else:
        return res
    
def safe_step(res):
    '''A ''safe'' wrapper for dealing with OpenAI gym's refactor.'''
    if len(res)==5:
        return res[0], res[1], res[2] or res[3], res[4]
    else:
        return res

class HydroSVDWrapper(FlowEnv):
    def __init__(self, config):
        flow_name = config.get('flow_name', 'Pinball')
        flow_config = config.get('flow_config', {})
        dt = config.get('dt', 1.0e-2)
        env_config = {
            "flow": getattr(hgym, flow_name),
            "flow_config": flow_config,
            "solver": hgym.SemiImplicitBDF,
            "solver_config": {"dt": dt},
            }
        
        super().__init__(env_config)
        
        self._max_steps = config.get('max_steps', 1000)
        self._n_steps = 0
        
        svd_load = config.get('svd_load', None)
        if svd_load:
            with open(svd_load, 'rb') as f:
                self._svd_basis = pickle.load(f)
        else:
            self._svd_basis = None
            
        self._n_svd = config.get('n_svd', self.observation_space.shape[0])
        
        self.observation_space = Box(low = -np.inf, 
                                     high=np.inf, 
                                     shape = (self._n_svd, ),
                                     )

    def _fit_svd_basis(self, X):
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        self._svd_basis = Vh
        
    def _project_svd(self, X):
        return (self._svd_basis[:self._n_svd] @ X.T).T
    
    def _recon_svd(self, Z):
        return (self._svd_basis[:self._n_svd].T @ Z.T).T
    
    def step(self, action, **kwargs):
        res = [*super().step(action, **kwargs)]
        obs = res[0]
        res[0] = self._project_svd(obs)
        self._n_steps += 1
        
        if self._n_steps >= self._max_steps:
            res[2] = True
        
        return res
    
    def reset(self, **kwargs):
        obs = [*super().reset(**kwargs)][0]   # super returns (obs, info)
        obs_proj = self._project_svd(obs) 
        
        self._n_steps = 0
        return obs_proj

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
        
