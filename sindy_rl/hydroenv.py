import numpy as np
from gymnasium.spaces import Box
from hydrogym import firedrake as hgym
from hydrogym.core import PDEBase
import firedrake as fd
import matplotlib.pyplot as plt


class HydroEnvWrapper(hgym.FlowEnv):
    '''
    Wrapper for HydroEnv class to allow for filtering observations
    and allowing the agent to submit actions less frequently than the
    solver dt. In particular, allows us to provide 1st-order control 
    that acts as a linear interpolation between agent interactions.
    '''
    def __init__(self, env_config=None):
        '''
        Schema can be kind of confusing
        env_config:
          hydro_config:             (Hydrogym specific config)
              flow: <hgym.env>
              solver: <hygm.solver>
              flow_config: <dict>
          control_freq: 50          (how many inner_steps should execute---a first-order hold control)
          n_skip_on_reset: 50       (how many inner_steps to provide a null action after a reset)
          max_episode_steps: 1000   (how many times to execute an outer step before terminating)
          use_filter: True          (whether to use a filtered observation)
          
        Attributes:
            hydro_state, prev_hydro state: 
                the hydrogym state and previous hydrogym states. 
                Can be used to calculate differences, etc. 
            state:
                the observation to be passed out of the step function.
        '''
        self.config = env_config
        hydro_config = env_config.get('hydro_config', {})
        super(HydroEnvWrapper, self).__init__(hydro_config)
        self.dyn_model = env_config.get('dyn_model', None)
        self.control_freq = int(env_config.get('control_freq', 50))
        self.n_skip_on_reset = int(env_config.get('n_skip_on_reset', 50)) 
        
        self.use_filter = env_config.get('use_filter', False)
        
        assert self.control_freq >= 1, 'Control frequency must be positive integery'
        
        self.max_episode_steps = env_config.get('max_episode_steps', 1000)
        
        self.action_space = Box(low = self.action_space.low, 
                                high = self.action_space.high)

        self.observation_space = Box(low = self.observation_space.low, 
                                     high = self.observation_space.high)

    def real_step(self, action):
        '''
        Take a step in the hydrogym environment at the solver dt.
        '''
        # keep track of previous state
        self.prev_hydro_state = self.hydro_state
        
        # take step
        hydro_state, rew, done, info = super(HydroEnvWrapper, self).step(action)
        self.hydro_state = np.array(hydro_state)
        
        if self.use_filter: 
            # apply a low-pass filter
            tau = self.flow.TAU
            dt = self.solver.dt
            self.hydro_state = self.prev_hydro_state + (dt/tau) * (self.hydro_state - self.prev_hydro_state)
        
        self.n_real_steps += 1
        return self.hydro_state, rew, done, info 

    def get_done(self):
        # if None, return False
        if self.max_episode_steps and (self.n_episode_steps >= self.max_episode_steps):
            return True
        else:
            return False

    def step(self, action):
        '''
        We wrap the Hydrogym step function by providing many intermediate steps
        '''
        self.action = action        
        
        # accumulate rewards from inner steps
        tot_rew = 0
        for i in range(self.control_freq):
            frac = float(i+1)/self.control_freq
            
            # interpolate between the previous and currect action
            # to smooth out the control. Final control executed 
            # should be the provided control from the agent.
            u = (1-frac) * self.prev_action + frac * self.action
            
            self.hydro_state, rew, done, info = self.real_step(u)
            tot_rew += rew
        
        self.state = self.hydro_state

        self.episode_reward += tot_rew
        self.n_episode_steps += 1
        done = self.get_done()
        
        self.prev_action = self.action
        
        return self.state, tot_rew, done, info

    def reset(self, seed=0, options=None, **reset_kwargs):
        '''
        Resetting the environment just reinitializes it to the loaded checkpoint.
            If n_skip_on_reset is not 0, then this runs a fixed number of inner steps
            with null control at the beginning of the episode. This is useful to warm start
            a filter and smoothing out the flow field if there are any weird perturbations. 
        '''
        # extract hydrogym state from FlowEnv reset
        self.hydro_state = np.array(super(HydroEnvWrapper, self).reset(**reset_kwargs))
        self.prev_hydro_state = self.hydro_state.copy()
        
        # book-keeping quantities
        self.episode_reward = 0
        self.n_episode_steps = 0
        self.n_real_steps = 0

        # sometimes beneficial to skip the first number of steps
        for i in range(self.n_skip_on_reset):
            self.hydro_state, rew, done, info = self.real_step(np.zeros(self.flow.ACT_DIM))
        
        # set the state to the most "recent" hydro_state
        self.state = self.hydro_state
        
        # NOTE: this assumes that we start a simulation with zero actuation!
        self.prev_action = 0.0 * self.action_space.sample()
        
        return np.array(self.state)


class CylinderLiftEnv(HydroEnvWrapper):
    '''
    A wrapper for the Hydrogym Cylinder env that just provides 
    (C_L, \dot C_L ) as an observation.
    '''
    def __init__(self, env_config=None):
        config = env_config.copy()
        
        default_flow_config =  {
            "flow": hgym.Cylinder,
            "solver": hgym.IPCS,
            "flow_config": {
                'mesh': 'medium',
                'actuator_integration': 'implicit',
                'Re': 100
            }
        }

        flow_config = config.get('hydro_config', {})
        default_flow_config.update(flow_config)
        config['hydro_config'] = default_flow_config
        
        self.obs_clip_val = 100
        super(CylinderLiftEnv, self).__init__(config)
        self._init_augmented_obs()
        
    def _init_augmented_obs(self):
        obs_dim = 2
        self.observation_space = Box( 
                                     low=-np.inf,
                                     high=np.inf,
                                     shape=(obs_dim,),
                                     dtype=PDEBase.ScalarType,
                                     )
    def get_CL_dot(self): 
        # only to be used by the real env!
        return (self.hydro_state - self.prev_hydro_state)[0]/self.solver.dt
    
    def clip_obs(self, state):
        '''
        Clips Observation between bounds
        '''
        return np.clip(state, -self.obs_clip_val, self.obs_clip_val)
    
    def build_augmented_obs(self):
        '''
        Builds augmented obs for the environment
        '''
        obs = [self.hydro_state[0], self.get_CL_dot()]
        return np.array(obs)

    def step(self, action):
        '''Overwrite just to build the augmented observation'''
        self.action = action
        self.state, rew, done, info = super(CylinderLiftEnv, self).step(action)
        self.state = self.build_augmented_obs()
        
        if self.obs_clip_val:
            self.state = self.clip_obs(self.state)

        return self.state, rew, done, info 
    
    def reset(self, **reset_kwargs):
        self.state = super(CylinderLiftEnv, self).reset(**reset_kwargs)
        self.state = self.build_augmented_obs()
        return self.state

class CylinderWrapper(CylinderLiftEnv):
    '''Additional wrapper needed since hydrogym not compliant with latest gymnasium'''
    def __init__(self, env_config=None):
        super().__init__(env_config)
        
    def step(self, action):
        obs, rew, done, info = super().step(action)
        
        truncated =  done
        terminated = done
        
        return obs, rew, terminated, truncated, info
    
    def reset(self, seed=0, options = None, **kwargs):
        obs = super().reset()
        return obs, {}