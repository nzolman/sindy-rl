import numpy as np
from gym.spaces import Box
from hydrogym import firedrake as hgym
from hydrogym.core import PDEBase


class SurrogateHydroEnv(hgym.FlowEnv):
    
    def __init__(self, env_config=None):
        '''
        Schema can be kind of confusing
        env_config:
          hydro_config:             (Hydrogym specific config)
              flow: <hgym.env>
              solver: <hygm.solver>
              flow_config: <dict>
          dyn_model: None           (dynamics model)
          control_freq: 50          (how many inner_steps should execute---a zero-hold control)
          n_skip_on_reset: 50       (how many inner_steps to provide a null action after a reset)
          max_episode_steps: 1000   (how many times to execute an outer step before terminating)
          
          
          TO-DO: time_delay_stack: 5    (dimension of TD embedding)
          TO-DO: Include C_L_dot
          
        '''

        hydro_config = env_config.get('hydro_config', {})
        super(SurrogateHydroEnv, self).__init__(hydro_config)
        self.dyn_model = env_config.get('dyn_model', None)
        self.control_freq = int(env_config.get('control_freq', 50))
        self.n_skip_on_reset = int(env_config.get('n_skip_on_reset', 50)) 
        
        assert self.control_freq >= 1, 'Control frequency must be positive integery'
        
        self.max_episode_steps = env_config.get('max_episode_steps', 1000)
        # self.time_delay_stack = 5
        

    def real_step(self, action):
        self.prev_state = self.state
        state, rew, done, info = super(SurrogateHydroEnv, self).step(action)
        self.state = np.array(state)
        self.real_steps += 1
        return self.state, rew, done, info 
    
    def surrogate_reward(self):
        # This needs to be implemeneted when subclassing
        raise NotImplementedError('Surrogate reward needs to be implemented per environment')
    
    def get_reward(self, **surrogate_kwargs):
        if self.dyn_model: 
            return self.surrogate_reward(**surrogate_kwargs)
        else:
            return super(SurrogateHydroEnv, self).get_reward()

    def get_done(self):
        # if None, return False
        if self.max_episode_steps and (self.n_episode_steps >= self.max_episode_steps):
            return True
        else:
            return False
        
    def action_map(self, action):
        return action
    
    def step(self, action):
        self.prev_state = self.state # keep track of previous state
        
        if self.dyn_model: 
            u = action
            self.state = self.dyn_model.predict(self.state, u)
            rew = self.get_reward()
            done = self.get_done()
            info = {}
        else: 
            # Apply inner step (only in  real env)
            #  `self.control_freq` number of times with constant action
            #  i.e. "zero-hold" control
            for i in range(self.control_freq):
                self.state, rew, done, info = self.real_step(action)
                
        
        self.episode_reward += rew
        self.n_episode_steps += 1
        done = self.get_done()
        
        return self.state, rew, done, info

    def reset(self, **reset_kwargs):
        self.state = np.array(super(SurrogateHydroEnv, self).reset(**reset_kwargs))
        self.prev_state = self.state.copy()
        
        self.episode_reward = 0
        self.n_episode_steps = 0
        self.real_steps = 0
        
        if not self.dyn_model:
            for i in range(self.n_skip_on_reset):
                self.state, rew, done, info = self.real_step(np.zeros(self.action_space.shape))
        
        return np.array(self.state)
    
class SurrogateCylinder(SurrogateHydroEnv):
    def __init__(self, env_config=None):
        # TO DO: generalize the scheme, 
        # i.e. update default flow config
        config = env_config.copy()
        
        default_flow_config =  {
            "flow": hgym.Cylinder,
            "solver": hgym.IPCS,
            "flow_config": {
                'mesh': 'coarse'
            }
        }
        
        config['hydro_config'] = default_flow_config
        self.use_omega = env_config.get('use_omega', False)  # angluar velocity
        self.use_CL_dot = env_config.get('use_CL_dot', False) # Coefficient of lift time derivative

        super(SurrogateCylinder, self).__init__(config)
        self._init_augmented_obs()
        
    def _init_augmented_obs(self):
        obs_dim = self.flow.OBS_DIM + self.use_omega + self.use_CL_dot
        self.observation_space = Box( 
                                     low=-np.inf,
                                     high=np.inf,
                                     shape=(obs_dim,),
                                     dtype=PDEBase.ScalarType,
                                     )
        
    def surrogate_reward(self):
        CL, CD = self.state
        return -1 * CD
    
    def get_omega(self):
        return self.flow.actuators[0].u.values()[0]
    
    def get_CL_dot(self): 
        return (self.state - self.prev_state)[0]/self.solver.dt
    
    def build_augmented_obs(self):
        '''
        Builds augmented obs for the REAL environment
        '''
        obs = [*self.state]
        
        if self.use_omega:
            obs.append(self.get_omega())
        if self.use_CL_dot:
            obs.append(self.get_CL_dot())
        
        return np.array(obs)
    
    def step(self, action):
        '''Overwrite just to build the augmented observation'''
        self.state, rew, done, info = super(SurrogateCylinder, self).step(action)
        
        if self.dyn_model:
            obs = self.state
        else:
            obs = self.build_augmented_obs()
        return obs, rew, done, info 
    
    def reset(self, **reset_kwargs):
        self.state = super(SurrogateCylinder, self).reset(**reset_kwargs)
        if self.dyn_model:

            # For the dynamics model, the state is probably going to be a
            # different shape than the real env. 
            self.state = self.build_augmented_obs()
            self.prev_state = self.state.copy()
        else:
            obs = self.build_augmented_obs()
            return obs
    
if __name__ == '__main__': 
    flow_env_config = {
        "flow": hgym.Cylinder,
        "solver": hgym.IPCS,
        "flow_config": {
            'mesh': 'fine' # coarse, medium, fine
        }
    }
    
    env_config = {
            'hydro_config': flow_env_config,
            'dyn_model': None         
    }
    env = SurrogateHydroEnv(env_config)
    obs = env.reset()
    print(obs)
    for i in range(10):
        print(env.step(0.0))
    