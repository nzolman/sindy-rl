import numpy as np
from gym.spaces import Box
from hydrogym import firedrake as hgym
from hydrogym.core import PDEBase
import firedrake as fd
import matplotlib.pyplot as plt


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
          
        Attributes:
            hydro_state, prev_hydro state: 
                the hydrogym state and previous hydrogym states. 
                Can be used to calculate differences, etc. 
            state:
                the observation to be passed out of the step function.
                This is true for both the dynamics model and the 
                real environment
        '''
        self.config = env_config
        hydro_config = env_config.get('hydro_config', {})
        super(SurrogateHydroEnv, self).__init__(hydro_config)
        self.dyn_model = env_config.get('dyn_model', None)
        self.control_freq = int(env_config.get('control_freq', 50))
        self.n_skip_on_reset = int(env_config.get('n_skip_on_reset', 50)) 
        
        assert self.control_freq >= 1, 'Control frequency must be positive integery'
        
        self.max_episode_steps = env_config.get('max_episode_steps', 1000)

    def real_step(self, action):
        '''
        Take a step in the hydrogym environment
        '''
        # keep track of previous state
        self.prev_hydro_state = self.hydro_state
        hydro_state, rew, done, info = super(SurrogateHydroEnv, self).step(action)
        self.hydro_state = np.array(hydro_state)
        
        self.n_real_steps += 1
        return self.hydro_state, rew, done, info 
    
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
        # placeholder function in case action needs to be remapped
        # (TO DO: Figure out how to get rid of this from pipeline)
        return action
    
    def step(self, action):
        
        # take action in surrogate environment
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
                self.hydro_state, rew, done, info = self.real_step(action)
            
            self.state = self.hydro_state

        self.episode_reward += rew
        self.n_episode_steps += 1
        done = self.get_done()
        
        return self.state, rew, done, info

    def reset(self, seed=0, **reset_kwargs):
        '''
        TO DO: Do _something_ with `seed`!
        '''
        # extract hydrogym state from FlowEnv reset
        self.hydro_state = np.array(super(SurrogateHydroEnv, self).reset(**reset_kwargs))
        self.prev_hydro_state = self.hydro_state.copy()
        
        # book-keeping quantities
        self.episode_reward = 0
        self.n_episode_steps = 0
        self.n_real_steps = 0
        
        if not self.dyn_model:
            # sometimes beneficial to skip the first number of steps
            for i in range(self.n_skip_on_reset):
                self.hydro_state, rew, done, info = self.real_step(np.zeros(self.flow.ACT_DIM))
        
        # set the state to the most "recent" hydro_state
        self.state = self.hydro_state
        
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
        
        flow_config = config.get('hydro_config', {})
        default_flow_config.update(flow_config)
        config['hydro_config'] = default_flow_config
        
        self.use_omega = config.get('use_omega', False)  # angluar velocity
        self.use_CL_dot = config.get('use_CL_dot', False) # Coefficient of lift time derivative
        self.obs_clip_val = 100
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
        # TO-DO: Figure out how to relax this reward condition
        CL, CD = self.state[:2]
        # !!!! JUST FOR TESTING
        ref = 1
        return -1 * np.abs(CL- ref)
    
    def get_omega(self):
        return self.flow.actuators[0].u.values()[0]
    
    def get_CL_dot(self): 
        # only to be used by the real env!
        return (self.hydro_state - self.prev_hydro_state)[0]/self.solver.dt
    
    def build_augmented_obs(self):
        '''
        Builds augmented obs for the REAL environment
        '''
        obs = [*self.hydro_state]
        
        if self.use_omega:
            obs.append(self.get_omega())
        if self.use_CL_dot:
            obs.append(self.get_CL_dot())
        
        return np.array(obs)
    
    def clip_obs(self, state):
        '''
        Clips Observation between bounds
        '''
        return np.clip(state, -self.obs_clip_val, self.obs_clip_val)
    
    def step(self, action):
        '''Overwrite just to build the augmented observation'''
        self.state, rew, done, info = super(SurrogateCylinder, self).step(action)
        
        if not self.dyn_model: 
            # augment the observation
            # NOTE: this isn't needed when there's a dynamics
            # model because the state should already have the full
            # size
            self.state = self.build_augmented_obs()
        
        if self.obs_clip_val:
            self.state = self.clip_obs(self.state)
            
        # !!!!NOTE THIS WAS JUST FOR TESTING!!!
        rew = self.surrogate_reward()
        return self.state, rew, done, info 
    
    def reset(self, **reset_kwargs):
        self.state = super(SurrogateCylinder, self).reset(**reset_kwargs)
        self.state = self.build_augmented_obs()
        return self.state

class SurrogatePinball(SurrogateHydroEnv):
    def __init__(self, env_config=None):
        # TO DO: generalize the scheme, 
        # i.e. update default flow config
        config = env_config.copy()
        
        default_flow_config =  {
            "flow": hgym.Pinball,
            "solver": hgym.IPCS,
            "flow_config": {
                'mesh': 'coarse'
            }
        }
        
        flow_config = config.get('hydro_config', {})
        default_flow_config.update(flow_config)
        config['hydro_config'] = default_flow_config
        
        self.use_omega = config.get('use_omega', False)  # angluar velocity
        self.use_CL_dot = config.get('use_CL_dot', False) # Coefficient of lift time derivative
        self.obs_clip_val = 100
        self.MAX_TORQUE = config.get('max_torque', 0.25)
        self.ACTION_MAX = config.get('action_bound', 10.0)
        self.CL_REF = np.array([0.0, 1.0, -1.0])
        super(SurrogatePinball, self).__init__(config)
        self._init_augmented_obs()
        
        # control config is an integer and dictates the configuration to input control.
        # if `control_mode` = 1:
        #  this actuates like (0, action, -1 * action).
        # if `control_mode` = 2:
        #  this actuates like (0, action[0], action[1])
        # if `control_mode' = 3:
        #  this is the _full_ actuation (action[0], action[1], action[2])
        self._control_mode = config.get('control_mode', 1)
        self._init_action_space()
             
    def _init_augmented_obs(self):
        # TO-DO: probably clean this up for the pinball
        obs_dim = 6 + 3*self.use_omega + 3*self.use_CL_dot
        self.observation_space = Box( 
                                     low=-np.inf,
                                     high=np.inf,
                                     shape=(obs_dim,),
                                     dtype=PDEBase.ScalarType,
                                     )
    def _init_action_space(self):
        act_dim = self._control_mode
        self.action_space = Box(low = -self.ACTION_MAX,
                                high = self.ACTION_MAX,
                                shape = (act_dim, ),
                                dtype=self.flow.ScalarType,
                                )
    
    def surrogate_reward(self):
        # TO-DO: Figure out how to relax this reward condition
        rew = 0
        CL = self.state[:3]
        CD = self.state[3:6] 
        
        # !!!! JUST FOR TESTING
        track_penalty = -1 * np.linalg.norm(CL - self.CL_REF)
        rew += track_penalty
        
        if self.use_CL_dot:
            CL_dot_coef = 1e-2
            CL_dot = self.state[-3:]
            dot_pen = -CL_dot_coef * np.linalg.norm(CL_dot)
            rew += dot_pen
        return rew
    
    def get_omega(self):
        return np.array([actuator.u.values()[0] for actuator in self.flow.actuators])
    
    def get_CL_dot(self): 
        # only to be used by the real env!
        return (self.hydro_state - self.prev_hydro_state)[0:3]/self.solver.dt
    
    def build_augmented_obs(self):
        '''
        Builds augmented obs for the REAL environment
        '''
        obs = [*self.hydro_state]
        
        if self.use_omega:
            obs += list(self.get_omega())
        if self.use_CL_dot:
            obs +=  list(self.get_CL_dot())
        
        return np.array(obs)
    
    def clip_obs(self, state):
        '''
        Clips Observation between bounds
        '''
        return np.clip(state, -self.obs_clip_val, self.obs_clip_val)
    
    def policy_to_physical(self, action):
        '''
        Bound the output of the policy to the output of the maximum torque
        '''
        scale_factor = self.MAX_TORQUE/self.ACTION_MAX
        return scale_factor*action
    
    def action_map(self, action):
        '''
        Maps `action` (dim=1,2,3) based off the control mode and
            to the corresponding `new_action` with three dimensions.
        '''
        if self._control_mode == 1:
            try:
                # handle if action is a single-element array
                action = action[0]
            except TypeError:
                action = action
            new_action = np.array([0, action, -1 * action])

        elif self._control_mode == 2:
            assert len(action) == 2, "control mode does not match action length"
            new_action = np.array([0, *action])

        elif self._control_mode == 3:
            assert len(action) == 3, "control mode does not match action length"
            new_action = action

        physical_action = self.policy_to_physical(new_action)
        return physical_action
    
    def step(self, action):
        '''Overwrite just to build the augmented observation'''
        self.action = action
        self.physical_action = self.action_map(self.action)
        self.state, rew, done, info = super(SurrogatePinball, self).step(self.physical_action)
        
        if not self.dyn_model: 
            # augment the observation
            # NOTE: this isn't needed when there's a dynamics
            # model because the state should already have the full
            # size
            self.state = self.build_augmented_obs()
        
        if self.obs_clip_val:
            self.state = self.clip_obs(self.state)
            
        # !!! NOTE THE INTENT OF THIS WAS JUST FOR TESTING!!!
        rew = self.surrogate_reward()
        return self.state, rew, done, info 
    
    def reset(self, **reset_kwargs):
        self.state = super(SurrogatePinball, self).reset(**reset_kwargs)
        self.state = self.build_augmented_obs()
        return self.state

    def render(self, mode="human", clim=None, levels=None, cmap="RdBu", **kwargs):
        if clim is None:
            clim = (-2, 2)
        if levels is None:
            levels = np.linspace(*clim, 10)
        vort = fd.project(fd.curl(self.flow.u), self.flow.pressure_space)
        im = fd.tricontourf(
            vort,
            cmap=cmap,
            levels=levels,
            vmin=clim[0],
            vmax=clim[1],
            extend="both",
            **kwargs,
        )
        
        # this is broken
        # for (x0, y0) in zip(self.flow.x0, self.flow.y0):
        #     cyl = plt.Circle((x0, y0), self.flow.rad, edgecolor="k", facecolor="gray")
        #     im.axes.add_artist(cyl)
    
        return im



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
    