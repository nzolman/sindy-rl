from gym.envs.mujoco.swimmer_v4 import SwimmerEnv
import numpy as np


_DEFAULT_ENV_BOUNDS = np.array([
        [-np.pi, np.pi],
        [-100/180 * np.pi, 100/180*np.pi],
        [-100/180 * np.pi, 100/180*np.pi],
        [-10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, 10.0],
])


class SwimmerWithBounds(SwimmerEnv):
    '''
    The intent of this class is to subclass the Swimmer
    Environment, but reset if we hit certain bounds.
    
    In particular, the environment will reset if we exceed bounds on the angles 
    (where the dynamics become discontinuous)
    '''
    def __init__(self, env_config=None):
        if env_config is None:
            env_config = {}
        env_kwargs = env_config.get('env_kwargs', {})
        super().__init__(**env_kwargs)
        
        self.noise = env_config.get('noise', None) or np.zeros(8)
        self.noise = np.array(self.noise)
        self.max_episode_steps = env_config.get('max_episode_steps', 1000)
        self.reward_threshold = env_config.get('reward_threshold', 360.0)
        
        self.reset_on_bounds = env_config.get('reset_on_bounds', True)
        self.bounds = _DEFAULT_ENV_BOUNDS

        self.n_episode_steps = 0
        
    def get_done(self, state):
        done = bool(
            self.n_episode_steps >= self.max_episode_steps
        )
        
        if self.reset_on_bounds:
            lower_bounds = np.any(state <= _DEFAULT_ENV_BOUNDS.T[0])
            upper_bounds = np.any(state >= _DEFAULT_ENV_BOUNDS.T[1])
            out_of_bounds = lower_bounds or upper_bounds
            done = done or out_of_bounds
        return done
    
    def step(self, action):
        '''
        To do: Figure out when we should be compliant with new vs. old gym API
        '''
        observation, reward, terminated, truncated, info = super().step(action)
        done = self.get_done(observation)
        self.n_episode_steps +=1
        
        observation += self.noise * np.random.normal(loc=0, scale=1, size=(8,))
        return observation, reward, (terminated or done), truncated, info
    
    def reset(self, **kwargs):
        # TO-DO: add noise on initial state
        self.n_episode_steps = 0
        return super().reset(**kwargs)

        
class SwimmerWithBoundsClassic(SwimmerWithBounds):
    def __init__(self, config):
        super().__init__(config)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward, (terminated or truncated), info
    
    def reset(self, **kwargs):
        obs, d = super().reset(**kwargs)
        return obs