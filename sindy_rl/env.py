import gymnasium
from gymnasium.spaces.box import Box
import numpy as np
from tqdm import tqdm
from sindy_rl import dynamics 

from sindy_rl import reward
from sindy_rl import registry

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
    
def replace_with_inf(arr, neg):
    '''helper function to replace an array with inf. Used for Box bounds'''
    replace_with = np.inf
    if neg:
        replace_with *= -1
    return np.nan_to_num(arr, nan=replace_with)


def rollout_env(env, policy, n_steps, n_steps_reset=np.inf, seed=None, verbose = False, env_callback = None):
    '''
    Step through an environment and produce rollouts.
    Arguments:
        env: gymnasium environment
        policy: sindy_rl.BasePolicy subclass
            The policy to rollout experience
        n_steps: int
            number of steps of experience to rollout
        n_steps_reset: int
            number of steps of experience to rollout before reset
        seed: int
            environment reset seed 
        verbose: bool
            whether to provide tqdm progress bar
        env_callback: fn(idx, env)
            optional function that is called after every step
    Returns:
        lists of obs, acts, rews trajectories
    '''
    if seed is not None:    
        obs_list = [safe_reset(env.reset(seed=seed))]
    else:
        obs_list = [safe_reset(env.reset())]

    act_list = []
    rew_list = []
    
    trajs_obs = []
    trajs_acts = []
    trajs_rews = []

    for i in tqdm(range(n_steps), disable=not verbose):
        
        # collect experience
        action = policy.compute_action(obs_list[-1])
        obs, rew, done, info = safe_step(env.step(action))
        act_list.append(action)
        obs_list.append(obs)
        rew_list.append(rew)
        
        # handle resets
        if done or len(obs_list) > n_steps_reset:
            obs_list.pop(-1)
            trajs_obs.append(np.array(obs_list))
            trajs_acts.append(np.array(act_list))
            trajs_rews.append(np.array(rew_list))

            obs_list = [safe_reset(env.reset())]
            act_list = []
            rew_list = []
        
        # env callback
        if env_callback:
            env_callback(i, env)
    
    # edge case if never hit reset
    if len(act_list) != 0:
        obs_list.pop(-1)
        trajs_obs.append(np.array(obs_list))
        trajs_acts.append(np.array(act_list))
        trajs_rews.append(np.array(rew_list))
    return trajs_obs, trajs_acts, trajs_rews


class BaseSurrogateEnv(gymnasium.Env):
    '''
    Base class for a surrogate environment. Essentially wraps real environment 
    and uses surrogate functions for the dynamics/rewards
    
    NOTE: original intent of allowing real-env is to be able to 
    use callbacks and/or intialize an evaluation environment
    using RLLib. It was not strictly necessary to use. HOWEVER,
    need a way to reset the environment, and the real-env makes that
    extremely easy.
    '''
    def __init__(self, config):
        '''config: dictionray config defining all aspects of the environment'''
        self.config = config
        
        # extract information from the config
        self.use_old_api = self.config.get('use_old_api', False) # True if using old openai gym API. Would not recommend.
        self.max_episode_steps = self.config.get('max_episode_steps', 1000)
        self.n_episode_steps = 0
        
        self.real_env_class = self.config.get('real_env_class', None)
        self.real_env_config = self.config.get('real_env_config', {})
        self.dynamics_model_config = self.config['dynamics_model_config']
        self.rew_model_config = self.config['rew_model_config']
        self._init_weights = self.config.get('init_weights', False)
        
        # whether to reset the surrogate environment by sampling a buffer
        # of previously collected data. Usually off-policy. The intention
        # is to make resetting go very fast by bypassing the full-order environment.
        self.reset_from_buffer = self.config.get('reset_from_buffer', False)
        self.buffer_dict = self.config.get('buffer_dict', {})
        
        self.use_real_env = False
        self.real_env = False
        self.obs = None
        self.action = None
        
        # whether to initiliaze the real environment.
        # it is recommended to avoid initializing expensive full-order models
        # and instead sample initial conditions from a buffer.
        if self.config.get('init_real_on_start', False):
            self.init_real_env()
        
        # whether to use the real environment instead of the surrogate
        # can be useful for debugging or evaluating
        if self.config.get('use_real_env', False):
            self.use_real_env = True
    
    def _init_act(self):
        '''Initialize the action space for the surrogate environment'''
        self.act_dim = self.config['act_dim']
        self.act_bounds = np.array(self.config.get('act_bounds'), dtype=float).T   
        self.act_bounds[0] = replace_with_inf(self.act_bounds[0], neg=True)
        self.act_bounds[1] = replace_with_inf(self.act_bounds[1], neg=False)
        self.action_space = Box(low=self.act_bounds[0], 
                                high = self.act_bounds[1])
        
    def _init_obs(self):
        '''Initiliaze the observation space for the surrogate environment'''
        self.obs_dim = self.config['obs_dim']
        self.obs_bounds = np.array(self.config.get('obs_bounds'), dtype=float).T   
        self.obs_bounds[0] = replace_with_inf(self.obs_bounds[0], neg=True)
        self.obs_bounds[1] = replace_with_inf(self.obs_bounds[1], neg=False)
        
        self.observation_space = Box(low= -np.inf * np.ones(self.obs_dim), #self.obs_bounds[0], 
                                    high = np.inf * np.ones(self.obs_dim) # self.obs_bounds[1])
                                    )
        
    def _init_dynamics_model(self):
        '''Initialize dynamics model'''
        
        # grab dynamics class from sindy_rl.dynamics
        dynamics_class = getattr(dynamics, self.dynamics_model_config['class'])
        self.dynamics_model = dynamics_class(self.dynamics_model_config['config'])
        
        # init weights
        # fitting is just to make things play nice until the model is actually fit.
        if self._init_weights: 
            x_tmp = np.ones((10, self.obs_dim))
            u_tmp = np.ones((10, self.act_dim))
            self.dynamics_model.fit([x_tmp], [u_tmp])

    def _init_rew_model(self):
        '''Initialize reward model'''
        
        # grab reward class from sindy_rl.reward
        rew_class = getattr(reward, self.rew_model_config['class'])
        self.rew_model = rew_class(self.rew_model_config['config'])
        
        # init weights
        # fitting is just to make things play nice until the model is actually fit.
        if self._init_weights: 
            x_tmp = np.ones((10, self.obs_dim))
            u_tmp = np.ones((10, self.act_dim))
            r_tmp = np.ones((10, 1))
            self.rew_model.fit([x_tmp], U=[u_tmp], Y=[r_tmp])

    def switch_on_real_env_(self):
        '''Use real env for step updates'''
        self.use_real_env = True
        
    def switch_off_real_env_(self):
        '''Use surrogate env for step updates'''
        self.use_real_env = False

    def init_real_env(self, env=None, reset=True, **reset_kwargs):
        '''
        Initialize real environment (for computation purposes, may not always want to init)
        
        Inputs:
            env: (gym.env-like)
                the real environment with gym.evn-like API (step, reset, etc.)
            reset: (bool)
                whether to reset the environment upon initialization
            reset_kwargs:
                seed: (int)
                    the seed to reset the enviornment. Only used if `reset`==True
        '''
        if env:
            self.real_env = env
        else:
            if isinstance(self.real_env_class, str):
                self.real_env_class = getattr(registry, self.real_env_class)
            
            self.real_env = self.real_env_class(self.real_env_config)

        if reset:
            self.real_env.reset(**reset_kwargs)
        return self.real_env 
    
    def is_trunc(self):
        '''Computes whether the episode has truncated'''
        raise NotImplementedError
    
    def is_term(self):
        '''Computes whether the episode has terminated'''
        raise NotImplementedError
    
    def _real_step(self, action):
        '''Use the full-order environment step'''
        res = self.real_env.step(action)
        if self.use_old_api:
            return safe_step(res)
        else:
            
            obs = res[0]
            rew = res[1]
            term = res[2]
            info = res[-1]
            return obs, rew, self.is_term(), self.is_trunc(), info 
    
    def step(self, action):
        '''Steps through the full-order or surrogate environment'''
        self.action = action
        self.n_episode_steps +=1
        
        if self.use_real_env:
            return self._real_step(self.action)
        
        next_obs = self.dynamics_model.predict(self.obs, self.action)
        rew = self.rew_model.predict(next_obs, self.action)
        
        self.obs = next_obs
        info = {}
        
        if self.use_old_api:
            done = self.is_trunc() or self.is_term()
            return self.obs, rew, done, info
        else:
            return self.obs, rew, self.is_term(), self.is_trunc(), info

    
    def reset(self, **reset_kwargs):
        '''Resets the environment'''
        self.n_episode_steps = 0
        
        if not self.reset_from_buffer:
            self.obs = safe_reset(self.real_env.reset(**reset_kwargs))
        
        else:
            buffer_obs = np.concatenate(self.buffer_dict['x'])
            buffer_idx = np.random.choice(len(buffer_obs))
            self.obs = buffer_obs[buffer_idx]
        
        info = {}
        if self.use_old_api:
            return self.obs
        else:
            return self.obs, info



class BaseEnsembleSurrogateEnv(BaseSurrogateEnv):
    '''Wrapper to initialize surrogate environment with Ensemble dynamics models'''
    def __init__(self, config):
        super().__init__(config)
        
        self.ensemble_modes = self.config.get('ensemble_modes', {'dyn': 'median', 
                                                                 'rew': 'median'})
        self.use_bounds = self.config.get('use_bounds', True)
        self._init_act()
        self._init_obs()
        
        self._init_dynamics_model()
        self._init_rew_model()
        self.set_ensemble_mode_(modes=self.ensemble_modes)
    
    def set_ensemble_mode_(self, modes=None, valid=False):
        '''Set the behavior of the ensemble model'''
        self.ensemble_modes = modes
        
        models = {'dyn': self.dynamics_model, 'rew': self.rew_model}
        mode_mapping = {'sample': 'set_rand_coef_', 
                        'mean': 'set_mean_coef_',
                        'median': 'set_median_coef_'}
        
        for model_name, mode in self.ensemble_modes.items():
            assert mode in ['sample', 'mean', 'median', None], f'Invalid dynamcis mode: {mode}'
            model = models[model_name]
            
            if mode is None:
                # Do Nothing
                continue
            else:
                # get he appropriate mapping and apply it
                fn_ = getattr(model, mode_mapping[mode])
                fn_(valid=valid)

  
    def update_models_(self, dynamics_weights=None, reward_weights=None):
        '''Wrapper for setting different model weights'''
        if dynamics_weights:
            self.dynamics_model.set_ensemble_coefs_(dynamics_weights)
        if reward_weights:
            self.rew_model.set_ensemble_coefs_(reward_weights)
        self.set_ensemble_mode_(modes=self.ensemble_modes)
    
    
    def is_trunc(self):
        trunc = (self.n_episode_steps >= self.max_episode_steps)
        return trunc
    
    def is_term(self):
        term = False
        if self.use_bounds:
            lower_bounds_done = np.any(self.obs <= self.obs_bounds[0])
            upper_bounds_done = np.any(self.obs >= self.obs_bounds[1])
            term = lower_bounds_done or upper_bounds_done
        return term
        
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        # resample if necessary
        self.set_ensemble_mode_(modes=self.ensemble_modes)
        return obs