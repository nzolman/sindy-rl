import logging
import os
import pickle
import pysindy as ps
import numpy as np
from ray.rllib.algorithms.registry import get_algorithm_class
from gym.wrappers import StepAPICompatibility

from sindy_rl.refactor import registry, dynamics, reward
from sindy_rl.refactor.policy import RLlibPolicyWrapper
from sindy_rl.refactor.dynamics import EnsembleSINDyDynamicsModel
from sindy_rl.refactor.reward import EnsembleSparseRewardModel
from sindy_rl.refactor.traj_buffer import MaxSamplesBuffer
from sindy_rl.refactor.env import rollout_env, BaseEnsembleSurrogateEnv
from sindy_rl.refactor.ray_utils import update_dyn_and_rew_models


#TO DO:
# [ ] Build off-policy Policy
# [ ] Docstrings
# [ ] Generalize beyond EnsembleSINDy dynamics/reward models. 
#     - Open up the config to include a class name

class BaseDynaSINDy:
    def __init__(self, config, logger = None):
        if logger is None:
            
            self.logger = logging.getLogger()
        else:
            self.logger  = logger
        self.config = config
        
        self.fit_freq = None
        self.drl_algo = None
        self.dynamics_model = None
        self.rew_model = None
        
        self.off_policy_buffer = None
        self.on_policy_buffer = None
        
        self.off_policy_pi = None
        self.on_policy_pi = None
        self.real_env = None
        
    def collect_data(self):
        raise NotImplementedError
    
    def fit_dynamics(self):
        raise NotImplementedError
    
    def fit_rew(self):
        raise NotImplementedError
    
    def train_algo(self):
        raise NotImplementedError
    
    def save_checkpoint(self):
        raise NotImplementedError
    
    def update_surrogate(self):
        raise NotImplementedError
    

class DynaSINDy(BaseDynaSINDy):
    def __init__(self, config, logger=None, init_drl=True):
        super().__init__(config=config, logger=logger)
        self.off_pi_buffer_config = config.get('off_policy_buffer')
        self.on_pi_buffer_config = config.get('on_policy_buffer')
        self.dyn_config = self.config['dynamics_model']
        self.rew_config = self.config['rew_model']
        self.n_dyn_updates = 0
        
        self._init_real_env()
        self._init_off_policy()
        self._init_data_buffers()
        self._init_dynamics_model()
        self._init_rew_model()
        if init_drl:
            self._init_drl()
        
    def _init_drl(self):
        '''Initialize DRL model and on-policy-pi'''
        self.logger.info('Setting up DRL algo...')
        drl_class = self.config['drl']['class']
        self.drl_config = self.config['drl']['config']
        drl_class, drl_default_config = get_algorithm_class(drl_class, 
                                                            return_config=True)

        # ensure the surrogate env has the same dynamics/reward models.
        self.drl_config['env_config'].update({
                            'dynamics_model_config': self.dyn_config,
                            'rew_model_config': self.rew_config,
                            'real_env_config': self.config['real_env']['config'],
                            'real_env_class': self.config['real_env']['class'],
                        })
        
        self.drl_config['env'] = BaseEnsembleSurrogateEnv 
        drl_default_config.update(self.drl_config)
        
        # TO-DO: figure out a better place to put this
        drl_default_config['model']["fcnet_hiddens"] = [64, 64]
        
        self.drl_algo = drl_class(config=drl_default_config)
        self.on_policy_pi = RLlibPolicyWrapper(self.drl_algo)
        self.logger.info('...DRL algo setup.')
        
    def _init_real_env(self):
        '''Initiliaze a copy of the real environment'''
        if isinstance(self.config['real_env'], dict):
            env_class = getattr(registry, self.config['real_env']['class'])
            env_config = self.config['real_env']['config']
            self.real_env = env_class(env_config)
        else:
            self.real_env = self.config['real_env']
        
        self.real_env = StepAPICompatibility(self.real_env, output_truncation_bool=False)
        self.real_env.reset()
        # TO-DO: initialize with seed?
        return self.real_env
        
        
    def _init_off_policy(self):
        '''Initialize the off-policy Policy'''
        # TO-DO: build this
        self.off_policy_pi = self.config['off_policy_pi']
        
    def _init_dynamics_model(self):
        '''Initialize dynamics model'''
        dynamics_class = getattr(dynamics, self.dyn_config['class'])
        self.dynamics_model = dynamics_class(self.dyn_config['config'])
        # self.dynamics_model = EnsembleSINDyDynamicsModel(self.dyn_config)
    
    def _init_rew_model(self):
        '''Initialize reward model'''
        rew_class = getattr(reward, self.rew_config['class'])
        self.rew_model = rew_class(self.rew_config['config'])
        # self.rew_model = EnsembleSparseRewardModel(self.rew_config)
        
    def _init_data_buffers(self):
        '''Initialize the data buffers'''
        self.logger.info('Initializing buffers...')
        
        self.off_policy_buffer = MaxSamplesBuffer(self.off_pi_buffer_config['config'])
        self.on_policy_buffer = MaxSamplesBuffer(self.on_pi_buffer_config['config'])
        
        off_policy_init = self.off_pi_buffer_config['init']

        # determine if need to collect data or just load from file
        if off_policy_init['type'] == 'file':
            self.off_policy_buffer.load(**off_policy_init['kwargs'])
        else:
            self.collect_data(self.off_policy_buffer,
                              self.real_env,
                              self.off_policy_pi, 
                              **off_policy_init['kwargs'])
            self.logger.info('Data collected.')

    def collect_data(self, buffer, env, pi, **rollout_kwargs):
        '''Collect data for a buffer with a given policy'''
        self.logger.info('Collecting data...')
        trajs_obs, trajs_acts, trajs_rews = rollout_env(env, pi, **rollout_kwargs)
        buffer.add_data(trajs_obs, trajs_acts, trajs_rews)
        return trajs_obs, trajs_acts, trajs_rews
    
    def fit_dynamics(self):
        '''Fit the dynamics model'''
        self.logger.info('fitting dynamics...')
        x_off, u_off, rew_off = self.off_policy_buffer.to_list()
        x_on, u_on, rew_on = self.on_policy_buffer.to_list()
        x = x_off + x_on
        u = u_off + u_on
        self.dynamics_model.fit(x, u)

    def fit_rew(self):
        '''The the reward model'''
        self.logger.info('fitting rew...')
        x_off, u_off, rew_off = self.off_policy_buffer.to_list()
        x_on, u_on, rew_on = self.on_policy_buffer.to_list()

        x = x_off + x_on
        u = u_off + u_on
        r = rew_off + rew_on
        
        self.rew_model.fit(x, Y=r, U=u)
        
    def train_algo(self):
        '''Train algorithm for one 'iteration'
        '''
        return self.drl_algo.train()
        
    
    def save_checkpoint(self, ckpt_num, save_dir):
        # setup directories
        # TO-DO: ensure this ckpt_num is the right number/format.
        # NOTE: this structure seems to have changed in newer versions of RLlib.
        # (e.g. ray==2.5.0)
        
        ckpt_dir = os.path.join(save_dir, f'checkpoint_{ckpt_num+1:06}')
        dyn_path = os.path.join(ckpt_dir, 'dyn_model.pkl')
        rew_path = os.path.join(ckpt_dir, 'rew_model.pkl')
        data_path = os.path.join(ckpt_dir,'on-pi_data.pkl')
        
        checkpoint = self.drl_algo.save(save_dir)
        
        with open(dyn_path, 'wb') as f:
            pickle.dump(self.dynamics_model, f)
            
        with open(rew_path, 'wb') as f:
            pickle.dump(self.rew_model, f)
            
        self.on_policy_buffer.save_data(data_path)
        return save_dir
    
    def update_surrogate(self):
        '''Updating surrogate model'''
        self.logger.info('Updating worker weights')
        dyn_weights = self.dynamics_model.get_coef_list()
        
        rew_weights = None
        if self.rew_model.can_update:
            rew_weights = self.rew_model.get_coef_list()
            
        self.drl_algo.workers.foreach_worker(update_dyn_and_rew_models(dyn_weights, rew_weights))
        self.n_dyn_updates += 1

    def get_buffer_metrics(self):
        n_samples_on_pi = self.on_policy_buffer.total_samples()
        n_samples_off_pi = self.off_policy_buffer.total_samples()
        n_traj_on_pi = len(self.on_policy_buffer)
        n_traj_off_pi = len(self.off_policy_buffer)
        total_samples = self.on_pi_buffer_config['collect']['n_steps'] * (self.n_dyn_updates - 1) + n_samples_off_pi
        buffer_metrics = {
            'n_dyn_updates': self.n_dyn_updates,
            'n_samples': n_samples_on_pi + n_samples_off_pi,
            'n_samples_on_pi': n_samples_on_pi,
            'n_sampels_off_pi': n_samples_off_pi,
            'n_traj': n_traj_on_pi  + n_traj_off_pi,
            'n_traj_on_pi': n_traj_on_pi,
            'n_traj_off_pi': n_traj_off_pi,
            'n_total_real': total_samples
        }
        return buffer_metrics