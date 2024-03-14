import logging
import os
import numpy as np
import time

from ray.rllib.algorithms.registry import ALGORITHMS as rllib_algos
from sindy_rl import dynamics, reward

from sindy_rl import registry
from sindy_rl.policy import RLlibPolicyWrapper
from sindy_rl.traj_buffer import MaxSamplesBuffer
from sindy_rl.env import rollout_env, BaseEnsembleSurrogateEnv
from sindy_rl.ray_utils import update_dyn_and_rew_models


class BaseDynaSINDy:
    '''Base class outlining dyna API'''
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
    
    def load_checkpoint(self, air_checkpoint):
        raise NotImplementedError
    
    def update_surrogate(self):
        raise NotImplementedError
    

class DynaSINDy(BaseDynaSINDy):
    '''Main interface for Dyna-style SINDy-RL'''
    def __init__(self, config, logger=None, init_drl=True):
        super().__init__(config=config, logger=logger)
        
        # separate different dictionary configuration
        self.off_pi_buffer_config = config.get('off_policy_buffer')
        self.on_pi_buffer_config = config.get('on_policy_buffer')
        self.dyn_config = self.config['dynamics_model']
        self.rew_config = self.config['rew_model']
        self.drl_config = self.config['drl']['config']
        self.n_dyn_updates = 0
        
        # whether to provide the buffer to the surrogate environment for resetting
        self.reset_from_buffer = config['drl']['config']['environment']['env_config'].get('reset_from_buffer', False)
        
        # initialize components
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
        drl_class_name = self.config['drl']['class']

        # currently only support DRL algorithms
        drl_class, drl_config_obj = rllib_algos.get(drl_class_name)()

        # ensure the surrogate env has the same dynamics/reward models.
        self.drl_config['environment']['env_config'].update({
                                                            'dynamics_model_config': self.dyn_config,
                                                            'rew_model_config': self.rew_config,
                                                            'real_env_config': self.config['real_env']['config'],
                                                            'real_env_class': self.config['real_env']['class'],
                                                        })
        self.drl_config['environment']['env'] = BaseEnsembleSurrogateEnv 
        
        rollout_config = self.drl_config.get('rollouts', {})
        resource_config = self.drl_config.get('resources', {})
        
        # prep the RLlib config object
        drl_config_obj = (drl_config_obj
                            .rl_module(_enable_rl_module_api=False)
                            .environment(**self.drl_config['environment'])
                            .training(**self.drl_config['training'],
                                      _enable_learner_api=False)
                            .evaluation(**self.drl_config['evaluation'])
                            .rollouts(**rollout_config)
                            .resources(**resource_config)
                        )
        
        # configure model architecture
        model_config = drl_config_obj.model
        fcnet_hiddens = self.config.get('fcnet_hiddens', [64, 64])
        model_config.update({'fcnet_hiddens': fcnet_hiddens})
        drl_config_obj.training(model=model_config)
        
        self.drl_algo = drl_config_obj.build()
        self.on_policy_pi = RLlibPolicyWrapper(self.drl_algo)
        self.logger.info('...DRL algo setup.')

    def _init_real_env(self):
        '''Initiliaze a copy of the real environment for evaluation/on-policy collection'''
        if isinstance(self.config['real_env'], dict):
            env_class = getattr(registry, self.config['real_env']['class'])
            env_config = self.config['real_env']['config']
            self.real_env = env_class(env_config)
        else:
            self.real_env = self.config['real_env']
        self.real_env.reset()
        return self.real_env
        
    def _init_off_policy(self):
        '''Initialize the off-policy Policy'''
        self.off_policy_pi = self.config['off_policy_pi']
        
    def _init_dynamics_model(self):
        '''Initialize dynamics model'''
        dynamics_class = getattr(dynamics, self.dyn_config['class'])
        self.dynamics_model = dynamics_class(self.dyn_config['config'])
    
    def _init_rew_model(self):
        '''Initialize reward model'''
        rew_class = getattr(reward, self.rew_config['class'])
        self.rew_model = rew_class(self.rew_config['config'])
        
    def _init_data_buffers(self):
        '''Initialize the data buffers'''
        self.logger.info('Initializing buffers...')
        
        self.off_policy_buffer = MaxSamplesBuffer(self.off_pi_buffer_config['config'])
        self.on_policy_buffer = MaxSamplesBuffer(self.on_pi_buffer_config['config'])
        
        off_policy_init = self.off_pi_buffer_config['init']

        # determine if need to collect data or just load from file
        if off_policy_init['type'] == 'file':
            self.off_policy_buffer.load_data(**off_policy_init['kwargs'])
        else:
            self.collect_data(self.off_policy_buffer,
                              self.real_env,
                              self.off_policy_pi, 
                              **off_policy_init['kwargs'])
            self.logger.info('Data collected.')
            
        if self.reset_from_buffer:
            buffer_dict = self.off_policy_buffer.to_dict()
            self.drl_config['environment']['env_config']['buffer_dict'] = buffer_dict

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
        '''Train algorithm for one 'iteration'''
        return self.drl_algo.train()

    def save_checkpoint(self, ckpt_num, save_dir):
        # setup directories
        ckpt_dir = os.path.join(save_dir, f'checkpoint_{ckpt_num+1:06}')
        checkpoint = self.drl_algo.save(save_dir)
        
        dyn_path = os.path.join(ckpt_dir, 'dyn_model.pkl')
        rew_path = os.path.join(ckpt_dir, 'rew_model.pkl')
        data_path_on = os.path.join(ckpt_dir,'on-pi_data.pkl')
        data_path_off = os.path.join(ckpt_dir,'off-pi_data.pkl')

        self.dynamics_model.save(dyn_path)
        self.rew_model.save(rew_path)

        self.on_policy_buffer.save_data(data_path_on)
        self.off_policy_buffer.save_data(data_path_off)
        
        check_dict = {  'epoch': ckpt_num,
                        'checkpoint_dir': ckpt_dir,
                        'algo_dir': checkpoint,
                        'dyn_model': self.dynamics_model,
                        'rew_model': self.rew_model,
                        'on-policy': self.on_policy_buffer,
                        'off-policy': self.off_policy_buffer
                      }
        return check_dict

    def load_checkpoint(self, check_dict):
        '''
        Restore dyna algorithm from saved checkpoint.
        NOTE: this can be a bit buggy with Tune's PBT
        '''
        ckpt_dir = check_dict['checkpoint_dir']
        algo_dir = check_dict['algo_dir']
        
        self.dynamics_model = check_dict['dyn_model']
        self.rew_model = check_dict['rew_model']
        self.on_policy_buffer = check_dict['on-policy']
        self.off_policy_buffer = check_dict['off-policy']
        

        # Hacky way to deal with the file not existing immediately for PBT...
        # Weirdly this seems to work. 
        n_tries = 12
        dt = 5
        for i in range(n_tries):
            try:
                self.drl_algo.restore(algo_dir)
                
                # not _entirely_ clear this is necessary. But better to be safe
                self.on_policy_pi = RLlibPolicyWrapper(self.drl_algo)
                break
            except FileNotFoundError:
                time.sleep(dt)
                
        self.update_surrogate()
        
        return ckpt_dir

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
        '''Define metrics that are used for Tune reporting, e.g. to tensorboard'''

        n_samples_on_pi = self.on_policy_buffer.total_samples()
        n_samples_off_pi = self.off_policy_buffer.total_samples()
        n_traj_on_pi = len(self.on_policy_buffer)
        n_traj_off_pi = len(self.off_policy_buffer)
        total_samples = self.on_pi_buffer_config['collect']['n_steps'] * (self.n_dyn_updates - 1) + n_samples_off_pi
        
        rews = self.on_policy_buffer.rew_traj_buffer
        if len(rews)!=0:
            last_on_policy_rew = np.sum(rews[-1])
            on_policy_mean_rew =  np.mean([r.sum() for r in rews])
        else:
            last_on_policy_rew = np.nan
            on_policy_mean_rew = np.nan
        
        buffer_metrics = {
            'n_dyn_updates': self.n_dyn_updates, # number of dynamics updates
            'n_samples': n_samples_on_pi + n_samples_off_pi, # current number of samples between the two buffers
            'n_samples_on_pi': n_samples_on_pi,
            'n_sampels_off_pi': n_samples_off_pi,
            'n_traj': n_traj_on_pi  + n_traj_off_pi, # current number of trajectories between the two buffers
            'n_traj_on_pi': n_traj_on_pi,
            'n_traj_off_pi': n_traj_off_pi,
            'n_total_real': total_samples, # total number of samples from the full-order environment
            'last_on_policy_ep_rew': last_on_policy_rew, # last on-policy episode rewar
            'on_policy_ep_mean_rew': on_policy_mean_rew # mean on-policy rew (from the current buffer)
        }
        return buffer_metrics