import ray
import ray.rllib.algorithms.ppo as ppo
import os
from ray.tune.logger import pretty_print
from ray import tune, air
from pprint import pprint
import random
import numpy as np
import torch
import pysindy as ps

from sindy_rl.dynamics import CartPoleGymDynamics, SINDyDynamics
from sindy_rl.data_utils import collect_random_data, split_by_steps
from sindy_rl.envs.swimmer import SwimmerSurrogate
from sindy_rl import _parent_dir

ray.init()

LOCAL_DIR = os.path.join(_parent_dir,'ray_results', 'swimmer')
reward_threshold = 360.0
N_EXPERIMENTS = 8
N_STEPS_TRAIN = 5000

EVAL_SEED = 0     # seed for evaluating all environments
TUNE_SEED = 42    # seed for setting Tune
TRAIN_SEED = 1 #tune.randint(1, 1000)  # seed for passing to all envs and policy
COLLECT_SEED = 2 #tune.randint(1, 1000)# seed for collecting initial data


EXP_NAME = f'SINDy_cub_affine_25_test_{N_STEPS_TRAIN}'

config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config['framework'] = 'torch'


# The seed passed to the trainable.
# and for initializing the neural network
config['seed'] = TRAIN_SEED

real_env_config = {'dyn_model': None}
real_env = SwimmerSurrogate(real_env_config)

def get_swimmer_affine_lib(poly_deg):
   polyLib = ps.PolynomialLibrary(degree=poly_deg, 
                                 include_bias=False, 
                                 include_interaction=False)
   affineLib = ps.PolynomialLibrary(degree=1, 
                                 include_bias=False, 
                                 include_interaction=False)

   #  n_state = 10
   n_state = 8
   n_control = 2

   inputs_per_library = np.array([
                  np.arange(n_state + n_control),
                  (n_state + n_control - 1) * np.ones(n_state + n_control)
                  ], dtype=int)
   inputs_per_library[0,-2:] = 0
   inputs_per_library[1, -2] = n_state + n_control - n_control

   tensor_array = np.array([[1, 1]])

   generalized_library = ps.GeneralizedLibrary(
      [polyLib, affineLib],
      tensor_array=tensor_array,
      inputs_per_library=inputs_per_library,
   )
   return generalized_library

quad_affine_lib = get_swimmer_affine_lib(poly_deg=3)
quad_affine_config = {
            'dt': real_env.dt,
            'optimizer': 'STLSQ',
            'optimizer_kwargs': {
                'threshold': 0.02,
                'alpha': 0.5
            },
            'feature_library': quad_affine_lib
        }


dyn_experiment_config = {  'is_true': False,
                           'dyn_class': SINDyDynamics,
                           'dyn_model_config': quad_affine_config,
                           'N_steps_collect': int(1e4),
                           'N_steps_train': N_STEPS_TRAIN, # how many steps to fit on
                           'collect_seed': COLLECT_SEED,   # seed for collecting the original data
                           'real_env': real_env, # environment to collect data from
                        }

# config['dyn_experiment_config'] = dyn_experiment_config


# class CustomAlgo(ppo.PPO):
#    '''
#    Just serves as a wrapper for the PPO trainable.
#    '''
#    def __init__(self, config = None, env = None, logger_creator = None, **kwargs):
#       dyn_config = config.get('dyn_experiment_config', None)

#       if dyn_config:
#          self.dyn_experiment_config = dyn_config
         
#          is_true_dyn = dyn_config.get('is_true', False)
         
#          if is_true_dyn:
#             config['env_config'] = {'dyn_model': None}
#          else: 
#             config['env_config'] = {'dyn_model': self.get_dyn_model(dyn_config)}

#          config.pop('dyn_experiment_config')
#       super().__init__(config, env, logger_creator, **kwargs)
   
#    def get_dyn_model(self, dyn_config):
#       N_steps_collect = dyn_config['N_steps_collect']
#       N_steps_train = dyn_config['N_steps_train']
#       seed = dyn_config['collect_seed']
#       real_env = dyn_config['real_env']
#       trajs_action, trajs_obs = collect_random_data(real_env, N_steps_collect, seed=seed)
#       x_train, u_train, x_test, u_test = split_by_steps(N_steps_train, trajs_action, trajs_obs)

#       # Train Dynamics Model
#       dyn_model_config = dyn_config['dyn_model_config']
#       dyn_class = dyn_config['dyn_class']
#       dyn_model = dyn_class(dyn_model_config)
      
#       dyn_model.fit(observations = x_train, actions=u_train)
#       self.dyn_model = dyn_model
#       return dyn_model

seed = dyn_experiment_config['collect_seed']
N_steps_collect = dyn_experiment_config['N_steps_collect']
N_steps_train = dyn_experiment_config['N_steps_train']

trajs_action, trajs_obs = collect_random_data(real_env, N_steps_collect, seed=seed)
x_train, u_train, x_test, u_test = split_by_steps(N_steps_train, trajs_action, trajs_obs)

dyn_model = SINDyDynamics(quad_affine_config)
dyn_model.fit(observations=x_train, actions=u_train)

config['env'] = SwimmerSurrogate
config['env_config'] = {'dyn_model': dyn_model, 'max_episode_steps': 25}
# config['env_config'] = {'dyn_model': None}

# Example: overriding env_config, exploration, etc:
config['evaluation_interval'] = 1
config["evaluation_config"] ={
                              "env_config": real_env_config,
                              "explore": False,
                              'seed': EVAL_SEED
                              }
if __name__ == '__main__': 
   stop_config = {
            # "training_iteration": 3, 
            'episode_reward_mean': reward_threshold
         }

   checkpoint_config = air.CheckpointConfig(checkpoint_at_end=True, 
                                             checkpoint_frequency=10)

   tune_config=tune.TuneConfig(num_samples=N_EXPERIMENTS)



   # this needs to be set to make sure the 
   #  tune.Tuner() creates the same configs
   #  between ray.init() runs. 
   random.seed(TUNE_SEED)
   np.random.seed(TUNE_SEED)
   torch.manual_seed(TUNE_SEED)

   results = tune.Tuner(
                        # CustomAlgo,
                        "PPO",
                        tune_config=tune_config,
                        param_space=config,
                        run_config=air.RunConfig(
                           name=EXP_NAME,
                           local_dir=LOCAL_DIR,
                           stop=stop_config,
                           checkpoint_config=checkpoint_config
                           )
                        ).fit()