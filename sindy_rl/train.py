import ray
import ray.rllib.algorithms.ppo as ppo
import os
from ray.tune.logger import pretty_print
from ray import tune, air
from pprint import pprint
import random
import numpy as np
import torch

from sindy_rl.dynamics import CartPoleGymDynamics, SINDyDynamics
from sindy_rl.data_utils import collect_random_data, split_by_steps
from sindy_rl.environment import CartSurrogate
from sindy_rl import _parent_dir

ray.init()

LOCAL_DIR = os.path.join(_parent_dir,'ray_results')
reward_threshold = 475
N_EXPERIMENTS = 8
N_STEPS_TRAIN = 25

EVAL_SEED = 0     # seed for evaluating all environments
TUNE_SEED = 42    # seed for setting Tune
TRAIN_SEED = tune.randint(1, 1000)  # seed for passing to all envs and policy
COLLECT_SEED = tune.randint(1, 1000)# seed for collecting initial data


EXP_NAME = f'SINDy_{N_STEPS_TRAIN}'

config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config['framework'] = 'torch'


# The seed passed to the trainable.
# and for initializing the neural network
config['seed'] = TRAIN_SEED

real_env_config = {'dyn_model': CartPoleGymDynamics()}
real_env = CartSurrogate(real_env_config)

dyn_experiment_config = {
                           'dyn_class': SINDyDynamics,
                           'dyn_model_config': {
                              'dt': 0.02,
                              'optimizer': 'STLSQ',
                              'optimizer_kwargs': {
                                 'threshold': 0.01,
                                 'alpha': 0.5
                              },
                           },
                           'N_steps_collect': int(1e4),
                           'N_steps_train': N_STEPS_TRAIN, # how many steps to fit on
                           'collect_seed': COLLECT_SEED,   # seed for collecting the original data
                           'real_env': real_env, # environment to collect data from

                        }

config['dyn_experiment_config'] = dyn_experiment_config

# TO-DO: 
# - provide a way to actually restore the dyn_config.
#     either by creating custom callback, restore(), or manually. 

class CustomAlgo(ppo.PPO):
   '''
   Just serves as a wrapper for the PPO trainable.
   '''
   def __init__(self, config = None, env = None, logger_creator = None, **kwargs):
      dyn_config = config.get('dyn_experiment_config', None)

      if dyn_config:
         self.dyn_experiment_config = dyn_config
         
         is_true_dyn = dyn_config.get('is_true', False)
         
         if is_true_dyn:
            config['env_config'] = {'dyn_model': CartPoleGymDynamics()}
         else: 
            config['env_config'] = {'dyn_model': self.get_dyn_model(dyn_config)}

         config.pop('dyn_experiment_config')
      super().__init__(config, env, logger_creator, **kwargs)
   
   def get_dyn_model(self, dyn_config):
      N_steps_collect = dyn_config['N_steps_collect']
      N_steps_train = dyn_config['N_steps_train']
      seed = dyn_config['collect_seed']
      real_env = dyn_config['real_env']
      trajs_action, trajs_obs = collect_random_data(real_env, N_steps_collect, seed=seed)
      x_train, u_train, x_test, u_test = split_by_steps(N_steps_train, trajs_action, trajs_obs)

      # Train Dynamics Model
      dyn_model_config = dyn_config['dyn_model_config']
      dyn_class = dyn_config['dyn_class']
      dyn_model = dyn_class(dyn_model_config)
      
      dyn_model.fit(observations = x_train, actions=u_train)
      self.dyn_model = dyn_model
      return dyn_model

config['env'] = CartSurrogate
config['env_config'] = {}


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
                        CustomAlgo,
                        tune_config=tune_config,
                        param_space=config,
                        run_config=air.RunConfig(
                           name=EXP_NAME,
                           local_dir=LOCAL_DIR,
                           stop=stop_config,
                           checkpoint_config=checkpoint_config
                           )
                        ).fit()