import ray
import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.sac as sac

from ray.rllib.models.torch.torch_action_dist import TorchBeta,  TorchSquashedGaussian
from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_action_dist("TorchSquashedGaussian", TorchSquashedGaussian)
ModelCatalog.register_custom_action_dist("TorchBeta", TorchBeta)

import os
from ray.tune.logger import pretty_print
from ray import tune, air
from pprint import pprint
import random
import numpy as np
import torch
import pysindy as ps

from sindy_rl.dynamics import CartPoleGymDynamics, SINDyDynamics, EnsembleSINDyDynamics
from sindy_rl.data_utils import collect_random_data, split_by_steps
from sindy_rl.envs.swimmer import SwimmerSurrogate
from sindy_rl import _parent_dir

ALGO = 'PPO'
BASELINE = True

if ALGO == 'PPO':
   config = ppo.DEFAULT_CONFIG.copy()
elif ALGO == 'SAC':
   config = sac.DEFAULT_CONFIG.copy()
   config['model']['custom_action_dist'] = 'TorchSquashedGaussian' 
else:
   raise NotImplementedError('Invalid Algo')

ray.init()

LOCAL_DIR = os.path.join(_parent_dir,'ray_results', 'swimmer')
# reward_threshold = 360.0
reward_threshold = 1000.0
N_EXPERIMENTS = 8
N_STEPS_TRAIN = 4000

EVAL_SEED = 0     # seed for evaluating all environments
TUNE_SEED = 42    # seed for setting Tune
TRAIN_SEED = 1 #tune.randint(1, 1000)  # seed for passing to all envs and policy
COLLECT_SEED = 2 #tune.randint(1, 1000)# seed for collecting initial data


EXP_NAME = f'SINDy_ensemble_ppo_baseline_2022-12-14'

config["num_gpus"] = 0
config["num_workers"] = 1
config['framework'] = 'torch'


# The seed passed to the trainable.
# and for initializing the neural network
config['seed'] = TRAIN_SEED

real_env_config = {'dyn_model': None, 
                   'mod_angles': True
                   }
real_env = SwimmerSurrogate(real_env_config)

def get_swimmer_affine_lib(poly_deg,n_state=11, n_control = 2, poly_int=False, tensor=False):
    polyLib = ps.PolynomialLibrary(degree=poly_deg, 
                                    include_bias=False, 
                                    include_interaction=poly_int)
    affineLib = ps.PolynomialLibrary(degree=1, 
                                    include_bias=False, 
                                    include_interaction=False)

    inputs_per_library = np.array([
                    np.arange(n_state + n_control),
                    (n_state + n_control - 1) * np.ones(n_state + n_control)
                    ], dtype=int)
    inputs_per_library[0,-2:] = 0
    inputs_per_library[1, -2] = n_state + n_control - n_control

    if tensor:
        tensor_array = np.array([[1, 1]])
    else:
        tensor_array = None 

    generalized_library = ps.GeneralizedLibrary(
        [polyLib, affineLib],
        tensor_array=tensor_array,
        inputs_per_library=inputs_per_library,
    )
    return generalized_library


trajs_action, trajs_obs = collect_random_data(real_env, n_steps=10000, seed =0, max_traj_len=100)
null_action, null_obs = collect_random_data(real_env, n_steps=8000, seed =1, use_null = 0*real_env.action_space.sample(), max_traj_len=100)
x_train, u_train, x_test, u_test = split_by_steps(N_train_steps=4000, trajs_action=trajs_action, trajs_obs = trajs_obs)

x_train += null_obs
u_train += null_action

n_state = len(x_train[0][0])
n_control = len(u_train[0][0])
quad_affine_lib = get_swimmer_affine_lib(poly_deg=2, n_state=n_state, n_control=n_control,
                                         poly_int=False,
                                         tensor=True
                                         )

base_optimizer = ps.STLSQ(threshold = 0.02, 
                          alpha = 0.5)

optimizer = ps.EnsembleOptimizer(base_optimizer, 
                                 bagging=True, 
                                 library_ensemble=True,
                                 n_models=100
                                 )

quad_affine_config = {
            # 'dt': true_env.dt,
            'optimizer': optimizer, 
            'feature_library': quad_affine_lib
        }

dyn_experiment_config = {  'is_true': False,
                           'dyn_class': SINDyDynamics,
                           'dyn_model_config': quad_affine_config,
                           'N_steps_collect': int(8e3),
                           'N_steps_train': N_STEPS_TRAIN, # how many steps to fit on
                           'collect_seed': COLLECT_SEED,   # seed for collecting the original data
                           'real_env': real_env, # environment to collect data from
                        }

dyn_model = EnsembleSINDyDynamics(quad_affine_config)
dyn_model.fit(observations=x_train, actions=u_train)

config['env'] = SwimmerSurrogate
config['env_config'] = {'dyn_model': dyn_model, 
                        'max_episode_steps': 1000,
                        'mod_angles': True
                        }
if BASELINE:
   config['env_config'] = real_env_config

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
            # 'episode_reward_mean': reward_threshold
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
   
   # env = config['env'](config['env_config'])
   # print(env.dyn_model.n_models)
   # env.reset()
   # for i in range(10):
   #    print(env.step(env.action_space.sample())[0])


   results = tune.Tuner(
                        # CustomAlgo,
                        ALGO,
                        tune_config=tune_config,
                        param_space=config,
                        run_config=air.RunConfig(
                           name=EXP_NAME,
                           local_dir=LOCAL_DIR,
                           stop=stop_config,
                           checkpoint_config=checkpoint_config
                           )
                        ).fit()