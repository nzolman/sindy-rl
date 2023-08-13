import logging
import os
import pickle
import pysindy as ps
import numpy as np
from ray.rllib.algorithms.registry import get_algorithm_class
from gym.wrappers import StepAPICompatibility

from sindy_rl.refactor import registry
from sindy_rl.refactor.policy import RLlibPolicyWrapper
from sindy_rl.refactor.dynamics import EnsembleSINDyDynamicsModel
from sindy_rl.refactor.reward import EnsembleSparseRewardModel
from sindy_rl.refactor.traj_buffer import MaxSamplesBuffer
from sindy_rl.refactor.env import rollout_env, BaseEnsembleSurrogateEnv
from sindy_rl.refactor.ray_utils import update_dyn_and_rew_models
from sindy_rl.refactor.dyna import DynaSINDy

if __name__ == '__main__': 
    import yaml
    import logging
    import ray
    from ray import tune, air
    from gym.wrappers import StepAPICompatibility
    from pprint import pprint
    
    from sindy_rl.refactor.swimmer import SwimmerWithBounds
    from sindy_rl.refactor.policy import RandomPolicy
    from sindy_rl import _parent_dir
    
    filename = '/home/nzolman/projects/sindy-rl/sindy_rl/refactor/baseline_dm_config.yml'
    # filename = '/home/nzolman/projects/sindy-rl/sindy_rl/refactor/baseline_config.yml'
    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    
    ip_head = os.environ.get('ip_head', None)
    ray.init(address=ip_head)
    print(ray.nodes())
    
    
    LOCAL_DIR =  os.path.join(_parent_dir, 'ray_results', config['exp_dir'])
    
    ray_config = config['ray_config']
    run_config=air.RunConfig(
        local_dir=LOCAL_DIR,
        **ray_config['run_config'],
        checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ray_config['checkpoint_freq']),
    )
    
    
    tune_config=tune.TuneConfig(**config['ray_config']['tune_config'])
    
    drl_class, drl_default_config = get_algorithm_class(config['drl']['class'], 
                                                        return_config=True)
    
    drl_config = config['drl']['config']
    drl_default_config.update(drl_config)
    drl_default_config['env'] = getattr(registry, drl_default_config['env'])
    drl_default_config['model']["fcnet_hiddens"] = [64, 64]
    
    tune.Tuner(
        config['drl']['class'],
        param_space=drl_default_config, # this is what is passed to the experiment
        run_config=run_config,
        tune_config=tune_config
    ).fit()
