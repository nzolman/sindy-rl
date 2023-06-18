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


def dyna_sindy(config):
    dyna_config = config
    train_iterations = config['n_train_iter']
    dyn_fit_freq = config['dyn_fit_freq']
    ckpt_freq = config['ray_config']['checkpoint_freq']
    
    # data collected (or loaded) upon initialization
    dyna = DynaSINDy(dyna_config)
    # setup the dynamics, reward, DRL algo push weights to surrogate
    # on remote workers
    dyna.fit_dynamics()
    dyna.fit_rew()
    dyna.update_surrogate()
    dyna.save_checkpoint(ckpt_num=-1, save_dir = tune.get_trial_dir())
    
    collect_dict = {}
    for n_iter in range(train_iterations):
        train_results = dyna.train_algo()
        if (n_iter % ckpt_freq) == ckpt_freq -1:
            dyna.save_checkpoint(ckpt_num=n_iter, save_dir = tune.get_trial_dir())
        if (n_iter % dyn_fit_freq) == dyn_fit_freq - 1:
            (trajs_obs, 
             trajs_acts, 
             trajs_rew) = dyna.collect_data(dyna.on_policy_buffer,
                                            dyna.real_env,
                                            dyna.on_policy_pi,
                                            **dyna_config['on_policy_buffer']['collect']
                                            #**kwargs
                                            )

            dyna.fit_dynamics()
            dyna.fit_rew()
            dyna.update_surrogate()
            
            collect_dict = {}
            collect_dict['mean_rew'] = np.mean([np.sum(rew) for rew in trajs_rew])
            collect_dict['mean_len'] = np.mean([len(obs) for obs in trajs_obs]) 
        train_results['traj_buffer'] = dyna.get_buffer_metrics()
        train_results['dyn_collect'] = collect_dict
        
        tune.report(**train_results)
        


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
    
    filename = '/home/nzolman/projects/sindy-rl/sindy_rl/refactor/dyna_config_cart.yml'
    with open(filename, 'r') as f:
        dyna_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    pprint(dyna_config)
    logging.basicConfig()
    logger = logging.getLogger('dyna-sindy')
    logger.setLevel(logging.INFO)

    # TO-DO: find a better place to automate this
    n_control = dyna_config['dynamics_model']['feature_library']['kwargs']['n_control']
    dyna_config['off_policy_pi'] = RandomPolicy(low=-1*np.ones(n_control), 
                                                high = np.ones(n_control), 
                                                seed=0)
    
    ip_head = os.environ.get('ip_head', None)
    ray.init(address=ip_head)
    print(ray.nodes())
    
    
    LOCAL_DIR =  os.path.join(_parent_dir, 'ray_results',dyna_config['exp_dir'])
    
    ray_config = dyna_config['ray_config']
    run_config=air.RunConfig(
        local_dir=LOCAL_DIR,
        **ray_config['run_config']
        # checkpoint_config=air.CheckpointConfig(checkpoint_frequency=1),
    )
    
    
    tune_config=tune.TuneConfig(**dyna_config['ray_config']['tune_config'])
    
    drl_class, drl_default_config = get_algorithm_class(dyna_config['drl']['class'], 
                                                        return_config=True)
    
    tune.Tuner(
        tune.with_resources(dyna_sindy, 
                            drl_class.default_resource_request(drl_default_config)
                            ),
        param_space=dyna_config, # this is what is passed to the experiment
        run_config=run_config,
        tune_config=tune_config
    ).fit()
