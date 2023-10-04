import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import random

from ray.rllib.algorithms.registry import ALGORITHMS as rllib_algos
from ray.tune.schedulers import PopulationBasedTraining

from sindy_rl.refactor import registry

if __name__ == '__main__': 
    import yaml
    import logging
    import ray
    from ray import tune, air
    
    from sindy_rl import _parent_dir
    
    filename = '/home/nzolman/projects/sindy-rl/sindy_rl/refactor/baseline_swimmer_config_test.yml'
    # filename = '/home/nzolman/projects/sindy-rl/sindy_rl/refactor/baseline_config.yml'
    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    LOCAL_DIR =  os.path.join(_parent_dir, 'ray_results', config['exp_dir'])
    
    ip_head = os.environ.get('ip_head', None)
    ray.init(address=ip_head,
             logging_level=logging.ERROR)
    print(ray.nodes())
    
    def explore(config):
        config['lambda_'] = np.clip(config['lambda_'], 0, 1)
        config['gamma'] = np.clip(config['gamma'], 0, 1)
        return config
    
    pbt_sched = None
    if config.get('use_pbt', False):
        pbt_config = config['pbt_config']
        hyper_mut = {}
        for key, val in pbt_config['hyperparam_mutations'].items():
            search_class = getattr(tune, val['search_class'])
            hyper_mut[key] = search_class(*val['search_space'])
        
        pbt_config['hyperparam_mutations'] = hyper_mut
            
        pbt_sched = PopulationBasedTraining(
                        **pbt_config,
                        custom_explore_fn=explore
                    )
    
    ray_config = config['ray_config']
    run_config=air.RunConfig(
        local_dir=LOCAL_DIR,
        **ray_config['run_config'],
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=ray_config['checkpoint_freq']),
    )
    
    tune_config=tune.TuneConfig(**config['ray_config']['tune_config'],
                                scheduler=pbt_sched,
                                )
    
    drl_class, drl_default_config = rllib_algos.get(config['drl']['class'])()
    drl_config = config['drl']['config']
    drl_config['environment']['env'] = getattr(registry, drl_config['environment']['env'])

    drl_default_config = (drl_default_config
                         .environment(**drl_config['environment'])
                         .training(**drl_config['training'])
                         .evaluation(**drl_config['evaluation'])
                        )
    
    model_config = drl_default_config.model
    fcnet_hiddens = config.get('fcnet_hiddens', [64, 64])
    model_config.update({'fcnet_hiddens': fcnet_hiddens})
    drl_default_config.training(model=model_config)
    
    tuner = tune.Tuner(
        config['drl']['class'],
        param_space=drl_default_config, # this is what is passed to the experiment
        run_config=run_config,
        tune_config=tune_config,
    )
    results = tuner.fit()

    print("best hyperparameters: ", results.get_best_result().config)