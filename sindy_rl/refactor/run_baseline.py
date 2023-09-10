import os
import numpy as np
from ray.rllib.algorithms.registry import ALGORITHMS as rllib_algos

from sindy_rl.refactor import registry

if __name__ == '__main__': 
    import yaml
    import logging
    import ray
    from ray import tune, air
    
    from sindy_rl import _parent_dir
    
    filename = '/home/nzolman/projects/sindy-rl/sindy_rl/refactor/baseline_dm_config_test.yml'
    # filename = '/home/nzolman/projects/sindy-rl/sindy_rl/refactor/baseline_config.yml'
    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    LOCAL_DIR =  os.path.join(_parent_dir, 'ray_results', config['exp_dir'])
    
    ip_head = os.environ.get('ip_head', None)
    ray.init(address=ip_head)
    print(ray.nodes())
    
    
    
    ray_config = config['ray_config']
    run_config=air.RunConfig(
        local_dir=LOCAL_DIR,
        **ray_config['run_config'],
        checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ray_config['checkpoint_freq']),
    )
    
    tune_config=tune.TuneConfig(**config['ray_config']['tune_config'])
    
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
    
    tune.Tuner(
        config['drl']['class'],
        param_space=drl_default_config, # this is what is passed to the experiment
        run_config=run_config,
        tune_config=tune_config
    ).fit()