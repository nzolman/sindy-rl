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
    
    filename = '/home/nzolman/projects/sindy-rl/sindy_rl/refactor/baseline_dm_config_test.yml'
    # filename = '/home/nzolman/projects/sindy-rl/sindy_rl/refactor/baseline_config.yml'
    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    LOCAL_DIR =  os.path.join(_parent_dir, 'ray_results', config['exp_dir'])
    
    ip_head = os.environ.get('ip_head', None)
    ray.init(address=ip_head)
    print(ray.nodes())
    
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=50,
        resample_probability=0.25,
        quantile_fraction=0.25,
        # synch=True,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
        #     "drl/config/training/lambda_": lambda: random.uniform(0.9, 1.0),
        #     "drl/config/training/clip_param": lambda: random.uniform(0.01, 0.5),
            # "lambda_": lambda: random.uniform(0.9, 1.0),
            # "clip_param": lambda: random.uniform(0.01, 0.5),
            "lr": tune.loguniform(1e-6, 1e-3),
            # "num_sgd_iter": lambda: random.randint(1, 30),
            # "sgd_minibatch_size": lambda: random.randint(128, 16384),
            # "train_batch_size": lambda: random.randint(2000, 160000),
        },
        # custom_explore_fn=explore,
    )
    
    ray_config = config['ray_config']
    run_config=air.RunConfig(
        local_dir=LOCAL_DIR,
        **ray_config['run_config'],
        checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ray_config['checkpoint_freq']),
    )
    
    tune_config=tune.TuneConfig(**config['ray_config']['tune_config'],
                                scheduler=pbt,
                                metric="episode_reward_mean",
                                mode="max",
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