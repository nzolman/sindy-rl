import ray
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.algorithms.ppo import PPOConfig
import os
from ray.tune.logger import pretty_print
from ray.tune import Trainable
from ray import tune, air
from pprint import pprint
import random
import numpy as np
import torch
import yaml

import pysindy as ps

from sindy_rl.sindy_utils import get_affine_lib
from sindy_rl.dynamics import SINDyDynamics, EnsembleSINDyDynamics
from sindy_rl.data_utils import collect_random_data, split_by_steps
from sindy_rl.envs.swimmer import SwimmerSurrogate
from sindy_rl import _parent_dir


"""Example of a custom experiment wrapped around an RLlib Algorithm."""
import argparse

import ray
from ray import tune
import ray.rllib.algorithms.ppo as ppo

parser = argparse.ArgumentParser()
parser.add_argument("--train-iterations", type=int, default=10)


_ENV_CLASS = SwimmerSurrogate
_ENV_CONFIG = {'dyn_model': None,
                'max_episode_steps': 1000,
                'mod_angles': True
                }

def collect_real_traj(env, 
                      collect_seed = 0, 
                      n_random_steps = 10000, 
                      n_null_steps = 10000,
                      max_traj_len = 100):
    '''
    Collect experience in the real environment
    '''
    # TO-DO: incorporate on-policy collection
    
    u_train, x_train  = collect_random_data(env, 
                                            n_steps=n_random_steps, 
                                            seed = collect_seed, 
                                            max_traj_len=max_traj_len)
    if n_null_steps:
        null_act = 0*env.action_space.sample()
        null_action, null_obs = collect_random_data(env, 
                                                    n_steps=n_null_steps, 
                                                    seed = collect_seed + 1, # different seed
                                                    use_null = null_act, 
                                                    max_traj_len=max_traj_len)
    else:
        null_action, null_obs = ([], [])
    
    x_train += null_obs
    u_train += null_action
    
    return x_train, u_train


def make_update_env_fn(env_conf):
    '''
    Updates the environment config
    Source [1] 
    
    [1] https://discuss.ray.io/t/update-env-config-in-workers/1831/3
    '''
    def update_env_conf(env):
        env.config.update(env_conf)
        env.game.configure(env.config)
        
    def update_env_fn(worker):
        worker.foreach_env(update_env_conf)

    return update_env_fn

def update_env_dyn_model(dyn_model):
    def update_env(env):
        # TO-DO: Is setattr safer? 
        env.dyn_model = dyn_model
        
    def update_env_fn(worker):
        worker.foreach_env(update_env)
    
    return update_env_fn

def test_print_attr(s=''):
    def print_env(env):
        print(s, 'dyn_model', env.dyn_model)
    
    def print_fn(worker):
        worker.foreach_env(print_env)
    
    return print_fn


def on_policy_real_traj(env, policy, n_steps=1000, seed=0, explore=False):
    obs_trajs = []
    obs_list = [env.reset(seed=seed)]
    
    act_trajs = []
    act_list = []
    
    for i in range(n_steps):
        obs = obs_list[-1]
        action = policy.compute_single_action(obs, explore=explore)
        obs, rew, done, info = env.step(action)
        
        if done or (i == n_steps -1):
            obs_list.pop(-1)
            
            obs_trajs.append(obs_list)
            act_trajs.append(act_list)
            
            obs_list = [env.reset()]
            act_list = []
    return act_trajs, obs_trajs
    
            
def setup_dyn_model(affine_config=None, base_config = None, ensemble_config=None):
    affine_kwargs = affine_config or dict(poly_deg= 2, 
                                          n_state = 8, 
                                          n_control = 2)
    base_kwargs = base_config or dict(threshold = 0.02, 
                                        alpha = 0.5)
    ensemble_kwargs = ensemble_config or dict(bagging=True, 
                                            library_ensemble=True,
                                            n_models=100
                                            )  
    
    dyn_lib = get_affine_lib(**affine_kwargs)
    base_optimizer = ps.STLSQ(**base_kwargs)

    optimizer = ps.EnsembleOptimizer(
                                    base_optimizer, 
                                    **ensemble_kwargs
                                    )    
    dyn_config = {
                # 'dt': true_env.dt,
                'optimizer': optimizer, 
                'feature_library': dyn_lib
                }
        
    dyn_model = EnsembleSINDyDynamics(dyn_config)
    return dyn_model
        
def experiment(config):
    drl_config = train_config['drl_config']
    n_dyn_updates = 0
    
    # TO-DO: Find a better place for this
    train_iterations = drl_config.pop("train-iterations")
    
    algo = ppo.PPO(config=drl_config)
    
    # a hack to get an environment to fit on
    env_config = drl_config['env_config']
    env = algo.workers.local_worker().env_creator(env_config)
    env.dyn_model = None  # ensure that we're using the true dynamics
    
    # collect trajectories
    # collect_config = dict(collect_seed = 0, 
    #                         n_random_steps = 10000, 
    #                         n_null_steps = 10000,
    #                         max_traj_len = 100)
    
    x_train, u_train = collect_real_traj(env, **config['init_collection'])

    # setup dyn model
    dyn_config = config['dyn_model_config']
    dyn_model = setup_dyn_model(**dyn_config)
    dyn_model.fit(x_train, u_train)
    
    algo.workers.foreach_worker(update_env_dyn_model(dyn_model))
    
    checkpoint = None
    train_results = {}

    # Train
    for i in range(train_iterations):
        train_results = algo.train()
        if i % 2 == 0 or i == train_iterations - 1:
            checkpoint = algo.save(tune.get_trial_dir())
        tune.report(**train_results)
        
    # algo.evaluation_workers.foreach_worker(test_print_attr('eval'))
    algo.stop()

    # # Manual Eval
    # config["num_workers"] = 0
    # eval_algo = ppo.PPO(config=config)
    # eval_algo.restore(checkpoint)
    # env = eval_algo.workers.local_worker().env

    # obs= env.reset()
    # done = False
    # eval_results = {"eval_reward": 0, "eval_eps_length": 0}
    # while not done:
    #     action = eval_algo.compute_single_action(obs)
    #     next_obs, reward, done, info = env.step(action)
    #     eval_results["eval_reward"] += reward
    #     eval_results["eval_eps_length"] += 1
    # results = {**train_results, **eval_results}
    # tune.report(results)

_TEMPLATES_PATH = os.path.join(_parent_dir, 'templates')
if __name__ == "__main__":
    from sindy_rl import envs as ENVS
    
    from pprint import pprint 
    template_fpath = os.path.join(_TEMPLATES_PATH, 'train_swimmer.yml')
    with open(template_fpath, 'r') as f: 
        train_config = yaml.safe_load(f)
    
    drl_config = train_config['drl_config']
    ray_config = train_config['ray_config']
    drl_config['environment']['env'] = getattr(ENVS, drl_config['environment']['env'] )

    # real_env_config = {
    #                     'dyn_model': None, 
    #                     'mod_angles': True
    #                 }
    # _EVAL_SEED = 0

    LOCAL_DIR = os.path.join(_parent_dir,'ray_results', 'tmp')
    # args = parser.parse_args()

    ray.init(num_cpus=3)
    # eval_config = {"env_config": _ENV_CONFIG.copy(),
    #                 "explore": False,
    #                 'seed': _EVAL_SEED}
    config = (ppo.PPOConfig()
                .environment(**drl_config['environment'])
                .framework(**drl_config['framework'])
                .evaluation(**drl_config['evaluation'])
    )
    # config = (ppo.PPOConfig()
    #             .environment(env=_ENV_CLASS, env_config=_ENV_CONFIG)
    #             .framework(framework='torch')
    #             .evaluation(evaluation_config=eval_config, 
    #                         evaluation_interval=1)
    # )
    
    # algo = config.build()
    
    
    
    config = config.to_dict()
    config["train-iterations"] = 2
    
    run_config=air.RunConfig(
        local_dir=LOCAL_DIR,
        **ray_config['run_config']
        # checkpoint_config=air.CheckpointConfig(checkpoint_frequency=1),
    )
    train_config['drl_config'] = config
    # pprint(train_config)
    
    tune.Tuner(
        tune.with_resources(experiment, ppo.PPO.default_resource_request(config)),
        param_space=train_config,
        run_config=run_config,
    ).fit()
    
    # # env = SwimmerSurrogate(env_config=_ENV_CONFIG)
    # # x_train, u_train = collect_real_traj(env, 0)