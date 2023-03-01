import ray
from ray import tune, air
import os
import random
import numpy as np
import yaml
import pickle
from collections import deque
from itertools import chain
import pysindy as ps
import pickle

from sindy_rl.sindy_utils import get_affine_lib
from sindy_rl.dynamics import EnsembleSINDyDynamics
from sindy_rl.data_utils import collect_random_data
from sindy_rl import _parent_dir


_TEMPLATES_PATH = os.path.join(_parent_dir, 'templates')

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
        obs_list.append(obs)
        
        # subtlety where the environment action may not
        # map back to the action in the dynamics model.
        mapped_action = env.action_map(action)
        act_list.append(mapped_action)
        if done or (i == n_steps -1):
            obs_list.pop(-1)
            
            obs_trajs.append(np.array(obs_list))
            act_trajs.append(np.array(act_list))
            
            obs_list = [env.reset()]
            act_list = []
    return obs_trajs, act_trajs
    
            
def setup_dyn_model(affine_config=None, base_config = None, ensemble_config=None):
    '''
    An example of setting up a dynamics model using SINDy with an
        affine-control dynamics library of the from 
            x_dot = p_1(x) + p_2(x)u
    '''
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

def num_samples(x_train):
    return np.sum([len(x) for x in x_train])


def experiment(config):
    SINDY_FIT_FREQ = config['sindy_fit']['fit_freq']
    CHECKPOINT_FREQ = config['ray_config']['checkpoint_freq']
    baseline = config['baseline']
    
    # TO-DO: Find a better place for this
    train_iterations = config.pop("train-iterations")
    
    drl_config = config['drl_config']
    drl_class = config['drl_class']
    
    # on-pi: on-policy, agent collects data
    # off-pi: off-policy, data random or null
    n_pi_collect = config['sindy_fit']['n_pi_collect']
    n_off_init = config['init_collection']['n_random_steps'] + config['init_collection']['n_null_steps']
    
    # Each element of the deque is the _set_ of trajectories from the update
    # Maximal number of updates is the total number of on-policy interactions (equal to off-policy by default)
    #  divided by the total number of steps being colleced. So if n_off_init = 12k (4k rand + 8k null)
    #  and n_pi_collect = 1000, max number of updates is 12. Now the number of trajectories per update
    #  can still fluctuate. But we'll discard all trajectories from an update once we've reached our maximum threshold
    on_pi_x_buffer = deque([], n_off_init//n_pi_collect)
    on_pi_u_buffer = deque([], n_off_init//n_pi_collect)
    
    off_pi_x_xp = []
    off_pi_u_xp = []
    on_pi_x_xp = []
    on_pi_u_xp = []
    x_buffer_xp = []
    u_buffer_xp = []

    n_dyn_updates = 0

    algo = drl_class(config=drl_config)
    
    # TO-DO: 
    # - Allow for loading of saved data
    # - Allow for loading of a model
    # - Replace n_control and n_state after finding out the size of x_train and u_train!!
    # probably have to use a ray store... unless we do this before we create the ray object.
    # be careful! If passing the data, it probably gets saved in a JSON somewhere. Could bloat
    if not baseline:
        # a hack to get an environment to fit on
        env_config = drl_config['env_config']
        env = algo.workers.local_worker().env_creator(env_config)
        env.dyn_model = None  # ensure that we're using the true dynamics

        # collect trajectories
        load_path = config['init_collection'].get('load_path',None)
        if load_path:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
                x_train, u_train = data['x'], data['u']
        else:
            x_train, u_train = collect_real_traj(env, **config['init_collection'])
        # setup dyn model
        dyn_config = config['dyn_model_config']
        
        # to-do: This might need to go inside an environment instead.
        dyn_model = setup_dyn_model(**dyn_config)
        dyn_model.fit(x_train, u_train)

        off_pi_u_xp += u_train
        off_pi_x_xp += x_train

        algo.workers.foreach_worker(update_env_dyn_model(dyn_model))
    else:
        dyn_model = None
    
    checkpoint = None
    train_results = {}
    
    # Train
    for i in range(train_iterations):
        train_results = algo.train()
        
        # checkpoint the agent and dynamics model
        if (i % CHECKPOINT_FREQ == 0) or (i == train_iterations - 1):
            checkpoint = algo.save(tune.get_trial_dir())
            dyn_path = os.path.join(tune.get_trial_dir(), f'checkpoint_{i+1:06}', 'dyn_model.pkl')
            with open(dyn_path, 'wb') as f:
                pickle.dump(dyn_model, f)
        
        # update the SINDy model
        # TO-DO: make it so we don't necessarily have to fit SINDy
        if (i % SINDY_FIT_FREQ) == (SINDY_FIT_FREQ - 1) and not baseline:
            n_dyn_updates += 1
            x_train, u_train = on_policy_real_traj(env, algo, 
                                                   n_steps=n_pi_collect, seed=0, 
                                                   explore=False)
            # add XP to deque
            on_pi_u_buffer.append(u_train)
            on_pi_x_buffer.append(x_train)
            
            # extract xp
            on_pi_x_xp = list(chain(*on_pi_x_buffer))
            on_pi_u_xp = list(chain(*on_pi_u_buffer))
            
            # combine on- and off-policy data
            x_buffer_xp =  on_pi_x_xp + off_pi_x_xp
            u_buffer_xp =  on_pi_u_xp + off_pi_u_xp
            
            dyn_model.fit(x_buffer_xp, u_buffer_xp)
            algo.workers.foreach_worker(update_env_dyn_model(dyn_model))
        
        
        # TO-DO: add in model metrics: sparsity, test score (on-policy, off-policy)
        if not baseline:
            n_samples_on_pi =  num_samples(on_pi_x_xp)
            n_samples_off_pi =  num_samples(off_pi_x_xp)
            n_traj_on_pi = len(on_pi_x_xp)
            n_traj_off_pi = len(off_pi_x_xp)
            sindy_results = {'n_dyn_updates': n_dyn_updates,
                                'n_samples': n_samples_on_pi + n_samples_off_pi,
                                'n_samples_on_pi': n_samples_on_pi,
                                'n_sampels_off_pi': n_samples_off_pi,
                                'n_traj': n_traj_on_pi  + n_traj_off_pi,
                                'n_traj_on_pi': n_traj_on_pi,
                                'n_traj_off_pi': n_traj_off_pi
                                }
            train_results['sindy'] = sindy_results
        tune.report(**train_results)
        
    algo.stop()

# TO-DO set seeds for collect and initialization

if __name__ == "__main__":
    from ray.rllib.algorithms.registry import get_algorithm_class
    from sindy_rl import envs as ENVS
    
    LOCAL_DIR =  os.path.join(_parent_dir, 'ray_results', 'tmp', 'pinball', '2023-02-21')
    # LOCAL_DIR =  os.path.join(_parent_dir, 'ray_results', 'cylinder', 'test', 'collect_C_L=1')
    # LOCAL_DIR = os.path.join(_parent_dir, 'ray_results', 'swimmer', '2023-01-21_ppo_baseline', 'reset_reward')
                            #  '2023-01-12_api_ppo_refit_ensemble_cubic_int_tensor_reset')
    _EVAL_SEED = 0
    
    # experiment configuration
    # template_fpath = os.path.join(_TEMPLATES_PATH, 'train_swimmer.yml')
    template_fpath = os.path.join(_TEMPLATES_PATH, 'pinball.yml')
    with open(template_fpath, 'r') as f: 
        exp_config = yaml.safe_load(f)
    
    drl_config = exp_config['drl_config']
    exp_name = exp_config['drl_class'] + '_' + drl_config['env']
    
    exp_config['ray_config']['run_config']['name'] = exp_name
    
    drl_class, drl_default_config = get_algorithm_class(exp_config['drl_class'], 
                                                        return_config=True)

    
    drl_config['env'] = getattr(ENVS, drl_config['env'] )
    # drl_config['evaluation_config']['seed'] = _EVAL_SEED
    drl_default_config.update(drl_config)

    exp_config['drl_class'] = drl_class
    exp_config['drl_config'] = drl_default_config
    

    # setup ray
    ip_head = os.environ.get('ip_head', None)
    ray.init(address=ip_head)
    print(ray.nodes())
    
    ray_config = exp_config['ray_config']
    run_config=air.RunConfig(
        local_dir=LOCAL_DIR,
        **ray_config['run_config']
        # checkpoint_config=air.CheckpointConfig(checkpoint_frequency=1),
    )
    

    tune_config=tune.TuneConfig(**exp_config['ray_config']['tune_config'])
    
    tune.Tuner(
        tune.with_resources(experiment, 
                            drl_class.default_resource_request(drl_default_config)
                            ),
        param_space=exp_config, # this is what is passed to the experiment
        run_config=run_config,
        tune_config=tune_config
    ).fit()