import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
import pickle
import json
from tqdm import tqdm

from pysindy import PolynomialLibrary

colors = sns.color_palette('colorblind')

from sindy_rl import _parent_dir

from sindy_rl.env import BaseEnsembleSurrogateEnv
from sindy_rl.policy import SparseEnsemblePolicy
from sindy_rl.sindy_utils import build_optimizer



def replace_strings(feat_mappings, text):
    '''Taken from https://stackoverflow.com/questions/6116978/how-to-replace-multiple-substrings-of-a-string '''
    
    # use these three lines to do the replacement
    rep = dict((re.escape(k), v) for k, v in feat_mappings.items()) 
    pattern = re.compile("|".join(feat_mappings.keys()))
    new_text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    new_text = '${}$'.format(new_text)
    return new_text


def env_from_config(config, new_config = False):
    if new_config:
        env_config = config['drl']['config']['environment']['env_config']
    else:
        env_config = config['drl']['config']['env_config']
    dyn_config = config['dynamics_model']
    rew_config = config['rew_model']
    
    env_config['dynamics_model_config'] = dyn_config
    env_config['rew_model_config'] = rew_config
    
    env = BaseEnsembleSurrogateEnv(env_config)
    
    return env


def get_models_from_prefix(exp_dir_rel, trial_prefix, trial_suffix, trial_idx, check_idx, return_policy = False, models_dir = 'ray_results'):
    exp_dir = os.path.join(_parent_dir, models_dir, exp_dir_rel)

    trial_dir = os.path.join(exp_dir,
                                f'{trial_prefix}_{trial_idx:05}_{trial_idx}_{trial_suffix}',
                            )

    return get_models(trial_dir, check_idx, return_policy=return_policy)


def get_models(trial_dir, check_idx, return_policy = False):

    json_path = os.path.join(trial_dir, 'params.json')

    ckpt_path = os.path.join(trial_dir, 
                            f'checkpoint_{check_idx:06}'
                            )
        
    dyn_path = os.path.join(trial_dir, 
                            f'checkpoint_{check_idx:06}',
                            'dyn_model.pkl'
                            )

    rew_path = os.path.join(trial_dir, 
                            f'checkpoint_{check_idx:06}',
                            'rew_model.pkl'
                            )
    
    with open(json_path, 'r') as f:
        config = json.load(f)
        
    env = env_from_config(config, new_config=True)

    with open(dyn_path, 'rb') as f:
        dyn_model = pickle.load(f)
        
    with open(rew_path, 'rb') as f:
        rew_model = pickle.load(f)
        
    with open(os.path.join(ckpt_path, 'on-pi_data.pkl'), 'rb') as f:
        pi_data = pickle.load(f)
        
    with open(os.path.join(ckpt_path, 'off-pi_data.pkl'), 'rb') as f:
        off_pi_data = pickle.load(f)
    
    data = {'on_pi': pi_data,
            'off_pi': off_pi_data}
        
    if not return_policy:
        return dyn_model, rew_model, data, env
    else:
        from ray.rllib.algorithms import Algorithm
        from sindy_rl.policy import RLlibPolicyWrapper

        algo = Algorithm.from_checkpoint(ckpt_path)
        policy = RLlibPolicyWrapper(algo)
        return dyn_model, rew_model, data, policy, env


def fit_policy(X_data, U_data, alpha = 1e-6, thresh = 1e-5, n_models = 20, poly_deg = 3, bounds = None, include_bias = False):
    optimizer_config = {
        'base_optimizer':{
        'name': 'STLSQ',
        'kwargs': {
            'alpha':  alpha, 
            'threshold': thresh,
        }
        },
        'ensemble':{
        'bagging': True,
        'library_ensemble': True,
        'n_models': n_models,
        }
    }
    optimizer = build_optimizer(optimizer_config)

    feature_library = PolynomialLibrary(degree=poly_deg, 
                                        include_bias=include_bias, 
                                        include_interaction=True)

    if bounds is not None:
        min_bounds, max_bounds = bounds
    else:
        min_bounds, max_bounds = None, None
    sparse_policy = SparseEnsemblePolicy(optimizer=optimizer, 
                                        feature_library=feature_library, 
                                        min_bounds=min_bounds, 
                                        max_bounds=max_bounds)
    
    
    sparse_policy.fit([X_data], [U_data])
    
    return sparse_policy   


def eval_policy(policy, X_data, U_data, clip_params = None, use_median = True):
    '''
    '''
    if use_median:
        policy.set_median_coef_()
    else:
        policy.set_mean_coef_()

    pred = policy.optimizer.coef_ @ policy.feature_library.transform(X_data).T
    pred = pred.T
    
    if clip_params is None:
        mse = np.mean((pred - U_data)**2)
    else:
        pred_clip = np.clip(pred, *clip_params)
        u_clip = np.clip(U_data, *clip_params)
        mse = np.mean((pred_clip - u_clip)**2)
    
    l1 = np.abs(policy.optimizer.coef_ ).sum()
    return mse, l1

def eval_wrapper(policy, X_data, U_data):
    evals = []
    n_coefs = len(policy.get_coef_list())
    for idx in range(n_coefs):
        policy.set_idx_coef_(idx)
        evals.append(eval_policy(policy, X_data, U_data, use_median=False))

    return np.array(evals)


hyperparams = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2]) 
hyper_mesh = np.meshgrid(hyperparams, hyperparams)
hyper_mesh = np.array([hyper_mesh[0].flatten(), hyper_mesh[1].flatten()]).T


def fit_policies_v(X_data, U_data, 
                   X_eval, U_eval, 
                   hyper_mesh, 
                   n_models=20, 
                   use_median = True, 
                   clip_params = None, 
                   poly_deg=3,
                   include_bias = False,
                   bounds = None): 
    '''
    Clip params are for evaluation: do we select best MSE based off whether we clip the action
    Bounds are whether we enforce bounds on the policy when computing new actions (e.g. during rollouts)
    '''
    policies = []
    
    for hyper in tqdm(hyper_mesh):
        policies.append(fit_policy(X_data, U_data, *hyper, n_models=n_models, poly_deg = poly_deg, bounds = bounds, include_bias=include_bias))
    evals = np.array([eval_policy(policy, X_eval, U_eval, 
                                  clip_params=clip_params, 
                                  use_median=use_median) for policy in policies])
    
    idx = np.argmin(evals[:,0])
    sparse_policy = policies[idx]

    sparse_policy.set_median_coef_()
    sparse_policy.print()
    
    return {'policies': policies, 'evals': evals, 
            'best_policy': sparse_policy,
           }