import warnings
warnings.filterwarnings('ignore')
import logging
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# colors = plt.cm.tab10.colors


import ray
from ray.rllib.algorithms.registry import get_algorithm_class

from sindy_rl.registry import DMCEnvWrapper
from sindy_rl.env import rollout_env
from sindy_rl.policy import RLlibPolicyWrapper


def get_dfs(exp_dir):
    q_string = os.path.join(exp_dir, '**', '*.csv')
    df_paths = sorted(glob.glob(q_string, recursive=True))

    df_list = []
    for path in df_paths:
        df_list.append(pd.read_csv(path))
        
    return df_list

def get_checkpoint_path(seed_dir, check_num):
    checks = sorted(glob.glob(os.path.join(seed_dir, f'checkpoint*', 'checkpoint*'),recursive=True))
    return checks[check_num]



def clean_ts(ts):
    ts_new = ts.copy()
    for i, val in enumerate(ts):
        if np.isnan(val) and i !=0:
            ts_new.iloc[i] = ts_new.iloc[i-1]
    return ts_new


def get_mean_data(df_list, key = 'evaluation/episode_reward_mean', t_key = 'num_agent_steps_sampled', win=10):
    max_t = max([df.shape[0] for df in df_list])
    max_t_idx = np.argmax([df.shape[0] for df in df_list])
    x_vals = [clean_ts(df[key]).values for df in df_list]
    x_ext = [np.concatenate([x, np.repeat(x[-1], max_t - len(x))]) for x in x_vals]
    x_proc = [x for x in x_ext]
    
    n_nan = np.isnan(x_proc[0]).sum()

    T = df_list[max_t_idx][t_key].values[n_nan:]
    x = np.array([x_p[n_nan:] for x_p in x_proc])

    med_x = np.median(x, axis=0)
    min_x = np.min(x, axis= 0)
    max_x = np.max(x, axis=0)
    q25_x = np.quantile(x, q=0.25, axis=0)
    q75_x = np.quantile(x, q = 0.75, axis=0)
    mean_x = np.mean(x, axis=0)
    return T, med_x, min_x, max_x, q25_x, q75_x, mean_x

def get_best_data(df_list, key = 'evaluation/episode_reward_mean', t_key = 'num_agent_steps_sampled', win=10):
    max_t = max([df.shape[0] for df in df_list])
    max_t_idx = np.argmax([df.shape[0] for df in df_list])

    x_vals = [clean_ts(df[key]).rolling(window=win).mean().values for df in df_list]
    x_ext = [np.concatenate([x, np.repeat(x[-1], max_t - len(x))]) for x in x_vals]
    x_proc = [x for x in x_ext]
    
    n_nan = np.isnan(x_proc[0]).sum()

    T = df_list[max_t_idx][t_key].rolling(window=win).mean().values[n_nan:]
    x = np.array([np.maximum.accumulate(x_p[n_nan:]) for x_p in x_proc])

    med_x = np.median(x, axis=0)
    min_x = np.min(x, axis= 0)
    max_x = np.max(x, axis=0)
    q25_x = np.quantile(x, q=0.25, axis=0)
    q75_x = np.quantile(x, q = 0.75, axis=0)
    mean_x = np.mean(x, axis=0)
    return T, med_x, min_x, max_x, q25_x, q75_x, mean_x


def get_data(df_list, key, t_key, mode='best', **kwargs):
    if mode == 'best': 
        return get_best_data(df_list, key=key, t_key=t_key, **kwargs)
    elif mode == 'mean':
        return get_mean_data(df_list, key=key, t_key=t_key, **kwargs)
    else:
        raise NotImplementedError