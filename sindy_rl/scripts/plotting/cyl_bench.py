# ---------------------------------------------
# Used for making Figure 7 in arXiv 
# ---------------------------------------------

import warnings
warnings.filterwarnings('ignore')
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd

colorblind_colors = sns.color_palette('colorblind')
colors = [colorblind_colors[3], colorblind_colors[0], colorblind_colors[1], colorblind_colors[2], colorblind_colors[4]]

from sindy_rl.viz.compare_trials import get_dfs, get_data
from sindy_rl import _parent_dir
from sindy_rl.viz.rllib_bench import plot_best_comparison_pbt, plot_quantile_comparison


if __name__ == "__main__":
    # make sure that data has been unzipped, etc.!
    ray_dir = os.path.join(_parent_dir, 'data', 'benchmarks', 'cylinder')
    cyl_names = {
                'Baseline PPO': 
                    'baseline_ppo',
                'SINDy-RL':
                    'sindy_rl',
                    }


    cyl_exp_dirs = {key: os.path.join(ray_dir, exp_name) 
                    for key, exp_name in cyl_names.items()}

    cyl_df_list_dict = {key: get_dfs(exp_dir) for key, exp_dir in cyl_exp_dirs.items()}
    cyl_seed_dirs = {key: sorted(glob.glob(os.path.join(exp_dir, '*/'))) for key, exp_dir in cyl_exp_dirs.items()}
    
    _ALPHA = 0.2
    LABELSIZE=20
    LEGENDSIZE=15
    linewidth=4

    win = 1
    _MODE = 'best' # refers to tracking the running "best" evaluation.
    _PLOT_KEYS = {'Baseline PPO': 'dyn_collect/mean_rew', 
                'SINDy-RL': 'dyn_collect/mean_rew', 
                }

    # which keys to use for the x-axis
    _TIME_KEYS = {'Baseline PPO': 'num_agent_steps_sampled', # Direclty from RLLib
                'SINDy-RL': 'traj_buffer/n_total_real',      # total number of real interactions from sindy-rl
                }

    # setup plotting limits
    _LEFT_TICKS = 2e4 * np.arange(5)
    _LEFT_LIM = (-1e3,8e4)
    _RIGHT_TICKS = 1.0 * 1e4 * np.arange(6)   

    df_list_dict = cyl_df_list_dict
    plot_keys = _PLOT_KEYS
    time_keys = _TIME_KEYS

    # make plot
    fig, ax = plt.subplots(1,1,figsize=(15,4), sharey=True)
    for i, (name, df_list) in enumerate(df_list_dict.items()):
        plot_key = plot_keys[name]
        t_key = time_keys[name]

        # get quantile data
        T, med_x, min_x, max_x, q25_x, q75_x, mean_x = get_data(df_list, 
                                                                plot_key, 
                                                                t_key = t_key,
                                                                mode = _MODE, 
                                                                win = win)

        color = colors[i]

        # add in horizontal line
        if 'Baseline' in name: 
            ax.axhline(med_x[-1], 0, T.max(), color=color, linestyle='dashed',  linewidth=linewidth,)

        # fill between quantiles
        ax.fill_between(T, q25_x, q75_x, color = color, alpha = _ALPHA)
        # plot median
        ax.plot(T, med_x, color = color, label = name,  linewidth=linewidth)

    ax.tick_params(labelsize=LABELSIZE)
    ax.set_xticks(1e4*np.arange(0,11,2))
    ax.set_xticklabels(10*np.arange(0,11,2))
    ax.set_xlim(-1000,102e3)

    # vertical lines corresponding to evaluations
    ax.axvline(3000 + (200/25) * 200, linestyle='dotted', linewidth = 3, color = colors[1])
    ax.axvline(1e5, linestyle='dotted', linewidth = 3, color = colors[0])


    # save figure
    figure_path = os.path.join(_parent_dir, 'figures')
    os.makedirs(figure_path, exist_ok=True)
    
    save_path = os.path.join(figure_path, 'cylinder_bench.png')
    plt.savefig(save_path, transparent=True, bbox_inches = "tight")
    
    
    # ---------------------------------------------
    # Used for making bottom half of Figure 7 in arXiv
    # and the evaluation plots in C.3 
    # ---------------------------------------------
        
    base_data_dir = os.path.join(_parent_dir, 'data/agents/cylinder/baseline/checkpoint_000025')
    base_fine_data_path = os.path.join(base_data_dir, 'traj_eval-fine.pkl')
    base_med_data_path = os.path.join(base_data_dir, 'traj_eval-med.pkl')
    
    with open(base_fine_data_path, 'rb') as f:
        baseline_fine_eval = pickle.load(f)
        
    with open(base_med_data_path, 'rb') as f:
        baseline_med_eval = pickle.load(f)


    dyna_data_dir = os.path.join(_parent_dir, 'data/agents/cylinder/dyna/checkpoint_000200')
    dyna_fine_data_path = os.path.join(dyna_data_dir, 'traj_eval-fine.pkl')
    dyna_med_data_path = os.path.join(dyna_data_dir, 'traj_eval-med.pkl')

    with open(dyna_fine_data_path, 'rb') as f:
        dyna_fine_eval = pickle.load(f)
        
    with open(dyna_med_data_path, 'rb') as f:
        dyna_med_eval = pickle.load(f)
        
    df = pd.DataFrame({'sindy_med': -10*dyna_med_eval['r'][-1],
                    'sindy_fine': -10*dyna_fine_eval['r'][-1],
                    'base_med': -10*baseline_med_eval['r'][-1],
                    'base_fine': -10*baseline_fine_eval['r'][-1]})


    win = 10
    ALPHA = 0.2
    LINEWIDTH = 4
    DOTTED_LINE = 3

    fig, ax = plt.subplots(1,1,figsize=(25,5))
    
    # convert to seconds and shift initial control to t=0
    t = np.arange(600)*0.1 - 10 
    
    # one-second moving average
    win = 10
    plt.plot(t,df['sindy_med'].rolling(window=win).mean()[:], c = colors[1], linewidth = LINEWIDTH)
    plt.axvline(0, color = 'k', linestyle = 'dashed', linewidth = DOTTED_LINE)
    plt.plot(t,df.base_med.rolling(window=win).mean()[:], c = colors[0], linewidth = LINEWIDTH)
    
    plt.tick_params(labelsize=30)
    plt.xlim(-10,)
    
    save_path = os.path.join(figure_path, 'cylinder_eval_medium.png')
    plt.savefig(save_path, transparent=True, bbox_inches = "tight")
    
    
    # fine-mesh
    fig, ax = plt.subplots(1,1,figsize=(25,5))
    win = 10
    plt.plot(t,df['sindy_fine'].rolling(window=win).mean()[:], c = colors[1], linewidth = LINEWIDTH)
    plt.axvline(0, color = 'k', linestyle = 'dashed', linewidth = DOTTED_LINE)
    plt.plot(t,df.base_fine.rolling(window=win).mean()[:], c = colors[0], linewidth = LINEWIDTH)
    
    plt.tick_params(labelsize=30)
    plt.xlim(-10,)
    
    save_path = os.path.join(figure_path, 'cylinder_eval_fine.png')
    plt.savefig(save_path, transparent=True, bbox_inches = "tight")