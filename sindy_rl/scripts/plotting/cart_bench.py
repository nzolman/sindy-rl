# ---------------------------------------------
# Used for making Figure 5 in arXiv 
# ---------------------------------------------

import warnings
warnings.filterwarnings('ignore')
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

colorblind_colors = sns.color_palette('colorblind')
colors = [colorblind_colors[3], colorblind_colors[0], colorblind_colors[1], colorblind_colors[2], colorblind_colors[4]]


from sindy_rl.viz.compare_trials import get_dfs
from sindy_rl import _parent_dir
from sindy_rl.viz.rllib_bench import plot_best_comparison_pbt, plot_quantile_comparison


if __name__ == "__main__":
    # make sure that data has been unzipped, etc.!
    ray_dir = os.path.join(_parent_dir, 'data','benchmarks', 'swingup')

    cart_1k_names = {
                'Baseline PPO': 
                    'baseline_ppo',
                'SINDy-RL (quadratic)':
                    'sindy_rl_quad',
                'SINDy-RL (linear)':
                    'sindy_rl_linear',
                'Dyna-NN':
                    'dyna_nn',
                'MB-MPO':
                    'mbmpo'
                    }

    # setup experiment directories
    cart_1k_exp_dirs = {key: os.path.join(ray_dir, exp_name) 
                    for key, exp_name in cart_1k_names.items()}

    # setup dataframes
    cart_1k_df_list_dict = {key: get_dfs(exp_dir) for key, exp_dir in cart_1k_exp_dirs.items()}
    
    # setup experiment seed directories
    cart_1k_seed_dirs = {key: sorted(glob.glob(os.path.join(exp_dir, '*/'))) for key, exp_dir in cart_1k_exp_dirs.items()}
    
    _ALPHA = 0.2
    LABELSIZE=20
    LEGENDSIZE=15

    # setup plotting modes
    win = 1
    _MODE = 'best' # refers to tracking the running "best" evaluation.
    _PLOT_KEYS = {'Baseline PPO': 'evaluation/episode_reward_mean', 
                'SINDy-RL (quadratic)': 'evaluation/episode_reward_mean', 
                'SINDy-RL (linear)': 'evaluation/episode_reward_mean', 
                'Dyna-NN': 'evaluation/episode_reward_mean', 
                'MB-MPO':'episode_reward_mean'
                }

    # which keys to use for the x-axis
    _TIME_KEYS = {'Baseline PPO': 'num_agent_steps_sampled',        # Direclty from RLLib
                'SINDy-RL (quadratic)': 'traj_buffer/n_total_real', # total number of real interactions from sindy-rl
                'SINDy-RL (linear)': 'traj_buffer/n_total_real',    # total number of real interactions from sindy-rl
                'Dyna-NN': 'traj_buffer/n_total_real' ,             # total number of real interactions from sindy-rl
                'MB-MPO':'info/num_steps_sampled'                   # Directly from RLLib, total # of real interactions from mbmpo
                }

    # setup plotting limits
    _LEFT_TICKS = 2e4 * np.arange(5)
    _LEFT_LIM = (-1e3,8e4)
    _RIGHT_TICKS = 1.0 * 1e6 * np.arange(6)

    # Make plots
    fig, axes = plot_quantile_comparison(cart_1k_df_list_dict, 
                            _PLOT_KEYS, _TIME_KEYS, _MODE, 
                                left_ticks=_LEFT_TICKS, right_ticks=_RIGHT_TICKS, left_lim=_LEFT_LIM, alpha=_ALPHA,
                                win = win, linewidth=3)

    # configure axes
    for ax in axes:
        ax.tick_params(axis="both", labelsize=LABELSIZE)
        ax.xaxis.get_offset_text().set_fontsize(LABELSIZE)
        
        # cleanup labels in post.
        # ax.set_xlabel('Thousands of Interactions in Full-Order Environment', fontsize=LABELSIZE) 
        ax.set_xlabel('')
        ax.set_ylabel('')
                
    axes[1].legend(fontsize=LEGENDSIZE)
    # axes[0].set_ylabel('Mean Evaluation Reward', fontsize=LABELSIZE)

    # save figure
    figure_path = os.path.join(_parent_dir, 'figures')
    os.makedirs(figure_path, exist_ok=True)
    
    save_path = os.path.join(figure_path, 'cart_bench.png')
    plt.savefig(save_path, transparent=True, bbox_inches = "tight")