# ---------------------------------------------
# Used for making Figure 6 in arXiv 
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
    ray_dir = os.path.join(_parent_dir, 'data', 'benchmarks', 'swimmer')

    swimmer_names = {
                'Baseline PPO + PBT': 
                    'baseline_pbt',
                'SINDy-RL + PBT': 
                    "sindy_rl_pbt",
                    }

    swimmer_exp_dirs = {key: os.path.join(ray_dir, exp_name) 
                    for key, exp_name in swimmer_names.items()}

    swimmer_df_list_dict = {key: get_dfs(exp_dir) for key, exp_dir in swimmer_exp_dirs.items()}
    swimmer_seed_dirs = {key: sorted(glob.glob(os.path.join(exp_dir, '*/'))) for key, exp_dir in swimmer_exp_dirs.items()}
    
    
    # setup plotting modes
    _ALPHA = 0.2
    LABELSIZE=20
    LEGENDSIZE=15
    swimmer_keys = {'Baseline PPO + PBT': 'episode_reward_mean',
                    'SINDy-RL + PBT': 'evaluation/episode_reward_mean'
                    }

    # Initial points used for data collection before training
    swimmer_t_init = {'Baseline PPO + PBT': 0,
                    'SINDy-RL + PBT': 12e3
                    }
    
    # Time conversion (since `plot_best_comparison_pbt` uses number of updates)
    swimmer_conversion = {'Baseline PPO + PBT': 4e3,    # raw 4000 interactions per RLlib policy updates
                        'SINDy-RL + PBT': 1000 / 5.0    # collect 1000 on-policy interactions per 5 RLlib policy updates
                    }

    # setup plotting limits
    SWIM_LEFT_TICKS = 1 * 1e6 * np.arange(7)
    SWIM_LEFT_LIM = (-1e3,1.25e6)
    SWIM_RIGHT_TICKS = 1.0 * 1e7 * np.arange(9)

    # Make plots
    fig, axes = plot_best_comparison_pbt(swimmer_df_list_dict, 
                            swimmer_keys, 
                            t_init_dict = swimmer_t_init,
                            t_conversion_dict=swimmer_conversion,
                            mode='best', # refers to tracking the running "best" evaluation.
                            left_ticks=SWIM_LEFT_TICKS, 
                            right_ticks=SWIM_RIGHT_TICKS, 
                            left_lim=SWIM_LEFT_LIM, 
                            alpha=_ALPHA,
                            linewidth=3,
                            win = 1)

    # configure axes
    for ax in axes:
        ax.tick_params(axis="both", labelsize=LABELSIZE)
        ax.xaxis.get_offset_text().set_fontsize(LABELSIZE)
        
        # cleanup labels in post.
        # ax.set_xlabel('Millions of Interactions in Full-Order Environment', fontsize=LABELSIZE)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
    axes[1].legend(fontsize=LEGENDSIZE)
    # axes[0].set_ylabel('Mean Evaluation Reward', fontsize=LABELSIZE)

    # save figure
    figure_path = os.path.join(_parent_dir, 'figures')
    os.makedirs(figure_path, exist_ok=True)
    
    save_path = os.path.join(figure_path, 'swim-pbt_bench.png')
    plt.savefig(save_path, transparent=True, bbox_inches = "tight")