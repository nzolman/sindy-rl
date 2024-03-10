import warnings
warnings.filterwarnings('ignore')
import logging
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import seaborn as sns
from copy import deepcopy

colors = sns.color_palette('colorblind') 

import ray
from ray.rllib.algorithms.registry import get_algorithm_class

from sindy_rl.registry import DMCEnvWrapper
from sindy_rl.env import rollout_env
from sindy_rl.policy import RLlibPolicyWrapper

from sindy_rl.viz.compare_trials import get_data, get_dfs
from sindy_rl import _parent_dir


def plot_quantile_comparison(df_list_dict,plot_keys, time_keys, mode, 
                             left_ticks=None, right_ticks=None, left_lim=None, alpha= 0.2,
                             unit_str = "Thousands",
                             unit_factor = 1e3,
                             linewidth=2,
                             **data_kwargs):
    fig, axes = plt.subplots(1,2,figsize=(15,5), sharey=True)
    axes = axes.flatten()

    for i, (name, df_list) in enumerate(df_list_dict.items()):
        plot_key = plot_keys[name]
        t_key = time_keys[name]

        T, med_x, min_x, max_x, q25_x, q75_x, mean_x = get_data(df_list, 
                                                                plot_key, 
                                                                t_key = t_key,
                                                                mode = mode, 
                                                                **data_kwargs)

        color = colors[i]
        for ax in axes:
            if 'Baseline' in name: 
                ax.hlines(med_x[-1], 0, T.max(), colors=color, linestyles='dashed',  linewidth=linewidth,)

            ax.fill_between(T, q25_x, q75_x, color = color, alpha = alpha)
            ax.plot(T, med_x, color = color, label = name,  linewidth=linewidth,)


    for ax in axes:
        ax.set_xlabel(f'{unit_str} of Interactions in Ground Truth Environment', fontsize=15)
        ax.tick_params(axis="both", labelsize=15)
        ax.xaxis.get_offset_text().set_fontsize(15)

    ax = axes[0]
    ax.set_xlim(*left_lim)
    xticks = left_ticks
    xtick_labels = [fr"{number/unit_factor:.0f}" for number in xticks]
    xtick_labels[0] = '0'
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel('Mean Evaluation Reward', fontsize=15)

    ax = axes[1]
    ax.legend( fontsize=15)
    xticks = right_ticks
    xtick_labels = [fr"{number/unit_factor:.0f}" for number in xticks]
    xtick_labels[0] = '0'
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    fig.tight_layout()
    return fig, axes


def plot_best_comparison_pbt(df_list_dict, 
                             plot_keys, 
                             t_init_dict,
                             t_conversion_dict,
                             mode, 
                             left_ticks=None, right_ticks=None, 
                             left_lim=None, alpha= 0.2,
                             unit_str = "Millions",
                             unit_factor = 1e6,
                              linewidth=2, **data_kwargs,):
    fig, axes = plt.subplots(1,2,figsize=(15,5), sharey=True)
    axes = axes.flatten()

    for i, (name, df_list) in enumerate(df_list_dict.items()):
        plot_key = plot_keys[name]
        t_key = 'training_iteration'
        N_agents = len(df_list)
        T0 = t_init_dict[name]
        t_conversion = t_conversion_dict[name]

        T, med_x, min_x, max_x, q25_x, q75_x, mean_x = get_data(df_list, 
                                                                plot_key, 
                                                                t_key = t_key,
                                                                mode = mode, 
                                                                **data_kwargs)
        T = (T0 + T*t_conversion)*N_agents
        color = colors[i]
        for ax in axes:
            if 'Baseline' in name: 
                ax.hlines(max_x[-1], 0, T.max(), colors=color, linestyles='dashed', linewidth=linewidth)

            ax.plot(T, max_x, color = color, label = name, linewidth=linewidth)


    for ax in axes:
        ax.set_xlabel(f'{unit_str} of Total Population Interactions in GT Environment', fontsize=15)
        ax.tick_params(axis="both", labelsize=15)
        ax.xaxis.get_offset_text().set_fontsize(15)

    ax = axes[0]
    ax.set_xlim(*left_lim)
    xticks = left_ticks
    xtick_labels = [fr"{number/unit_factor:.0f}" for number in xticks]
    xtick_labels[0] = '0'
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel('Mean Evaluation Reward', fontsize=15)

    ax = axes[1]
    ax.legend( fontsize=15)
    xticks = right_ticks
    xtick_labels = [fr"{number/unit_factor:.0f}" for number in xticks]
    xtick_labels[0] = '0'
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)


    fig.tight_layout()
    return fig, axes
