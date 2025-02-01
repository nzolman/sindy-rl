import os
import numpy as np
import pickle
from tqdm import tqdm
import yaml

import hydrogym.firedrake as hgym
import firedrake as fd

from sindy_rl.registry import HydroSVDWrapper
from sindy_rl.policy import SparseEnsemblePolicy
from sindy_rl.env import rollout_env
from sindy_rl.traj_buffer import BaseTrajectoryBuffer
from sindy_rl import _data_dir, _parent_dir


# --------- Configure Env ---------
train_Re = 100
eval_Re = 100
dt = 1e-2
flow_name = 'Pinball'

load_dir = os.path.join(_data_dir, 
                        'unsteady_checkpoints', 
                        f'Pinball_medium_Re={eval_Re}'
                        )

if eval_Re > 130:
    restart_path = os.path.join(load_dir, 
                            '0040000.ckpt')
else:

    restart_path = os.path.join(load_dir, 
                            '0020000.ckpt')

flow_config = {
    "mesh": "medium",
    "Re": eval_Re,
    "observation_type": "velocity_probes",
    "restart": [restart_path]
}


env_config = {
                'flow': flow_name,
                'flow_config': flow_config,
                'dt': 1.0e-2,
                'n_svd': 10,
                'svd_load': os.path.join(_data_dir, f'unsteady_checkpoints/Pinball_medium_Re={train_Re}/svd_vh.pkl'),
                'max_steps': int(1e5)
            }
# -----------------------------------



# --------- Configure Agent ---------
USE_MEAN = True
baseline = False
agent_num = 11
checkpoint = 300

exp_prefix = 'dyna_sindy_0de06'
exp_suffix = '2024-12-29_14-41-57'
exp_dir_rel = 'pinball_new/test_pinball-1229_17k_5_traj'
exp_dir = os.path.join(_parent_dir, 'ray_results', exp_dir_rel)
trial_dir = os.path.join(exp_dir,
                            f'{exp_prefix}_{agent_num:05}_{agent_num}_{exp_suffix}',
                        )

if baseline:
    exp_baseline = 'pinball_baseline'
    save_baseline = 'baseline'
else:
    exp_baseline = 'pinball_new'
    save_baseline = 'dyna'

ckpt_path = os.path.join(trial_dir, 
                        f'checkpoint_{checkpoint:06}',
                        'sparse_policy.pkl'
                        )
checkpoint_dir = ckpt_path
with open(checkpoint_dir, 'rb') as f:
    sparse_policy = pickle.load(f)

coef_type = 'med'
if USE_MEAN:
    sparse_policy.set_mean_coef_()
    print('using mean coef!')
    coef_type='mean'
print('Policy Loaded!')
# -----------------------------------



# ------------ Save + Log ------------
save_dir = os.path.join(_data_dir, 'eval', 'pinball', 
                        f'Re={train_Re}_sparse_{coef_type}',
                        f'2024-12-29_{save_baseline}_{eval_Re}',
                        f'{agent_num:02}-{checkpoint:06}'
                        )

os.makedirs(save_dir, exist_ok=True)


config_log = {'trial_dir': trial_dir,
                'ckpt_path': checkpoint_dir,
                'save_dir': save_dir,
                'restart_path': restart_path,
                'env_config': env_config,
                'train_Re': train_Re,
                'eval_Re': eval_Re
                }

log_path = os.path.join(save_dir, 'log.yml')
with open(log_path, 'w') as f:
    yaml.dump(config_log, f)
print(f'Logging: {log_path}')
# -----------------------------------
    


env = HydroSVDWrapper(env_config)
env.reset()

SAVE_FREQ = 100


def callback(step_idx, env, save_freq = SAVE_FREQ):
    if (step_idx % save_freq) == 0:
        checkpath = os.path.join(save_dir, f'flow_{step_idx:06d}.ckpt')
        env.flow.save_checkpoint(checkpath)
        print('checkpoint saved!', checkpath)


CONTROL_STEPS = int(1e4)
obs_list, act_list, rew_list = rollout_env(env, sparse_policy, n_steps = CONTROL_STEPS, 
                                            env_callback=callback, verbose=True)
buffer  = BaseTrajectoryBuffer()
buffer.add_data(obs_list, act_list, rew_list)
buffer.save_data(os.path.join(save_dir, 
                                'traj_data.pkl'))
