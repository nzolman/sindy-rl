# mpiexec -np 16 python eval_env.py

import os
import glob
import yaml
from tqdm import tqdm
import numpy as np
import pickle
import json
os.environ['OMP_NUM_THREADS']='1'

import firedrake as fd
import ray

try:
    comm = fd.COMM_WORLD
    rank = comm.Get_rank()
except:
    print('error!')
    rank = 0

import ray.rllib.algorithms.ppo as ppo
from hydrogym import firedrake as hgym

from sindy_rl import _parent_dir
from sindy_rl.envs.hydroenv import SurrogateCylinder, SurrogateCylinderLIFT, \
                                    SurrogateHydroEnv, SurrogatePinball

def get_save_paths(exp_dir, checkpoint):
    config_path = os.path.join(exp_dir, 'params.json')
    ckpt_dir = os.path.join(exp_dir, f'checkpoint_{checkpoint:06}')
    ckpt_path = os.path.join(ckpt_dir, f'checkpoint-{checkpoint}')
    dyn_path = os.path.join(ckpt_dir, 'dyn_model.pkl')
    return config_path, ckpt_path, dyn_path, ckpt_dir

def load_config(config_path):
    with open(config_path, 'rb') as f: 
        d = json.load(f)
    return d

def load_checkpoint(ckpt_path, drl_config, env_class):
    agent = ppo.PPO(config=drl_config, env=env_class)
    agent.restore(ckpt_path)
    return agent
    
ray_dir = os.path.join('/home/firedrake/sindy-rl/ray_results/')
run_dir = os.path.join(ray_dir, 
                        'cylinder',
                        '2023-04-21',
                        'LIFT_rew=CL_amp_dt=1e-3_freq=10_Re=100_Tf=30_low_sine_trunc2',
                        # '2023-05-03',
                        # 'LIFT_rew=CL_amp_dt=1e-3_freq=10_Re=100_Tf=30_low_sine_trunc_proxy',
                        'PPO_SurrogateCylinderLIFT',
                        # f'experiment_adcdc_{n_trial:05}_{n_trial}_2023-05-01_21-46-55'
                        )
trial_dirs = [exp for exp in glob.glob(os.path.join(run_dir, '*/')) if 'experiment' in exp]
trial_dirs.sort()
print(f'Found {len(trial_dirs)} trial dirs: ', trial_dirs)
if len(trial_dirs)==0:
    raise ValueError('No trial dirs found')

save_root = os.path.join('/home/firedrake/sindy-rl/data/hydro/',
                         'cylinder',
                         'eval',
                         '2023-04-21',
                         'Re=100'
)

for n_trial, trial_dir in enumerate(trial_dirs):
    if n_trial < 12: # skip if already ran
        continue

    save_dir = os.path.join(save_root,
                            f'trial-{n_trial:02}')
    print('Making Dir:', save_dir)
    os.makedirs(save_dir, exist_ok=True)

    _RE=100

    ckpt_num = 101
    num_steps_eval = 3000
    # restart_path = '/home/firedrake/sindy-rl/data/hydro/cylinder/2023-04-07_medium/Re=400_dt=1e-3/snapshots/no_control_100000.ckpt'
    restart_path = '/home/firedrake/sindy-rl/data/hydro/cylinder/2023-03-15_medium/no_control_10000.ckpt'

    config_path, ckpt_path, dyn_path, ckpt_dir = get_save_paths(trial_dir, ckpt_num)
    exp_config = load_config(config_path)


    env_class = SurrogateCylinderLIFT
    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update({'env': env_class,
                        'env_config': exp_config['drl_config']['env_config'],
                        'framework': 'torch'
                        }
                    )
    print('loading agent')
    agent = load_checkpoint(ckpt_path, ppo_config, env_class)
    print('agent loaded')

    print('starting agent eval')

    env_eval_class = SurrogateCylinder
    env_eval_config = {
        'dyn_model':  None,
        'control_freq': 10,  
        'n_skip_on_reset': 5,
        'max_episode_steps': num_steps_eval,
        'use_omega': True,
        'use_CL_dot': True,
        'hydro_config': {
            'flow_config':{
                'Re': _RE,
                'mesh': 'medium',
                'restart': restart_path,
            },
            'solver_config':{
                'dt': 1.0e-3
            },
        }

    }

    env = env_eval_class(env_eval_config)

    obs_list = [env.reset()]
    action_list = []
    save_freq = 100
    for i in tqdm(range(num_steps_eval-1)):
        
        CL = obs_list[-1][0]
        dCL = obs_list[-1][-1]
        obs = np.array([CL, dCL])
                            
        action = agent.compute_single_action(obs, explore=False)
        obs_list.append(env.step(action)[0])
        
        action_list.append(action)
        
        # periodically save data (just in case)
        if i % save_freq == (save_freq -1):
            save_obs = np.array(obs_list[:-1])
            save_act = np.array(action_list)
            agent_data = {'x': [save_obs], 'u': [save_act]}
            with open(os.path.join(save_dir, f'agent_eval_{i:05}.pkl'), 'wb') as f:
                pickle.dump(agent_data, f)

    # save final data
    obs_list.pop(-1)
    obs_list = np.array(obs_list)
    action_list = np.array(action_list)
    agent_data = {'x': [obs_list], 'u': [action_list]}
    with open(os.path.join(save_dir, f'agent_eval_final.pkl'), 'wb') as f:
        pickle.dump(agent_data, f)
