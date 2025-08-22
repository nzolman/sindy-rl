# -------------------------------------------------
# Used for evaluating an RLlib policy.
# -------------------------------------------------

import os
import yaml
from ray.rllib.algorithms import Algorithm

from sindy_rl.policy import RLlibPolicyWrapper, RandomPolicy, SwitchAfterT
from sindy_rl.hydroenv import CylinderLiftEnv
from sindy_rl.env import rollout_env
from sindy_rl.traj_buffer import BaseTrajectoryBuffer
from sindy_rl import _parent_dir

if __name__ == '__main__':
    _RE = 100
    _DT = 1e-2
    trial_idx = 14 # 14 # 0 #13
    check_idx = 530 # 25 # 200 # 200
    MESH = 'medium'
    baseline = False
    use_filter = True
    
    NO_CONTROL_STEPS = 100
    CONTROL_STEPS = 600


    exp_dir = os.path.join(_parent_dir, 'ray_results',
                           'cylinder_new/dyna_cylinder_refit=25_on-collect=200_filter'
                          )
    trial_dir = os.path.join(exp_dir,
                             f'dyna_sindy_df8d9_{trial_idx:05}_{trial_idx}_2024-02-05_00-09-48'
                            )
    
    ckpt_path = os.path.join(trial_dir, 
                             f'checkpoint_{check_idx:06d}'
                             )

    save_dir = os.path.join(_parent_dir, 'data', 'hydro', 'cylinder', 'eval', 
                            '2024-02-05_filter',
                            f'dt=1e-2_long_RE={_RE}_{MESH}_agent={trial_idx}_check={check_idx}_baseline={baseline}_filter={use_filter}'
                            )

    os.makedirs(save_dir, exist_ok=True)

    if MESH == 'fine': 
        restart_path = os.path.join(_parent_dir, 'data/hydro/cylinder',
                                 '2023-12-11_fine/Re=100_dt=1e-3/snapshots/no_control_340000.ckpt'
                                )
    elif MESH == 'medium': 
        restart_path = os.path.join(_parent_dir, 'data/hydro/cylinder',
                                 '2023-12-13_medium/Re=100_dt=1e-3/snapshots/no_control_487500.ckpt'
                                )
    
    flow_env_config = {
                'flow_config': {
                    'actuator_integration': 'implicit',
                    'Re': _RE,
                    'mesh': MESH,
                    'restart': restart_path,
                },
                'solver_config': {
                    'dt': _DT
                },
            }  
    
    env_config = {'hydro_config': flow_env_config,
                 'n_skip_on_reset': 5,
                 'control_freq': 10,
                 'max_episode_steps': 40000,
                  'use_filter': True
                 }
    

    config_log = {'trial_dir': trial_dir,
                  'ckpt_path': ckpt_path,
                  'save_dir': save_dir,
                  'restart_path': restart_path,
                  'env_config': env_config
                 }
    
    log_path = os.path.join(save_dir, 'log.yml')
    with open(log_path, 'w') as f:
        yaml.dump(config_log, f)

    env = CylinderLiftEnv(env_config)
    
    rllib_algo = Algorithm.from_checkpoint(ckpt_path)
    nn_policy = RLlibPolicyWrapper(rllib_algo)
    print('Policy Loaded!')
    
    null_policy = RandomPolicy(env.action_space)
    null_policy.magnitude = 0
    
    switch_policy = SwitchAfterT(t_switch= NO_CONTROL_STEPS, 
                                 policies=[null_policy, nn_policy])

    def callback(step_idx, env, save_freq = 1):
        if (step_idx % save_freq) == 0:
            checkpath = os.path.join(save_dir, f'flow_{step_idx:06d}.ckpt')
            env.flow.save_checkpoint(checkpath)
            print('checkpoint saved!', checkpath)

    obs_list, act_list, rew_list = rollout_env(env, switch_policy, n_steps = CONTROL_STEPS, 
                                               env_callback=callback, verbose=True)
    buffer  = BaseTrajectoryBuffer()
    buffer.add_data(obs_list, act_list, rew_list)
    buffer.save_data(os.path.join(save_dir, 
                                  'traj_data.pkl'))