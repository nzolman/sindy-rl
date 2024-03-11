# -------------------------------------------------
# Used for evaluating a sparse dictionary policy.
# -------------------------------------------------
import os
import pickle

from sindy_rl.policy import RLlibPolicyWrapper, RandomPolicy, SwitchAfterT
from sindy_rl.hydroenv import CylinderLiftEnv
from sindy_rl.env import rollout_env
from sindy_rl.traj_buffer import BaseTrajectoryBuffer
from sindy_rl import _parent_dir


if __name__ == '__main__':
    _RE = 100
    _DT = 1e-2
    NULL_STEPS = 100
    N_STEPS = 600
    MESH = 'fine'
    use_filter = True
    
    work_dir = os.path.join(_parent_dir, 'data', 'hydro', 'cylinder', 'eval', 
                            '2024-02-10_sparse_filter'
                            )
    save_dir = os.path.join(work_dir, f'flow_checkpoints_{MESH}_sparse')
    
    load_path = os.path.join(work_dir,
                             'spare_policy_filter_trial=0_ckpt=200.pkl'
                            )
    
    with open(load_path, 'rb') as f:
        sparse_policy = pickle.load(f)

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
                 'use_filter': use_filter
                 }

    env = CylinderLiftEnv(env_config)

    null_policy = RandomPolicy(env.action_space)
    null_policy.magnitude = 0
    
    sparse_policy.set_mean_coef_()
    switch_policy = SwitchAfterT(t_switch= NULL_STEPS, 
                                 policies=[null_policy, sparse_policy])

    def callback(step_idx, env, save_freq = 1):
        if (step_idx % save_freq) == 0:
            checkpath = os.path.join(save_dir, f'flow_{step_idx:06d}.ckpt')
            env.flow.save_checkpoint(checkpath)
            print('checkpoint saved!', checkpath)

    obs_list, act_list, rew_list = rollout_env(env, switch_policy, n_steps = N_STEPS, 
                                               env_callback=callback, verbose=True)
    buffer  = BaseTrajectoryBuffer()
    buffer.add_data(obs_list, act_list, rew_list)
    buffer.save_data(os.path.join(save_dir, 
                                  'traj_data.pkl'))