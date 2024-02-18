import os
import yaml
import pickle

from ray.rllib.algorithms import Algorithm

from sindy_rl.refactor.policy import RLlibPolicyWrapper, RandomPolicy, SwitchAfterT
from sindy_rl.refactor.hydroenv import PinballLiftEnv
from sindy_rl.refactor.env import rollout_env
from sindy_rl.refactor.traj_buffer import BaseTrajectoryBuffer
from sindy_rl import _parent_dir


if __name__ == '__main__':
    _RE = 30
    _DT = 1e-2
    _FLOW = 'square'
    NO_CONTROL_STEPS = 200
    CONTROL_STEPS = 1200
    
    work_dir = os.path.join(_parent_dir, 'data', 'hydro', 'pinball', 'eval', 
                            '2024-01-14_sparse'
                            )
    save_dir = os.path.join(work_dir, 'flow_checkpoints')
    
    load_path = os.path.join(work_dir,
                             'sparse_policy_trial=4_check=30.pkl')
    
    with open(load_path, 'rb') as f:
        sparse_policy = pickle.load(f)

    os.makedirs(save_dir, exist_ok=True)
    
    restart_path = os.path.join(_parent_dir, 'data/hydro/pinball',
                             '2023-10-14_fine_instability/Re=30_dt=1e-3/snapshots/no_control_500000.ckpt'
                            )
    
    
    
    flow_env_config = {
                'flow': _FLOW,
                'flow_config': {
                    'actuator_integration': 'implicit',
                    'Re': _RE,
                    'mesh': 'fine',
                    'restart': restart_path,
                },
                'solver_config': {
                    'dt': _DT
                },
            }  
    
    env_config = {'hydro_config': flow_env_config,
                 'n_skip_on_reset': 5,
                 'control_freq': 10,
                 'max_episode_steps': 8000,
                 }


    
    env = PinballLiftEnv(env_config)

    null_policy = RandomPolicy(env.action_space)
    null_policy.magnitude = 0
    
    sparse_policy.set_mean_coef_()
    switch_policy = SwitchAfterT(t_switch= NO_CONTROL_STEPS, 
                                 policies=[null_policy, sparse_policy])

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