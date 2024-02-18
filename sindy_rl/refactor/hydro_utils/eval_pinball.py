#  mpiexec -np 12 python eval_cylinder.py
# NOTE: Doesn't appear to be able to run while
#       an rllib experiment is training.

import os
import yaml

from ray.rllib.algorithms import Algorithm

from sindy_rl.refactor.policy import RLlibPolicyWrapper, RandomPolicy, SwitchAfterT
from sindy_rl.refactor.hydroenv import PinballLiftEnv
from sindy_rl.refactor.env import rollout_env
from sindy_rl.refactor.traj_buffer import BaseTrajectoryBuffer
from sindy_rl import _parent_dir


if __name__ == '__main__':
    import ray
    try:
        ray.init(log_to_driver=False)
    except Exception as e:
        print(e)

    _RE = 120
    _DT = 5e-3 #1e-2
    _FLOW = 'square'
    NO_CONTROL_STEPS = 200
    CONTROL_STEPS = 1200
    
    trial_idx = 4
    check_idx = 6

    # exp_dir = os.path.join(_parent_dir, 'ray_results/pinball_new/',
    #                        # 'dyna_pinball_refit=25_on-collect=200_classic_fixed'
    #                        # 'dyna_pinball_refit=25_on-collect=200_square'
    #                       'dyna_pinball_refit=25_on-collect=200_Re=30_lift-track_new-params_new-quad-dyn',
    #                       )

    exp_dir = os.path.join(_parent_dir, 'ray_results/pinball_test/',
                            'baseline_test_rollout=300_Re=120_square_new-param',
                          )
    
    trial_dir = os.path.join(exp_dir,
                                 # f'dyna_sindy_67a0b_{trial_idx:05}_{trial_idx}_2023-12-30_05-22-20/',
                             # f'dyna_sindy_a886f_{trial_idx:05}_{trial_idx}_2023-12-28_14-44-52/',
                             # f'dyna_sindy_d48cd_{trial_idx:05}_{trial_idx}_2024-01-27_23-21-42',
                            f'dyna_sindy_c00a5_0000{trial_idx}_{trial_idx}_2024-01-28_16-46-14',
                               )
    
    ckpt_path = os.path.join(trial_dir, 
                             f'checkpoint_{check_idx:06d}'
                             )

    # save_dir = os.path.join(_parent_dir, 'data', 'hydro', 'pinball', 'eval', 
    #                         '2024-01-14',
    #                         f'dt=1e-2_long_RE={_RE}_agent={trial_idx}_check={check_idx}_{_FLOW}'
    #                         )
    save_dir = os.path.join('/local/nzolman/sindy-rl/data/hydro/tmp_pinball/',
                            'eval',
                            '2024-01-28',
                            # f'dt=1e-2_long_RE={_RE}_agent={trial_idx}_check={check_idx}_{_FLOW}'
                            f'dt=5e-3_long_RE={_RE}_agent={trial_idx}_check={check_idx}_{_FLOW}'
                           )
    os.makedirs(save_dir, exist_ok=True)
    

    restart_path = os.path.join(_parent_dir, 'data/hydro/pinball',
                             # '2023-10-14_fine_instability/Re=30_dt=1e-3/snapshots/no_control_500000.ckpt'
                            '/home/firedrake/sindy-rl/data/hydro/pinball/2024-01-20_chaos/Re=120_dt=1e-2/snapshots/no_control_25000.ckpt'
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

    config_log = {'trial_dir': trial_dir,
                  'ckpt_path': ckpt_path,
                  'save_dir': save_dir,
                  'restart_path': restart_path,
                  'env_config': env_config
                 }
    log_path = os.path.join(save_dir, 'log.yml')
    with open(log_path, 'w') as f:
        yaml.dump(config_log, f)
    
    env = PinballLiftEnv(env_config)
    
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