#  mpiexec -np 12 python eval_cylinder.py
# NOTE: Doesn't appear to be able to run while
#       an rllib experiment is training.

import os
from ray.rllib.algorithms import Algorithm

from sindy_rl.refactor.policy import RLlibPolicyWrapper, RandomPolicy, SwitchAfterT
from sindy_rl.refactor.hydroenv import CylinderLiftEnv
from sindy_rl.refactor.env import rollout_env



if __name__ == '__main__':
    parent_dir = '/home/firedrake/sindy-rl/'
    exp_dir = os.path.join(parent_dir,
                           'ray_results/test/dyna_cylinder_refit=25_on-collect=2k/'
                           )
    trial_dir = os.path.join(exp_dir,
                             'dyna_sindy_30229_00004_4_2023-10-04_05-02-16')
    
    ckpt_path = os.path.join(trial_dir, 
                             'checkpoint_000100'
                             )
    
    save_dir = os.path.join(parent_dir, 'data', 'hydro', 'cylinder', 'eval', 
                            '2023-10-04', 
                            'example'
                            )
    os.makedirs(save_dir, exist_ok=True)
    
    flow_config = {
                    'Re': 100,
                    'mesh': 'medium',
                    'restart': '/home/firedrake/sindy-rl/data/hydro/cylinder/2023-10-02_medium/Re=100_dt=1e-3/snapshots/no_control_95000.ckpt',
                    'solver_config': {
                        'dt': 1.0e-3
                    },
    }           
    
    env_config = {'hydro_config': flow_config,
                 'n_skip_on_reset': 50,
                 'control_freq': 10,
                 'max_episode_steps': 4000,
                 }

    env = CylinderLiftEnv(env_config)
    
    rllib_algo = Algorithm.from_checkpoint(ckpt_path)
    nn_policy = RLlibPolicyWrapper(rllib_algo)
    print('Policy Loaded!')
    
    null_policy = RandomPolicy(env.action_space)
    null_policy.magnitude = 0
    
    switch_policy = SwitchAfterT(t_switch= 1000, 
                                 policies=[null_policy, nn_policy])

    def callback(step_idx, env):
        if (step_idx % 10) == 0:
            checkpath = os.path.join(save_dir, f'flow_{step_idx:05d}.ckpt')
            env.flow.save_checkpoint(checkpath)
            print('checkpoint saved!', checkpath)

    obs_list, act_list, rew_list = rollout_env(env, switch_policy, n_steps = 4000, env_callback=callback, verbose=True)