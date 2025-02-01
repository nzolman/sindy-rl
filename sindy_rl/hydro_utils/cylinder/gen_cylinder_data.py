# ----------------------------------------------------
# Used for generating off-policy data for SINDy
# ----------------------------------------------------

import os
os.environ['OMP_NUM_THREADS']='1'
from tqdm import tqdm
import numpy as np
import pickle

from matplotlib import pyplot as plt
plt.style.use('ggplot')

from hydrogym import firedrake as hgym
from sindy_rl import _parent_dir
from sindy_rl.hydroenv import CylinderLiftEnv
from sindy_rl.env import rollout_env
from sindy_rl.policy import OpenLoopRandRest, OpenLoopSinRest
from sindy_rl.traj_buffer import BaseTrajectoryBuffer

def save_plot(env, idx, fname='no_control_{:05d}.png', dir_name=None):
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    env.render(axes=axes)
    axes.set_title(f'Coarse Mesh Step: {idx} Timestep: {env.solver.t}')
    fig.tight_layout()
    fig_name = os.path.join(dir_name, fname.format(idx+1))
    plt.savefig(fig_name, bbox_inches="tight")
    plt.close()

def gen_random_then_none(checkpoint,
                         env_class=CylinderLiftEnv,
                       n_steps=1000, 
                       n_none=1000,
                       seed = 0,
                       control_freq=50, 
                       mesh = 'coarse',
                       n_skip=10,
                       Re=100,
                       dt=1e-2
                       ):
    '''
    NOTE: THIS USES the wrapper API!
    '''

    flow_config = {
        "flow": hgym.Cylinder,
        "flow_config": {
            'actuator_integration': 'implicit',
            'mesh':mesh,
            'Re': Re,
            'restart': checkpoint
        },
        "solver": hgym.IPCS,
        "solver_config":
            {
                'dt': dt
            }
    }

    env_config = {
            'hydro_config': flow_config,
            'dyn_model': None,
            'use_omega': True,
            'use_CL_dot': True, 
            'control_freq': control_freq,   
            'n_skip_on_reset': n_skip, 
            'max_episode_steps': 2*n_steps
    }

    env = env_class(env_config=env_config)

    np.random.seed(seed)
    env.action_space.seed(seed)
    
    policy = OpenLoopRandRest(steps_rest=n_steps, 
                              action_space = env.action_space)
    
    x_list, u_list, rew_list = rollout_env(env, policy, 
                                           n_steps=n_steps + n_none,
                                           verbose=True
                                           )
    
 
    return x_list, u_list, rew_list 


def gen_sine_then_none(checkpoint,
                       env_class = CylinderLiftEnv,
                       n_steps=1000,
                       n_none=1000, 
                       k_int = 1,
                       seed = 0,
                       control_freq=50, 
                       mesh = 'coarse',
                       n_skip=10,
                       Re=100,
                       dt=1e-2,
                       use_filter = False
                       ):
    '''
    NOTE: THIS USES the wrapper API!
    
    TO-DO: Make the timing more precise since there is the `n_skip`
    parameter
    '''
       
    flow_config = {
        "flow": hgym.Cylinder,
        "flow_config": {
            'actuator_integration': 'implicit',
            'mesh':mesh,
            'Re': Re,
            'restart': checkpoint
        },
        "solver": hgym.IPCS,
        "solver_config":
            {
                'dt': dt
            }
    }

    env_config = {
            'hydro_config': flow_config,
            'dyn_model': None,
            'use_omega': True,
            'use_CL_dot': True, 
            'control_freq': control_freq,   
            'n_skip_on_reset': n_skip, 
            'max_episode_steps': 2*n_steps,
            'use_filter': use_filter
    }

    env = env_class(env_config=env_config)

    np.random.seed(seed)
    env.action_space.seed(seed)
    
    # random parameters defining the sine wave
    amp = np.random.uniform(low=0.25, high=1.0)
    phase = np.random.uniform(low=0, high=np.pi)
    offset = np.random.uniform(-np.pi/2 + 1, np.pi/2 - 1)
    f0 = 5.556
    # k_int = seed + 1 #np.random.randint(low=1,high=5)
    
    Dt = (env.control_freq * env.solver.dt)
    t_rest = n_steps * Dt
    
    policy = OpenLoopSinRest(t_rest, dt= Dt, 
                             amp=amp, phase=phase, 
                             offset=offset, f0=f0, k=k_int)
    
    x_list, u_list, rew_list = rollout_env(env, policy, 
                                           n_steps=n_steps + n_none,
                                           verbose=True
                                           )
    return x_list, u_list, rew_list

def gen_trajs(n_trajs = 5, seed=0, control_type='random', save_path=None, **kwargs):
    traj_buffer = BaseTrajectoryBuffer()
    for i in range(n_trajs):
        print(f'Starting Traj {i}')
        try:
            if control_type == 'random':
                x, u, rew = gen_random_then_none(seed = seed+i, **kwargs)
            elif control_type == 'sine':
                x, u, rew = gen_sine_then_none(k_int = 1/(i + 1), seed = seed+i, **kwargs)
            else:
                raise NotImplementedError('invalid control_type')
            
            traj_buffer.add_data(x, u, rew)
            
            # iteratively save the data
            traj_buffer.save_data(save_path)

        except Exception as e:
            print(f'FAILED {i}', e)
            continue
    return traj_buffer


    
if __name__ == '__main__':
    _RE = 100
    _DT = 1e-2
    N_TRAJ = 5
    # N_TRAJ = 1
    _N_SKIP = 5
    _CONTROL_FREQ =  10 # 10 Hz
    _MESH = 'medium'
    _FILTER = True
    
    _N_STEPS = 500
    _N_NONE = 100
    
    data_dir = os.path.join(_parent_dir, 
                            'data/hydro/cylinder/'
                            )
    
    load_data_dir = os.path.join(data_dir ,
                            # f'2023-10-02_medium/Re={_RE}_dt=1e-3/'
                            f'2023-12-13_medium/Re={_RE}_dt=1e-3/'
                            )
    check_path = os.path.join(load_data_dir, 
                              'snapshots',
                              # 'no_control_95000.ckpt'
                              # 'no_control_77500.ckpt'
                              'no_control_487500.ckpt'
                              )
    
#     save_data_dir = os.path.join(data_dir ,
#                             f'2023-10-04_medium/Re={_RE}_dt=1e-3_implict/'
#                             )
    
    save_data_dir = os.path.join(data_dir ,
                            # f'2023-11-02_medium/Re={_RE}_dt=1e-3_implict/'
                            # f'2023-12-13_medium/Re={_RE}_dt=1e-3'
                            f'2024-02-03_medium-filter/Re={_RE}_dt=1e-2'
                            )
    
    # save_dir = os.path.join(save_data_dir, 'control')
    save_dir = os.path.join(save_data_dir, 'holdout')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 
                            #  'traj=8_freq=10_steps=10000_sine_ctrl_then_none-04-07_low_freq.pkl')
                             # f'env=CL_traj=5_freq=10_dt=1e-3_res=med_Re={_RE}_steps=6000_low-sine-ctrl-then_none.pkl'
                             # f'none=10k_2.pkl'
                            # 'test.pkl'
                             # f'traj={N_TRAJ}_freq=10_dt=1e-2_steps={_N_STEPS}_none={_N_NONE}.pkl'
                             'traj_holdout.pkl'
                             # 'traj_data.pkl'
                             )

    rand_then_none_kwargs = dict(
                                checkpoint = check_path,
                                env_class = CylinderLiftEnv,
                                n_steps = _N_STEPS,
                                # n_steps=5000,
                                # n_steps=10,
                                # n_none=1000,
                                n_none=_N_NONE,
                                # n_none=10,
                                control_freq=_CONTROL_FREQ,
                                n_skip = _N_SKIP,
                                mesh=_MESH,
                                dt=_DT,
                                Re=_RE,
                                use_filter = _FILTER
                    )
    
    data = gen_trajs(n_trajs = N_TRAJ, 
                     seed = 1, #1,
                     control_type='sine',
                     save_path=save_path,
                     **rand_then_none_kwargs)
    