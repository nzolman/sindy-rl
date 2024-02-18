import os
os.environ['OMP_NUM_THREADS']='1'
from tqdm import tqdm
import numpy as np
import pickle
import firedrake as fd

from matplotlib import pyplot as plt
plt.style.use('ggplot')

from hydrogym import firedrake as hgym
from sindy_rl import _parent_dir
from sindy_rl.refactor.hydroenv import PinballLiftEnv
from sindy_rl.refactor.env import rollout_env
from sindy_rl.refactor.policy import OpenLoopRandRest, OpenLoopSinRest
from sindy_rl.refactor.traj_buffer import BaseTrajectoryBuffer

# def save_plot(env, idx, fname='{:05d}.png', dir_name=None):
#     fig, axes = plt.subplots(2,1, figsize=(10,10))
#     axes = axes.flatten()

#     flow = env.flow
#     vort = fd.project(fd.curl(flow.u), flow.pressure_space)

#     clim = (-2,2)
#     im1 = fd.tripcolor(vort, 
#                     cmap = 'RdBu',
#                     vmin=clim[0],
#                     vmax=clim[1],
#                     axes = axes[0])


#     levels = np.linspace(*clim, 10)
#     im2 = fd.tripcolor(
#                 env.flow.p,
#                 cmap='RdBu',
#                 vmin=clim[0],
#                 vmax=clim[1],
#                 axes=axes[1]
#             )


#     axes[0].set_title('Vorticity', fontsize=20)
#     axes[1].set_title('Pressure', fontsize=20)
    
#     for ax in axes:
#         ax.set_ylim(-5,5)
#         ax.set_xlim(-5.0,20.0)
    
#     save_name = os.path.join(dir_name, fname)
#     fig.savefig(save_name, bbox_inches="tight")
#     plt.close()

def gen_random_then_none(checkpoint,
                         env_class=PinballLiftEnv,
                         flow = hgym.Pinball,
                       n_steps=1000, 
                       n_none=1000,
                       seed = 0,
                       control_freq=50, 
                       mesh = 'fine',
                       n_skip=10,
                       Re=30,
                       dt=1e-2,
                       render_freq = 10
                       ):
    '''
    NOTE: THIS USES the wrapper API!
    '''

    flow_config = {
        "flow": flow,
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
                                           verbose=True, 
                                           )
    
    return x_list, u_list, rew_list 


def gen_sine_then_none(checkpoint,
                       env_class = PinballLiftEnv,
                       flow = hgym.Pinball,
                       n_steps=1000,
                       n_none=1000, 
                       k_int = 1,
                       seed = 0,
                       control_freq=50, 
                       mesh = 'fine',
                       n_skip=10,
                       Re=30,
                       dt=1e-2,
                       check_freq=10,
                       dir_name=None
                       ):
    '''
    NOTE: THIS USES the wrapper API!
    
    TO-DO: Make the timing more precise since there is the `n_skip`
    parameter
    '''
       
    flow_config = {
        "flow": flow,
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
    
    # random parameters defining the sine wave
    amp = np.random.uniform(low=0.25, high=1.0)
    phase = np.random.uniform(low=0, high=np.pi)
    offset = np.random.uniform(-np.pi/2 + 1, np.pi/2 - 1)
    f0 = 5.556
    # k_int = seed + 1 #np.random.randint(low=1,high=5)
    
    Dt = (env.control_freq * env.solver.dt)
    t_rest = n_steps * Dt
    
    # periodically checkpoint results
    def check_callback(i, env):
        idx = i
        if (idx % check_freq) == (check_freq -1) or (idx == 0):
            ckpt_name = os.path.join(dir_name, f'{seed}-{idx:05}.ckpt')
            env.flow.save_checkpoint(ckpt_name)
            print(env.flow.max_cfl(dt))
        return None

    
    policy = OpenLoopSinRest(t_rest, dt= Dt, 
                             amp=amp, phase=phase, 
                             offset=offset, f0=f0, k=k_int)
    
    x_list, u_list, rew_list = rollout_env(env, policy, 
                                           n_steps=n_steps + n_none,
                                           verbose=True,
                                           env_callback=check_callback
                                           )
    return x_list, u_list, rew_list

def gen_trajs(n_trajs = 5, seed=0, control_type='sine', save_path=None, **kwargs):
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
    from sindy_rl.refactor.hydroenv import PinballFlowSq, PinballFlowRefSq
    # _RE = 30
    _RE = 120
    _DT_exp = 3
    front_dt = 5
    _DT = front_dt * (10**(-_DT_exp))
    control_freq = 20
    flow = 'square' #PinballFlowSq # PinballFlowRefSq # hgym.Pinball
    flow_name = 'min_sq'

    data_dir = os.path.join(_parent_dir, 
                            'data/hydro/pinball/'
                            )

    load_data_dir = os.path.join(data_dir ,
                            # f'2023-10-14_fine_instability/Re=30_dt=1e-3/'
                            f'2024-01-20_chaos/Re=120_dt=1e-2/'
                            # f'2024-01-20_chaos/Re=100_dt=1e-2/'
                            )
    check_path = os.path.join(load_data_dir, 
                              'snapshots',
                              # 'no_control_500000.ckpt'
                              'no_control_25000.ckpt'
                              # 'no_control_50000.ckpt'
                              )
    
    tmp_save = '/local/nzolman/sindy-rl/data/hydro/tmp_pinball'
    
    save_data_dir = os.path.join(
                            tmp_save,
                            # data_dir ,
                            # f'2023-12-23_fine/Re={_RE}_dt=1e-{_DT_exp}/'
                            # f'2024-01-20_chaos/Re={_RE}_dt={front_dt}e-{_DT_exp}/'
                            f'2024-01-31_tau=1-27_icm=0-25/Re={_RE}_dt={front_dt}e-{_DT_exp}/'
                            )
    save_dir = os.path.join(save_data_dir, 'control', flow_name)
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 
                            #  f'env=CL_traj=5_freq=10_dt=1e-3_res=med_Re={_RE}_steps=600_low-sine-ctrl-then_none.pkl'
                            f'dt={front_dt}e-{_DT_exp}_10hz.pkl'
                             )
    
    rand_then_none_kwargs = dict(
                                checkpoint = check_path,
                                env_class = PinballLiftEnv,
                                flow = flow,
                                n_steps=500,
                                n_none=100, 
                                control_freq=control_freq, #10 hz
                                n_skip = 5,
                                mesh='fine',
                                dt=_DT,
                                Re=_RE,
                                check_freq=1,
                                dir_name=save_dir
                    )
    
    data = gen_trajs(n_trajs = 5, 
                     seed = 0,
                     control_type='sine',
                     save_path=save_path,
                     **rand_then_none_kwargs)
    