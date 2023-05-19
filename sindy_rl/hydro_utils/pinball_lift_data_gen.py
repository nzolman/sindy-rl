import os
os.environ['OMP_NUM_THREADS']='1'
from tqdm import tqdm
import numpy as np
import pickle
import traceback
# import sys

from matplotlib import pyplot as plt
plt.style.use('ggplot')

from hydrogym import firedrake as hgym
from sindy_rl import _parent_dir
from sindy_rl.envs.hydroenv import SurrogatePinballLIFT, SurrogatePinball

def save_plot(env, idx, fname='no_control_{:05d}.png', dir_name=None):
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    env.render(axes=axes)
    axes.set_title(f'Coarse Mesh Step: {idx} Timestep: {env.solver.t}')
    fig.tight_layout()
    fig_name = os.path.join(dir_name, fname.format(idx+1))
    plt.savefig(fig_name)
    plt.close()
    

def make_no_control_checkpoints(save_dir, 
                                mesh, 
                                n_steps=10000, 
                                render_freq=1000,
                                Re=30,
                                dt=1e-3,
                                restart_num=None,
                                n_excite=0):
    '''
    NOTE: Uses the Flow Env API, not the surrogate cylinder!
    '''
    if restart_num:
        restart_path = os.path.join(save_dir,f'no_control_{restart_num:05d}.ckpt')
        print('restarting from', restart_path)
    else:
        restart_path = None
    env_config = {
        "flow": hgym.Pinball,
        "flow_config": {
            'mesh':mesh,
            'Re': Re,
            'restart': restart_path,
        },
        "solver": hgym.IPCS,
        "solver_config":
            {
                'dt': dt
            }
    }
    env = hgym.FlowEnv(env_config)
    env.reset()
    
    produce_checkpoints(env, 
                        n_steps=n_steps, 
                        render_freq=render_freq,
                        dir_name=save_dir,
                        fname = 'no_control_{:05d}',
                        restart_num=restart_num, 
                        n_excite=n_excite)
    return env

def produce_checkpoints(env, 
                        n_steps, 
                        dir_name=None, 
                        fname='no_control_{:05d}',
                        render_freq=1000,
                        restart_num = 0,
                        n_excite=0):
    
    os.makedirs(dir_name, exist_ok=True)
    
    for i in tqdm(range(restart_num, n_steps)):
        if (i - restart_num) < n_excite:
            action = (1e-2)*np.ones_like(env.action_space.sample())
        # be wary of this
        else: 
            action = 0 * env.action_space.sample()
        env.step(action)
        if (i % render_freq) == (render_freq -1):
            # save_plot(env, idx=i, 
            #         fname=fname + '.png', 
            #         dir_name=dir_name)
            ckpt_name = os.path.join(dir_name, fname + '.ckpt').format(i+1)
            env.flow.save_checkpoint(ckpt_name)
    print('done.')
    
def gen_random_then_none(checkpoint,
                         env_class=SurrogatePinballLIFT,
                       n_steps=1000, 
                       n_none=1000,
                       seed = 0,
                       control_freq=50, 
                       mesh = 'fine',
                       n_skip=10,
                       Re=30,
                       dt=1e-3
                       ):
    '''
    NOTE: THIS USES the surrogate API!
    '''

    flow_config = {
        "flow": hgym.Pinball,
        "flow_config": {
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
            'control_mode': 1,
            'n_skip_on_reset': n_skip, 
            'max_episode_steps': 2*n_steps
    }

    env = env_class(env_config=env_config)

    np.random.seed(seed)
    env.action_space.seed(seed)
    
    u_list = []
    x_list = [env.reset()]
    
    # take random control
    for i in tqdm(range(n_steps)):
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)
        u_list.append(action)
        x_list.append(obs)

    # relax, no control
    for i in tqdm(range(n_none)):
        action = 0*env.action_space.sample()
        obs, _, _, _ = env.step(action)
        u_list.append(action)
        x_list.append(obs)
    x_list.pop(-1)
    return np.array(x_list), np.array(u_list)


def gen_sine_then_none(checkpoint,
                       env_class = SurrogatePinballLIFT,
                       n_steps=1000,
                       n_none=1000, 
                       k_int = 1,
                       seed = 0,
                       control_freq=50, 
                       mesh = 'fine',
                       n_skip=10,
                       Re=30,
                       dt=1e-3,
                       ):
    '''
    NOTE: THIS USES the surrogate API!
    '''
       
    flow_config = {
        "flow": hgym.Pinball,
        "flow_config": {
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
            'control_mode': 1,
            'n_skip_on_reset': n_skip, 
            'max_episode_steps': 2*n_steps
    }

    env = env_class(env_config=env_config)

    np.random.seed(seed)
    env.action_space.seed(seed)
    
    u_list = []
    x_list = [env.reset()]
    t=0
    # random parameters defining the sine wave
    amp = np.random.uniform(low=0.25, high=1.0)
    phase = np.random.uniform(low=0, high=np.pi)
    offset = np.random.uniform(-np.pi/2 + 1, np.pi/2 - 1)
    f0 = 5.556
    # k_int = seed + 1 #np.random.randint(low=1,high=5)
    
    # take random control
    for i in tqdm(range(n_steps)):
        t = i * env.solver.dt * env.control_freq
        action = amp*np.sin(t*(2*np.pi*k_int)/f0 - phase) + offset
        obs, _, _, _ = env.step(action)
        u_list.append(np.array([action]))
        x_list.append(obs)

    # relax, no control
    for i in tqdm(range(n_none)):
        action = 0*env.action_space.sample()
        obs, _, _, _ = env.step(action)
        u_list.append(action)
        x_list.append(obs)
    x_list.pop(-1)
    return np.array(x_list), np.array(u_list)

def gen_trajs(n_trajs = 5, seed=0, control_type='random', save_path=None, **kwargs):
    x_lists = []
    u_lists = []
    for i in range(n_trajs):
        print(f'Starting Traj {i}')
        try:
            if control_type == 'random':
                x, u = gen_random_then_none(seed = seed+i, **kwargs)
            elif control_type == 'sine':
                x, u = gen_sine_then_none(k_int = 1/(i + 1), seed = seed+i, **kwargs)
            else:
                raise NotImplementedError('invalid control_type')
            x_lists.append(x)
            u_lists.append(u)
            data = {'x': x_lists, 'u': u_lists}
            
            with open(save_path, 'wb') as f:
                print(f'Saving Traj {i}')
                pickle.dump(data, f)
        except Exception as e:
            print(f'FAILED {i}', traceback.format_exc())
            continue
    return data
    
if __name__ == '__main__':
    _RE = 30
    _DT = 1e-3
    # save_dir = os.path.join(_parent_dir, 
    #                         'data/hydro/pinball/', 
    #                         f'2023-05-10_fine/Re={_RE}_dt=1e-3/snapshots')
    
    # os.makedirs(save_dir, exist_ok=True)
    # make_no_control_checkpoints(save_dir=save_dir, 
    #                             mesh='fine',
    #                             n_steps=int(3e5),
    #                             Re=_RE,
    #                             dt=_DT,
    #                             render_freq=2500,
    #                             restart_num = 200000,
    #                             # n_excite=1000
    #                             )
    
    data_dir = os.path.join(_parent_dir, 
                            'data/hydro/pinball/', 
                            f'2023-05-10_fine/Re={_RE}_dt=1e-3/'
                            # f'2023-04-07_medium/Re={_RE}_dt=1e-3/control'
                            # f'2023-03-15_medium/'
                            )
    check_path = os.path.join(data_dir, 
                              'snapshots',
                              'no_control_300000.ckpt'
                            #   'no_control_100000.ckpt'
                              )
    save_dir = os.path.join(data_dir, 
                             'control')
    save_path = os.path.join(save_dir,
                            #  'traj=8_freq=10_steps=10000_sine_ctrl_then_none-04-07_low_freq.pkl')
                             f'env=CL_CD_traj=5_freq=10_dt=1e-3_res=med_Re={_RE}_steps=6000_low-sine-ctrl-then_none.pkl'
                             )
    
    print('making dir: ', save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    rand_then_none_kwargs = dict(
                                checkpoint = check_path,
                                env_class = SurrogatePinball,
                                n_steps=5000,
                                # n_steps=10,
                                n_none=1000, 
                                # n_none=10,
                                control_freq=10, #100 hz when dt=1e-3
                                n_skip = 10,
                                mesh='fine',
                                dt=_DT,
                                Re=_RE,
                    )
    
    data = gen_trajs(n_trajs = 5, 
                     seed = 0,
                     control_type='sine',
                     save_path=save_path,
                     **rand_then_none_kwargs)
    