# ----------------------------------------------------
# Used for generating off-policy data for SINDy
# Compatibale with the updated version of Hydrogym
# ----------------------------------------------------

import os
import yaml
from tqdm import tqdm
import numpy as np
import pickle
import traceback


from matplotlib import pyplot as plt
plt.style.use('ggplot')

from hydrogym import firedrake as hgym

from sindy_rl import _parent_dir, _data_dir
from sindy_rl.env import rollout_env
from sindy_rl.policy import OpenLoopRandRest, OpenLoopSinRest, OpenLoopRandRest
from sindy_rl.traj_buffer import BaseTrajectoryBuffer


output_dir = os.path.join(_data_dir, 'control_checkpoints')



def save_data(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def gen_ctrl_then_none(env,
                       n_steps=1000, n_none=1000,
                       seed = 0, k_int = 1, save_freq = 100,zero_hold_n=10,
                       save_dir='',
                       ctrl_type = 'sine'
                       ):
    '''
    '''
    np.random.seed(seed)
    env.action_space.seed(seed)
    env.reset()
    
    u_dim = env.action_space.shape[0]
    
    Dt = env.solver.dt
    t_rest = n_steps * Dt
    if ctrl_type == 'sine': 
        k_int = k_int * np.ones(u_dim)
        
        # random parameters defining the sine wave
        max_amp = (env.flow.MAX_CONTROL_UP - env.flow.MAX_CONTROL_LOW)/2
        mid = (env.flow.MAX_CONTROL_LOW + env.flow.MAX_CONTROL_UP)/2 
        
        amp = np.random.uniform(low=0.25, high=0.75, size=(u_dim,)) * max_amp
        phase = np.random.uniform(low=-np.pi, high=np.pi, size=(u_dim,))
        
        offset = mid + 0.25 * max_amp * np.random.uniform(-1, 1, size=(u_dim,)) 
        
        f0 = 5.556
    
        policy = OpenLoopSinRest(t_rest, dt= Dt, 
                                amp=amp, phase=phase, 
                                offset=offset, f0=f0, k=k_int
                                )
    else: 
        policy = OpenLoopRandRest(action_space = env.action_space, 
                                     zero_hold_n=zero_hold_n, steps_rest = n_steps,
                                     seed=seed)
    
    def save_callback(idx, env):
        if (idx % save_freq == 0):
            save_path = os.path.join(save_dir,f'{idx:07}.ckpt')
            env.flow.save_checkpoint(save_path)
    
    x_list, u_list, rew_list = rollout_env(env, policy, 
                                           n_steps=n_steps + n_none,
                                           verbose=True,
                                           env_callback=save_callback
                                           )
    data = {'x': x_list, 'u': u_list, 'r': rew_list}
    save_path = os.path.join(save_dir, 'traj.pkl')
    
    save_data(data, save_path)
    return x_list, u_list, rew_list


def save_log(save_dir, flow_name, flow_config, dt, n_trajs, save_freq, traj_kwargs):
    '''log config used to generate traj'''
    
    log_config = dict(flow=flow_name, flow_config=flow_config, 
                      n_trajs=n_trajs, save_freq=save_freq,
                      dt=dt, traj_kwargs=traj_kwargs)
    
    save_path = os.path.join(save_dir, 'log.yaml')
    with open(save_path, 'w') as f:
        yaml.safe_dump(log_config, f)
    

def gen_trajs(flow_name, flow_config, traj_kwargs = {},
              n_trajs = 5, save_freq=100,
              seed=0, dt=1e-2, max_k_int = 5):
    ctrl_type = traj_kwargs.get('ctrl_type', 'sine')
    
    # build config
    env_config = {
        "flow": getattr(hgym, flow_name),
        "flow_config": flow_config,
        "solver": hgym.SemiImplicitBDF,
        "solver_config": {"dt": dt},
    }
    
    mesh = flow_config.get('mesh', '')
    Re = flow_config.get('Re')
    
    env = hgym.FlowEnv(env_config)
    
    # setup dirs and log
    save_dir = os.path.join(output_dir,  f'{flow_name}_{mesh}_Re={Re}_{ctrl_type}')
    os.makedirs(save_dir, exist_ok=True)
    save_log(save_dir, flow_name, flow_config, dt, n_trajs, save_freq, traj_kwargs)
    
    # iterate through the number of trajectories
    traj_buffer = BaseTrajectoryBuffer()
    for i in range(n_trajs):
        
        # setup traj dir
        print(f'Starting Traj {i}')
        traj_dir = os.path.join(save_dir, f'{i:03}')
        os.makedirs(traj_dir, exist_ok=True)
        
        
        # generate traj
        try:
            x, u, rew = gen_ctrl_then_none(
                            env, 
                            k_int = 1/((i%max_k_int) + 1), # only affects the sine policy
                            seed = seed+i, 
                            save_freq = save_freq, 
                            save_dir=traj_dir,
                            **traj_kwargs
                            )
            traj_buffer.add_data(x, u, rew)
            
            save_path = os.path.join(save_dir, 'traj.pkl')
            # iteratively save the data
            traj_buffer.save_data(save_path)

        except Exception as e:
            print(f'FAILED {i}', e)
            print(traceback.format_exc())
            continue
    return traj_buffer


def gen_cylinder(Re):
    ckpt_path = os.path.join(_data_dir, 
                             f'unsteady_checkpoints/Cylinder_medium_Re={Re}/0010000.ckpt'
                             )
    flow_config = {
        "mesh": "medium",
        "Re": Re,
        "observation_type": "velocity_probes",
        "restart": [ckpt_path]
    }
    
    traj_kwargs = {
         'n_steps':int(6e3), # 60 seconds
         'n_none': 1000,     # 10 seconds
    }
    gen_trajs('Cylinder', flow_config, 
              traj_kwargs = traj_kwargs,
              n_trajs = 5, 
              save_freq=100,
              seed=0, 
              dt=1e-2)
    

def gen_rot_cylinder(Re):
    ckpt_path = os.path.join(_data_dir, 
                             f'unsteady_checkpoints/Cylinder_medium_Re={Re}/0010000.ckpt'
                             )
    flow_config = {
        "mesh": "medium",
        "Re": Re,
        "observation_type": "velocity_probes",
        "restart": [ckpt_path]
    }
    
    traj_kwargs = {
         'n_steps':int(6e3), # 60 seconds
         'n_none': 1000,     # 10 seconds
    }
    gen_trajs('RotaryCylinder', flow_config, 
              traj_kwargs = traj_kwargs,
              n_trajs = 5, 
              save_freq=100,
              seed=0, 
              dt=1e-2)
    
def gen_pinball(Re=100):
    ckpt_path = os.path.join(_data_dir, 
                             f'unsteady_checkpoints/Pinball_medium_Re={Re}/0020000.ckpt'
                             )
    flow_config = {
        "mesh": "medium",
        "Re": Re,
        "observation_type": "velocity_probes",
        "restart": [ckpt_path]
    }
    
    traj_kwargs = {
         'ctrl_type': 'random',
         'zero_hold_n': 5,   # 20 Hz
         'n_steps': 3000,    # 30 seconds
         'n_none':  500,      # 5 seconds
    }
    gen_trajs('Pinball', flow_config, 
              traj_kwargs = traj_kwargs,
              n_trajs = 5, 
              save_freq=100,
              seed=0, 
              dt=1e-2)
    
if __name__== '__main__':
    gen_pinball(Re=100)