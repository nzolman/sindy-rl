# Example Usage: 
# mpiexec -np 12 python gen_cylinder_checkpoints.py

import os
os.environ['OMP_NUM_THREADS']='1'
from tqdm import tqdm
import numpy as np
import pickle

from matplotlib import pyplot as plt
plt.style.use('ggplot')

from hydrogym import firedrake as hgym
from sindy_rl import _parent_dir


def save_plot(env, idx, fname='no_control_{:05d}.png', dir_name=None):
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    env.render(axes=axes)
    axes.set_title(f'Coarse Mesh Step: {idx} Timestep: {env.solver.t}')
    fig.tight_layout()
    fig_name = os.path.join(dir_name, fname.format(idx+1))
    plt.savefig(fig_name, bbox_inches="tight")
    plt.close()
    

def make_no_control_checkpoints(save_dir, 
                                mesh, 
                                n_steps=10000, 
                                render_freq=1000,
                                Re=100,
                                dt=1e-2):
    '''
    NOTE: Uses the Flow Env API, not the surrogate cylinder!
    '''
    env_config = {
        "flow": hgym.Cylinder,
        "flow_config": {
            'mesh':mesh,
            'Re': Re
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
                        fname = 'no_control_{:05d}')
    return env

def produce_checkpoints(env, 
                        n_steps, 
                        dir_name=None, 
                        fname='no_control_{:05d}',
                        render_freq=1000):
    
    os.makedirs(dir_name, exist_ok=True)
    
    for i in tqdm(range(n_steps)):
        action = 0
        env.step(action)
        if (i % render_freq) == (render_freq -1):
            save_plot(env, idx=i, 
                    fname=fname + '.png', 
                    dir_name=dir_name)
            ckpt_name = os.path.join(dir_name, fname + '.ckpt').format(i+1)
            env.flow.save_checkpoint(ckpt_name)
    print('done.')


if __name__ == '__main__':
    _RE = 100
    _DT = 1e-3
    save_dir = os.path.join(_parent_dir, 
                            'data/hydro/cylinder/', 
                            f'2023-10-02_medium/Re={_RE}_dt=1e-3/snapshots')
    
    make_no_control_checkpoints(save_dir=save_dir, 
                                mesh='medium',
                                n_steps=int(1e5),
                                Re=_RE,
                                dt=_DT,
                                render_freq=2500
                                )