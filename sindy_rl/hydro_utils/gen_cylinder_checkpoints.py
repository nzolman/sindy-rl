# ----------------------------------------------------
# Used for generating checkpoints for the Cylinder env
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
                                dt=1e-2,
                               make_plot=False):
    '''
    NOTE: Uses the Flow Env API, not the surrogate cylinder!
    '''
    env_config = {
        "flow": hgym.Cylinder,
        "flow_config": {
            'mesh':mesh,
            'Re': Re,
            'actuator_integration': 'implicit'    
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
                        make_plot = make_plot)
    return env

def produce_checkpoints(env, 
                        n_steps, 
                        dir_name=None, 
                        fname='no_control_{:05d}',
                        render_freq=1000,
                        make_plot = False):
    
    os.makedirs(dir_name, exist_ok=True)
    
    for i in tqdm(range(n_steps)):
        action = 0
        env.step(action)
        if (i % render_freq) == (render_freq -1):
            if make_plot:
                save_plot(env, idx=i, 
                        fname=fname + '.png', 
                        dir_name=dir_name)
            ckpt_name = os.path.join(dir_name, fname + '.ckpt').format(i+1)
            env.flow.save_checkpoint(ckpt_name)
    print('done.')


if __name__ == '__main__':
    _RE = 100
    _DT = 1e-3
    _MESH = 'medium'
    _DATE = '2023-12-13'
    save_dir = os.path.join(_parent_dir, 
                            'data/hydro/cylinder/', 
                            f'{_DATE}_{_MESH}/Re={_RE}_dt=1e-3/snapshots')
    
    make_no_control_checkpoints(save_dir=save_dir, 
                                mesh=_MESH,
                                n_steps=int(1e6),
                                Re=_RE,
                                dt=_DT,
                                render_freq=2500,
                                make_plot=False
                                )