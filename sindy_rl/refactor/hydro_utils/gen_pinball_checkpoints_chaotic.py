import os
os.environ['OMP_NUM_THREADS']='1'
from tqdm import tqdm
import numpy as np
import pickle

from matplotlib import pyplot as plt
plt.style.use('ggplot')

from hydrogym import firedrake as hgym
from sindy_rl import _parent_dir
from sindy_rl.refactor.policy import RandomPolicy
from sindy_rl.refactor.env import rollout_env
from sindy_rl.refactor.traj_buffer import BaseTrajectoryBuffer


def save_plot(env, idx, fname='no_control_{:05d}.png', dir_name=None):
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    im = env.render(axes=axes)
    axes.set_title(f'Mesh Step: {idx} Timestep: {env.solver.t}')
    fig_name = os.path.join(dir_name, fname.format(idx+1))
    fig.savefig(fig_name, bbox_inches="tight")
    plt.close()
    

def make_no_control_checkpoints(save_dir, 
                                mesh, 
                                n_steps=10000, 
                                render_freq=1000,
                                Re=30,
                                prev_checkpoint=None,
                                prev_idx = 0,
                                dt=1e-2):
    '''
    NOTE: Uses the Flow Env API
    '''
    env_config = {
        "flow": hgym.Pinball,
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
    if prev_checkpoint is not None:
        env_config['flow_config']['restart'] = prev_checkpoint
    env = hgym.FlowEnv(env_config)
    env.reset()
    
    produce_checkpoints(env, 
                        n_steps=n_steps, 
                        render_freq=render_freq,
                        dir_name=save_dir,
                        prev_idx=prev_idx,
                        fname = 'no_control_{:05d}')
    return env

def produce_checkpoints(env, 
                        n_steps, 
                        dir_name=None, 
                        prev_idx=0, 
                        fname='no_control_{:05d}',
                        render_freq=1000):
    
    os.makedirs(dir_name, exist_ok=True)
    
    def check_callback(i, env):
        idx = i + prev_idx
        if (idx % render_freq) == (render_freq -1) or (idx == 0):
            save_plot(
                        env, 
                        idx=idx, 
                        fname=fname + '.png', 
                        dir_name=dir_name
                    )
            ckpt_name = os.path.join(dir_name, fname + '.ckpt').format(idx+1)
            env.flow.save_checkpoint(ckpt_name)
        return None
    
    policy = RandomPolicy(env.action_space)
    policy.magnitude = 0.0
    obs_list, act_list, rew_list = rollout_env(env, policy, n_steps=n_steps, 
                                               verbose=True, env_callback= check_callback)
    
    buffer  = BaseTrajectoryBuffer()
    buffer.add_data(obs_list, act_list, rew_list)
    
    buffer.save_data(os.path.join(dir_name, f'traj_{prev_idx + len(obs_list[0])}.pkl'))

    print('done.')
    
    
def excite_instability(env, magnitude=0.1, freq=10, n_steps = 1000):
    policy =  RandomPolicy(env.action_space)
    policy.magnitude=magnitude
    obs_list=[]
    act_list = []
    rew_list = []
    for i in tqdm(range(n_steps)):
        if i % freq == 0: 
            act = policy.compute_action(None)
            
            # let's only actuate the back two cylinders with antisymmetric control
            act[0] = 0.0
            act[1] = -1.0 * act[2]

        res = env.step(act)
        obs_list.append(res[0])
        rew_list.append(res[1])
        act_list.append(act)
    return np.array(obs_list), np.array(act_list), np.array(rew_list)

def instability_check(save_dir, 
                    mesh, 
                    Re=30,
                    prev_checkpoint=None,
                    instability_kwargs=None,
                    dt=1e-2):
    '''
    NOTE: Uses the Flow Env API
    '''
    
    os.makedirs(save_dir, exist_ok=True)
    
    env_config = {
        "flow": hgym.Pinball,
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
    if prev_checkpoint is not None:
        env_config['flow_config']['restart'] = prev_checkpoint
    env = hgym.FlowEnv(env_config)
    env.reset()
    
    kwargs = instability_kwargs or {}
    obs, acts, rews = excite_instability(env, **kwargs)
    
    ckpt_name = os.path.join(save_dir, 'instability' + '.ckpt')
    env.flow.save_checkpoint(ckpt_name)
    
    buffer  = BaseTrajectoryBuffer()
    buffer.add_data([obs], [acts], [rews])
    buffer.save_data(os.path.join(save_dir, f'traj.pkl'))
    

if __name__ == '__main__':
    _RE = 140
    _DT = 1e-2

    prev_checkpoint = os.path.join(_parent_dir, 
                                   # 'data/hydro/pinball/2023-10-14_fine_instability/Re=30_dt=1e-3/snapshots/no_control_500000.ckpt'
                                   'data/hydro/pinball/2024-01-20_chaos/Re=120_dt=1e-2/snapshots/no_control_25000.ckpt'
                                  )
    

    save_dir = os.path.join(_parent_dir, 
                            'data/hydro/pinball/', 
                            f'2024-01-20_chaos/Re={_RE}_dt=1e-2/snapshots')

    make_no_control_checkpoints(save_dir=save_dir, 
                                mesh='fine',
                                n_steps=int(2.5e4),
                                Re=_RE,
                                dt=_DT,
                                prev_checkpoint=prev_checkpoint,
                                prev_idx=0,
                                render_freq=250
                                )