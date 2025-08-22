import hydrogym.firedrake as hgym
import firedrake as fd
import os
import yaml
import pickle
from tqdm import tqdm

from sindy_rl import _parent_dir, _data_dir

load_dir = os.path.join(_data_dir, 'steady_checkpoints')
output_dir = os.path.join(_data_dir, 'unsteady_checkpoints')

def save_config(config, save_path): 
    with open(save_path, 'w') as f:
        yaml.safe_dump(config, f)

def save_data(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

def gen_unsteady(flow_name, flow_config, start_idx = 0, dt=1e-2, n_control=100, n_no_control=int(1e4), ckpt_freq=100, sample_seed = 0):
    
    env_config = {
        "flow": getattr(hgym, flow_name),
        "flow_config": flow_config,
        "solver": hgym.SemiImplicitBDF,
        "solver_config": {"dt": dt},
    }
    mesh = flow_config.get('mesh', '')
    Re = flow_config.get('Re')
    save_dir = os.path.join(output_dir, f'{flow_name}_{mesh}_Re={Re}')
    os.makedirs(save_dir, exist_ok=True)
    
    env = hgym.FlowEnv(env_config)
    env.reset()
    
    env.action_space.seed(sample_seed)
    
    log_config = {
        'flow': flow_name,
        'n_control': n_control,
        'n_no_control': n_no_control,
        'ckpt_freq': ckpt_freq,
        **flow_config
        }
    log_path = os.path.join(save_dir, 'log.yaml')
    save_config(log_config, log_path)
    
    print('starting control')
    u_list, obs_list, r_list = control(env, start_idx=start_idx, n_steps= n_control, 
                                       save_dir=save_dir, ckpt_freq=ckpt_freq
                                       )

    data_path = os.path.join(save_dir, 'control.pkl')
    data = {'u': u_list,
            'x': obs_list,
            'r': r_list}
    save_data(data, data_path)
    
    print('starting no control')
    u_list_no, obs_list_no, r_list_no = no_control(env, start_idx=n_control, n_steps= n_no_control, 
                                                    save_dir=save_dir, ckpt_freq=ckpt_freq
                                                    )
    data_path = os.path.join(save_dir, 'no_control.pkl')
    data = {'u': u_list_no,
            'x': obs_list_no,
            'r': r_list_no}
    save_data(data, data_path)
    

def no_control(env, start_idx, n_steps, ckpt_freq, save_dir=''):
    u_list = []
    obs_list = []
    r_list = []
    for i in tqdm(range(n_steps)):
        u = 0 * env.action_space.sample()
        u_list.append(u)
        obs, r, _, _ =  env.step(u)
        
        u_list.append(u)
        obs_list.append(obs)
        r_list.append(r)
        
        idx = start_idx + i        
        if i % ckpt_freq == 0:

            save_path = os.path.join(save_dir, f'{idx:07}.ckpt')
            env.flow.save_checkpoint(save_path)
            data_path = os.path.join(save_dir, 'no_control.pkl')
            data = {'u': u_list,
                    'x': obs_list,
                    'r': r_list}
            save_data(data, data_path)
    return u_list, obs_list, r_list

def control(env, start_idx, n_steps, ckpt_freq, save_dir=''):
    u_list = []
    obs_list = []
    r_list = []
    for i in tqdm(range(n_steps)):
        u = env.action_space.sample()
        u_list.append(u)
        obs, r, _, _ =  env.step(u)
        
        u_list.append(u)
        obs_list.append(obs)
        r_list.append(r)
        
        idx = start_idx + i
        if i % ckpt_freq == 0:
            save_path = os.path.join(save_dir, f'{idx:07}.ckpt')
            env.flow.save_checkpoint(save_path)
            data_path = os.path.join(save_dir, 'control.pkl')
            data = {'u': u_list,
                    'x': obs_list,
                    'r': r_list}
            save_data(data, data_path)
    return u_list, obs_list, r_list


def gen_unsteady_cylinder():
    ckpt_path = os.path.join(load_dir, 'Cylinder-medium-steady_Re=100.ckpt')
    
    flow_config = {
        "mesh": "medium",
        "Re": 100,
        "observation_type": "velocity_probes",
        "restart": [ckpt_path]
    }
    
    return gen_unsteady(flow_name='Cylinder', flow_config=flow_config, 
                        start_idx = 0, 
                        dt=1e-2, 
                        n_control=100, 
                        n_no_control=int(1e4), 
                        ckpt_freq=100
                        )
    
def gen_unsteady_pinball(Re=100, dt=1e-2, n_control = 100, n_no_control = int(2e4),ckpt_freq=100):
    ckpt_path = os.path.join(load_dir, f'Pinball-medium-steady_Re={Re}.ckpt')
    flow_config = {
        "mesh": "medium",
        "Re": Re,
        "observation_type": "velocity_probes",
        "restart": [ckpt_path]
    }
    
    return gen_unsteady(flow_name='Pinball', flow_config=flow_config, 
                        start_idx = 0, 
                        dt=dt, 
                        n_control=n_control, 
                        n_no_control=n_no_control, 
                        ckpt_freq=ckpt_freq
                        )

if __name__ == '__main__':
    
    gen_unsteady_pinball(Re=300, 
                         dt = 5e-3, 
                         n_no_control=int(4e4), 
                         n_control = 200, 
                         ckpt_freq=200
                         )
    

    
    
