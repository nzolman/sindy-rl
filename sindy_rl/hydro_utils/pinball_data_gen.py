import os
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import time
import pickle
import firedrake as fd
from gym.spaces import Box

comm = fd.COMM_WORLD
rank = comm.Get_rank()

from hydrogym import firedrake as hgym

from sindy_rl.envs.hydroenv import SurrogateCylinder, SurrogateHydroEnv, SurrogatePinball

flow_env_config =  {
            "flow_config": {
                'mesh': 'coarse',
                # 'restart': './tmp_1.h5'
            },
        }

env_config = {
        'hydro_config': flow_env_config,
        'dyn_model': None,
        'use_omega': True,
        'use_CL_dot': True, 
        'control_freq': 50,   
        'n_skip_on_reset': 50,     
        'max_episode_steps': 10000,
        'control_mode': 3
}

N_COLLECT = 100
MAX_TORQUE = 0.25
fname = 'coarse-pinball_ctrl_freq=50_torque=0.25_collec=100-rand_100-null'

t0 = time.time()
env = SurrogatePinball(env_config = env_config)
env.action_space = Box(-MAX_TORQUE, MAX_TORQUE,(3,), np.float64)

env.action_space.seed(0)
obs_list = [env.reset()]
u_list = []

for i in range(N_COLLECT):
    action = env.action_space.sample()
    obs = env.step(action)[0]
    obs_list.append(obs)
    u_list.append(action)

for i in range(N_COLLECT):
    action = np.zeros(3)
    obs = env.step(action)[0]
    obs_list.append(obs)
    u_list.append(action)

obs_list.pop(-1)
obs_list = np.array(obs_list)
u_list = np.array(u_list)


data = {'x': [obs_list], 
        'u': [u_list]}

# save data
with open(fname + '.pkl', 'wb') as f:
    pickle.dump(data,f)

# save checkpoint
env.flow.save_checkpoint(fname + '.ckpt')

if rank == 0:
    t1 = time.time() - t0
    print('time: ', t1)
    print('len: ', len(obs_list))
