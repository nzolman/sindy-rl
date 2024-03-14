# ----------------------------------------------------
# Used for generating data with no control
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
from sindy_rl.hydro_utils.gen_cylinder_data import gen_trajs

    
if __name__ == '__main__':
    _RE = 100
    _DT = 1e-2
    N_TRAJ = 1
    _N_SKIP = 5
    _CONTROL_FREQ =  10 # 10 Hz
    _MESH = 'medium'
    
    _N_STEPS = 0
    _N_NONE = int(1e4)
    
    data_dir = os.path.join(_parent_dir, 
                            'data/hydro/cylinder/'
                            )

    
    save_data_dir = os.path.join(data_dir ,
                            f'2024-01-20_medium_null/Re={_RE}_dt=1e-2'
                            )
    
    save_dir = os.path.join(save_data_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 
                             'traj_null.pkl'
                             )
    check_path = None
    rand_then_none_kwargs = dict(
                                checkpoint = check_path,
                                env_class = CylinderLiftEnv,
                                n_steps = _N_STEPS,
                                n_none=_N_NONE,
                                # n_none=10,
                                control_freq=_CONTROL_FREQ,
                                n_skip = _N_SKIP,
                                mesh=_MESH,
                                dt=_DT,
                                Re=_RE,
                    )
    
    data = gen_trajs(n_trajs = N_TRAJ, 
                     seed = 1,
                     control_type='sine',
                     save_path=save_path,
                     **rand_then_none_kwargs)
    