import numpy as np
from copy import deepcopy
import os
import pickle

# from pysindy import PolynomialLibrary, CustomLibrary, FourierLibrary

from sindy_rl import _parent_dir

from sindy_rl.env import rollout_env, BaseEnsembleSurrogateEnv
from sindy_rl.policy import RandomPolicy, RLlibPolicyWrapper, FixedPolicy

from sindy_rl.sparse_policy.utils import get_models, fit_policies_v, hyper_mesh


def dist_in_hull(points, n):
    '''
    For sampling distributions uniformly inside the convex hull of a set.
    Source: https://stackoverflow.com/questions/59073952/how-to-get-uniformly-distributed-points-in-convex-hull
    '''
    dims = points.shape[-1]
    hull = points[ConvexHull(points).vertices]
    deln = hull[Delaunay(hull).simplices]

    vols = np.abs(det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)    
    sample = np.random.choice(len(vols), size = n, p = vols / vols.sum())

    return np.einsum('ijk, ij -> ik', deln[sample], dirichlet.rvs([1]*(dims + 1), size = n))



def fit_pin(env, dyn_model, policy, unbounded_policy,  hyper_mesh, 
            data,
            n_dyn_rollout=50000, 
            n_reset = 100,
            n_resamples = 2,
            seed=0, 
            use_median = False, 
            poly_deg = 2, 
            include_bias=True, 
            ics = 'gauss',
            post_traj_noise = 0.2,
            n_state = 10):
    
    # randomly generate initial conditions from off policy
    CLIP_BOUNDS = np.array([[-1.0], [1.0]])
    u_max = CLIP_BOUNDS.max()
    np.random.seed(seed)
    if ics == 'gauss': 
        noise_scale =  0.5 
        ic_X = np.random.normal(loc=np.zeros(n_state), scale=noise_scale,
                                size = (n_dyn_rollout, n_state)
                                )
    elif ics == 'uniform':
        low = -1.0
        high = 1.0
        ic_X = np.random.uniform(low = low, high = high,
                                size = (n_dyn_rollout, n_state)
                                )
    elif ics == 'convex': 
        ic_X = dist_in_hull(data['on_pi']['x'][-1], n_dyn_rollout)

    else:
        ic_X = np.concatenate(data['off_pi']['x'])
    
        noise_scale =  0.1
        ic_X += np.random.normal(loc=np.zeros(n_state), scale=noise_scale,
                                size = ic_X.shape
                                )

    env.dynamics_model = deepcopy(dyn_model)


    env.dynamics_model.set_median_coef_()

    env.use_real_env = False
    env.reset_from_buffer = True
    env.buffer_dict = {'x': [ic_X]}

    env.reset()
    traj_obs, traj_acts, traj_rews = rollout_env(env, policy, 
                                                n_steps = n_dyn_rollout, 
                                                n_steps_reset=n_reset, verbose = True,
                                                seed=seed+1)
    # randomly generate noise on top of this
    X_data = np.concatenate(traj_obs)
    X_data = np.concatenate([X_data for n in range(n_resamples)])

    noise_scale =   post_traj_noise
    noise =  np.random.normal(loc = np.zeros(n_state),
                                scale = noise_scale,
                                size = X_data.shape)
    X_data = np.concatenate([
                            X_data + noise])
    
    U_data = unbounded_policy.algo.compute_actions(X_data, explore=False)[0]
    
    # turns out RLlib has a defaults scale for the output between -1 and 1
    # however, by using "unsquash" it also clips the values. So we'll manually
    # renormalize here
    U_data *= u_max
    
    # we're clipping values much larger than u_max in case things get excessive.
    # But ideally we'd be able to give some wiggle room to the polynomials
    U_data = np.clip(U_data, -10, 10)

    traj_X_normal = X_data
    traj_U_normal = U_data

    traj_X_eval = np.concatenate(data['on_pi']['x']) 
    traj_U_eval = unbounded_policy.algo.compute_actions(traj_X_eval, explore=False)[0]
    traj_U_eval *= u_max
    traj_U_eval = np.clip(traj_U_eval, -10, 10)

        
    normal_traj_data = {
            'X_data': traj_X_normal,
            'U_data': traj_U_normal,
            'X_eval': traj_X_eval,
            'U_eval': traj_U_eval,
        }

    data_dicts = {
                    'normal_traj': normal_traj_data,
                }
    

    # sweep through hyper params
    val = normal_traj_data
    
    res_dict = fit_policies_v(**val, hyper_mesh=hyper_mesh, n_models = 20, 
                            use_median = use_median, clip_params = CLIP_BOUNDS, bounds = CLIP_BOUNDS,
                            poly_deg = poly_deg, include_bias= include_bias)
    sparse_policy = res_dict['best_policy']
    
    return res_dict, normal_traj_data, traj_obs


if __name__ == '__main__': 
    from tqdm import tqdm
    trial_dir = os.path.join(_parent_dir, 'data/agents/pinball/dyna')
    check_idx = 200

    pin_dyn, pin_rew, pin_data, pin_policy, pin_env = get_models(
                                                                trial_dir, 
                                                                check_idx,
                                                                return_policy=True
                                                            )

    pin_unbounded_policy = RLlibPolicyWrapper(pin_policy.algo.get_policy(), mode='policy')

    pin_res_dict, pin_normal_traj_data, traj_obs  = fit_pin(pin_env, 
                                                            pin_dyn, 
                                                            pin_policy,
                                                            pin_unbounded_policy,  
                                                            hyper_mesh, 
                                                            pin_data,
                                                            n_dyn_rollout=10000, 
                                                            n_reset = 100,
                                                            n_resamples=2,
                                                            seed=0, 
                                                            use_median = False, 
                                                            poly_deg = 2,
                                                            include_bias=True, 
                                                            ics = 'off',
                                                            post_traj_noise=0.1,
                                                            n_state=10
                                                            )

    pin_sparse_policy = pin_res_dict['best_policy']


    save_dir = ckpt_path

    sparse_path = os.path.join(save_dir, f'sparse_policy.pkl')

    with open(sparse_path, 'wb') as f:
        pickle.dump(pin_sparse_policy,f)
