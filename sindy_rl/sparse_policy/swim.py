import os
import pickle
import numpy as np
from copy import deepcopy

from sindy_rl import _parent_dir
from sindy_rl.env import rollout_env
from sindy_rl.policy import RLlibPolicyWrapper
from sindy_rl.sparse_policy.utils import get_models, fit_policies_v, hyper_mesh

import ray
try:
    ray.init(log_to_driver=False)
except Exception as e:
    print(e)


if __name__ == '__main__':
    trial_dir = os.path.join(_parent_dir, 'data/agents/swimmer/dyna')
    
    # load models, data, and env
    swim_dyn, swim_rew, swim_data, swim_policy, swimmer_env = get_models(
                                                                        trial_dir,
                                                                        check_idx=1250,
                                                                        return_policy=True
                                                                )
    swim_unbounded_policy = RLlibPolicyWrapper(swim_policy.algo.get_policy(), mode='policy')


    def fit_swim(env, dyn_model, policy, unbounded_policy, hyper_mesh, n_dyn_rollout=50000, 
                seed=0, use_median = False, poly_deg = 3):
        '''
        Essentially a wrapper for fit_policies_v
        
        Arguments:
            env: sindy_rl.env.BaseEnsembleSurrogateEnv
                environment to run
            dyn_model: sindy_rl.dynamics.EnsembleSINDyDynamicsModel
                surrogate dynamics
            policy: sindy_rl.policy.RLlibPolicyWrapper
                policy to fit using rllib "algorithm" API, where actions are clipped
            unbounded_policy: sindy_rl.policy.RLlibPolicyWrapper
                policy using policy API, where actions are _NOT_ clipped
            hyper_mesh:
                np.meshgrid of hyper parameters to sweep over.
            n_dyn_rollout:
                number of surrogate experience points to generate
            use_median:
                whether to use the median ensemble coefficients (True) or the mean (False)
            poly_deg: maximal degree of polynomial for sparse policy.
        '''
        
        # randomly generate initial conditions similar to the SwimmerEnv
        np.random.seed(seed)
        N_ICs = 400

        ic_X = np.random.normal(loc=0, scale=0.2,
                                size = (N_ICs, 8)
                                )

        # rollout trajectories using the SINDy dynamics
        env.dynamics_model = deepcopy(dyn_model)
        env.dynamics_model.set_median_coef_()
        env.use_real_env = False
        env.reset_from_buffer = True
        env.buffer_dict = {'x': [ic_X]}
        env.reset()
        traj_obs, traj_acts, traj_rews = rollout_env(env, 
                                                    policy, 
                                                    n_steps = n_dyn_rollout, 
                                                    n_steps_reset=200, verbose = True,
                                                    seed=seed+1
                                                    )

        # randomly generate noise on top of the surrogate trajs
        n_resamples = 2
        X = np.concatenate(traj_obs)
        n_train = int(0.8*len(X)) # train data is the first 80%
        X_data = X[:n_train]
        X_data = np.concatenate([X_data for n in range(n_resamples)])

        noise_scale = 1e-1
        noise =  np.random.normal(loc = 0,
                                    scale = noise_scale,
                                    size = X_data.shape)
        X_data = np.concatenate([
                                X_data + noise])

        # evaluate the NN control on these trajectories
        U_data = unbounded_policy.algo.compute_actions(X_data, explore=False)[0]
        U_data = np.clip(U_data, -5, 5)

        traj_X_normal = X_data
        traj_U_normal = U_data
        
        # evaluation data is the last 20% of data
        traj_X_eval = X[n_train:]
        traj_U_eval = unbounded_policy.algo.compute_actions(traj_X_eval, explore=False)[0]
        traj_U_eval = np.clip(traj_U_eval, -5, 5)

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
        CLIP_BOUNDS = np.array([[-1.0, -1.0], [1.0, 1.0]])
        res_dict = fit_policies_v(**val, hyper_mesh=hyper_mesh, n_models = 20, 
                                use_median = use_median, clip_params = CLIP_BOUNDS, 
                                bounds = CLIP_BOUNDS, poly_deg = poly_deg)
        sparse_policy = res_dict['best_policy']
        
        return res_dict, normal_traj_data



    # fit the policies (sweep through hyper parameters)
    # and select best one
    swim_res_dict, swim_normal_traj_data = fit_swim(swimmer_env, swim_dyn, 
                                                swim_policy, swim_unbounded_policy, 
                                                hyper_mesh,
                                                n_dyn_rollout=10000,
                                                poly_deg=3,
                                                seed=0)

    swim_sparse_policy = swim_res_dict['best_policy']

    # example of saving the policy
    with open('swim_sparse_policy.pkl', 'wb') as f:
        pickle.dump(swim_sparse_policy, f)


    swim_sparse_policy.set_mean_coef_()

    swimmer_env.real_env.reset_on_bounds = False
    eval_obs, eval_acts, eval_rews = rollout_env(swimmer_env.real_env, 
                                                policy=swim_sparse_policy, 
                                                n_steps = 15000, verbose=True, 
                                                seed=42)
    
    perf = np.median([r.sum() for r in eval_rews])
    
    swim_baseline_obs, swim_baseline_acts, swim_baseline_rews = rollout_env(swimmer_env.real_env, 
                                                                policy=swim_policy, 
                                                                n_steps = 15000, 
                                                                verbose=True,
                                                                seed=42)
    baseline_perf = np.median([r.sum() for r in swim_baseline_rews])
    
    # print comparison of rewards
    print(f'baseline: {baseline_perf}, sparse: {perf}')

